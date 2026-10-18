# Copyright (c) Shanghai AI Lab. All rights reserved.
from typing import Sequence
import math
import os
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp

logger = logging.getLogger(__name__)

T_MAX = 8192
HEAD_SIZE = 64


import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_path = os.path.join(current_dir, 'cuda_v6')
if cuda_path not in sys.path:
    sys.path.append(cuda_path)

try:
    import wkv6_cuda
except ImportError:
    wkv6_cuda = None



def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def resize_pos_embed(pos_embed, src_shape, dst_shape, mode='bicubic', num_extra_tokens=0):
    """纯 PyTorch 实现的位置编码缩放"""
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
    b, l, c = pos_embed.shape
    pos_embed_weight = pos_embed[:, num_extra_tokens:, :]
    pos_embed_weight = pos_embed_weight.reshape(
        1, src_shape[0], src_shape[1], c).permute(0, 3, 1, 2)
    pos_embed_weight = F.interpolate(
        pos_embed_weight, size=dst_shape, mode=mode, align_corners=False)
    pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
    if num_extra_tokens > 0:
        extra_tokens = pos_embed[:, :num_extra_tokens, :]
        pos_embed_weight = torch.cat((extra_tokens, pos_embed_weight), dim=1)
    return pos_embed_weight


class PatchEmbed(nn.Module):
    """纯 PyTorch 实现的 PatchEmbed"""

    def __init__(self, in_channels=3, input_size=224, embed_dims=768, conv_type='Conv2d', kernel_size=16, stride=16,
                 bias=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, bias=bias)
        img_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.init_out_size = (img_size[0] // kernel_size, img_size[1] // kernel_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        out_size = (x.shape[2], x.shape[3])  # H, W after conv
        x = x.flatten(2).transpose(1, 2)
        return x, out_size


# =========================================================================
# 3. VRWKV 核心算子逻辑
# =========================================================================
class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H

            # 🚀 核心修复：强制转为 float32 (Float) 并保持内存连续
            r = r.float().contiguous()
            k = k.float().contiguous()
            v = v.float().contiguous()
            w = w.float().contiguous()
            u = u.float().contiguous()

            # 注意：这里的 w 已经是 float32 了，可以直接算 exp
            ew = (-torch.exp(w)).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)

            y = torch.empty((B, T, C), device=r.device, dtype=torch.float32, memory_format=torch.contiguous_format)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H

            # 🚀 核心修复：梯度反传时，也必须强制转为 float32
            gy = gy.float().contiguous()

            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32,
                             memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32,
                             memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32,
                             memory_format=torch.contiguous_format)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32,
                             memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32,
                             memory_format=torch.contiguous_format)

            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu)


def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    if wkv6_cuda is None:
        raise RuntimeError(
            "wkv6_cuda is not available. Build the V-RWKV CUDA extension in "
            "src/nn/backbone/vrwkv/cuda_v6 before running VRWKV blocks."
        )
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)


def q_shift_multihead(input, shift_pixel=1, head_dim=HEAD_SIZE,
                      patch_resolution=None, with_cls_token=False):
    B, N, C = input.shape  # [B, N, C]
    assert C % head_dim == 0
    assert head_dim % 4 == 0
    if with_cls_token:
        cls_tokens = input[:, [-1], :]
        input = input[:, :-1, :]
    input = input.transpose(1, 2).reshape(
        B, -1, head_dim, patch_resolution[0], patch_resolution[1])  # [B, n_head, head_dim , H, W]
    B, _, _, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, :, 0:int(head_dim * 1 / 4), :, shift_pixel:W] = \
        input[:, :, 0:int(head_dim * 1 / 4), :, 0:W - shift_pixel]
    output[:, :, int(head_dim / 4):int(head_dim / 2), :, 0:W - shift_pixel] = \
        input[:, :, int(head_dim / 4):int(head_dim / 2), :, shift_pixel:W]
    output[:, :, int(head_dim / 2):int(head_dim / 4 * 3), shift_pixel:H, :] = \
        input[:, :, int(head_dim / 2):int(head_dim / 4 * 3), 0:H - shift_pixel, :]
    output[:, :, int(head_dim * 3 / 4):int(head_dim), 0:H - shift_pixel, :] = \
        input[:, :, int(head_dim * 3 / 4):int(head_dim), shift_pixel:H, :]
    if with_cls_token:
        output = output.reshape(B, C, N - 1).transpose(1, 2)
        output = torch.cat((output, cls_tokens), dim=1)
    else:
        output = output.reshape(B, C, N).transpose(1, 2)
    return output


class VRWKV_SpatialMix_V6(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, init_mode='fancy', key_norm=False, with_cls_token=False,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd

        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        assert self.head_size == HEAD_SIZE
        self.device = None
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        self.shift_func = eval(shift_mode)

        self.key = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.value = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        self.gate = nn.Linear(self.n_embd, self.attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(self.attn_sz, n_embd, bias=False)

        self.ln_x = nn.GroupNorm(self.n_head, self.attn_sz, eps=1e-5)
        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    ddd[0, 0, i] = i / self.n_embd

                # fancy time_mix
                self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
                self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

                TIME_MIX_EXTRA_DIM = 32  # generate TIME_MIX for w,k,v,r,g
                self.time_maa_w1 = nn.Parameter(torch.zeros(self.n_embd, TIME_MIX_EXTRA_DIM * 5).uniform_(-1e-4, 1e-4))
                self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, self.n_embd).uniform_(-1e-4, 1e-4))

                # fancy time_decay
                decay_speed = torch.ones(self.attn_sz)
                for n in range(self.attn_sz):
                    decay_speed[n] = -6 + 5 * (n / (self.attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, self.attn_sz))

                TIME_DECAY_EXTRA_DIM = 64
                self.time_decay_w1 = nn.Parameter(torch.zeros(self.n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
                self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, self.attn_sz).uniform_(-1e-4, 1e-4))

                tmp = torch.zeros(self.attn_sz)
                for n in range(self.attn_sz):
                    zigzag = ((n + 1) % 3 - 1) * 0.1
                    tmp[n] = ratio_0_to_1 * (1 - (n / (self.attn_sz - 1))) + zigzag

                self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution,  decay_delta=None):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()

        xx = self.shift_func(x, self.shift_pixel, patch_resolution=patch_resolution,
                             with_cls_token=self.with_cls_token) - x   # 变化量

        # 当前 token + 时序差分”生成的 5路动态参数
        xxx = x + xx * self.time_maa_x  # [B, T, C]  x + (变化量 × 可学习权重)
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)     # [5, B*T, TIME_MIX_EXTRA_DIM]
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)         # [5, B, T, C]
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)  # 混合系数



        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))  # 总门控

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        # [B, T, C]
        w = self.time_decay + ww  # 衰减因子

        if decay_delta is not None:
            w = w + decay_delta.to(device=w.device, dtype=w.dtype)

        return r, k, v, g, w

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x, patch_resolution=None, decay_delta=None):
        def _inner_forward(x, decay_delta=None):
            B, T, C = x.size()
            self.device = x.device

            r, k, v, g, w = self.jit_func(x, patch_resolution, decay_delta)
            x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w, u=self.time_faaaa)
            if self.key_norm is not None:
                x = self.key_norm(x)
            return self.jit_func_2(x, g)

        if self.with_cp and x.requires_grad:
            if decay_delta is None:
                x = cp.checkpoint(_inner_forward, x)
            else:
                x = cp.checkpoint(_inner_forward, x, decay_delta)
        else:
            x = _inner_forward(x, decay_delta)
        return x

    # def forward(self, x, patch_resolution=None):
    #     def _inner_forward(x):
    #         B, T, C = x.size()
    #         self.device = x.device
    #
    #         r, k, v, g, w = self.jit_func(x, patch_resolution)
    #         x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w, u=self.time_faaaa)
    #         if self.key_norm is not None:
    #             x = self.key_norm(x)
    #         return self.jit_func_2(x, g)
    #
    #     if self.with_cp and x.requires_grad:
    #         x = cp.checkpoint(_inner_forward, x)
    #     else:
    #         x = _inner_forward(x)
    #     return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, hidden_rate=4, init_mode='fancy', key_norm=False,
                 with_cls_token=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd
        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        assert self.head_size == HEAD_SIZE
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        self.shift_func = eval(shift_mode)

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():  # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            xx = self.shift_func(x, self.shift_pixel, patch_resolution=patch_resolution,
                                 with_cls_token=self.with_cls_token)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv  #
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                 shift_pixel=1, drop_path=0., hidden_rate=4, init_mode='fancy',
                 init_values=None, post_norm=False, key_norm=False, with_cls_token=False,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix_V6(n_embd, n_head, n_layer, layer_id, shift_mode,
                                       shift_pixel, init_mode, key_norm=key_norm,
                                       with_cls_token=with_cls_token)

        self.ffn = VRWKV_ChannelMix(n_embd, n_head, n_layer, layer_id, shift_mode,
                                    shift_pixel, hidden_rate, init_mode, key_norm=key_norm,
                                    with_cls_token=with_cls_token)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    # def forward(self, x, patch_resolution=None,decay_delta=None):
    #     def _inner_forward(x):
    #         if self.layer_id == 0:
    #             x = self.ln0(x)
    #         if self.post_norm:
    #             if self.layer_scale:
    #                 x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution, decay_delta)))
    #                 x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
    #             else:
    #                 x = x + self.drop_path(self.ln1(self.att(x, patch_resolution, decay_delta)))
    #                 x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
    #         else:  # 前置归一化
    #             if self.layer_scale:
    #                 x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution, decay_delta))
    #                 x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
    #             else:
    #                 x = x + self.drop_path(self.att(self.ln1(x), patch_resolution, decay_delta))
    #                 x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
    #         return x
    #
    #     if self.with_cp and x.requires_grad:
    #         x = cp.checkpoint(_inner_forward, x)
    #     else:
    #         x = _inner_forward(x)
    #     return x

    def forward(self, x, patch_resolution=None, decay_delta=None):
        def _inner_forward(x, decay_delta=None):
            if self.layer_id == 0:
                x = self.ln0(x)

            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution, decay_delta)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution, decay_delta)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution, decay_delta))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution, decay_delta))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))

            return x

        if self.with_cp and x.requires_grad:
            if decay_delta is None:
                x = cp.checkpoint(_inner_forward, x)
            else:
                x = cp.checkpoint(_inner_forward, x, decay_delta)
        else:
            x = _inner_forward(x, decay_delta)
        return x