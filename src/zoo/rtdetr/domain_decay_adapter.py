import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainDecayAdapter(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=80, max_delta=0.5, proto_tau=0.07):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_delta = max_delta
        self.proto_tau = proto_tau

        self.slot_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.proto_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.teacher_proj = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.fuse_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.scale = nn.Parameter(torch.zeros(1))

    def _zero_projected_signal(self, memory, projector, input_dim):
        dummy = memory.new_zeros(*memory.shape[:-1], input_dim)
        return projector(dummy) * 0.0

    def _align_masks(self, masks, patch_resolution, target_tokens):
        if masks is None or masks.size(-1) == target_tokens:
            return masks

        b, s, t = masks.shape
        h, w = patch_resolution
        src_hw = int(math.sqrt(t))

        if src_hw * src_hw == t:
            masks_2d = masks.reshape(b * s, 1, src_hw, src_hw)
            masks_2d = F.interpolate(masks_2d, size=(h, w), mode="bilinear", align_corners=False)
            return masks_2d.reshape(b, s, h * w)

        return F.interpolate(masks, size=target_tokens, mode="linear", align_corners=False)

    def _slot_signal(self, memory, context, patch_resolution):
        slots = context.get("slots", None)
        masks = context.get("slot_masks", None)

        b, t, _ = memory.shape
        if slots is None or masks is None:
            return self._zero_projected_signal(memory, self.slot_proj, self.hidden_dim + 1)

        masks = self._align_masks(masks, patch_resolution, t)
        masks = masks.to(dtype=memory.dtype, device=memory.device)
        slots = slots.to(dtype=memory.dtype, device=memory.device)

        masks = masks / (masks.sum(dim=1, keepdim=True) + 1e-6)
        slot_context = torch.einsum("bst,bsd->btd", masks, slots)

        entropy = -(masks.clamp_min(1e-6) * masks.clamp_min(1e-6).log()).sum(dim=1, keepdim=True)
        entropy = entropy.transpose(1, 2)

        return self.slot_proj(torch.cat([slot_context, entropy], dim=-1))

    def _prototype_signal(self, memory, context):
        prototypes = context.get("prototypes", None)

        if prototypes is None or prototypes.numel() == 0:
            return self._zero_projected_signal(memory, self.proto_proj, self.hidden_dim + 1)

        prototypes = prototypes.to(dtype=memory.dtype, device=memory.device)
        if prototypes.abs().sum() == 0:
            return self._zero_projected_signal(memory, self.proto_proj, self.hidden_dim + 1)

        mem_norm = F.normalize(memory, dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)

        sim = torch.einsum("btd,cd->btc", mem_norm, proto_norm)
        prob = F.softmax(sim / self.proto_tau, dim=-1)

        weighted_proto = torch.einsum("btc,cd->btd", prob, proto_norm)
        proto_dist = mem_norm - weighted_proto
        proto_uncertainty = 1.0 - prob.max(dim=-1, keepdim=True).values

        return self.proto_proj(torch.cat([proto_dist, proto_uncertainty], dim=-1))

    def _teacher_signal(self, memory, context):
        teacher_logits = context.get("teacher_logits", None)

        b, t, _ = memory.shape
        if teacher_logits is None:
            return self._zero_projected_signal(memory, self.teacher_proj, self.num_classes)

        teacher_logits = teacher_logits.to(dtype=memory.dtype, device=memory.device)

        probs = teacher_logits.sigmoid()
        entropy = -(probs * (probs + 1e-6).log() + (1 - probs) * (1 - probs + 1e-6).log())
        entropy = entropy.mean(dim=1)

        if entropy.size(-1) > self.num_classes:
            entropy = entropy[..., :self.num_classes]
        elif entropy.size(-1) < self.num_classes:
            entropy = F.pad(entropy, (0, self.num_classes - entropy.size(-1)))

        entropy = entropy.unsqueeze(1).expand(b, t, self.num_classes)
        return self.teacher_proj(entropy)

    def forward(self, memory, patch_resolution, context=None):
        if context is None:
            return None

        slot_signal = self._slot_signal(memory, context, patch_resolution)
        proto_signal = self._prototype_signal(memory, context)
        teacher_signal = self._teacher_signal(memory, context)

        fused = torch.cat([slot_signal, proto_signal, teacher_signal], dim=-1)
        delta = self.fuse(fused) * self.fuse_gate(fused)

        return self.scale * self.max_delta * torch.tanh(delta)
