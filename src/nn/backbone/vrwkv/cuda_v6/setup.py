import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取当前 setup.py 所在的绝对目录
this_dir = os.path.dirname(os.path.abspath(__file__))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. VRWKV requires CUDA to compile.")

HEAD_SIZE = 64
T_MAX = 8192

ext_modules = [
    CUDAExtension(
        name='wkv6_cuda',
        sources=[
            os.path.join(this_dir, 'wkv6_op.cpp'),
            os.path.join(this_dir, 'wkv6_cuda.cu')
        ],
        define_macros=[
            ('_N_', str(HEAD_SIZE)),  
            ('_T_', str(T_MAX))       
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '-Xptxas', '-O3',
                '--use_fast_math',
                '--extra-device-vectorization',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__',
            ]
        }
    )
]

setup(
    name='wkv6_cuda',
    version='1.0',
    description='CUDA Extension for Vision-RWKV v6',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)
