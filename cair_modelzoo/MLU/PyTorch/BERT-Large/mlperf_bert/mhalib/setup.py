import torch
import setuptools
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch_mlu.utils.cpp_extension import MLUExtension

import warnings
warnings.filterwarnings("ignore")

setup(
    name='mhalib',
    ext_modules=[
        MLUExtension(
            name='mhalib',
            sources=['mha_mlu_funcs.cpp'],
            extra_compile_args={
                'cxx': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
})

