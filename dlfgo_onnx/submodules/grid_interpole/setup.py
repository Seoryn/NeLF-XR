from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="grid_interpole",
    packages=['grid_interpole'],
    ext_modules=[
        CUDAExtension(
            name="grid_interpole._C",
            sources=[
            "cuda/grid_interpole.cu",
            "cuda/bindings.cpp"],
            )
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)