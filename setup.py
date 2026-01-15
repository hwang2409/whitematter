from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import os

# pybind11 include path will be added by build_ext

# Detect platform-specific flags
extra_compile_args = ['-std=c++17', '-O3', '-ffast-math']
extra_link_args = []

if sys.platform == 'darwin':
    # macOS
    extra_compile_args += ['-mcpu=apple-m1', '-Xpreprocessor', '-fopenmp',
                           '-I/opt/homebrew/opt/libomp/include']
    extra_link_args += ['-L/opt/homebrew/opt/libomp/lib', '-lomp']
elif sys.platform == 'linux':
    extra_compile_args += ['-fopenmp', '-march=native']
    extra_link_args += ['-fopenmp']

class BuildExt(build_ext):
    """Custom build_ext to add pybind11 include path."""
    def build_extensions(self):
        import pybind11
        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())
        super().build_extensions()

ext_modules = [
    Extension(
        'whitematter',
        sources=[
            'whitematter_py.cpp',
            'tensor.cpp',
            'layer.cpp',
            'loss.cpp',
            'optimizer.cpp',
            'serialize.cpp',
        ],
        include_dirs=['.'],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='whitematter',
    version='0.1.0',
    author='Henry',
    description='Whitematter ML inference module',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    python_requires='>=3.8',
    install_requires=[
        'pybind11>=2.10',
    ],
)
