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
    # Use portable flags for containers, -march=native can cause issues
    extra_compile_args += ['-fopenmp']
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
            os.path.join('..', 'bindings', 'whitematter_py.cpp'),
            os.path.join('..', 'core', 'tensor.cpp'),
            os.path.join('..', 'core', 'layer.cpp'),
            os.path.join('..', 'core', 'loss.cpp'),
            os.path.join('..', 'core', 'optimizer.cpp'),
            os.path.join('..', 'core', 'serialize.cpp'),
        ],
        include_dirs=[
            '.',
            os.path.join('..', 'core'),
            os.path.join('..', 'bindings'),
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

def _read_readme():
    root = os.path.dirname(os.path.abspath(__file__))
    readme = os.path.join(root, '..', 'README.md')
    if os.path.isfile(readme):
        with open(readme, encoding='utf-8') as f:
            return f.read()
    return 'Lightweight neural network framework with GPU support.'

setup(
    name='whitematter',
    version='0.1.0',
    author='Whitematter Contributors',
    description='Lightweight neural network framework with GPU support',
    long_description=_read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/hwang2409/whitematter',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    python_requires='>=3.8',
    install_requires=[
        'pybind11>=2.10',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
