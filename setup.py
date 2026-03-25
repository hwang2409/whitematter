"""Build the whitematter C++ extension from repo root. Use from root: pip install ."""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import re
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def _read_version():
    """Read version from pyproject.toml to avoid duplication."""
    pyproject = os.path.join(ROOT, "pyproject.toml")
    with open(pyproject, encoding="utf-8") as f:
        for line in f:
            m = re.match(r'^version\s*=\s*"([^"]+)"', line.strip())
            if m:
                return m.group(1)
    raise RuntimeError("Unable to find version in pyproject.toml")

extra_compile_args = ["-std=c++17", "-O3", "-ffast-math"]
extra_link_args = []

if sys.platform == "darwin":
    extra_compile_args += [
        "-mcpu=apple-m1",
        "-Xpreprocessor",
        "-fopenmp",
        "-I/opt/homebrew/opt/libomp/include",
    ]
    extra_link_args += ["-L/opt/homebrew/opt/libomp/lib", "-lomp"]
elif sys.platform == "linux":
    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-fopenmp"]


class BuildExt(build_ext):
    def build_extensions(self):
        import pybind11
        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())
        super().build_extensions()


ext_modules = [
    Extension(
        "whitematter",
        sources=[
            os.path.join(ROOT, "bindings", "whitematter_py.cpp"),
            os.path.join(ROOT, "core", "memory_pool.cpp"),
            os.path.join(ROOT, "core", "autograd.cpp"),
            os.path.join(ROOT, "core", "broadcast.cpp"),
            os.path.join(ROOT, "core", "tensor.cpp"),
            os.path.join(ROOT, "core", "loss.cpp"),
            os.path.join(ROOT, "core", "optimizer.cpp"),
            os.path.join(ROOT, "core", "serialization", "serialize.cpp"),
            os.path.join(ROOT, "core", "dataloader.cpp"),
            os.path.join(ROOT, "core", "model_zoo.cpp"),
            os.path.join(ROOT, "core", "serialization", "onnx_export.cpp"),
            os.path.join(ROOT, "core", "serialization", "onnx_import.cpp"),
            os.path.join(ROOT, "core", "device.cpp"),
            os.path.join(ROOT, "core", "layers", "linear.cpp"),
            os.path.join(ROOT, "core", "layers", "activations.cpp"),
            os.path.join(ROOT, "core", "layers", "conv.cpp"),
            os.path.join(ROOT, "core", "layers", "normalization.cpp"),
            os.path.join(ROOT, "core", "layers", "embedding.cpp"),
            os.path.join(ROOT, "core", "layers", "recurrent.cpp"),
            os.path.join(ROOT, "core", "layers", "attention.cpp"),
            os.path.join(ROOT, "core", "layers", "sequential.cpp"),
            os.path.join(ROOT, "core", "ops", "simd_ops_avx.cpp"),
            os.path.join(ROOT, "core", "ops", "simd_ops_neon.cpp"),
            os.path.join(ROOT, "core", "ops", "simd_ops_fallback.cpp"),
            os.path.join(ROOT, "core", "ops", "matmul_cpu.cpp"),
            os.path.join(ROOT, "core", "ops", "im2col.cpp"),
            os.path.join(ROOT, "core", "ops", "conv_ops.cpp"),
            os.path.join(ROOT, "core", "ops", "augmentation.cpp"),
            os.path.join(ROOT, "core", "metal", "metal_stub.cpp"),
            os.path.join(ROOT, "core", "cuda", "cuda_stub.cpp"),
        ],
        include_dirs=[
            ROOT,
            os.path.join(ROOT, "core"),
            os.path.join(ROOT, "core", "ops"),
            os.path.join(ROOT, "bindings"),
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]


def _read_readme():
    readme = os.path.join(ROOT, "README.md")
    if os.path.isfile(readme):
        with open(readme, encoding="utf-8") as f:
            return f.read()
    return "Lightweight neural network framework with GPU support."


setup(
    name="whitematter",
    version=_read_version(),
    author="Whitematter Contributors",
    description="Lightweight neural network framework with GPU support",
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/hwang2409/whitematter",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    python_requires=">=3.8",
    install_requires=["pybind11>=2.10"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
