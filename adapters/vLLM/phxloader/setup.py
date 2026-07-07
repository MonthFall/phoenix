"""Build script for phxloader pybind11 extension."""

import os
import subprocess
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Try to find libphoenix and liburing
def find_library(name, search_paths=None):
    """Try to find a library using pkg-config or search paths."""
    # Try pkg-config first
    try:
        result = subprocess.run(
            ["pkg-config", "--libs", name],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try common paths
    if search_paths is None:
        search_paths = [
            "/usr/lib", "/usr/lib64", "/usr/local/lib", "/usr/local/lib64",
        ]

    for path in search_paths:
        for ext in [".so", ".a"]:
            if os.path.exists(os.path.join(path, f"lib{name}{ext}")):
                return f"-L{path} -l{name}"
    return None


# Find Phoenix project root
# adapters/vLLM/phxloader/setup.py → parent×4 = Phoenix/
PHOENIX_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LIBPHOENIX_DIR = PHOENIX_ROOT / "libphoenix"
LIBPHOENIX_INCLUDE = LIBPHOENIX_DIR / "include"
LIBPHOENIX_LIB = PHOENIX_ROOT / "build"

# CUDA
cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

ext_modules = [
    Pybind11Extension(
        "phxloader._phxloader",
        sources=[
            "src/phx_loader.cpp",
            "src/bindings.cpp",
        ],
        include_dirs=[
            str(LIBPHOENIX_INCLUDE),
            str(Path(__file__).resolve().parent / "src"),
            os.path.join(cuda_home, "include"),
        ],
        library_dirs=[
            str(LIBPHOENIX_LIB),
            os.path.join(cuda_home, "lib64"),
            os.path.join(cuda_home, "lib"),
        ],
        libraries=["phoenix", "uring", "cuda", "cudart"],
        extra_compile_args=["-std=c++17", "-O2", "-fPIC"],
        extra_link_args=["-std=c++17"],
    ),
]

setup(
    name="phxloader",
    version="2.2.0",
    description="Phoenix DMA loader for GPU Direct Storage",
    packages=["phxloader"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[],
    python_requires=">=3.10",
)
