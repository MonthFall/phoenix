# phxloader_v1

Phoenix DMA loader V1 for GPU Direct Storage.

A pybind11-based Python extension that provides GPU Direct Storage (GDS) loading via the Phoenix DMA engine. It parses safetensors files and loads tensor data directly into GPU memory using DMA, bypassing the CPU.

## Structure

```
phxloader_v1/
├── phxloader_v1/          # Python package
│   ├── __init__.py
│   └── safetensors_parser.py
├── src/                   # C++ extension source
│   ├── bindings.cpp       # pybind11 bindings
│   ├── phx_loader_v1.cpp  # DMA loader implementation
│   └── phx_loader_v1.h    # Header
├── setup.py
├── install.sh
└── README.md
```

## Build & Install

```bash
# Prerequisites: pybind11, CUDA toolkit, libphoenix, liburing
bash install.sh
```

Or manually:

```bash
pip install . --no-deps
```

## Usage

```python
from phxloader_v1 import PhxLoaderV1, parse_safetensor_header

header = parse_safetensor_header("model.safetensors")
loader = PhxLoaderV1(...)
```
