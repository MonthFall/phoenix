# phxloader_v2

Phoenix DMA loader V2 for GPU Direct Storage.

Next-generation DMA loader (C++ extension under development). The Python package structure is in place; the pybind11 extension will be added when the V2 implementation is ready.

## Structure

```
phxloader_v2/
├── phxloader_v2/          # Python package
│   └── __init__.py
├── src/                   # C++ extension source (TODO)
├── setup.py
├── install.sh
└── README.md
```

## Status

**Work in progress.** The V2 C++ extension has not been implemented yet. `setup.py` references `src/phx_loader_v2.cpp` and `src/bindings.cpp` which do not exist yet. Install will fail until those are added.

## Planned Usage

```python
from phxloader_v2 import PhxLoaderV2, parse_safetensor_header
```
