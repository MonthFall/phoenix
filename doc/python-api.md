# Python API — `phxfs`

The `python/phxfs` package wraps `libphoenix` (via `ctypes` against `libphoenix.so`) so Python applications can use Phoenix without writing C++.

## Features

- `ctypes` wrapper around the `libphoenix` shared library.
- A class-oriented interface for Phoenix file operations.

## Installation

```bash
cd python
python setup.py install
```

## Usage

1. Build `libphoenix.so` (see [install.md](install.md)).
2. Deploy `libphoenix.so` to `/usr/lib64/` (or any library search path).
3. Verify the import:

```bash
python -c "import phxfs; print(phxfs.__file__)"
```

## License

Apache-2.0
