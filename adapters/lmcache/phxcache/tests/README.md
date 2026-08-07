# phxcache Unit Tests

## Overview

phxcache is the Phoenix DMA adapter for LMCache. It wraps the `phxfs_read_batch` / `phxfs_write_batch` (batch I/O) and `phxfs_read` / `phxfs_write` (single I/O) APIs to provide direct storage read/write capabilities via the phxfs device.

This test suite covers **30 tests** across two tiers:

| Tier | File | Tests | Hardware | Description |
|------|------|-------|----------|-------------|
| Tier 2 | `test_import.py` | 16 | Compiled extension, no device | Import and API-surface validation |
| Tier 3 | `test_phxcache.py` | 11 | Device + phxfs | PhxCache core functionality |
| Tier 3 | `test_phxfile.py` | 4 | Device + phxfs | PhxFile single-file I/O |

## Requirements

| Dependency | Requirement |
|------------|-------------|
| Python environment | phxcache extension installed (`bash install.sh`) |
| libphoenix.so | `/usr/local/lib/libphoenix.so` (latest version with 6-param `phxfs_read`) |
| Device | Required for hardware tests (e.g. Nvidia GPU、AMD GPU) |
| phxfs device | `/dev/phxfs_dev0` (required for hardware tests) |
| pytest | ≥ 7.0 |

## How to Run

```bash
# Run all tests (hardware tests auto-skip if no device)
bash run_tests.sh

# Run import tests only (no device needed)
bash run_tests.sh --python

# Run hardware tests only
bash run_tests.sh --hardware

# Verbose output
bash run_tests.sh --verbose
```

Manual invocation:

```bash
# All tests
python -P -m pytest tests/ -v --import-mode=importlib

# Import tests only
python -P -m pytest tests/test_import.py -v --import-mode=importlib

# Hardware tests only
python -P -m pytest tests/test_phxcache.py tests/test_phxfile.py -v --import-mode=importlib
```

> **Note:** The `-P` flag (do not add cwd to `sys.path`) and `--import-mode=importlib` (avoid `conftest.py` conflicts) are required to prevent the source directory from shadowing the installed package.

## Test Files

### `conftest.py` — Shared Fixtures

Provides infrastructure for all test modules:

| Component | Purpose |
|-----------|---------|
| `_check_hardware()` | Checks if `phxcache` is importable and a device is available (does not instantiate PhxCache) |
| `skip_no_hardware` | pytest marker that auto-skips when no device/phxfs is available |
| `phx_cache` | Session-scoped `PhxCache(device_id=0)` instance; skips if init fails |
| `gpu_buffer` | Function-scoped 4 MiB device buffer (`torch.empty(4M, uint8, cuda:0)`) |
| `aligned_cpu_buffer` | Function-scoped 4 MiB 4K-aligned CPU buffer (for O_DIRECT writes) |
| `alloc_aligned_cpu()` | Helper: allocate a 4K-aligned CPU tensor |
| `temp_file` | Temp file path (under pytest `tmp_path`) |

### `test_import.py` — Import & API Surface (16 tests, no hardware)

**Objective**: Verify the compiled extension imports correctly and all public classes expose the expected methods and properties. No objects are instantiated.

**Method**: Pure `import` + `hasattr` assertions.

| Test | Verifies |
|------|----------|
| `test_import_phxcache` | `from phxcache import PhxCache, PhxFile` succeeds |
| `test_module_docstring` | Module has a non-empty docstring |
| `test_all_exports` | `__all__` contains `PhxCache` and `PhxFile` |
| `test_phxcache_has_method` ×5 (parametrized) | `PhxCache` has `regmem`, `deregmem`, `close`, `read_batch`, `write_batch` |
| `test_phxcache_has_property` ×2 (parametrized) | `PhxCache` has `device_id`, `page_size` properties |
| `test_phxfile_has_method` ×5 (parametrized) | `PhxFile` has `read`, `write`, `close`, `__enter__`, `__exit__` |
| `test_phxfile_nonexistent_path_raises` | Opening a non-existent file path raises an exception |

### `test_phxcache.py` — PhxCache Hardware Tests (11 tests, device + phxfs required)

**Objective**: Verify `PhxCache` construction, properties, regmem/deregmem lifecycle, batch read/write round-trip data integrity, and close semantics.

**Method**: Uses `phx_cache` (session-scoped instance) and `gpu_buffer` / `aligned_cpu_buffer` fixtures. Batch read/write tests open files with O_DIRECT, write `torch.arange` data, DMA-read back to device, and verify with `torch.equal`.

| Test | Data | Verifies |
|------|------|----------|
| `test_cache_init_and_device_id` | — | `device_id >= 0` |
| `test_cache_page_size_positive` | — | `page_size` is a power of two, ≥ 4096 |
| `test_regmem_returns_valid_addr` | 4 MiB device buffer | `regmem` returns a non-zero address |
| `test_deregmem_idempotent` | 4 MiB device buffer | Dereg before reg and double-dereg do not raise |
| `test_regmem_multiple_regions` | 2× 2 MiB device buffers | Both independent registrations return non-zero addresses |
| `test_write_batch_and_read_batch_round_trip` | `torch.arange(page_size, uint8)` written to aligned CPU buffer → O_DIRECT write to file → DMA read to device | `write_batch` returns `nbytes`, `read_batch` returns `nbytes`, `torch.equal` data match |
| `test_read_batch_empty` | Empty request list | Returns empty list |
| `test_write_batch_empty` | Empty request list | Returns empty list |
| `test_close_idempotent` | — | `close()` twice does not raise |
| `test_invalid_device_raises` | `device_id=99999` | Raises exception + device context reset (**must run last in module**) |

### `test_phxfile.py` — PhxFile Hardware Tests (4 tests, device + phxfs required)

**Objective**: Verify `PhxFile` context manager, single-I/O read/write, and read-write round-trip data integrity.

**Method**: Uses `phx_cache` to register a device buffer, opens files with O_DIRECT via `PhxFile`, performs DMA read/write with `phxfs_read` / `phxfs_write`, and verifies data with `torch.equal`.

| Test | Data | Verifies |
|------|------|----------|
| `test_context_manager_closes_fd` | Empty file (`ftruncate(page_size)`) | `with PhxFile(...)` context manager works correctly |
| `test_phxfile_read` | `torch.arange(page_size, uint8)` written via `write_batch` | `PhxFile.read` returns `page_size`, data matches |
| `test_phxfile_write` | `torch.arange(page_size, uint8)` device buffer | `PhxFile.write` returns `page_size`, on-disk data matches |
| `test_phxfile_read_write_round_trip` | `torch.arange(page_size, uint8)` | write → read round-trip, data matches |
