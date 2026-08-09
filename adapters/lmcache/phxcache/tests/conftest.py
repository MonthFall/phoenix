"""Shared fixtures and hardware detection for phxcache unit tests."""

import os

import pytest
import torch


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cuda: tests that require a real device + phxfs device",
    )


# ---------------------------------------------------------------------------
# Hardware availability detection (session scope, evaluated once)
# ---------------------------------------------------------------------------

def _check_hardware() -> bool:
    """Check that phxcache can be imported and CUDA is available.

    Does NOT instantiate PhxCache (opening/closing the phxfs device at
    import time can interfere with subsequent CUDA operations).  The
    actual instantiation happens in the ``phx_cache`` fixture, which
    skips on failure.
    """
    try:
        import phxcache  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


HAS_HARDWARE = _check_hardware()

skip_no_hardware = pytest.mark.skipif(
    not HAS_HARDWARE,
    reason="no device or phxfs available (set up phxfs + compile phxcache)",
)


# ---------------------------------------------------------------------------
# Aligned CPU buffer helper
# ---------------------------------------------------------------------------

def alloc_aligned_cpu(nbytes: int, alignment: int = 4096) -> "torch.Tensor":
    """Allocate a CPU uint8 tensor whose data_ptr is aligned to *alignment*."""
    import ctypes
    # Over-allocate and slice to aligned offset
    raw = torch.empty(nbytes + alignment, dtype=torch.uint8)
    ptr = raw.data_ptr()
    offset = (alignment - (ptr % alignment)) % alignment
    return raw[offset:offset + nbytes]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def phx_cache():
    """Session-scoped PhxCache instance on CUDA device 0.

    Automatically closed at session teardown.  Skips if PhxCache cannot
    be instantiated (e.g. phxfs device not available).
    """
    try:
        from phxcache import PhxCache
    except ImportError:
        pytest.skip("phxcache not installed")
    try:
        cache = PhxCache(device_id=0)
    except Exception as e:
        pytest.skip(f"PhxCache init failed: {e}")
    yield cache
    cache.close()


@pytest.fixture
def gpu_buffer():
    """Function-scoped 4 MiB device buffer (uint8) for DMA tests."""
    buf = torch.empty(4 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    buf.zero_()
    yield buf
    del buf
    torch.cuda.empty_cache()


@pytest.fixture
def aligned_cpu_buffer():
    """Function-scoped 4 MiB page-aligned CPU buffer for O_DIRECT writes."""
    return alloc_aligned_cpu(4 * 1024 * 1024, alignment=4096)


@pytest.fixture
def temp_file(tmp_path):
    """A temp file path (not yet created) inside the pytest tmp_path."""
    return str(tmp_path / "phxcache_test.bin")


@pytest.fixture
def temp_file_aligned(tmp_path):
    """A temp file path for O_DIRECT tests (will be created with 4K alignment)."""
    return str(tmp_path / "phxcache_aligned.bin")
