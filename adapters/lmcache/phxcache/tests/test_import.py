"""Import and API-surface tests for phxcache (no hardware required).

These tests verify that the compiled extension can be imported and that
the public classes expose the expected methods and properties.  They do
**not** instantiate PhxCache / PhxFile (which requires a phxfs device).
"""

import os

import pytest


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def test_import_phxcache():
    """``from phxcache import PhxCache, PhxFile`` must succeed."""
    from phxcache import PhxCache, PhxFile  # noqa: F401


def test_module_docstring():
    import phxcache
    assert phxcache.__doc__ is not None
    assert len(phxcache.__doc__) > 0


def test_all_exports():
    """``__all__`` should list PhxCache and PhxFile."""
    from phxcache import __all__
    assert "PhxCache" in __all__
    assert "PhxFile" in __all__


# ---------------------------------------------------------------------------
# PhxCache API surface
# ---------------------------------------------------------------------------

# Methods that PhxCache must expose (bound via pybind11).
PHXCACHE_METHODS = [
    "regmem",
    "deregmem",
    "close",
    "read_batch",
    "write_batch",
]

PHXCACHE_PROPERTIES = [
    "device_id",
    "page_size",
]


@pytest.mark.parametrize("method_name", PHXCACHE_METHODS)
def test_phxcache_has_method(method_name):
    from phxcache import PhxCache
    assert hasattr(PhxCache, method_name), (
        f"PhxCache missing method '{method_name}'"
    )


@pytest.mark.parametrize("prop_name", PHXCACHE_PROPERTIES)
def test_phxcache_has_property(prop_name):
    from phxcache import PhxCache
    assert hasattr(PhxCache, prop_name), (
        f"PhxCache missing property '{prop_name}'"
    )


# ---------------------------------------------------------------------------
# PhxFile API surface
# ---------------------------------------------------------------------------

PHXFILE_METHODS = [
    "read",
    "write",
    "close",
    "__enter__",
    "__exit__",
]


@pytest.mark.parametrize("method_name", PHXFILE_METHODS)
def test_phxfile_has_method(method_name):
    from phxcache import PhxFile
    assert hasattr(PhxFile, method_name), (
        f"PhxFile missing method '{method_name}'"
    )


# ---------------------------------------------------------------------------
# Constructor error handling
# ---------------------------------------------------------------------------
# NOTE: PhxCache(device_id=invalid) calls phxfs_find_dev which internally
# calls cuDeviceGetPCIBusId — an invalid device ordinal corrupts the CUDA
# context.  Therefore, the invalid-device test lives in test_phxcache.py
# (hardware-dependent) where we can reset the CUDA context afterwards.


def test_phxfile_nonexistent_path_raises(tmp_path):
    """Opening a non-existent file should raise."""
    from phxcache import PhxFile
    # PhxFile's constructor calls ::open() before touching the cache, so a
    # bad path raises even with a dummy object that only provides
    # ``device_id``.
    class _FakeCache:
        device_id = 0

    with pytest.raises(Exception):
        PhxFile(_FakeCache(), str(tmp_path / "does_not_exist.bin"), os.O_RDONLY)
