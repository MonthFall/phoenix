"""Hardware-dependent tests for PhxCache (require device + phxfs device).

All tests in this module are decorated with ``@skip_no_hardware`` and will
be skipped automatically when no phxfs device / device is available.
"""

import os

import pytest
import torch

from .conftest import skip_no_hardware


# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------

@skip_no_hardware
def test_cache_init_and_device_id(phx_cache):
    """PhxCache should report a non-negative device_id."""
    assert phx_cache.device_id >= 0


@skip_no_hardware
def test_cache_page_size_positive(phx_cache):
    """page_size must be a positive integer (e.g. 65536 for NVIDIA)."""
    ps = phx_cache.page_size
    assert isinstance(ps, int)
    assert ps > 0
    # Common page sizes are powers of two ≥ 4096
    assert (ps & (ps - 1)) == 0, "page_size should be a power of two"
    assert ps >= 4096


# ---------------------------------------------------------------------------
# regmem / deregmem lifecycle
# ---------------------------------------------------------------------------

@skip_no_hardware
def test_regmem_returns_valid_addr(phx_cache, gpu_buffer):
    """regmem should return a non-zero target_addr for a device buffer."""
    ptr = gpu_buffer.data_ptr()
    size = gpu_buffer.nbytes
    target = phx_cache.regmem(ptr, size)
    assert target != 0
    phx_cache.deregmem(ptr, size)


@skip_no_hardware
def test_deregmem_idempotent(phx_cache, gpu_buffer):
    """Deregistering an unregistered region should be a no-op (no raise)."""
    ptr = gpu_buffer.data_ptr()
    size = gpu_buffer.nbytes
    # dereg before reg — should not raise
    phx_cache.deregmem(ptr, size)
    # reg then dereg twice — second should be a no-op
    phx_cache.regmem(ptr, size)
    phx_cache.deregmem(ptr, size)
    phx_cache.deregmem(ptr, size)


@skip_no_hardware
def test_regmem_multiple_regions(phx_cache):
    """Registering two distinct device buffers should succeed independently."""
    buf1 = torch.empty(2 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    buf2 = torch.empty(2 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    t1 = phx_cache.regmem(buf1.data_ptr(), buf1.nbytes)
    t2 = phx_cache.regmem(buf2.data_ptr(), buf2.nbytes)
    assert t1 != 0
    assert t2 != 0
    phx_cache.deregmem(buf1.data_ptr(), buf1.nbytes)
    phx_cache.deregmem(buf2.data_ptr(), buf2.nbytes)
    del buf1, buf2
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# write_batch / read_batch round-trip
# ---------------------------------------------------------------------------

@skip_no_hardware
def test_write_batch_and_read_batch_round_trip(
    phx_cache, gpu_buffer, aligned_cpu_buffer, temp_file,
):
    """Write CPU data to a file via write_batch, then DMA-read it back to device.

    Uses O_DIRECT for both write and read.  The CPU buffer is page-aligned
    (from the ``aligned_cpu_buffer`` fixture) to satisfy O_DIRECT alignment
    requirements.
    """
    page_size = phx_cache.page_size
    nbytes = page_size  # one page

    # Prepare aligned CPU source data — use the aligned buffer directly
    # (do NOT clone, as clone() may return a non-aligned allocation)
    cpu_data = aligned_cpu_buffer[:nbytes]
    cpu_data.copy_(torch.arange(nbytes, dtype=torch.uint8))
    expected = torch.arange(nbytes, dtype=torch.uint8)

    # --- Write phase: CPU buffer → file (O_DIRECT) ---
    fd = os.open(temp_file, os.O_CREAT | os.O_WRONLY | os.O_DIRECT)
    try:
        phx_cache.regmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
        write_reqs = [(
            fd,                          # fd
            cpu_data.data_ptr(),         # buf_ptr (CPU, aligned)
            0,                           # buf_offset
            nbytes,                      # nbytes
            0,                           # f_offset
        )]
        results = phx_cache.write_batch(write_reqs)
        assert len(results) == 1
        assert results[0] == nbytes, (
            f"write_batch returned {results[0]}, expected {nbytes}"
        )
    finally:
        os.close(fd)

    # --- Read phase: file → device buffer (DMA, O_DIRECT) ---
    gpu_buffer.zero_()
    fd = os.open(temp_file, os.O_RDONLY | os.O_DIRECT)
    try:
        read_reqs = [(
            fd,          # fd
            0,           # buf_offset in device buf
            nbytes,      # nbytes
            0,           # f_offset
        )]
        results = phx_cache.read_batch(gpu_buffer.data_ptr(), read_reqs)
        assert len(results) == 1
        assert results[0] == nbytes, (
            f"read_batch returned {results[0]}, expected {nbytes}"
        )
    finally:
        os.close(fd)
        phx_cache.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)

    # Verify data integrity
    got = gpu_buffer[:nbytes].cpu()
    assert torch.equal(got, expected), "Data mismatch after write→read round-trip"


@skip_no_hardware
def test_read_batch_empty(phx_cache):
    """read_batch with an empty request list should return an empty list."""
    result = phx_cache.read_batch(0, [])
    assert result == []


@skip_no_hardware
def test_write_batch_empty(phx_cache):
    """write_batch with an empty request list should return an empty list."""
    result = phx_cache.write_batch([])
    assert result == []


# ---------------------------------------------------------------------------
# close semantics
# ---------------------------------------------------------------------------

@skip_no_hardware
def test_close_idempotent():
    """Calling close() twice should not raise."""
    from phxcache import PhxCache
    cache = PhxCache(device_id=0)
    cache.close()
    cache.close()  # second close — should be a no-op


# ---------------------------------------------------------------------------
# Invalid device handling (must run LAST — corrupts CUDA context)
# ---------------------------------------------------------------------------

@skip_no_hardware
def test_invalid_device_raises():
    """An out-of-range device_id should raise.

    WARNING: phxfs_find_dev(invalid_id) calls cuDeviceGetPCIBusId with an
    invalid ordinal, which sets the CUDA error sticky bit.  We reset the
    device context afterwards so subsequent tests are not affected.

    This test is deliberately placed at the end of the module.
    """
    from phxcache import PhxCache
    with pytest.raises(Exception):
        PhxCache(device_id=99999)

    # Reset CUDA context — clear the sticky error from invalid device access
    torch.cuda.synchronize()
    # Force a harmless CUDA call on device 0 to reset the context
    try:
        _ = torch.empty(1, device="cuda:0")
    except Exception:
        # If device 0 is still broken, try synchronizing again
        torch.cuda.synchronize()
