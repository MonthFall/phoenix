"""Hardware-dependent tests for PhxFile (require device + phxfs device).

All tests use the ``phx_cache`` session fixture and are skipped when no
hardware is available.

PhxFile.read and PhxFile.write both pass ``dev_`` (the phxfs device ID ≥ 0)
to phxfs_read/phxfs_write.  This means the buffer **must** be a device address
registered via ``regmem`` — not a plain CPU address.
(Use ``PhxCache.write_batch`` with ``device_id = -1`` for CPU buffers.)
"""

import os

import pytest
import torch

from .conftest import skip_no_hardware


@skip_no_hardware
def test_context_manager_closes_fd(phx_cache, temp_file):
    """Using PhxFile as a context manager should close the fd on exit."""
    from phxcache import PhxFile

    page_size = phx_cache.page_size
    with open(temp_file, "wb") as f:
        f.truncate(page_size)

    with PhxFile(phx_cache, temp_file, os.O_RDONLY | os.O_DIRECT) as pf:
        assert pf is not None

    pf.close()


@skip_no_hardware
def test_phxfile_read(phx_cache, gpu_buffer, temp_file):
    """Write data to file via write_batch, then DMA-read via PhxFile.read.

    The file must be created with O_DIRECT (via write_batch) because
    phxfs_read requires O_DIRECT on the fd.  PhxFile.read reads into a
    registered device buffer.
    """
    from phxcache import PhxFile
    from .conftest import alloc_aligned_cpu

    page_size = phx_cache.page_size

    # Create file with write_batch (O_DIRECT + aligned CPU buffer)
    cpu_data = alloc_aligned_cpu(page_size)
    cpu_data.copy_(torch.arange(page_size, dtype=torch.uint8))
    expected = torch.arange(page_size, dtype=torch.uint8)

    fd = os.open(temp_file, os.O_CREAT | os.O_WRONLY | os.O_DIRECT)
    try:
        phx_cache.write_batch([(
            fd, cpu_data.data_ptr(), 0, page_size, 0,
        )])
    finally:
        os.close(fd)

    # DMA-read via PhxFile into registered device buffer
    phx_cache.regmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
    try:
        gpu_buffer.zero_()
        with PhxFile(phx_cache, temp_file, os.O_RDONLY | os.O_DIRECT) as pf:
            ret = pf.read(
                gpu_buffer.data_ptr(),
                0,          # buf_offset
                page_size,  # nbyte
                0,          # f_offset
            )
        assert ret == page_size

        got = gpu_buffer[:page_size].cpu()
        assert torch.equal(got, expected), "PhxFile.read data mismatch"
    finally:
        phx_cache.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)


@skip_no_hardware
def test_phxfile_write(phx_cache, gpu_buffer, temp_file):
    """Write device data to file via PhxFile.write, then verify on disk.

    PhxFile.write passes dev_ to phxfs_write, so the buffer must be a
    registered device address.
    """
    from phxcache import PhxFile

    page_size = phx_cache.page_size

    phx_cache.regmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
    try:
        gpu_data = gpu_buffer[:page_size]
        gpu_data.copy_(torch.arange(page_size, dtype=torch.uint8))

        with PhxFile(phx_cache, temp_file, os.O_CREAT | os.O_WRONLY | os.O_DIRECT) as pf:
            ret = pf.write(
                gpu_buffer.data_ptr(),
                0,          # buf_offset
                page_size,  # nbyte
                0,          # f_offset
            )
        assert ret == page_size
    finally:
        phx_cache.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)

    # Verify file contents with standard I/O
    with open(temp_file, "rb") as f:
        disk_data = f.read(page_size)
    assert len(disk_data) == page_size
    expected = torch.arange(page_size, dtype=torch.uint8)
    assert torch.equal(
        torch.frombuffer(bytes(disk_data[:page_size]), dtype=torch.uint8),
        expected,
    ), "PhxFile.write data mismatch"


@skip_no_hardware
def test_phxfile_read_write_round_trip(phx_cache, gpu_buffer, temp_file):
    """Full round-trip: write device data via PhxFile, read back via PhxFile."""
    from phxcache import PhxFile

    page_size = phx_cache.page_size
    expected = torch.arange(page_size, dtype=torch.uint8)

    phx_cache.regmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
    try:
        # Write device data to file via PhxFile
        gpu_buffer[:page_size].copy_(expected)
        with PhxFile(phx_cache, temp_file, os.O_CREAT | os.O_WRONLY | os.O_DIRECT) as pf:
            ret = pf.write(gpu_buffer.data_ptr(), 0, page_size, 0)
        assert ret == page_size

        # Read back to device via PhxFile
        gpu_buffer.zero_()
        with PhxFile(phx_cache, temp_file, os.O_RDONLY | os.O_DIRECT) as pf:
            ret = pf.read(gpu_buffer.data_ptr(), 0, page_size, 0)
        assert ret == page_size

        got = gpu_buffer[:page_size].cpu()
        assert torch.equal(got, expected), "Round-trip data mismatch"
    finally:
        phx_cache.deregmem(gpu_buffer.data_ptr(), gpu_buffer.nbytes)
