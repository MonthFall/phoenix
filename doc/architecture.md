# Architecture

Phoenix refactors the I/O stack for **GPU Direct Storage (GDS)** so that data moves from storage straight into accelerator memory via DMA, without being staged through CPU host memory ("phony buffers"). It is structured as a thin middleware layer that AI applications integrate through *adapters*.

## Layers

```
            AI Application (vLLM, lmcache, ...)
                      │  adapter  (e.g. phxloader)
                      ▼
        ┌─────────────────────────────┐
        │  libphoenix  (user library) │   phxfs_open / regmem / read / write
        │  python/phxfs (ctypes bind) │
        └──────────────┬──────────────┘
                       │  ioctl / mmap  (char device /dev/phxfs_devN)
                       ▼
        ┌─────────────────────────────┐
        │   phxfs  (kernel module)    │   P2P map GPU BAR, DMA into GPU mem
        └──────────────┬──────────────┘
                       │  nvidia_p2p_get_pages / PCIe P2P
                       ▼
                 GPU / xPU memory  ◄──── DMA from NVMe / NFS storage
```

## Data path (no phony buffer)

1. The application allocates a GPU buffer and registers it with Phoenix via `phxfs_regmem`. Phoenix `mmap`s a character device and uses `ioctl(PHXFS_IOCTL_MAP)` so the GPU memory is mapped into a host VMA backed by the GPU's PCIe BAR (via `ZONE_DEVICE`).
2. A `phxfs_read` / `phxfs_write` issues the file I/O directly against that VMA. The storage controller DMAs data straight into GPU memory — the CPU never touches the payload.
3. `phxfs_deregmem` releases the mapping.

## Component responsibilities

| Path | Responsibility |
| --- | --- |
| `module/` | `phxfs` Linux kernel module: GPU BAR remap, per-GPU char device, `mmap`/`ioctl` P2P mapping |
| `libphoenix/` | User-space C/C++ library wrapping the char device; buffer registration and synchronous/async I/O |
| `python/` | `phxfs` Python package (ctypes) exposing the library to Python |
| `adapters/` | Integration with AI frameworks. `adapters/vllm/` ships `phxloader` for safetensors weight loading |
| `example/` | Minimal end-to-end usage example |
| `benchmarks/` | Performance evaluation binaries (breakdown, end-to-end, kvcache, micro, safetensor) |
| `tests/` | Correctness tests (`.cu`) |
| `scripts/` | Helper/evaluation scripts |
| `third-party/fio/` | fio for I/O testing |

## Supported environment (tested)

- **OS**: Ubuntu 22.04
- **Kernel**: Linux 6.1 (see [install.md](install.md) for details)
- **Accelerator**: NVIDIA GPU with CUDA 12.4 and the open `nvidia-fs` driver
- **Storage**: NVMe-of / NFS (local NVMe also supported for the direct path)
- **NPU**: experimentally proven feasible; not yet supported by the current code (see [roadmap.md](roadmap.md))
