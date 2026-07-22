# Architecture

Phoenix refactors the I/O stack for **GPU Direct Storage (GDS)** so that data moves from storage straight into accelerator memory via DMA, without being staged through CPU host memory ("phony buffers"). It is structured as a thin middleware layer that AI applications integrate through *adapters*.

## Layers

```
            AI Application (vLLM, lmcache, ...)
                      │  adapter  (e.g. phxloader, via pybind11)
                      ▼
        ┌─────────────────────────────┐
        │  libphoenix  (user library) │   phxfs_open / regmem / read / write
        │  DevConnector (per-vendor)  │   vendor device lookup + page size
        └──────────────┬──────────────┘
                       │  ioctl / mmap  (char device /dev/phxfs_devN)
                       ▼
        ┌─────────────────────────────┐
        │   phxfs  (kernel module)    │   P2P map GPU BAR, DMA into GPU mem
        │   P2P backend (per-vendor)  │
        └──────────────┬──────────────┘
                       │  vendor P2P API (e.g. nvidia_p2p_get_pages) / PCIe P2P
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
| `module/` | `phxfs` Linux kernel module: GPU BAR remap, per-GPU char device, `mmap`/`ioctl` P2P mapping; `phxfs-backend.*` selects the vendor P2P backend at build time |
| `libphoenix/` | User-space C/C++ library wrapping the char device; buffer registration and synchronous/async I/O; `connectors/` holds the vendor-specific `DevConnector` |
| `adapters/` | Integration with AI frameworks via pybind11. `adapters/vLLM/` ships `phxloader` for safetensors weight loading |
| `test/` | Correctness + performance tests (`test_regmem`, `test_io`, `test_batch`) |

## Multi-vendor support

Both the kernel module and `libphoenix` use a single compile-time switch, `PHXFS_VENDOR` (default `NVIDIA`), to select the accelerator vendor:

```shell
cmake -DPHXFS_VENDOR=NVIDIA ../   # default
cmake -DPHXFS_VENDOR=AMD ../      # requires module/amd-backend.c + libphoenix/connectors/amd_connector.cpp
cmake -DPHXFS_VENDOR=HUAWEI ../   # requires module/huawei-backend.c + libphoenix/connectors/huawei_connector.cpp
```

Core code (kernel: `phxfs.c`/`phxfs-mem.c`; user library: `phx_device.cpp`/`phx_mem.cpp`/`phx_io.cpp`) never references vendor APIs directly — it calls through `phxfs_p2p` (kernel) / `devconn` (user library) function-pointer tables. Adding a new vendor only requires implementing one backend file per layer; see `doc/kernel-module.md` and `doc/libphoenix.md`.

## Supported environment (tested)

- **OS**: Ubuntu 22.04 / TencentOS
- **Kernel**: Linux 6.1 (see [install.md](install.md) for details)
- **Accelerator**: NVIDIA GPU with CUDA 12.4+ and the open `nvidia-fs` driver
- **Storage**: NVMe-of / NFS (local NVMe also supported for the direct path)
- **Other vendors (AMD, Huawei NPU)**: backend interface is ready; vendor-specific implementations not yet shipped (see [roadmap.md](roadmap.md))
