# Roadmap

Phoenix is evolving from a research artifact (SC'25) into a long-term open-source middleware for storage→xPU I/O. The roadmap below is split into **engineering** (hardening for production) and **research** (new capabilities) tracks.

## Engineering

- **Async I/O via `io_uring`**: replace the current `cudaLaunchHostFunc + pread/pwrite` path in `integration.cpp` with `io_uring`, including verification of GUP compatibility for `ZONE_DEVICE` / `PCI_P2PDMA` pages on the target kernel.
- **Large-region registration**: `>32 GiB` single `regmem` still fails because the mapping descriptor uses `kmalloc`; convert it to `kvalloc` (verify `kvfree` is safe in the vendor driver callback context).
- **Broader kernel support**: validate and document supported kernel versions beyond the tested 6.1; provide DKMS packaging for the `phxfs` module.
- **Packaging & distribution**: produce wheel / deb artifacts for `libphoenix` and `phxloader`; CI to build and smoke-test on reference kernels.
- **Correctness regression suite**: expand `test/` (e.g. `>2 GiB` registration + offset verification) and wire into CI.
- **Bug intelligence (MCP)**: a lightweight local collector (`scripts/collect_bug_info.sh`) already structures bug reports. A future **MCP server** will aggregate bug reports across users to build a shared knowledge base for faster triage. *Planned, not yet implemented.*

## Research

- **Multi-vendor P2P backends**: the kernel `phxfs_p2p_ops` and user-library `DevConnector` interfaces are in place (`PHXFS_VENDOR=NVIDIA|AMD|HUAWEI`); implementing the AMD (`amd-backend.c`/`amd_connector.cpp`) and Huawei NPU (`huawei-backend.c`/`huawei_connector.cpp`) backends is the next step — direct storage→NPU transfer has been experimentally proven feasible.
- **Multi-framework adapters**: lmcache KV-cache acceleration; further AI-data-loader offloads.
- **Policy & scheduling**: adaptive chunking, multi-GPU striping, and QoS-aware DMA scheduling.
- **Filesystem breadth**: broader backend coverage (more distributed/storage backends) with unchanged application APIs.

## How to contribute

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) and the bug-report template under `.github/ISSUE_TEMPLATE/`.
