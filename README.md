# Phoenix

**Direct I/O middleware from storage systems to xPU accelerators (GPU / NPU).**

Phoenix refactors the I/O stack for GPU Direct Storage so that data moves from storage straight into accelerator memory via DMA — **bypassing CPU host-memory bounce buffers** ("phony buffers"). AI applications integrate through adapters, getting I/O acceleration with minimal changes.

> 📄 Built on the SC'25 paper *"Phoenix: A Refactored I/O Stack for GPU Direct Storage without Phony Buffers"*.
Phoenix is now a long-term open-source middleware for storage→xPU I/O, with adapters for AI data (vLLM done; lmcache planned).
## 📰 News


- **2026.7.10 phxloader released** — adapter for **vLLM**: safetensors weight loading via DMA straight into GPU memory (`--load-format phxsafetensors`).
- **2025.12** — Phoenix paper accepted at [SC'25](https://doi.org/10.1145/3712285.3759862).

## What is Phoenix

Phoenix is a direct hub between storage and xPU — accelerators (GPU/NPU) and AI apps plug in, and data streams straight through:

```
      AI App (vLLM, lmcache, …)
                  │ adapter
                  ▼
   storage ──► Phoenix ──► xPU (GPU / NPU)
        no detour through host memory
```

The kernel module (`phxfs`) remaps GPU PCIe BAR memory and serves P2P mappings; the user library (`libphoenix`) and Python bindings expose simple registration/I/O APIs; adapters plug Phoenix into AI frameworks.

## Features & supported matrix

| Area | Status | Notes |
| --- | --- | --- |
| Core I/O (storage→GPU DMA, no phony buffer) | ✅ Implemented | `phxfs` + `libphoenix` |
| Python bindings | ✅ Implemented | `python/phxfs` (ctypes) |
| vLLM model loading | ✅ Implemented | `adapters/vllm/phxloader` (V2.2) |
| lmcache KV-cache acceleration | 🚧 Roadmap | see [roadmap](doc/roadmap.md) |
| NPU (non-NVIDIA) support | 🔬 Research | experimentally proven, not yet in code |
| Async I/O (`io_uring`) | 🚧 Roadmap | currently `pread`/`pwrite` + CUDA host func |

 
**Environment (tested)**

- OS: Ubuntu 22.04 · Kernel: Linux 6.1 · NVIDIA driver 550.54 (open + `nvidia-fs`) · CUDA 12.4 · MLNX_OFED 24.10
- Storage backends: NVMe-of, NFS (local NVMe supported for the direct path)
- Accelerator: NVIDIA GPU (CUDA). NPU: not yet supported by current code.

## Roadmap

Engineering (production hardening) and research (new capabilities) tracks are detailed in [doc/roadmap.md](doc/roadmap.md) — including async `io_uring` I/O, >32 GiB registration, broader kernel/DKMS packaging, NPU support, and a future MCP-based shared bug knowledge base.

## Quick install & demo

### 1. Build kernel module + user library

```shell
# Default: NVIDIA GPU (no extra flags needed)
mkdir -p build && cd build && cmake ../ && make -j

# For other vendors (not yet implemented, interface ready):
# cmake -DPHXFS_VENDOR=AMD ../
# cmake -DPHXFS_VENDOR=HUAWEI ../
```

This produces:
- `build/libphoenix.so` — user-space library
- `build/module/phoenixfs.ko` — kernel module
- `build/bin/test_regmem`, `build/bin/test_io` — test programs

### 2. Insert kernel module

```shell
# Run nvidia-smi first to wake up the GPU BAR, then insert
nvidia-smi
sudo make insmod

# Verify: 8 GPU devices should appear
ls /dev/phxfs_dev*
# Check dmesg for BAR remap messages
dmesg | tail -20
```

### 3. Run tests

```shell
# Memory registration lifecycle (32 checks)
./bin/test_regmem 0          # GPU 0

# I/O correctness + performance (19 checks + throughput)
./bin/test_io 0              # GPU 0
```

## Documentation

| Doc | Link |
| --- | --- |
| Architecture & data path | [doc/architecture.md](doc/architecture.md) |
| Build / install / demo | [doc/install.md](doc/install.md) |
| Kernel module (`phxfs`) | [doc/kernel-module.md](doc/kernel-module.md) |
| User library (`libphoenix`) | [doc/libphoenix.md](doc/libphoenix.md) |
| Python API | [doc/python-api.md](doc/python-api.md) |
| Adapters (vLLM / lmcache) | [doc/adapters.md](doc/adapters.md) |
| Troubleshooting | [doc/troubleshooting.md](doc/troubleshooting.md) |
| Roadmap | [doc/roadmap.md](doc/roadmap.md) |
| AI developer orientation | [AGENTS.md](AGENTS.md) |

## Partners

*Partnership section — to be announced. We welcome storage vendors, accelerator vendors, and AI-framework teams to collaborate. Contact us (below).*

## Contact & community

*Contact information to be added (maintainers / mailing list / Slack or WeChat group).* For bug reports, please use the issue template and the `scripts/collect_bug_info.sh` collector — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Cite

If you use Phoenix in your research, please cite our SC'25 paper:

```bibtex
@inproceedings {sc25Phoenix,
  author = {Jianqin Yan, Shi Qiu, Yina Lv, Yifan Hu, Hao Chen, Zhirong Shen, Xin Yao, Renhai Chen, Jiwu Shu, Gong Zhang, and Yiming Zhang.},
  title = {Phoenix:A Refactored I/O Stack for GPU Direct Storage without Phony Buffers.},
  booktitle = {The International Conference for High Performance Computing, Networking, Storage and Analysis (SC '25)},
  year = {2025},
  address = {St Louis, MO, USA},
  publisher = {ACM}
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
