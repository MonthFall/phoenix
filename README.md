# Phoenix

**Direct I/O middleware from storage systems to xPU accelerators (GPU / NPU).**

Phoenix refactors the I/O stack for GPU Direct Storage so that data moves from storage straight into accelerator memory via DMA — **bypassing CPU host-memory bounce buffers** ("phony buffers"). AI applications integrate through adapters, getting I/O acceleration with minimal changes.

> 📄 Built on the SC'25 paper *"Phoenix: A Refactored I/O Stack for GPU Direct Storage without Phony Buffers"*.
Phoenix is now a long-term open-source middleware for storage→xPU I/O, with adapters for AI data (vLLM done; lmcache planned).
## 📰 News


- **2026.7.10 phxloader released** — adapter for **vLLM**: safetensors weight loading via DMA straight into GPU memory (`--load-format phxsafetensors`).
- **2025.12** — Phoenix paper accepted at [SC'25](https://doi.org/10.1145/3712285.3759862).

## What is Phoenix

Phoenix is a thin middleware layer between storage and accelerators:

```
   AI App (vLLM, lmcache, …)  ──adapter──►  libphoenix  ──ioctl/mmap──►  phxfs (kernel)
        │                                                                       │ nvidia_p2p / PCIe P2P
        └──────────────────────── DMA: storage ───────────────► GPU / xPU memory
```

The kernel module (`phxfs`) remaps GPU PCIe BAR memory and serves P2P mappings; the user library (`libphoenix`) and Python bindings expose simple registration/I/O APIs; adapters plug Phoenix into AI frameworks.

## Features & supported matrix

要加上支持什么模型。md支持什么样的展示方案？是否可以支持drawio？我想以一颗树的形式，从底层向上生长。 清晰的展示 支持底层的内核版本， IO技术， xPU 型号，应用场景，应用框架，模型。然后通过点击来扩展这个树。

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

```shell
# 1. prerequisites: NVIDIA GDS + MLNX_OFED (see doc/install.md)
# 2. build
mkdir -p build && cd build && cmake ../ && make -j
# 3. install kernel module (run nvidia-smi first)
sudo make insmod
# 4. quick demo
sudo ./bin/example <file_path> <io_size> <mode>
```

For vLLM weight loading:

```shell
cd adapters/vllm/phxloader && bash install.sh
# then launch vLLM with: --load-format phxsafetensors
```

Full instructions: [doc/install.md](doc/install.md).

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
