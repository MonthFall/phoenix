# Phoenix Documentation

Welcome to the Phoenix documentation. Phoenix is a middleware that enables **direct I/O transfer from storage systems to xPU accelerators (GPU/NPU)**, bypassing CPU bounce buffers ("phony buffers"). It exposes a kernel module, a user-space library, Python bindings, and adapters for AI applications (e.g. vLLM model loading).

## Document index

| Document | Description |
| --- | --- |
| [architecture.md](architecture.md) | System architecture, data path, and directory responsibilities |
| [install.md](install.md) | Prerequisites, build, kernel-module install, and quick demo |
| [kernel-module.md](kernel-module.md) | `phxfs` kernel module design and interfaces |
| [libphoenix.md](libphoenix.md) | User-space C/C++ library API (`libphoenix`) |
| [python-api.md](python-api.md) | Python bindings (`phxfs`) |
| [adapters.md](adapters.md) | Application adapters: vLLM (available) and lmcache (roadmap) |
| [troubleshooting.md](troubleshooting.md) | Kernel-module install troubleshooting |
| [roadmap.md](roadmap.md) | Engineering tasks and research directions |

## Quick links

- Project README: [`README.md`](../README.md)
- Contributor guide: [`CONTRIBUTING.md`](../CONTRIBUTING.md)
- AI developer orientation: [`AGENTS.md`](../AGENTS.md)
