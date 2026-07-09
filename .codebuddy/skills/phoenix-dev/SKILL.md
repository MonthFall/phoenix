---
name: phoenix-dev
description: >
  Developer workflow skill for the Phoenix project (storage→xPU direct I/O middleware).
  Use when the user wants to build/install/test the phxfs kernel module, work on the
  libphoenix user library or Python bindings, integrate an AI application adapter
  (vLLM / lmcache), or understand/collect/report a bug. Triggers include "build phoenix",
  "insmod", "kernel module", "libphoenix", "phxloader", "vLLM adapter", "report a bug",
  "collect bug info", or any task touching module/, libphoenix/, python/, adapters/.
---

# Phoenix developer skill

Phoenix is middleware for direct I/O from storage to xPU (GPU/NPU) via DMA, bypassing CPU
host memory. Three layers: `phxfs` kernel module, `libphoenix` user library (+ `python/phxfs`
bindings), and `adapters/` for AI apps. **Always start by reading `AGENTS.md` at the repo
root** for the architecture, directory map, and gotchas — do not read the whole tree.

## Workflow 1 — Environment adaptation: build / install / test the kernel module

1. Read `doc/install.md` and `doc/troubleshooting.md`.
2. Confirm prerequisites: NVIDIA GDS + MLNX_OFED, kernel source for the running kernel,
   CUDA 12.4, liburing. Run `nvidia-smi` so the NVIDIA driver is loaded.
3. Build:
   ```shell
   mkdir -p build && cd build && cmake ../ && make -j
   ```
   Skip the module with `cmake -Dno_module=true ../`.
4. Install: `sudo make insmod`. On "Operation not permitted", the GPU BAR is held — see
   `doc/troubleshooting.md` (free `/dev/nvidia*` users, unload `nvidia_drm`).
5. Test: run `sudo ./bin/example <file> <size> <mode>` from `build/`, or a benchmark under
   `benchmarks/`. Inspect `dmesg | grep phxfs` for module messages.
6. Remove when done: `sudo make rmmod`.

## Workflow 2 — Work on the user library / Python bindings

- `libphoenix/` is the C/C++ API (`phxfs_open/close`, `regmem/deregmem`, `read/write`).
- `python/phxfs` is a ctypes wrapper over `libphoenix.so`; rebuild the lib, then
  `cd python && python setup.py install`.
- Async I/O today is `cudaLaunchHostFunc + pread/pwrite` (NOT io_uring yet — planned).

## Workflow 3 — Upper-layer application integration

- vLLM: see `doc/adapters.md`. The `adapters/vllm/phxloader` package exposes `PhxLoader`
  and a `phxsafetensors` vLLM `load_format`. Pattern: parse model/weights → register a
  shared GPU buffer via `libphoenix` → `phxfs_read` DMA into GPU → `deregmem`.
- New adapters (e.g. lmcache): register a GPU buffer through `libphoenix` and transfer with
  `phxfs_read`/`phxfs_write`; follow the vLLM adapter structure. Place under `adapters/<app>/`.

## Workflow 4 — Understand and report a bug

1. Reproduce and capture `dmesg | grep phxfs`, module load state, and the failing command.
2. Run the structured collector:
   ```shell
   bash scripts/collect_bug_info.sh
   ```
3. Open an issue with `.github/ISSUE_TEMPLATE/bug_report.md`, pasting the collector output
   and clear reproduction steps.
4. (Future) A shared MCP server will aggregate these reports across users for faster triage
   — see `doc/roadmap.md`. Until then, keep reports in issues.

## Hard constraints / gotchas

- Only one driver may own the GPU PCIe BAR at a time.
- `regmem` > 32 GiB still fails (kmalloc-limited descriptor); large I/O is chunked at
  `PHXFS_IO_CHUNK` (1 GiB) for `read`/`write`.
- Code is NVIDIA-only today; NPU support is research, not implemented.
- `scripts/*.sh`/`*.py` hardcode dataset paths and are env-specific; not part of the build.
