# Adapters

Adapters connect Phoenix to AI applications. Each adapter depends on `libphoenix` (via `pybind11` bindings) to perform direct storage→xPU transfers for a specific framework.

## vLLM — `phxloader` (available)

`adapters/vllm/phxloader` provides GPU Direct Storage loading of `safetensors` model weights via Phoenix's DMA engine. It parses the safetensors header, plans reads into read-groups, registers a shared GPU buffer, and DMAs weights straight into GPU memory — bypassing CPU host memory.

**Current release: V2.2**

### Version evolution

- **V1 — contiguous synchronous DMA**: single contiguous DMA of the whole data section; one GPU buffer per file; synchronous I/O. (Historical; code preserved on the `tencent-backup` branch.)
- **V2 — batch DMA + shared buffer**: splits each file into read-groups (adjacent tensors with gap < 64 KiB merged), reuses one shared GPU buffer via `regmem`/`deregmem`, one `phxfs_read` per group. (Historical; code preserved on the `tencent-backup` branch.)
- **V2.1 — double buffer + async DMA**: dual GPU buffers (bufA/bufB) with background DMA on a C++ thread, overlapping `yield(copy_)` to hide yield latency; adds a DMA timer.
- **V2.2 (current) — official release + API rename**: functionally identical to V2.1; API renamed for clarity (`read_into_registered` → `load_tensors_into_buffer`, `read_into_registered_async` → `load_tensors_into_buffer_async`, class `PhxLoaderV2` → `PhxLoader`); hardened safetensors header parsing. Published as the `phxloader` package (no version suffix).

### Install

```bash
# Prerequisites: conda env active, libphoenix built (build/libphoenix.so)
cd adapters/vllm/phxloader
bash install.sh
```

### Use

```bash
# vLLM launch flag
--load-format phxsafetensors
```
```python
from vllm import LLM
llm = LLM(model="...", load_format="phxsafetensors")
```

### API

```python
from phxloader import PhxLoader

loader = PhxLoader(device_id=0)
loader.regmem(gpu_ptr, size)
loader.load_tensors_into_buffer(path, gpu_ptr, batch)        # synchronous
loader.load_tensors_into_buffer_async(path, gpu_ptr, batch)  # background DMA
loader.wait_dma()
loader.reset_dma_timer(); loader.get_dma_seconds()
loader.deregmem(gpu_ptr, size)
loader.close()
```

### Dependencies

`libphoenix`, `liburing`, CUDA, `pybind11`, PyTorch.

## lmcache (roadmap)

Integration with [lmcache](https://github.com/LMCache/LMCache) for KV-cache offload/loading acceleration is planned. See [roadmap.md](roadmap.md).
