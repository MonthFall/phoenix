# lmcache adapter (roadmap)

This directory will host the Phoenix adapter for [lmcache](https://github.com/LMCache/LMCache), enabling KV-cache offload/loading acceleration via direct storage→GPU DMA.

**Status: planned — not yet implemented.** Track progress in [doc/roadmap.md](../doc/roadmap.md).

When implemented, the adapter will follow the same pattern as `adapters/vllm/phxloader`: register a GPU buffer through `libphoenix` and transfer KV-cache blocks with `phxfs_read`/`phxfs_write`, bypassing CPU host memory.
