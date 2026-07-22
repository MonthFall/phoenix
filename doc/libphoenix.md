# LibPhoenix

`libphoenix` is the user-space C/C++ library that simplifies interaction with the `phxfs` kernel module. It manages device metadata and GPU buffer registration/unregistration.

## Multi-vendor DevConnector

Vendor-specific calls (device discovery via CUDA/HIP/CANN) are abstracted behind `struct devconn_ops` (`libphoenix/connectors/devconnector.h`). The active connector is selected at **compile time** via `PHXFS_VENDOR` (default `NVIDIA`) and exposed through the global `devconn` pointer. Core file `phoenix.cpp` calls only through `devconn->find_device()` / `page_size` and never includes vendor headers (e.g. `cuda.h`).

`libphoenix/connectors/nvidia_connector.cpp` implements the NVIDIA connector. Adding a new vendor means writing `<vendor>_connector.cpp` and pointing `devconn` at it; no other user-library file needs to change.

## Driver management

### `phxfs_open`
```c++
int phxfs_open(int deviceID);
```
Opens the character device for `deviceID`, initializes and stores the metadata required for later buffer registration.

### `phxfs_close`
```c++
int phxfs_close(int deviceID);
```
Releases all metadata associated with `deviceID`.

## Buffer management

### `phxfs_regmem`
```c++
int phxfs_regmem(int device_id, const void *addr, size_t len, void **target_addr);
```
Registers a memory region (`addr`, `len`) for `device_id`. It `mmap`s a VMA from the char device and issues `ioctl(PHXFS_IOCTL_MAP)` to map the GPU memory. On success, `target_addr` receives the mapped (host-remapped) address used for I/O.

Core flow:
```c++
int phxfs_regmem(int device_id, const void *addr, size_t len, void **target_addr) {
    struct phxfs_bdev *pb = get_phxfs_bdev(device_id);
    struct phxfs_p2p_map *p2p_map = malloc(sizeof(*p2p_map));
    p2p_map->vaddrs = mmap(NULL, len, PROT_READ|PROT_WRITE, MAP_SHARED, pb->bdev_fd, 0);
    int ret = __phxfs_regmem(pb, (u64)addr, (u64)p2p_map->vaddrs, len);
    if (ret < 0) { munmap(p2p_map->vaddrs, len); return ret; }
    *target_addr = p2p_map->vaddrs;
    insert_phxfs_mmap_node(pb, p2p_map);
    return 0;
}
```

### `phxfs_deregmem`
```c++
int phxfs_deregmem(int device_id, const void *addr, size_t len);
```
Unregisters a previously registered region: removes the kernel mapping via `ioctl(PHXFS_IOCTL_UNMAP)`, then `munmap`s the user-space VMA.

## Single-request I/O

`phxfs_read` / `phxfs_write` transfer data directly between a file descriptor and the registered (GPU-backed) VMA:

```c++
ssize_t phxfs_read (phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);
ssize_t phxfs_write(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);
```

`buf` may point anywhere inside a registered region; the host DMA address is resolved as `vaddr + (buf - registered_base) + buf_offset`. Large transfers are chunked at `PHXFS_IO_CHUNK` (1 GiB) to stay under the kernel's `MAX_RW_COUNT`. These calls require a registered GPU buffer and hold an internal reference on the mapping for the duration of the transfer, so a concurrent `phxfs_deregmem` cannot unmap it mid-I/O.

## Batch I/O

For workloads that issue many independent transfers (e.g. KV-cache retrieve/store, weight loading), the batch API submits a whole set of requests in one call. This removes the per-request syscall overhead of looping over `phxfs_read`/`phxfs_write` and lets the storage layer service requests concurrently.

### Request descriptor

```c++
typedef struct phxfs_io_req {
    int      fd;          // open file descriptor (O_DIRECT recommended)
    int      device_id;   // phxfs device the buf is registered on (GPU);
                          // ignored / set to -1 for plain CPU buffers
    void    *buf;         // GPU addr (registered) or CPU addr
    off_t    buf_offset;  // byte offset within buf
    size_t   nbytes;      // transfer length
    off_t    f_offset;    // file offset
    ssize_t  result;      // OUT: bytes transferred, or negative errno
} phxfs_io_req_t;
```

Each request's `buf` is resolved independently to a DMA-able host address:

- **Registered GPU buffer** — if `(device_id, buf)` falls inside a `phxfs_regmem` registration, the transfer targets the host-mapped P2P VMA. The mapping is reference-held for the batch's duration.
- **Plain CPU buffer** — otherwise `buf` is treated as an ordinary host address (e.g. pinned staging memory). Use `device_id = -1` to force this path.

### Synchronous batch

```c++
int phxfs_read_batch (phxfs_io_req_t *reqs, int n);
int phxfs_write_batch(phxfs_io_req_t *reqs, int n);
```

Submits all `n` requests, blocks until every one completes, and fills each `reqs[i].result`. Returns `0` if every request transferred exactly `nbytes`; otherwise the number of failed requests (`>0`); or a negative errno on a submission-level engine error. Requests whose buffer cannot be resolved are marked `result = -EFAULT` and never handed to the engine.

### Asynchronous batch (compute / I/O overlap)

```c++
phxfs_batch_t *phxfs_batch_submit_read (phxfs_io_req_t *reqs, int n);
phxfs_batch_t *phxfs_batch_submit_write(phxfs_io_req_t *reqs, int n);
int            phxfs_batch_wait(phxfs_batch_t *handle);
```

`submit` queues the batch on the internal NUMA pool and returns an opaque handle immediately; the caller can run GPU compute meanwhile, then `phxfs_batch_wait` blocks for completion, fills results, and frees the handle. `wait` returns the same value convention as the synchronous batch. At most one async batch is in flight **per NUMA node** (the batch owns that node's pool between submit and wait); different nodes run concurrently.

There is no vendor stream involved: the application drives its own CUDA/HIP stream and interleaves `submit` → compute → `wait` as it sees fit. (The former stream-based `phxfs_read_async` / `phxfs_write_async` were removed; this belongs on the application/adapter side.)

### Lifetime & concurrency contract

Everything a request references must stay valid until the call (sync) or `phxfs_batch_wait` (async) returns:

- `reqs[i].fd` must remain open — fds are **not** `dup()`'d internally.
- CPU buffers must not be freed.
- Registered GPU buffers are protected by an internal reference; a concurrent `phxfs_deregmem` on them blocks until the batch completes rather than unmapping under an in-flight transfer.

A batch may mix requests targeting different devices (each is resolved and issued correctly), but the whole batch is scheduled on the NUMA pool of the **first** resolved request's device. Requests to other NUMA nodes still complete correctly but may incur cross-NUMA DMA — submit one batch per device for peak bandwidth.

## I/O engines & NUMA thread pool

The batch path is built on two internal, pluggable layers:

- **I/O engine** (`libphoenix/io_engine.h`) — selected once at library load, in preference order `io_uring → sync`. The `io_uring` engine keeps a per-thread ring (QD 1024) with a pre-allocated slice/iovec scratch pool and a sliding-window submit/reap pipeline; it uses the non-fixed `OP_READ`/`OP_WRITE` path (same O_DIRECT DIO route as `pread`, no second long-term pin on device pages). The `sync` engine is a `pread`/`pwrite` loop and is the always-available fallback (also used per-thread if a worker's ring cannot be created). `phxfs_io_engine_name()` reports the active engine, for diagnostics/tests.
- **NUMA thread pool** (`libphoenix/io_pool.cpp`) — one pool of worker threads per NUMA node, pinned to that node's CPUs. A batch is striped round-robin across the target node's workers, each running the engine on its own ring, so a single blocking call can saturate the array while the caller crosses the Python GIL only once. A `libaio` engine can slot in behind the same interface later.
