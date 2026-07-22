# LibPhoenix

`libphoenix` is the user-space C/C++ library that simplifies interaction with the `phxfs` kernel module. It manages device metadata and GPU buffer registration/unregistration.

## Multi-vendor DevConnector

Vendor-specific calls (device discovery via CUDA/HIP/CANN) are abstracted behind `struct devconn_ops` (`libphoenix/connectors/devconnector.h`). The active connector is selected at **compile time** via `PHXFS_VENDOR` (default `NVIDIA`) and exposed through the global `devconn` pointer. Core files (`phx_device.cpp`, `phx_mem.cpp`, `phx_io.cpp`) call only through `devconn->find_device()` / `page_size` and never include vendor headers (e.g. `cuda.h`).

`libphoenix/connectors/nvidia_connector.cpp` implements the NVIDIA connector. Adding a new vendor means writing `<vendor>_connector.cpp` and pointing `devconn` at it; no other user-library file needs to change.

## Driver management

### `phxfs_open`
```c++
int phxfs_open(int deviceID);
```
Opens the character device for `deviceID`, initializes and stores the metadata required for later buffer registration. Opens are reference-counted: a second `phxfs_open` on the same device only adds a client reference.

### `phxfs_close`
```c++
int phxfs_close(int deviceID);
```
Drops one client reference on `deviceID`. The last close waits for in-flight operations to drain, then unmaps every registration and closes the device. A concurrent `phxfs_open` on a draining device fails with `-EBUSY` (retryable).

## Buffer management

### `phxfs_regmem`
```c++
int phxfs_regmem(int device_id, const void *addr, size_t len, void **target_addr);
```
Registers a memory region (`addr`, `len`) for `device_id`: `mmap`s a VMA from the char device, then issues `ioctl(PHXFS_IOCTL_MAP)` to pin the GPU pages into it. Both `addr` and `len` must be non-zero and 64 KiB (device page) aligned. On success, `target_addr` receives the host-mapped address — an **internal handle for reference only**; the I/O calls identify a buffer by its original device address `addr`, never by `target_addr`.

Registration semantics: an exact-duplicate registration (same `addr` + `len`, still live) is reference-counted and reused (deregister once per register); any other overlap with a live registration is rejected with `-EINVAL`.

### `phxfs_deregmem`
```c++
int phxfs_deregmem(int device_id, const void *addr, size_t len);
```
Drops one reference on the registration. The last reference waits for in-flight I/O on the region to drain, then removes the kernel mapping via `ioctl(PHXFS_IOCTL_UNMAP)` and `munmap`s the user-space VMA.

## Single-request I/O

`phxfs_read` / `phxfs_write` transfer data directly between a file descriptor and the registered (GPU-backed) VMA:

```c++
ssize_t phxfs_read (phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);
ssize_t phxfs_write(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);
```

`fid.deviceID` selects the buffer the same way as the batch API below: `>= 0` means `buf` must lie inside a registration on that phxfs device; `< 0` means `buf` is a plain CPU (host) address. For a registered buffer, `buf` may point anywhere inside the region; the host DMA address is resolved as `vaddr + (buf - registered_base) + buf_offset`, and an internal reference on the mapping is held for the transfer's duration, so a concurrent `phxfs_deregmem` cannot unmap it mid-I/O. Large transfers are chunked at `PHXFS_IO_CHUNK` (1 GiB) to stay under the kernel's `MAX_RW_COUNT`.

## Batch I/O

For workloads that issue many independent transfers (e.g. KV-cache retrieve/store, weight loading), the batch API submits a whole set of requests in one call. This removes the per-request syscall overhead of looping over `phxfs_read`/`phxfs_write` and lets the storage layer service requests concurrently.

### Request descriptor

```c++
typedef struct phxfs_io_req {
    int      fd;          // open file descriptor (O_DIRECT recommended)
    int      device_id;   // >=0: phxfs device the buf is registered on;
                          //  <0: plain CPU buffer
    void    *buf;         // GPU addr (registered) or CPU addr
    off_t    buf_offset;  // byte offset within buf
    size_t   nbytes;      // transfer length
    off_t    f_offset;    // file offset
    ssize_t  result;      // OUT: bytes transferred, or negative errno
} phxfs_io_req_t;
```

Each request's `buf` is resolved independently to a DMA-able host address:

- **Registered GPU buffer (`device_id >= 0`)** — `buf` must lie inside a `phxfs_regmem` registration on that device, or the request fails with `-EFAULT` (there is no silent CPU fallback). The mapping is reference-held for the batch's duration.
- **Plain CPU buffer (`device_id < 0`)** — `buf` is used as an ordinary host address (e.g. pinned staging memory).

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
int            phxfs_batch_destroy(phxfs_batch_t *handle);
```

`submit` queues the batch on the internal worker pool and returns an opaque handle immediately; the caller can run GPU compute meanwhile, then `phxfs_batch_wait` blocks for completion, fills results, and frees the handle. `wait` returns the same value convention as the synchronous batch. `phxfs_batch_destroy` abandons a batch whose results are not needed: it waits for in-flight I/O to quiesce, then frees the handle without copying results back.

Submitted batches queue up on a bounded FIFO and run one at a time with the pool's full worker set, so several submits can be pipelined ahead (e.g. layerwise prefetch); if the queue is full, submit fails with `NULL` + `errno == EBUSY`. An empty batch (`n <= 0`) returns a valid handle whose `wait` returns `0`, matching the synchronous API.

There is no vendor stream involved: the application drives its own CUDA/HIP stream and interleaves `submit` → compute → `wait` as it sees fit. (The former stream-based `phxfs_read_async` / `phxfs_write_async` were removed; this belongs on the application/adapter side.)

### Lifetime & concurrency contract

Everything a request references must stay valid until the call (sync) or `phxfs_batch_wait` (async) returns:

- `reqs[i].fd` must remain open — fds are **not** `dup()`'d internally.
- CPU buffers must not be freed.
- Registered GPU buffers are protected by an internal reference; a concurrent `phxfs_deregmem` on them blocks until the batch completes rather than unmapping under an in-flight transfer.

A batch may freely mix requests targeting different devices and CPU buffers; all requests share the same worker pool.

## I/O engines & worker pool

The batch path is built on two internal, pluggable layers:

- **I/O engine** (`libphoenix/io_engine/io_engine.h`) — selected once at library load, in preference order `io_uring → sync`. The `io_uring` engine keeps a per-thread ring (QD 1024) with a pre-allocated slice/iovec scratch pool and a sliding-window submit/reap pipeline; it uses the non-fixed `OP_READ`/`OP_WRITE` path (same O_DIRECT DIO route as `pread`, no second long-term pin on device pages). The `sync` engine is a `pread`/`pwrite` loop and is the always-available fallback (also used per-thread if a worker's ring cannot be created). `phxfs_io_engine_name()` reports the active engine, for diagnostics/tests.
- **Worker pool** (`libphoenix/io_engine/io_pool.cpp`) — a single pool of worker threads, each running the engine on its own ring. A batch is striped round-robin across the workers, so a single blocking call can saturate the array while the caller crosses the Python GIL only once. Submits queue up (bounded FIFO) so a caller can pipeline several async batches ahead. There is deliberately no NUMA pinning: the P2P transfer is device-to-device DMA that never touches host RAM, so which CPU/node issues the I/O is irrelevant to the data path. The pool registers a `pthread_atfork` child handler — a forked child lazily re-creates the pool on first use (in-flight async handles do not survive fork).
