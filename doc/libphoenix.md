# LibPhoenix

`libphoenix` is the user-space C/C++ library that simplifies interaction with the `phxfs` kernel module. It manages device metadata and GPU buffer registration/unregistration.

## Multi-vendor DevConnector

Vendor-specific calls (device discovery via CUDA/HIP/CANN, async stream launch) are abstracted behind `struct devconn_ops` (`libphoenix/connectors/devconnector.h`). The active connector is selected at **compile time** via `PHXFS_VENDOR` (default `NVIDIA`) and exposed through the global `devconn` pointer. Core files (`phoenix.cpp`, `integration.cpp`) call only through `devconn->find_device()` / `launch_async()` / `page_size` and never include vendor headers (e.g. `cuda.h`).

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

## I/O

`phxfs_read` / `phxfs_write` transfer data directly between a file descriptor and the registered (GPU-backed) VMA. Large transfers are chunked at `PHXFS_IO_CHUNK` (1 GiB) to stay under the kernel's `MAX_RW_COUNT`.

### Async I/O

`phxfs_read_async` / `phxfs_write_async` accept a vendor-agnostic `void *stream` handle and launch the transfer via `devconn->launch_async()` (e.g. `cudaLaunchHostFunc` for NVIDIA). If no connector supports async launch, the call falls back to synchronous execution. Native `io_uring` support is on the [roadmap](roadmap.md).
