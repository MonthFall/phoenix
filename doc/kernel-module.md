# Kernel Module — `phxfs`

The `phxfs` kernel module is the core of Phoenix. It remaps each GPU device's PCIe BAR memory into kernel space via `ZONE_DEVICE`, creates one character device per GPU (`/dev/phxfs_devN`), and serves `mmap` / `ioctl` requests that establish peer-to-peer (P2P) mappings between GPU memory and a user-space VMA.

## Initialization

On `insmod`, `phxfs_init` performs:

1. Load NVIDIA P2P symbols (`nvfs_nvidia_p2p_init`).
2. Build the GPU info table and count devices (`gpu_info_table`, `npu_num`).
3. For each GPU, discover its PCIe BAR and `devm_memremap` the BAR into kernel space (`phxfs_ctrl_init`).
4. Create a character device per GPU (`phxfs_cdev_init`).
5. Initialize the hash table that tracks registered GPU memory regions (`phxfs_mbuffer_init`).

## Uninitialization

On `rmmod`, `phxfs_exit` deletes the char devices, unmaps the BAR memory, releases NVIDIA P2P symbols, and destroys the device class / char-device region.

## Character device interface

Three base operations are exposed to user space:

### `open`
Saves the device metadata for `deviceID` into `file->private_data`.

### `mmap`
Sets VMA flags (`VM_MIXEDMAP`, non-cached) and registers the VMA in the hash table via `phxfs_add_phony_buffer`. The mapping is lazy — it does not pin GPU memory yet.

### `ioctl`
- `PHXFS_IOCTL_MAP`: map a device (GPU) address into the user-space VMA. Internally `phxfs_map_dev_addr` → `phxfs_map_dev_addr_inner` calls `nvidia_p2p_get_pages` to pin GPU pages, then `vm_insert_pages` to insert them into the VMA.
- `PHXFS_IOCTL_UNMAP`: release the mapping (`phxfs_map_dev_release`).

## Build & install

See [install.md](install.md) and [troubleshooting.md](troubleshooting.md).
