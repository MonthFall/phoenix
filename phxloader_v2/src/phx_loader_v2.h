#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// Forward declarations for phoenix C API
extern "C" {
int phxfs_open(int device_id);
int phxfs_close(int device_id);
int phxfs_find_dev_for_cuda_gpu(int cuda_gpu_id);
ssize_t phxfs_read(struct phxfs_fileid fid, void *buf, off_t buf_offset,
                    ssize_t nbyte, off_t f_offset);
int phxfs_regmem(int device_id, const void *addr, size_t len,
                 void **target_addr);
int phxfs_deregmem(int device, const void *addr, size_t len);
}

class PhxLoaderV2 {
public:
    explicit PhxLoaderV2(int cuda_device_id);
    ~PhxLoaderV2();

    // Non-copyable
    PhxLoaderV2(const PhxLoaderV2 &) = delete;
    PhxLoaderV2 &operator=(const PhxLoaderV2 &) = delete;

    /// Register GPU memory for DMA. Returns CPU mapped address.
    /// Size is internally aligned up to 64K (HUGE_PAGE_SIZE).
    uintptr_t regmem(void *gpu_ptr, size_t size);

    /// Deregister GPU memory previously registered with regmem().
    void deregmem(void *gpu_ptr, size_t size);

    /// Batch read multiple (buf_offset, file_offset, nbytes) entries from
    /// a single file into the registered GPU buffer. Each entry's
    /// buf_offset, file_offset, and nbytes must be 4K-aligned (handled by
    /// the Python side via build_read_groups). Serial phxfs_read per entry.
    void read_into_registered(
        const std::string &path, uintptr_t gpu_ptr,
        const std::vector<std::tuple<off_t, off_t, size_t>> &batch);

    /// Close the phxfs device. Called automatically by destructor.
    void close();

private:
    int dev_;            // phxfs device id
    int cuda_device_;    // original CUDA device id
    bool initialized_;

    // Tracks registered memory: gpu_ptr -> (aligned_size, cpu_target_addr)
    std::unordered_map<uintptr_t, std::pair<size_t, uintptr_t>> reg_map_;
};
