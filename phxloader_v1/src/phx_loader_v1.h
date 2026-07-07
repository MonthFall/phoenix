#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

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

class PhxLoaderV1 {
public:
    explicit PhxLoaderV1(int cuda_device_id);
    ~PhxLoaderV1();

    // Non-copyable
    PhxLoaderV1(const PhxLoaderV1 &) = delete;
    PhxLoaderV1 &operator=(const PhxLoaderV1 &) = delete;

    /// Register GPU memory for DMA. Returns CPU mapped address.
    /// Size is internally aligned up to 64K (HUGE_PAGE_SIZE).
    uintptr_t regmem(void *gpu_ptr, size_t size);

    /// Deregister GPU memory previously registered with regmem().
    void deregmem(void *gpu_ptr, size_t size);

    /// Read the entire data section of a safetensors file in one DMA.
    /// gpu_ptr must point to a registered GPU buffer.
    /// data_offset = header_size (byte offset of data section in file).
    /// data_size = max_end (byte length of data section).
    /// Returns pre_padding = data_offset - align_down(data_offset, 4096),
    /// so the actual tensor data starts at gpu_ptr + pre_padding.
    off_t read_data_section(const std::string &path, uintptr_t gpu_ptr,
                            off_t data_offset, size_t data_size);

    /// Close the phxfs device. Called automatically by destructor.
    void close();

private:
    int dev_;            // phxfs device id
    int cuda_device_;    // original CUDA device id
    bool initialized_;

    // Tracks registered memory: gpu_ptr -> (aligned_size, cpu_target_addr)
    std::unordered_map<uintptr_t, std::pair<size_t, uintptr_t>> reg_map_;
};
