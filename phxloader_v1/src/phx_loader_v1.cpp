#include "phx_loader_v1.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <system_error>

// phoenix C API header
#include <phoenix.h>

namespace {

constexpr size_t PAGE_SIZE = 4096;
constexpr size_t GPU_PAGE_SIZE = 64 * 1024;  // HUGE_PAGE_SIZE in phoenix

inline size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

inline size_t align_down(size_t val, size_t alignment) {
    return val & ~(alignment - 1);
}

}  // namespace

PhxLoaderV1::PhxLoaderV1(int cuda_device_id)
    : cuda_device_(cuda_device_id), initialized_(false) {
    dev_ = phxfs_find_dev_for_cuda_gpu(cuda_device_id);
    if (dev_ < 0) {
        throw std::runtime_error(
            "PhxLoaderV1: phxfs_find_dev_for_cuda_gpu(" +
            std::to_string(cuda_device_id) + ") failed with " +
            std::to_string(dev_));
    }

    int ret = phxfs_open(dev_);
    if (ret < 0) {
        throw std::runtime_error("PhxLoaderV1: phxfs_open(" +
                                 std::to_string(dev_) + ") failed with " +
                                 std::to_string(ret));
    }
    initialized_ = true;
}

PhxLoaderV1::~PhxLoaderV1() {
    if (initialized_) {
        try {
            close();
        } catch (...) {
            // Suppress exceptions in destructor
        }
    }
}

uintptr_t PhxLoaderV1::regmem(void *gpu_ptr, size_t size) {
    // Align size up to GPU_PAGE_SIZE (64K), required by phxfs_regmem
    size_t aligned_size = align_up(size, GPU_PAGE_SIZE);

    void *target_addr = nullptr;
    int ret = phxfs_regmem(dev_, gpu_ptr, aligned_size, &target_addr);
    if (ret < 0) {
        throw std::runtime_error(
            "PhxLoaderV1::regmem: phxfs_regmem failed with " +
            std::to_string(ret) + ", size=" + std::to_string(size) +
            ", aligned_size=" + std::to_string(aligned_size));
    }

    auto key = reinterpret_cast<uintptr_t>(gpu_ptr);
    reg_map_[key] = {aligned_size, reinterpret_cast<uintptr_t>(target_addr)};

    return reinterpret_cast<uintptr_t>(target_addr);
}

void PhxLoaderV1::deregmem(void *gpu_ptr, size_t size) {
    auto key = reinterpret_cast<uintptr_t>(gpu_ptr);
    auto it = reg_map_.find(key);
    if (it == reg_map_.end()) {
        return;  // Not registered, nothing to do
    }

    size_t aligned_size = it->second.first;
    int ret = phxfs_deregmem(dev_, gpu_ptr, aligned_size);
    if (ret < 0) {
        // Log but don't throw in cleanup path
        fprintf(stderr,
                "PhxLoaderV1::deregmem: phxfs_deregmem failed with %d\n", ret);
    }

    reg_map_.erase(it);
}

off_t PhxLoaderV1::read_data_section(const std::string &path, uintptr_t gpu_ptr,
                                  off_t data_offset, size_t data_size) {
    // Verify gpu_ptr is registered
    auto it = reg_map_.find(gpu_ptr);
    if (it == reg_map_.end()) {
        throw std::runtime_error(
            "PhxLoaderV1::read_data_section: gpu_ptr " +
            std::to_string(gpu_ptr) + " not registered");
    }

    // Open file with O_DIRECT for DMA
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        throw std::system_error(errno, std::system_category(),
                                "PhxLoaderV1::read_data_section: open(" + path +
                                    ")");
    }

    try {
        // 4K-align the file offset down (O_DIRECT requirement)
        off_t aligned_offset = align_down(data_offset, PAGE_SIZE);
        off_t pre_padding = data_offset - aligned_offset;

        // Get file size to cap aligned_end
        struct stat st;
        if (fstat(fd, &st) != 0) {
            throw std::system_error(errno, std::system_category(),
                                    "PhxLoaderV1::read_data_section: fstat(" +
                                        path + ")");
        }
        off_t file_size = st.st_size;

        // 4K-align the read end up, but cap at file size
        off_t aligned_end = align_up(data_offset + data_size, PAGE_SIZE);
        if (aligned_end > file_size) {
            aligned_end = align_up(file_size, PAGE_SIZE);
        }
        size_t aligned_read_size = aligned_end - aligned_offset;

        // Sanity: we must have read enough to cover the requested data
        off_t data_end_in_file = data_offset + data_size;
        if (file_size < data_end_in_file) {
            throw std::runtime_error(
                "PhxLoaderV1::read_data_section: requested data extends beyond "
                "file size, data_end=" + std::to_string(data_end_in_file) +
                ", file_size=" + std::to_string(file_size) +
                ", path=" + path);
        }

        phxfs_fileid_t fid;
        fid.fd = fd;
        fid.deviceID = dev_;

        // Single DMA read for the entire data section
        // buf_offset = 0: data starts at the beginning of the GPU buffer,
        // with pre_padding bytes of header garbage at the front.
        ssize_t result = phxfs_read(
            fid, reinterpret_cast<void *>(gpu_ptr),
            0,                  // buf_offset
            aligned_read_size,  // total aligned size
            aligned_offset);    // aligned file offset

        if (result < 0) {
            throw std::runtime_error(
                "PhxLoaderV1::read_data_section: phxfs_read failed with " +
                std::to_string(result) + ", path=" + path +
                ", offset=" + std::to_string(aligned_offset) +
                ", size=" + std::to_string(aligned_read_size));
        }

        // Tolerate short reads for the tail padding region.
        // The actual data we need ends at (data_offset + data_size), which
        // is guaranteed to be within what was read (checked above).
        if (static_cast<size_t>(result) < static_cast<size_t>(data_end_in_file - aligned_offset)) {
            throw std::runtime_error(
                "PhxLoaderV1::read_data_section: short read, read only " +
                std::to_string(result) + " bytes from offset " +
                std::to_string(aligned_offset) +
                ", but need at least " +
                std::to_string(data_end_in_file - aligned_offset) +
                " bytes to cover requested data, path=" + path);
        }

        ::close(fd);
        return pre_padding;
    } catch (...) {
        ::close(fd);
        throw;
    }
}

void PhxLoaderV1::close() {
    if (!initialized_) {
        return;
    }

    // Deregister any remaining registered memory
    for (const auto &[gpu_ptr, info] : reg_map_) {
        void *ptr = reinterpret_cast<void *>(gpu_ptr);
        int ret = phxfs_deregmem(dev_, ptr, info.first);
        if (ret < 0) {
            fprintf(stderr,
                    "PhxLoaderV1::close: phxfs_deregmem failed with %d\n", ret);
        }
    }
    reg_map_.clear();

    int ret = phxfs_close(dev_);
    if (ret < 0) {
        fprintf(stderr, "PhxLoaderV1::close: phxfs_close failed with %d\n", ret);
    }
    initialized_ = false;
}
