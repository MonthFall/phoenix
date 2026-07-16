#include "phx_cache.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <system_error>

namespace {

constexpr size_t GPU_PAGE_SIZE = 64 * 1024;  // HUGE_PAGE_SIZE in phoenix

inline size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

}  // namespace

// ---------------------------------------------------------------------------
// PhxCache
// ---------------------------------------------------------------------------

PhxCache::PhxCache(int cuda_gpu_id)
    : cuda_device_(cuda_gpu_id), initialized_(false) {
    dev_ = phxfs_find_dev_for_cuda_gpu(cuda_gpu_id);
    if (dev_ < 0) {
        throw std::runtime_error(
            "PhxCache: phxfs_find_dev_for_cuda_gpu(" +
            std::to_string(cuda_gpu_id) + ") failed with " +
            std::to_string(dev_));
    }

    int ret = phxfs_open(dev_);
    if (ret < 0) {
        throw std::runtime_error("PhxCache: phxfs_open(" +
                                 std::to_string(dev_) + ") failed with " +
                                 std::to_string(ret));
    }
    initialized_ = true;
}

PhxCache::~PhxCache() {
    if (initialized_) {
        try {
            close();
        } catch (...) {
        }
    }
}

uintptr_t PhxCache::regmem(uintptr_t gpu_addr, size_t size) {
    size_t aligned_size = align_up(size, GPU_PAGE_SIZE);

    void *target_addr = nullptr;
    int ret = phxfs_regmem(dev_, reinterpret_cast<const void *>(gpu_addr),
                           aligned_size, &target_addr);
    if (ret < 0) {
        throw std::runtime_error(
            "PhxCache::regmem: phxfs_regmem failed with " +
            std::to_string(ret) + ", size=" + std::to_string(size) +
            ", aligned_size=" + std::to_string(aligned_size));
    }

    auto mapped = reinterpret_cast<uintptr_t>(target_addr);
    reg_map_[gpu_addr] = {aligned_size, mapped};
    return mapped;
}

void PhxCache::deregmem(uintptr_t gpu_addr, size_t size) {
    auto it = reg_map_.find(gpu_addr);
    if (it == reg_map_.end()) {
        return;  // Not registered, nothing to do
    }

    size_t aligned_size = it->second.first;
    int ret = phxfs_deregmem(dev_, reinterpret_cast<const void *>(gpu_addr),
                             aligned_size);
    if (ret < 0) {
        fprintf(stderr,
                "PhxCache::deregmem: phxfs_deregmem failed with %d\n", ret);
    }
    reg_map_.erase(it);
}

void PhxCache::close() {
    if (!initialized_) {
        return;
    }

    // Deregister any remaining registered memory
    for (const auto &[gpu_addr, info] : reg_map_) {
        int ret = phxfs_deregmem(dev_, reinterpret_cast<const void *>(gpu_addr),
                                 info.first);
        if (ret < 0) {
            fprintf(stderr,
                    "PhxCache::close: phxfs_deregmem failed with %d\n", ret);
        }
    }
    reg_map_.clear();

    int ret = phxfs_close(dev_);
    if (ret < 0) {
        fprintf(stderr, "PhxCache::close: phxfs_close failed with %d\n", ret);
    }
    initialized_ = false;
}

// ---------------------------------------------------------------------------
// PhxFile
// ---------------------------------------------------------------------------

PhxFile::PhxFile(const PhxCache &cache, const std::string &path, int flags)
    : owns_fd_(false) {
    int fd = ::open(path.c_str(), flags);
    if (fd < 0) {
        throw std::system_error(errno, std::system_category(),
                                "PhxFile: open(" + path + ")");
    }
    fid_.fd = fd;
    fid_.deviceID = cache.device_id();
    owns_fd_ = true;
}

PhxFile::~PhxFile() {
    if (owns_fd_) {
        try {
            close();
        } catch (...) {
        }
    }
}

ssize_t PhxFile::read(uintptr_t buf, off_t buf_offset, ssize_t nbyte,
                      off_t f_offset) {
    ssize_t result = phxfs_read(
        fid_,
        reinterpret_cast<void *>(buf),
        buf_offset,
        nbyte,
        f_offset);
    if (result < 0) {
        throw std::runtime_error(
            "PhxFile::read: phxfs_read failed with " +
            std::to_string(result) + ", buf_offset=" +
            std::to_string(buf_offset) + ", nbyte=" +
            std::to_string(nbyte) + ", f_offset=" +
            std::to_string(f_offset));
    }
    return result;
}

std::shared_ptr<AsyncReadResult> PhxFile::read_async(uintptr_t buf, size_t nbytes,
                                                     off_t offset, uintptr_t stream) {
    // Heap-allocate: phxfs_read_async stores a raw pointer to bytes_done_
    // and updates it asynchronously when the DMA completes.  The object
    // must survive until the caller calls stream.synchronize().
    auto result = std::make_shared<AsyncReadResult>();
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    cudaError_t err = phxfs_read_async(
        fid_,
        reinterpret_cast<void *>(buf),
        nbytes,
        offset,
        result->ptr(),
        cu_stream);

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("PhxFile::read_async: phxfs_read_async failed: ") +
            cudaGetErrorString(err));
    }
    return result;
}

ssize_t PhxFile::write(uintptr_t buf, off_t buf_offset, ssize_t nbyte,
                       off_t f_offset) {
    ssize_t result = phxfs_write(
        fid_,
        reinterpret_cast<void *>(buf),
        buf_offset,
        nbyte,
        f_offset);
    if (result < 0) {
        throw std::runtime_error(
            "PhxFile::write: phxfs_write failed with " +
            std::to_string(result));
    }
    return result;
}

std::shared_ptr<AsyncReadResult> PhxFile::write_async(uintptr_t buf, size_t nbytes,
                                                      off_t offset, uintptr_t stream) {
    auto result = std::make_shared<AsyncReadResult>();
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    cudaError_t err = phxfs_write_async(
        fid_,
        reinterpret_cast<void *>(buf),
        nbytes,
        offset,
        result->ptr(),
        cu_stream);

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("PhxFile::write_async: phxfs_write_async failed: ") +
            cudaGetErrorString(err));
    }
    return result;
}

void PhxFile::close() {
    if (owns_fd_) {
        ::close(fid_.fd);
        owns_fd_ = false;
    }
}
