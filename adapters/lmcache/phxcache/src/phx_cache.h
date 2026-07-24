#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>


#include <phoenix.h>

/// Encapsulates a Phoenix device connection (phxfs_open / phxfs_close)
/// and device memory registration (phxfs_regmem / phxfs_deregmem).
class PhxCache {
public:
    /// Initialize: phxfs_find_dev + phxfs_open
    /// \param device_id  Vendor-specific device index (CUDA/HIP/NPU)
    explicit PhxCache(int device_id);
    ~PhxCache();

    // Non-copyable
    PhxCache(const PhxCache &) = delete;
    PhxCache &operator=(const PhxCache &) = delete;

    /// Register device memory for Phoenix DMA.
    /// Size is internally aligned up to the device page size.
    /// \return target_addr (DMA-mapped address) to pass to read
    uintptr_t regmem(uintptr_t dev_addr, size_t size);

    /// Deregister device memory previously registered with regmem().
    void deregmem(uintptr_t dev_addr, size_t size);

    int device_id() const { return dev_; }

    /// Return the device page size in bytes (vendor-specific, e.g. 64 KiB
    /// for NVIDIA). Queried from phxfs at runtime.
    uint64_t page_size() const;

    /// Close the phxfs device. Called automatically by destructor.
    void close();

    /// Batch read: submit N read requests concurrently via phxfs_read_batch.
    /// All reads target the same registered GPU buffer (buf_base) at different
    /// offsets.  Worker pool + io_uring handles concurrency.
    /// \param buf_base  registered GPU base pointer (same for all requests)
    /// \param reqs      list of (fd, buf_offset, nbytes, f_offset)
    /// \return list of bytes-read per request (or negative errno)
    std::vector<ssize_t> read_batch(
        uintptr_t buf_base,
        const std::vector<std::tuple<int, off_t, size_t, off_t>> &reqs);

    /// Batch write: submit N write requests concurrently via phxfs_write_batch.
    /// Each request writes from its own CPU buffer (no phxfs_regmem needed).
    /// \param reqs  list of (fd, buf_ptr, buf_offset, nbytes, f_offset)
    /// \return list of bytes-written per request (or negative errno)
    std::vector<ssize_t> write_batch(
        const std::vector<std::tuple<int, uintptr_t, off_t, size_t, off_t>> &reqs);

private:
    int dev_;             // phxfs device id
    int device_id_;       // original vendor device id (CUDA/HIP/NPU)
    bool initialized_;

    // Tracks registered memory: dev_addr -> (aligned_size, target_addr)
    std::unordered_map<uintptr_t, std::pair<size_t, uintptr_t>> reg_map_;
};

/// Encapsulates a file opened for Phoenix I/O.
/// Manages the POSIX fd and constructs phxfs_fileid_t for phxfs_read.
class PhxFile {
public:
    /// Open a file and construct phxfs_fileid_t{fd, device_id}
    /// \param cache  PhxCache instance (provides device_id)
    /// \param path   File path
    /// \param flags  open() flags (e.g. O_RDONLY | O_DIRECT)
    PhxFile(const PhxCache &cache, const std::string &path, int flags);
    ~PhxFile();

    // Non-copyable
    PhxFile(const PhxFile &) = delete;
    PhxFile &operator=(const PhxFile &) = delete;

    /// Synchronous read: phxfs_read(fid, buf, buf_offset, nbyte, f_offset)
    /// \return bytes read, or negative on error
    ssize_t read(uintptr_t buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);

    /// Synchronous write: phxfs_write(fid, buf, buf_offset, nbyte, f_offset)
    ssize_t write(uintptr_t buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);

    void close();

private:
    phxfs_fileid_t fid_;
    bool owns_fd_;
};
