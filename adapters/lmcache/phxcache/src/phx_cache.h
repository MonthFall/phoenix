#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

// phoenix.h includes <cuda_runtime.h> before its own extern "C" block,
// so we must NOT wrap it in extern "C" here — otherwise cuda_runtime.h's
// C++ templates would be trapped inside C linkage.
#include <phoenix.h>

/// Result of an async read/write operation.
/// The bytes_done field is only valid after the CUDA stream synchronize().
///
/// **Lifetime**: ``read_async`` / ``write_async`` return a
/// ``shared_ptr<AsyncReadResult>`` because ``phxfs_read_async`` stores a
/// raw pointer to ``bytes_done_`` and updates it asynchronously when the
/// DMA completes.  The object must therefore live on the heap and survive
/// until the caller synchronises the stream.  Returning by value would
/// destroy the stack-local object, leaving ``phxfs_read_async`` with a
/// dangling pointer.
class AsyncReadResult {
public:
    AsyncReadResult() : bytes_done_(-1) {}

    ssize_t bytes_done() const { return bytes_done_; }
    ssize_t *ptr() { return &bytes_done_; }

private:
    ssize_t bytes_done_;
};

/// Encapsulates a Phoenix device connection (phxfs_open / phxfs_close)
/// and GPU memory registration (phxfs_regmem / phxfs_deregmem).
class PhxCache {
public:
    /// Initialize: phxfs_find_dev_for_cuda_gpu + phxfs_open
    /// \param cuda_gpu_id  CUDA GPU index (e.g. 0 for cuda:0)
    explicit PhxCache(int cuda_gpu_id);
    ~PhxCache();

    // Non-copyable
    PhxCache(const PhxCache &) = delete;
    PhxCache &operator=(const PhxCache &) = delete;

    /// Register GPU memory for Phoenix DMA.
    /// Size is internally aligned up to 64K (GPU_PAGE_SIZE).
    /// \return target_addr (DMA-mapped address) to pass to read/read_async
    uintptr_t regmem(uintptr_t gpu_addr, size_t size);

    /// Deregister GPU memory previously registered with regmem().
    void deregmem(uintptr_t gpu_addr, size_t size);

    int device_id() const { return dev_; }

    /// Close the phxfs device. Called automatically by destructor.
    void close();

private:
    int dev_;             // phxfs device id
    int cuda_device_;     // original CUDA device id
    bool initialized_;

    // Tracks registered memory: gpu_addr -> (aligned_size, target_addr)
    std::unordered_map<uintptr_t, std::pair<size_t, uintptr_t>> reg_map_;
};

/// Encapsulates a file opened for Phoenix I/O.
/// Manages the POSIX fd and constructs phxfs_fileid_t for phxfs_read/read_async.
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

    /// Asynchronous read: phxfs_read_async(fid, buf, nbytes, offset, &bytes_done, stream)
    /// Submits DMA to the CUDA stream. Caller must synchronize stream before
    /// checking result.bytes_done.
    /// \param buf    GPU buffer address (must be registered via regmem)
    /// \param nbytes Number of bytes to read
    /// \param offset File offset to read from
    /// \param stream CUstream handle (uintptr_t from torch.cuda.Stream.cuda_stream)
    /// \return shared_ptr<AsyncReadResult> — check .bytes_done after
    ///         stream.synchronize().  Heap-allocated so the internal
    ///         bytes_done pointer remains valid until the DMA completes.
    std::shared_ptr<AsyncReadResult> read_async(uintptr_t buf, size_t nbytes,
                                                 off_t offset, uintptr_t stream);

    /// Synchronous write: phxfs_write(fid, buf, buf_offset, nbyte, f_offset)
    /// (Provided for completeness; PhxBackend store uses POSIX write instead)
    ssize_t write(uintptr_t buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);

    /// Asynchronous write: phxfs_write_async(fid, buf, nbytes, offset, &bytes_done, stream)
    std::shared_ptr<AsyncReadResult> write_async(uintptr_t buf, size_t nbytes,
                                                  off_t offset, uintptr_t stream);

    void close();

private:
    phxfs_fileid_t fid_;
    bool owns_fd_;
};
