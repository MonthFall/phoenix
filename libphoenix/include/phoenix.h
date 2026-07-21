#ifndef __PHOENIX_H__
#define __PHOENIX_H__
#include <cstddef>
#include <cstdint>
#include <stdint.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct phxfs_fileid {
    int fd;
    int deviceID;
} phxfs_fileid_t;

int phxfs_open(int device_id);
int phxfs_close(int device_id);

/* Returns the device page size in bytes (NVIDIA=64KB, etc.). */
uint64_t phxfs_get_page_size(void);

/*
 * Map a vendor-specific device ID to a phxfs device index.
 * The vendor is selected at build time (PHXFS_VENDOR).
 *   NVIDIA: device_id is a CUDA device ID
 *   AMD:    device_id is a HIP device ID
 *   Huawei: device_id is an NPU ID
 */
int phxfs_find_dev(int device_id);

/* Backward-compatible wrapper for existing NVIDIA callers */
static inline int phxfs_find_dev_for_cuda_gpu(int cuda_gpu_id) {
    return phxfs_find_dev(cuda_gpu_id);
}

ssize_t phxfs_read(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);
ssize_t phxfs_write(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);

int phxfs_regmem(int device_id, const void *addr, size_t len, void **target_addr);
int phxfs_deregmem(int device, const void *addr, size_t len);

/*
 * Async I/O — vendor-agnostic.
 *   stream: vendor stream handle (CUstream / hipStream_t / ...)
 *   Returns 0 on success.
 */
int phxfs_read_async(phxfs_fileid_t fid,
                     void *buf,
                     size_t nbytes, off_t offset,
                     ssize_t *bytes_done,
                     void *stream);

int phxfs_write_async(phxfs_fileid_t fid,
                      void *buf,
                      size_t nbytes, off_t offset,
                      ssize_t *bytes_done,
                      void *stream);

/* ------------------------------------------------------------------ *
 * Batch I/O
 *
 * A single call submits N independent I/O requests to the underlying
 * async engine (io_uring today; libaio / sync-fallback selected at
 * runtime). This removes the per-request syscall overhead of looping
 * over phxfs_read/phxfs_write, and lets the storage layer service the
 * requests concurrently.
 *
 * Each request targets a buffer that may be either:
 *   - a GPU buffer previously registered with phxfs_regmem (resolved to
 *     its host-mapped P2P VMA internally), or
 *   - an ordinary CPU buffer (e.g. pinned host memory used by a store
 *     path). The engine handles both transparently.
 * ------------------------------------------------------------------ */
typedef struct phxfs_io_req {
    int      fd;          /* open file descriptor (O_DIRECT recommended) */
    int      device_id;   /* phxfs device the buf is registered on (GPU); */
                          /* ignored for plain CPU buffers */
    void    *buf;         /* GPU addr (registered) or CPU addr */
    off_t    buf_offset;  /* byte offset within buf */
    size_t   nbytes;      /* transfer length */
    off_t    f_offset;    /* file offset */
    ssize_t  result;      /* OUT: bytes transferred, or negative errno */
} phxfs_io_req_t;

/*
 * Synchronous batch: submit all N requests, wait for completion, fill
 * each req->result. Returns 0 if every request completed with
 * result == nbytes; otherwise returns the number of failed requests
 * (>0), or a negative value on a submission-level error.
 */
int phxfs_read_batch(phxfs_io_req_t *reqs, int n);
int phxfs_write_batch(phxfs_io_req_t *reqs, int n);

/*
 * Asynchronous batch — submit returns immediately after queuing the batch
 * on the internal NUMA pool; the caller can run GPU compute meanwhile, then
 * call phxfs_batch_wait() to block for completion and get per-request
 * results. Enables compute/I/O overlap.
 *
 *   reqs MUST remain valid until phxfs_batch_wait() returns.
 *   At most one async batch may be in flight per NUMA node (the batch owns
 *   that node's pool between submit and wait).
 *
 * phxfs_batch_submit_read/write return an opaque handle (NULL on error).
 * phxfs_batch_wait returns the number of failed requests (>=0), or negative
 * on error, and frees the handle.
 */
typedef struct phxfs_batch phxfs_batch_t;

phxfs_batch_t *phxfs_batch_submit_read(phxfs_io_req_t *reqs, int n);
phxfs_batch_t *phxfs_batch_submit_write(phxfs_io_req_t *reqs, int n);
int phxfs_batch_wait(phxfs_batch_t *handle);

/*
 * Name of the active I/O engine selected at runtime
 * ("io_uring", "sync", ...). For diagnostics / tests.
 */
const char *phxfs_io_engine_name(void);

#ifdef __cplusplus
}
#endif

#endif
