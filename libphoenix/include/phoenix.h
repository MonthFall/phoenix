#ifndef __PHOENIX_H__
#define __PHOENIX_H__
/* Public C API: use C headers only so this compiles under a plain C compiler
 * as well as C++ (P1-8). */
#include <stddef.h>
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

/*
 * Register/deregister a device (GPU) buffer for direct I/O.
 *
 * IMPORTANT (P2-5): read/write and the batch API identify a buffer by its
 * original DEVICE address (`addr` here, `buf` there) — NOT by `*target_addr`.
 * `*target_addr` is an internal host-mapped handle returned for reference; do
 * not pass it back as the I/O buffer or the lookup will miss. `addr` must be
 * nonzero, `len` nonzero and 64KiB-aligned; overlapping a live registration
 * is rejected, while an exact-duplicate registration is reference-counted and
 * reused (deregister once per register).
 */
int phxfs_regmem(int device_id, const void *addr, size_t len, void **target_addr);
int phxfs_deregmem(int device, const void *addr, size_t len);

ssize_t phxfs_read(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);
ssize_t phxfs_write(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);

/* ------------------------------------------------------------------ *
 * Batch I/O
 *
 * A single call submits N independent I/O requests to the underlying
 * async engine (io_uring today; libaio / sync-fallback selected at
 * runtime). This removes the per-request syscall overhead of looping
 * over phxfs_read/phxfs_write, and lets the storage layer service the
 * requests concurrently.
 *
 * The target buffer is selected by the SIGN of device_id:
 *   - device_id >= 0: a GPU buffer previously registered with phxfs_regmem
 *     on that phxfs device. `buf` must lie inside a registration or the
 *     request fails (-EFAULT) — there is no silent CPU fallback.
 *   - device_id  < 0: an ordinary CPU buffer (e.g. pinned host memory used
 *     by a store path); `buf` is used as a plain host address.
 *
 * A batch may mix requests targeting different devices. Requests are grouped
 * by the NUMA node of their target GPU and each group runs on that node's
 * pool, so a mixed-device batch does not force every request onto the first
 * device's node (this applies to both the synchronous and asynchronous batch).
 * ------------------------------------------------------------------ */
typedef struct phxfs_io_req {
    int      fd;          /* open file descriptor (O_DIRECT recommended) */
    int      device_id;   /* >=0: phxfs GPU device the buf is registered on; */
                          /*  <0: plain CPU buffer */
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
 *
 * Lifetime contract (applies to both sync and async batches): every
 * resource each request references must stay valid until the call (sync)
 * or phxfs_batch_wait() (async) returns:
 *   - reqs[i].fd must remain open (fds are NOT dup'd internally; if you
 *     close and the number is reused, I/O may hit the wrong file);
 *   - the CPU buffer (for host-buffer requests) must not be freed;
 *   - for registered GPU buffers, the phxfs_regmem registration is held by
 *     an internal reference for the batch's duration — a concurrent
 *     phxfs_deregmem() on it blocks until the batch completes rather than
 *     unmapping under an in-flight transfer.
 */
int phxfs_read_batch(phxfs_io_req_t *reqs, int n);
int phxfs_write_batch(phxfs_io_req_t *reqs, int n);

/*
 * Asynchronous batch — submit returns immediately after queuing the batch
 * on the internal NUMA pool; the caller can run GPU compute meanwhile, then
 * call phxfs_batch_wait() to block for completion and get per-request
 * results. Enables compute/I/O overlap.
 *
 *   reqs (and every resource it references — see the lifetime contract on
 *   phxfs_read_batch above: fd, CPU buffer, GPU registration) MUST remain
 *   valid until phxfs_batch_wait()/phxfs_batch_destroy() returns, and its
 *   fields MUST NOT be modified after submit.
 *   The batch is partitioned by target NUMA node and one sub-batch is queued
 *   per node. At most one async batch may be in flight per NUMA node; submit
 *   is NON-BLOCKING. If EVERY target node is busy, submit returns NULL with
 *   errno == EBUSY and runs nothing. If only SOME target nodes are busy,
 *   submit still queues the free nodes and returns a handle; the busy nodes'
 *   requests are reported as -EBUSY by phxfs_batch_wait, so the caller can
 *   retry exactly those without redoing already-committed I/O.
 *
 * phxfs_batch_submit_read/write return an opaque handle, or NULL on error
 * (errno == EBUSY if a target node is busy, ENOMEM on allocation failure,
 * ENOTSUP if the worker pool is unavailable).
 * phxfs_batch_wait returns the number of failed requests (>=0), or negative
 * on error, copies per-request results back, and frees the handle.
 * phxfs_batch_destroy abandons a batch: it waits for in-flight I/O to quiesce
 * (submitted I/O cannot be cancelled), then frees the handle WITHOUT copying
 * results back — for callers that drop a batch without needing its results.
 */
typedef struct phxfs_batch phxfs_batch_t;

phxfs_batch_t *phxfs_batch_submit_read(phxfs_io_req_t *reqs, int n);
phxfs_batch_t *phxfs_batch_submit_write(phxfs_io_req_t *reqs, int n);
int phxfs_batch_wait(phxfs_batch_t *handle);
int phxfs_batch_destroy(phxfs_batch_t *handle);

/*
 * Name of the active I/O engine selected at runtime
 * ("io_uring", "sync", ...). For diagnostics / tests.
 */
const char *phxfs_io_engine_name(void);

#ifdef __cplusplus
}
#endif

#endif
