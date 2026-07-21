#ifndef __PHXFS_IO_ENGINE_H__
#define __PHXFS_IO_ENGINE_H__

/*
 * Pluggable batch I/O engine abstraction (internal to libphoenix).
 *
 * The public phxfs_read_batch/phxfs_write_batch entry points dispatch to
 * whichever engine was selected at runtime. Engines are ranked by
 * preference; the first one that probes successfully wins:
 *
 *   io_uring  -> best on kernels that support it (default)
 *   libaio    -> (future) alternative async framework
 *   sync      -> pread/pwrite loop, always-available fallback
 *
 * Each request's target buffer has already been resolved to a host
 * virtual address the kernel can DMA into (GPU P2P VMA or CPU buffer),
 * so engines only deal with (fd, host_addr, nbytes, f_offset).
 */

#include <sys/types.h>
#include <cstddef>

enum phxfs_io_op {
    PHXFS_IO_READ = 0,
    PHXFS_IO_WRITE = 1,
};

/* One resolved request: host_addr is ready to hand to the kernel. */
struct phxfs_io_op_req {
    int      fd;
    void    *host_addr;   /* resolved DMA-able host VA */
    size_t   nbytes;
    off_t    f_offset;
    ssize_t  result;      /* OUT */
};

struct phxfs_io_engine {
    const char *name;

    /* Probe availability. Return 0 if usable, <0 otherwise. Called once. */
    int (*probe)(void);

    /*
     * Submit and complete a whole batch. Fills each req->result.
     * Returns 0 if all succeeded (result == nbytes), or the count of
     * failed requests (>0), or negative on a submission-level error.
     */
    int (*submit_batch)(struct phxfs_io_op_req *reqs, int n, enum phxfs_io_op op);
};

/*
 * Return the active engine, probing/selecting on first use.
 * Never returns NULL (sync engine is the guaranteed fallback).
 */
const struct phxfs_io_engine *phxfs_io_engine_get(void);

/* Engine instances (defined in their respective .cpp files). */
extern const struct phxfs_io_engine phxfs_io_engine_sync;
#ifdef PHXFS_HAVE_LIBURING
extern const struct phxfs_io_engine phxfs_io_engine_uring;
#endif

/* ------------------------------------------------------------------ *
 * NUMA-aware batch thread pool
 *
 * A batch of already-resolved requests is fanned out (round-robin) across
 * the worker threads pinned to a given NUMA node, each running the active
 * engine's submit_batch on its own thread_local ring. This lets a single
 * top-level call saturate the device without the caller managing threads
 * (important for the Python/pybind path, which must cross the GIL once).
 *
 * Requests are routed to the pool of the NUMA node that owns the target
 * GPU, avoiding cross-NUMA DMA.
 * ------------------------------------------------------------------ */

#define PHXFS_MAX_NUMA_NODES   8
#define PHXFS_THREADS_PER_NODE 4

/*
 * Run a whole batch synchronously via the pool for `numa_node`.
 * Fills each req->result. Returns the number of failed requests (>=0),
 * or negative on a pool-level error. Falls back to inline execution on
 * the calling thread if the pool is unavailable.
 */
int phxfs_pool_run(struct phxfs_io_op_req *reqs, int n, enum phxfs_io_op op,
                   int numa_node);

/* Opaque async handle for submit/wait (compute–I/O overlap). */
struct phxfs_batch_handle;

#endif /* __PHXFS_IO_ENGINE_H__ */
