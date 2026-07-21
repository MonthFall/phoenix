/*
 * io_uring batch engine.
 *
 * Submits a whole batch of reads/writes with a single io_uring_submit(),
 * then reaps all completions — eliminating the per-request syscall of the
 * sync engine and letting the storage layer service requests concurrently.
 *
 * Concurrency & runtime-cost:
 *   - Per-thread ring + scratch pool (thread_local). A single submitting
 *     thread saturates only ~10 GiB/s on a 4-disk RAID0; multiple worker
 *     threads (e.g. one per GPU) each get their own ring and submit
 *     concurrently with no lock/contention, reaching the array's limit.
 *   - The ring (QD=1024) and pool are created lazily on a thread's first
 *     batch and reused for its lifetime; no per-call setup/teardown.
 *   - slice/iovec scratch is a fixed per-thread pool, so the hot path
 *     performs no malloc.
 *
 * Buffer handling: requests carry already-resolved host addresses (GPU
 * P2P VMA or CPU buffer). We deliberately use the *non-fixed* OP_READ /
 * OP_READV path (not registered buffers / READ_FIXED): it goes through the
 * same O_DIRECT DIO path as pread — which already works for phxfs GPU
 * VMAs — and avoids taking a second FOLL_LONGTERM pin on device pages.
 */
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <liburing.h>

#include "io_engine.h"

/* Per-request chunking cap (MAX_RW_COUNT ~= 2GiB). KV I/O rarely hits it. */
#define PHXFS_IO_CHUNK (1024ULL * 1024 * 1024)  /* 1 GiB */

/* Ring depth. KV-cache retrieve issues many concurrent small reads. */
#define PHXFS_URING_QD 1024

/* Pre-allocated scratch pool. One slot per in-flight slice; batches with
 * more slices than this are processed in multiple waves reusing the pool. */
#define PHXFS_SLICE_POOL 16384

/*
 * A "slice" is one SQE covering [off, off+len) of request `idx`.
 * Each request is fanned out into <=chunk slices so no single op exceeds
 * PHXFS_IO_CHUNK.
 */
struct slice {
    int    idx;
    size_t off;
    size_t len;
};

/* Per-thread ring + scratch (thread_local => lock-free concurrent submit). */
static thread_local struct io_uring t_ring;
static thread_local bool            t_ring_ready = false;
static thread_local struct slice    t_slices[PHXFS_SLICE_POOL];
static thread_local struct iovec    t_iovs[PHXFS_SLICE_POOL];

/* OP_READ vs OP_READV capability — process-wide, resolved once. */
static bool g_have_rw = false;
static bool g_uring_avail = false;

/*
 * Determine whether OP_READ (single-buffer) is accepted, using a harmless
 * zero-length read. IORING_REGISTER_PROBE is unreliable on some backported
 * kernels, so we test the opcode directly. Falls back to OP_READV. Also
 * confirms io_uring is usable at all (g_uring_avail).
 */
static void detect_caps(void) {
    struct io_uring ring;
    if (io_uring_queue_init(2, &ring, 0) < 0)
        return;  /* g_uring_avail stays false */
    g_uring_avail = true;
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    char c;
    io_uring_prep_read(sqe, /*fd*/ 0, &c, 0, 0);  /* 0-length: no data moved */
    if (io_uring_submit(&ring) == 1) {
        struct io_uring_cqe *cqe;
        if (io_uring_wait_cqe(&ring, &cqe) == 0) {
            if (cqe->res != -EINVAL && cqe->res != -EOPNOTSUPP)
                g_have_rw = true;
            io_uring_cqe_seen(&ring, cqe);
        }
    }
    io_uring_queue_exit(&ring);
}

/* Lazily create this thread's ring on first use. */
static bool thread_ring(void) {
    if (t_ring_ready)
        return true;
    if (!g_uring_avail)
        return false;
    if (io_uring_queue_init(PHXFS_URING_QD, &t_ring, 0) < 0)
        return false;
    t_ring_ready = true;
    return true;
}

static int uring_probe(void) {
    detect_caps();
    return g_uring_avail ? 0 : -1;
}

/* Prep a single slice into an sqe (OP_READ or OP_READV per capability). */
static void prep_op(struct io_uring_sqe *sqe, struct phxfs_io_op_req *r,
                    size_t off, size_t len, enum phxfs_io_op op,
                    struct iovec *iov) {
    char *addr = (char *)r->host_addr + off;
    off_t foff = r->f_offset + (off_t)off;
    if (g_have_rw) {
        if (op == PHXFS_IO_READ)
            io_uring_prep_read(sqe, r->fd, addr, (unsigned)len, foff);
        else
            io_uring_prep_write(sqe, r->fd, addr, (unsigned)len, foff);
    } else {
        iov->iov_base = addr;
        iov->iov_len = len;
        if (op == PHXFS_IO_READ)
            io_uring_prep_readv(sqe, r->fd, iov, 1, foff);
        else
            io_uring_prep_writev(sqe, r->fd, iov, 1, foff);
    }
}

/* Build slices for [lo, hi) requests into this thread's pool. */
static size_t build_slices(struct phxfs_io_op_req *reqs, int lo, int hi) {
    size_t s = 0;
    for (int i = lo; i < hi && s < PHXFS_SLICE_POOL; i++) {
        size_t rem = reqs[i].nbytes, off = 0;
        if (rem == 0) {
            t_slices[s++] = {i, 0, 0};
            continue;
        }
        while (rem && s < PHXFS_SLICE_POOL) {
            size_t len = rem > PHXFS_IO_CHUNK ? PHXFS_IO_CHUNK : rem;
            t_slices[s++] = {i, off, len};
            off += len;
            rem -= len;
        }
    }
    return s;
}

/* Account one completion into the owning request's result. */
static inline void account(struct phxfs_io_op_req *reqs, struct slice *sl,
                           int res) {
    int idx = sl->idx;
    if (res < 0) {
        if (reqs[idx].result >= 0)
            reqs[idx].result = res;       /* first errno wins */
    } else if (reqs[idx].result >= 0) {
        reqs[idx].result += res;
    }
}

/*
 * Submit `nslices` from this thread's pool as a *pipeline*: keep up to QD
 * slices in flight at all times. Each time a completion is reaped, one new
 * slice is queued, so the device queue never drains between "waves". This
 * is what lets a single ring approach the array's bandwidth ceiling.
 */
static int run_slices(struct phxfs_io_op_req *reqs, size_t nslices,
                      enum phxfs_io_op op) {
    size_t next = 0;       /* next slice to submit */
    size_t inflight = 0;
    size_t done = 0;

    /* Prime the pipeline up to QD. */
    while (next < nslices && inflight < PHXFS_URING_QD) {
        struct slice *sl = &t_slices[next];
        struct io_uring_sqe *sqe = io_uring_get_sqe(&t_ring);
        if (!sqe)
            break;
        prep_op(sqe, &reqs[sl->idx], sl->off, sl->len, op, &t_iovs[next]);
        io_uring_sqe_set_data(sqe, sl);
        next++;
        inflight++;
    }
    if (inflight && io_uring_submit(&t_ring) < 0)
        return -errno;

    /* Steady state: reap one, refill one. */
    while (done < nslices) {
        struct io_uring_cqe *cqe;
        int ret = io_uring_wait_cqe(&t_ring, &cqe);
        if (ret < 0)
            return ret;
        account(reqs, (struct slice *)io_uring_cqe_get_data(cqe), cqe->res);
        io_uring_cqe_seen(&t_ring, cqe);
        inflight--;
        done++;

        /* Refill as many as the ring allows, then submit in one shot. */
        int added = 0;
        while (next < nslices && inflight < PHXFS_URING_QD) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&t_ring);
            if (!sqe)
                break;
            struct slice *sl = &t_slices[next];
            prep_op(sqe, &reqs[sl->idx], sl->off, sl->len, op, &t_iovs[next]);
            io_uring_sqe_set_data(sqe, sl);
            next++;
            inflight++;
            added++;
        }
        if (added && io_uring_submit(&t_ring) < 0)
            return -errno;
    }
    return 0;
}

static int uring_submit_batch(struct phxfs_io_op_req *reqs, int n,
                              enum phxfs_io_op op) {
    if (n <= 0)
        return 0;
    if (!thread_ring())
        return -EIO;

    for (int i = 0; i < n; i++)
        reqs[i].result = 0;

    int rc = 0;
    /*
     * Process requests in blocks whose slice count fits the pool. Common
     * case (n <= pool, each req <= chunk) is a single block.
     */
    int i = 0;
    while (i < n) {
        int j = i;
        size_t need = 0;
        while (j < n) {
            size_t rem = reqs[j].nbytes;
            size_t s = rem ? (rem + PHXFS_IO_CHUNK - 1) / PHXFS_IO_CHUNK : 1;
            if (need + s > PHXFS_SLICE_POOL)
                break;
            need += s;
            j++;
        }
        if (j == i)  /* single request larger than pool: force it alone */
            j = i + 1;

        size_t ns = build_slices(reqs, i, j);
        rc = run_slices(reqs, ns, op);
        if (rc < 0)
            break;
        i = j;
    }

    if (rc < 0)
        return rc;

    int failed = 0;
    for (int k = 0; k < n; k++)
        if (reqs[k].result != (ssize_t)reqs[k].nbytes)
            failed++;
    return failed;
}

const struct phxfs_io_engine phxfs_io_engine_uring = {
    "io_uring",
    uring_probe,
    uring_submit_batch,
};
