/*
 * NUMA-aware batch I/O thread pool.
 *
 * One pool per NUMA node, PHXFS_THREADS_PER_NODE worker threads each, pinned to
 * that node's CPUs. Work is submitted as independent "jobs"; each job is run by
 * ALL of the node's workers cooperatively (round-robin stripe across its own
 * thread_local io_uring ring), so a single job still saturates the array's
 * bandwidth.
 *
 * Concurrency model (P2-2 — bounded FIFO queue per node):
 *   - Each node has a bounded FIFO queue of jobs. submit() enqueues a job and
 *     returns immediately (non-blocking); it only fails with EBUSY when the
 *     queue is full. This supports pipelining: a caller can submit several
 *     batches, overlap compute, then wait each handle — no more "one slot held
 *     until wait()" stalling the next submit.
 *   - Jobs run ONE AT A TIME (the worker group serves the queue head, then
 *     advances to the next), so every job keeps the full worker/bandwidth width
 *     — we deliberately do NOT split the workers across concurrent jobs.
 *   - Each job carries its own result state (failed/err/done); results are read
 *     by that job's wait(), decoupled from when the scheduling slot is freed.
 *   - A worker participates in the current job iff its lane < that job's stride;
 *     only participating workers touch the request array and decrement pending.
 */
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "io_engine.h"

/* Bounded in-flight jobs per node (queued + running). Enough to pipeline a few
 * batches for compute/I/O overlap without letting a runaway producer queue
 * unbounded work. */
#define PHXFS_POOL_QUEUE_CAP 16

/* One unit of work. Independent, heap- or stack-allocated by the submitter; its
 * result fields are filled by the workers and read by the matching wait(). */
struct pool_job {
    struct phxfs_io_op_req *reqs;
    int              n;
    enum phxfs_io_op op;
    int              failed;   /* OUT: accumulated failure count (under mtx) */
    int              err;      /* OUT: first engine-level (<0) error, else 0 */
    bool             done;     /* set once all participating workers finished */
    struct pool_job *next;     /* FIFO link (waiting queue) */
};

struct worker {
    pthread_t tid;
    int       node;
    int       lane;            /* 0..THREADS_PER_NODE-1 */
};

struct node_pool {
    struct worker   workers[PHXFS_THREADS_PER_NODE];
    int             nthreads;

    pthread_mutex_t mtx;
    pthread_cond_t  work_cv;   /* new current job / stop -> workers */
    pthread_cond_t  done_cv;   /* a job completed -> waiters */
    pthread_cond_t  free_cv;   /* a queue slot freed -> blocked submitters */

    struct pool_job *q_head, *q_tail;  /* waiting jobs (excludes cur) */
    struct pool_job *cur;              /* job the workers are running now */
    long             cur_gen;          /* bumped whenever cur changes */
    int              stride;           /* participating workers for cur */
    int              pending;          /* participating workers still running */
    int              inflight;         /* jobs in the system (waiting + cur) */
    bool             stop;
};

static struct node_pool g_pools[PHXFS_MAX_NUMA_NODES];
static int              g_num_nodes = 0;
static bool             g_pool_ready = false;
static pthread_once_t   g_once = PTHREAD_ONCE_INIT;

/* Run this worker's stripe of `job`: reqs[lane], reqs[lane+stride], ...
 * Accumulates this lane's failure count / first engine error into out params. */
static void worker_run_stripe(struct pool_job *job, int lane, int stride,
                              int *out_failed, int *out_err) {
    const struct phxfs_io_engine *eng = phxfs_io_engine_get();

    enum { CHUNK = 1024 };
    struct phxfs_io_op_req local[CHUNK];
    int back_idx[CHUNK];

    int failed = 0, err = 0;
    int base = lane;
    while (base < job->n) {
        int cnt = 0;
        for (int i = base; i < job->n && cnt < CHUNK; i += stride) {
            local[cnt] = job->reqs[i];
            back_idx[cnt] = i;
            cnt++;
        }
        if (cnt == 0)
            break;

        int rc = eng->submit_batch(local, cnt, job->op);
        if (rc < 0) {
            /* Engine-level failure: record the first one, mark this lane's
             * requests, but do NOT fold into a positive failure count. */
            if (err == 0)
                err = rc;
            for (int k = 0; k < cnt; k++)
                job->reqs[back_idx[k]].result = rc;
            failed += cnt;
        } else {
            for (int k = 0; k < cnt; k++) {
                job->reqs[back_idx[k]].result = local[k].result;
                if (local[k].result != (ssize_t)local[k].nbytes)
                    failed++;
            }
        }
        base += stride * CHUNK;
    }
    *out_failed = failed;
    *out_err = err;
}

/* Promote the queue head to `cur` and wake its workers. Caller holds mtx.
 * If the queue is empty, cur becomes NULL; under stop that also wakes idle
 * workers so they can exit. */
static void start_next_locked(struct node_pool *p) {
    struct pool_job *job = p->q_head;
    if (!job) {
        p->cur = NULL;
        if (p->stop)
            pthread_cond_broadcast(&p->work_cv);
        return;
    }
    p->q_head = job->next;
    if (!p->q_head)
        p->q_tail = NULL;
    job->next = NULL;

    int stride = p->nthreads;
    if (stride > job->n)
        stride = job->n;
    if (stride < 1)
        stride = 1;

    p->cur     = job;
    p->cur_gen++;
    p->stride  = stride;
    p->pending = stride;
    pthread_cond_broadcast(&p->work_cv);
}

static void *worker_main(void *arg) {
    struct worker *w = (struct worker *)arg;
    struct node_pool *p = &g_pools[w->node];

    /* Pin to this NUMA node's CPUs. Scan ALL possible CPU IDs, not just
     * 0..nproc-1: CPU IDs can be sparse or have low-numbered CPUs offline, so
     * _SC_NPROCESSORS_ONLN is not a valid upper bound and would miss a node's
     * high-numbered CPUs (P2-7). node%d/cpuC exists iff CPU C is on this node. */
    cpu_set_t set;
    CPU_ZERO(&set);
    bool any = false;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/node/node%d/cpu%d", w->node, c);
        if (access(path, F_OK) == 0) { CPU_SET(c, &set); any = true; }
    }
    if (any && pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
        fprintf(stderr, "phxfs pool: setaffinity(node %d) failed: %s\n",
                w->node, strerror(errno));

    long seen = 0;
    pthread_mutex_lock(&p->mtx);
    for (;;) {
        while (!(p->stop && p->cur == NULL) &&
               !(p->cur_gen != seen && p->cur != NULL && w->lane < p->stride))
            pthread_cond_wait(&p->work_cv, &p->mtx);

        /* Exit only once the queue is fully drained (P1-7): while a job is
         * current we run it even under stop, so `pending` is always decremented
         * and no waiter is left hanging. */
        if (p->stop && p->cur == NULL) {
            pthread_mutex_unlock(&p->mtx);
            return NULL;
        }

        seen = p->cur_gen;
        struct pool_job *job = p->cur;
        int lane = w->lane, stride = p->stride;
        pthread_mutex_unlock(&p->mtx);

        int lfailed = 0, lerr = 0;
        worker_run_stripe(job, lane, stride, &lfailed, &lerr);

        pthread_mutex_lock(&p->mtx);
        job->failed += lfailed;
        if (job->err == 0 && lerr < 0)
            job->err = lerr;
        if (--p->pending == 0) {
            job->done = true;
            pthread_cond_broadcast(&p->done_cv);   /* wake this job's waiter */
            p->inflight--;                          /* job leaves the system */
            pthread_cond_signal(&p->free_cv);       /* a queue slot freed */
            start_next_locked(p);                   /* run next, or cur=NULL */
        }
    }
}

static void pool_init(void) {
    for (int nd = 0; nd < PHXFS_MAX_NUMA_NODES; nd++) {
        char path[96];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", nd);
        if (access(path, F_OK) != 0)
            break;
        g_num_nodes = nd + 1;
    }
    if (g_num_nodes < 1)
        g_num_nodes = 1;

    for (int nd = 0; nd < g_num_nodes; nd++) {
        struct node_pool *p = &g_pools[nd];
        pthread_mutex_init(&p->mtx, NULL);
        pthread_cond_init(&p->work_cv, NULL);
        pthread_cond_init(&p->done_cv, NULL);
        pthread_cond_init(&p->free_cv, NULL);
        p->q_head = p->q_tail = NULL;
        p->cur = NULL;
        p->cur_gen = 0;
        p->stride = 0;
        p->pending = 0;
        p->inflight = 0;
        p->stop = false;
        p->nthreads = PHXFS_THREADS_PER_NODE;
        for (int t = 0; t < p->nthreads; t++) {
            p->workers[t].node = nd;
            p->workers[t].lane = t;
            if (pthread_create(&p->workers[t].tid, NULL, worker_main,
                               &p->workers[t]) != 0) {
                p->nthreads = t;
                break;
            }
        }
    }
    g_pool_ready = true;
}

/* Resolve node + validate pool. Returns NULL if inline fallback needed. */
static struct node_pool *pool_for(int numa_node) {
    pthread_once(&g_once, pool_init);
    if (numa_node < 0 || numa_node >= g_num_nodes)
        numa_node = 0;
    struct node_pool *p = &g_pools[numa_node];
    if (!g_pool_ready || p->nthreads <= 0)
        return NULL;
    return p;
}

/*
 * Enqueue `job` on `p` (bounded FIFO). blocking=false returns -EBUSY when the
 * queue is full (async submit); blocking=true waits for a free slot (sync).
 * Returns 0 on success, -1 with errno set (EBUSY / ESHUTDOWN) otherwise.
 */
static int job_enqueue(struct node_pool *p, struct pool_job *job, bool blocking) {
    pthread_mutex_lock(&p->mtx);
    if (p->stop) {
        pthread_mutex_unlock(&p->mtx);
        errno = ESHUTDOWN;
        return -1;
    }
    while (p->inflight >= PHXFS_POOL_QUEUE_CAP) {
        if (!blocking) {
            pthread_mutex_unlock(&p->mtx);
            errno = EBUSY;
            return -1;
        }
        pthread_cond_wait(&p->free_cv, &p->mtx);
        if (p->stop) {
            pthread_mutex_unlock(&p->mtx);
            errno = ESHUTDOWN;
            return -1;
        }
    }
    job->done = false;
    job->failed = 0;
    job->err = 0;
    job->next = NULL;
    if (p->q_tail) p->q_tail->next = job; else p->q_head = job;
    p->q_tail = job;
    p->inflight++;
    if (p->cur == NULL)             /* idle: promote this job to current */
        start_next_locked(p);
    pthread_mutex_unlock(&p->mtx);
    return 0;
}

/* Wait for `job` to complete; returns failure count or first engine error. */
static int job_wait(struct node_pool *p, struct pool_job *job) {
    pthread_mutex_lock(&p->mtx);
    while (!job->done)
        pthread_cond_wait(&p->done_cv, &p->mtx);
    int failed = job->failed, err = job->err;
    pthread_mutex_unlock(&p->mtx);
    return err < 0 ? err : failed;
}

int phxfs_pool_run(struct phxfs_io_op_req *reqs, int n, enum phxfs_io_op op,
                   int numa_node) {
    if (n <= 0)
        return 0;

    struct node_pool *p = pool_for(numa_node);
    if (!p) {
        const struct phxfs_io_engine *eng = phxfs_io_engine_get();
        return eng->submit_batch(reqs, n, op);
    }

    /* Synchronous: stack job (valid until we wait it), blocking enqueue. */
    struct pool_job job;
    memset(&job, 0, sizeof(job));
    job.reqs = reqs;
    job.n = n;
    job.op = op;
    if (job_enqueue(p, &job, /*blocking=*/true) != 0) {
        const struct phxfs_io_engine *eng = phxfs_io_engine_get();  /* shutting down */
        return eng->submit_batch(reqs, n, op);
    }
    return job_wait(p, &job);
}

/* ------------------------------------------------------------------ *
 * Async: submit enqueues and returns immediately; wait blocks for that job.
 * Multiple jobs may be queued per node (bounded), enabling pipelining.
 * ------------------------------------------------------------------ */
struct phxfs_pool_async {
    struct node_pool *p;
    struct pool_job   job;      /* embedded; enqueued by address */
    bool  inline_done;
    int   inline_failed;
};

struct phxfs_pool_async *phxfs_pool_submit(struct phxfs_io_op_req *reqs, int n,
                                           enum phxfs_io_op op, int numa_node,
                                           bool blocking) {
    struct phxfs_pool_async *h =
        (struct phxfs_pool_async *)calloc(1, sizeof(*h));
    if (!h)
        return NULL;

    if (n <= 0) {
        h->inline_done = true;
        h->inline_failed = 0;
        return h;
    }

    struct node_pool *p = pool_for(numa_node);
    if (!p) {
        /* Async needs the worker pool. Do NOT run the batch inline here: that
         * would make "submit returns immediately" a lie (P1-1). Fail so the
         * caller can fall back to the synchronous batch API instead. */
        free(h);
        errno = ENOTSUP;
        return NULL;
    }

    h->p = p;
    h->job.reqs = reqs;
    h->job.n = n;
    h->job.op = op;
    /* Enqueue on the bounded FIFO. blocking=false (async) returns EBUSY only
     * when the queue is FULL (not merely non-empty), so pipelining several
     * batches now succeeds (P2-2). blocking=true is used by the sync mixed-NUMA
     * path to reserve a slot. */
    if (job_enqueue(p, &h->job, blocking) != 0) {
        free(h);            /* errno set by job_enqueue (EBUSY / ESHUTDOWN) */
        return NULL;
    }
    return h;
}

int phxfs_pool_wait(struct phxfs_pool_async *h) {
    if (!h)
        return -EINVAL;
    int rc;
    if (h->inline_done)
        rc = h->inline_failed;
    else
        rc = job_wait(h->p, &h->job);
    free(h);
    return rc;
}

/*
 * Stop and join every worker at library unload. New submits are refused
 * (ESHUTDOWN); already-queued jobs are drained so their waiters never hang
 * (P1-7). Joining lets each worker return from worker_main(), running its
 * thread_local ring's destructor (io_uring_queue_exit), so no ring leaks and no
 * unloaded code runs later. Safe if the pool was never initialised.
 */
__attribute__((destructor))
static void phxfs_pool_shutdown(void) {
    if (!g_pool_ready)
        return;
    for (int nd = 0; nd < g_num_nodes; nd++) {
        struct node_pool *p = &g_pools[nd];
        pthread_mutex_lock(&p->mtx);
        p->stop = true;
        pthread_cond_broadcast(&p->work_cv);   /* wake idle workers to drain/exit */
        pthread_cond_broadcast(&p->free_cv);   /* wake blocked submitters */
        pthread_mutex_unlock(&p->mtx);
    }
    for (int nd = 0; nd < g_num_nodes; nd++) {
        struct node_pool *p = &g_pools[nd];
        for (int t = 0; t < p->nthreads; t++)
            pthread_join(p->workers[t].tid, NULL);
    }
    g_pool_ready = false;
}
