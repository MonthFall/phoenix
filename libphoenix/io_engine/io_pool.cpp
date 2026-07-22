/*
 * Batch I/O thread pool.
 *
 * A fixed set of PHXFS_POOL_THREADS worker threads, each running the active
 * engine's submit_batch on its own thread_local ring. A batch is split
 * round-robin across the workers so N rings issue I/O concurrently and reach
 * the storage array's aggregate bandwidth ceiling, while the caller makes one
 * (blocking or async) call and crosses the Python GIL once.
 *
 * Workers are NOT pinned to any NUMA node: the P2P transfer this pool exists
 * to drive is device-to-device DMA that never touches host RAM, so the CPU a
 * worker runs on has no effect on that data path (see io_engine.h). The only
 * job of this pool is fanning a batch out across enough independent rings.
 *
 * Scheduling model (bounded FIFO queue):
 *   - Submitted jobs queue up (bounded capacity) and run one at a time, each
 *     using the full worker set — so every job still gets the pool's full
 *     concurrency. submit() enqueues and returns immediately; it only fails
 *     with EBUSY when the queue itself is full. This supports pipelining:
 *     a caller can submit several batches, overlap compute, then wait each
 *     handle, without a single global "one batch in flight" stall.
 *   - Each job carries its own result state (failed/err/done); results are
 *     read by that job's wait(), decoupled from when the job leaves the
 *     queue.
 *   - A worker participates in the current job iff its lane < that job's
 *     stride; only participating workers touch the request array and
 *     decrement pending.
 */
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <pthread.h>

#include "io_engine.h"

/* Bounded in-flight jobs (queued + running). Enough to pipeline a few batches
 * for compute/I/O overlap without letting a runaway producer queue unbounded
 * work. */
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
    int       lane;            /* 0..PHXFS_POOL_THREADS-1 */
};

struct pool_state {
    struct worker   workers[PHXFS_POOL_THREADS];
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

static struct pool_state g_pool;
static bool              g_pool_ready = false;
static pthread_once_t    g_once = PTHREAD_ONCE_INIT;

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
static void start_next_locked(struct pool_state *p) {
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
    struct pool_state *p = &g_pool;

    long seen = 0;
    pthread_mutex_lock(&p->mtx);
    for (;;) {
        while (!(p->stop && p->cur == NULL) &&
               !(p->cur_gen != seen && p->cur != NULL && w->lane < p->stride))
            pthread_cond_wait(&p->work_cv, &p->mtx);

        /* Exit only once the queue is fully drained: while a job is current we
         * run it even under stop, so `pending` is always decremented and no
         * waiter is left hanging. */
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

/*
 * Called in the child after fork(): the parent's worker threads did not
 * survive, so the pool's mutex/condvars and g_once are stale. Reset everything
 * to "never initialised" so the child lazily re-creates the pool on first use.
 */
static void pool_atfork_child(void) {
    g_pool_ready = false;
    g_once = PTHREAD_ONCE_INIT;
    g_pool.nthreads = 0;
    g_pool.cur = NULL;
    g_pool.q_head = g_pool.q_tail = NULL;
    g_pool.inflight = 0;
    g_pool.stop = false;
}

static void pool_init(void) {
    struct pool_state *p = &g_pool;
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
    p->nthreads = PHXFS_POOL_THREADS;
    for (int t = 0; t < p->nthreads; t++) {
        p->workers[t].lane = t;
        if (pthread_create(&p->workers[t].tid, NULL, worker_main,
                           &p->workers[t]) != 0) {
            p->nthreads = t;
            break;
        }
    }
    g_pool_ready = true;

    /*
     * fork() only copies the calling thread; the worker threads vanish but
     * g_pool_ready / g_once survive. Without a reset the child would enqueue
     * a job, broadcast work_cv, and block forever (no workers to process it).
     * pthread_atfork(child) resets the pool to its pre-init state so the
     * child lazily re-creates it on first use.
     */
    pthread_atfork(NULL, NULL, pool_atfork_child);
}

/* Validate the pool. Returns NULL if inline fallback needed. */
static struct pool_state *pool_get(void) {
    pthread_once(&g_once, pool_init);
    if (!g_pool_ready || g_pool.nthreads <= 0)
        return NULL;
    return &g_pool;
}

/*
 * Enqueue `job` on `p` (bounded FIFO). blocking=false returns -EBUSY when the
 * queue is full (async submit); blocking=true waits for a free slot (sync).
 * Returns 0 on success, -1 with errno set (EBUSY / ESHUTDOWN) otherwise.
 */
static int job_enqueue(struct pool_state *p, struct pool_job *job, bool blocking) {
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
static int job_wait(struct pool_state *p, struct pool_job *job) {
    pthread_mutex_lock(&p->mtx);
    while (!job->done)
        pthread_cond_wait(&p->done_cv, &p->mtx);
    int failed = job->failed, err = job->err;
    pthread_mutex_unlock(&p->mtx);
    return err < 0 ? err : failed;
}

int phxfs_pool_run(struct phxfs_io_op_req *reqs, int n, enum phxfs_io_op op) {
    if (n <= 0)
        return 0;

    struct pool_state *p = pool_get();
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
 * Multiple jobs may be queued (bounded), enabling pipelining.
 * ------------------------------------------------------------------ */
struct phxfs_pool_async {
    struct pool_state *p;
    struct pool_job     job;      /* embedded; enqueued by address */
    bool  inline_done;
    int   inline_failed;
};

struct phxfs_pool_async *phxfs_pool_submit(struct phxfs_io_op_req *reqs, int n,
                                           enum phxfs_io_op op, bool blocking) {
    struct phxfs_pool_async *h =
        (struct phxfs_pool_async *)calloc(1, sizeof(*h));
    if (!h)
        return NULL;

    if (n <= 0) {
        h->inline_done = true;
        h->inline_failed = 0;
        return h;
    }

    struct pool_state *p = pool_get();
    if (!p) {
        /* Async needs the worker pool. Do NOT run the batch inline here: that
         * would make "submit returns immediately" a lie. Fail so the caller
         * can fall back to the synchronous batch API instead. */
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
     * batches succeeds. blocking=true is used by the sync path when it wants
     * to reserve a slot. */
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
 * (ESHUTDOWN); already-queued jobs are drained so their waiters never hang.
 * Joining lets each worker return from worker_main(), running its
 * thread_local ring's destructor (io_uring_queue_exit), so no ring leaks and
 * no unloaded code runs later. Safe if the pool was never initialised.
 */
__attribute__((destructor))
static void phxfs_pool_shutdown(void) {
    if (!g_pool_ready)
        return;
    struct pool_state *p = &g_pool;
    pthread_mutex_lock(&p->mtx);
    p->stop = true;
    pthread_cond_broadcast(&p->work_cv);   /* wake idle workers to drain/exit */
    pthread_cond_broadcast(&p->free_cv);   /* wake blocked submitters */
    pthread_mutex_unlock(&p->mtx);
    for (int t = 0; t < p->nthreads; t++)
        pthread_join(p->workers[t].tid, NULL);
    g_pool_ready = false;
}
