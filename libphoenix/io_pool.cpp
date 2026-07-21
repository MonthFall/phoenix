/*
 * NUMA-aware batch I/O thread pool.
 *
 * One pool per NUMA node, PHXFS_THREADS_PER_NODE worker threads each,
 * pinned to that node's CPUs. A batch is split round-robin across the
 * node's workers; every worker runs the active engine's submit_batch on
 * its own thread_local ring, so N rings issue I/O concurrently and reach
 * the array's bandwidth ceiling — while the caller only makes one
 * (blocking) call and crosses the Python GIL once.
 *
 * A single global lock serialises whole-batch calls (one batch in flight
 * at a time, others queue), matching the "call owns the pool" model.
 */
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "io_engine.h"

/* ---- per-node worker ---- */
struct worker {
    pthread_t       tid;
    int             node;
    int             lane;          /* 0..THREADS_PER_NODE-1 */
    /* work item (set by dispatcher, consumed by worker) */
    struct phxfs_io_op_req *reqs;  /* shared base array */
    int             n;             /* total requests in the batch */
    int             stride;        /* == threads active for this batch */
    enum phxfs_io_op op;
    int             failed;        /* OUT: failures in this lane */
};

struct node_pool {
    struct worker   workers[PHXFS_THREADS_PER_NODE];
    int             nthreads;
    /* dispatch sync */
    pthread_mutex_t mtx;
    pthread_cond_t  work_cv;       /* wake workers */
    pthread_cond_t  done_cv;       /* workers -> dispatcher */
    int             generation;    /* incremented per batch */
    int             pending;       /* workers still running */
    bool            stop;
    bool            active;        /* has a batch assigned this generation */
};

static struct node_pool g_pools[PHXFS_MAX_NUMA_NODES];
static int              g_num_nodes = 0;
static bool             g_pool_ready = false;
static pthread_mutex_t  g_call_lock = PTHREAD_MUTEX_INITIALIZER; /* one batch */
static pthread_once_t   g_once = PTHREAD_ONCE_INIT;

/* Run this worker's stripe: reqs[lane], reqs[lane+stride], ... */
static void worker_run_stripe(struct worker *w) {
    const struct phxfs_io_engine *eng = phxfs_io_engine_get();

    /* Gather this lane's requests into a compact array (indices lane,
     * lane+stride, ...). We reorder pointers via a small local buffer of
     * copies, then scatter results back. To avoid per-call malloc we cap
     * with a stack chunk and loop. */
    enum { CHUNK = 1024 };
    struct phxfs_io_op_req local[CHUNK];
    int back_idx[CHUNK];

    w->failed = 0;
    int base = w->lane;
    while (base < w->n) {
        int cnt = 0;
        for (int i = base; i < w->n && cnt < CHUNK; i += w->stride) {
            local[cnt] = w->reqs[i];
            back_idx[cnt] = i;
            cnt++;
        }
        if (cnt == 0)
            break;

        int rc = eng->submit_batch(local, cnt, w->op);
        if (rc < 0) {
            for (int k = 0; k < cnt; k++)
                w->reqs[back_idx[k]].result = rc;
            w->failed += cnt;
        } else {
            for (int k = 0; k < cnt; k++) {
                w->reqs[back_idx[k]].result = local[k].result;
                if (local[k].result != (ssize_t)local[k].nbytes)
                    w->failed++;
            }
        }
        base += w->stride * CHUNK;
    }
}

static void *worker_main(void *arg) {
    struct worker *w = (struct worker *)arg;
    struct node_pool *p = &g_pools[w->node];

    /* Pin to this NUMA node's CPUs. */
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    cpu_set_t set;
    CPU_ZERO(&set);
    bool any = false;
    for (long c = 0; c < ncpu && c < CPU_SETSIZE; c++) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/node/node%d/cpu%ld", w->node, c);
        if (access(path, F_OK) == 0) { CPU_SET(c, &set); any = true; }
    }
    if (any)
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);

    int seen = 0;
    pthread_mutex_lock(&p->mtx);
    for (;;) {
        while (!p->stop && (!p->active || p->generation == seen))
            pthread_cond_wait(&p->work_cv, &p->mtx);
        if (p->stop) {
            pthread_mutex_unlock(&p->mtx);
            return NULL;
        }
        seen = p->generation;
        pthread_mutex_unlock(&p->mtx);

        worker_run_stripe(w);

        pthread_mutex_lock(&p->mtx);
        if (--p->pending == 0)
            pthread_cond_signal(&p->done_cv);
    }
}

static void pool_init(void) {
    /* Determine node count from sysfs. */
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
        p->generation = 0;
        p->pending = 0;
        p->stop = false;
        p->active = false;
        p->nthreads = PHXFS_THREADS_PER_NODE;
        for (int t = 0; t < p->nthreads; t++) {
            p->workers[t].node = nd;
            p->workers[t].lane = t;
            if (pthread_create(&p->workers[t].tid, NULL, worker_main,
                               &p->workers[t]) != 0) {
                p->nthreads = t;  /* fewer threads than desired */
                break;
            }
        }
    }
    g_pool_ready = true;
}

int phxfs_pool_run(struct phxfs_io_op_req *reqs, int n, enum phxfs_io_op op,
                   int numa_node) {
    if (n <= 0)
        return 0;

    pthread_once(&g_once, pool_init);

    if (numa_node < 0 || numa_node >= g_num_nodes)
        numa_node = 0;
    struct node_pool *p = &g_pools[numa_node];

    if (!g_pool_ready || p->nthreads <= 0) {
        /* Fallback: run inline on the calling thread. */
        const struct phxfs_io_engine *eng = phxfs_io_engine_get();
        return eng->submit_batch(reqs, n, op);
    }

    /* Serialise whole-batch calls: one batch owns the pool at a time. */
    pthread_mutex_lock(&g_call_lock);

    int stride = p->nthreads;
    if (stride > n)
        stride = n;  /* don't spawn more lanes than requests */

    pthread_mutex_lock(&p->mtx);
    for (int t = 0; t < stride; t++) {
        p->workers[t].reqs = reqs;
        p->workers[t].n = n;
        p->workers[t].stride = stride;
        p->workers[t].op = op;
        p->workers[t].failed = 0;
    }
    p->pending = stride;
    p->active = true;
    p->generation++;
    pthread_cond_broadcast(&p->work_cv);

    while (p->pending > 0)
        pthread_cond_wait(&p->done_cv, &p->mtx);
    p->active = false;
    pthread_mutex_unlock(&p->mtx);

    int failed = 0;
    for (int t = 0; t < stride; t++)
        failed += p->workers[t].failed;

    pthread_mutex_unlock(&g_call_lock);
    return failed;
}

/* ------------------------------------------------------------------ *
 * Async: submit returns immediately, wait blocks for completion.
 *
 * One batch owns the pool between submit and wait (g_call_lock is held
 * across the pair), so at most one async batch is in flight per node —
 * exactly the compute-overlap model requested. The handle records which
 * node/stride to join.
 * ------------------------------------------------------------------ */
struct phxfs_pool_async {
    int node;
    int stride;
    bool inline_done;   /* ran inline (pool unavailable) */
    int inline_failed;
};

struct phxfs_pool_async *phxfs_pool_submit(struct phxfs_io_op_req *reqs, int n,
                                           enum phxfs_io_op op, int numa_node) {
    struct phxfs_pool_async *h =
        (struct phxfs_pool_async *)calloc(1, sizeof(*h));
    if (!h)
        return NULL;

    pthread_once(&g_once, pool_init);
    if (numa_node < 0 || numa_node >= g_num_nodes)
        numa_node = 0;
    h->node = numa_node;
    struct node_pool *p = &g_pools[numa_node];

    if (n <= 0) {
        h->inline_done = true;
        h->inline_failed = 0;
        return h;
    }
    if (!g_pool_ready || p->nthreads <= 0) {
        const struct phxfs_io_engine *eng = phxfs_io_engine_get();
        h->inline_done = true;
        h->inline_failed = eng->submit_batch(reqs, n, op);
        return h;
    }

    /* Acquire the pool for this batch; released in phxfs_pool_wait. */
    pthread_mutex_lock(&g_call_lock);

    int stride = p->nthreads;
    if (stride > n)
        stride = n;
    h->stride = stride;

    pthread_mutex_lock(&p->mtx);
    for (int t = 0; t < stride; t++) {
        p->workers[t].reqs = reqs;
        p->workers[t].n = n;
        p->workers[t].stride = stride;
        p->workers[t].op = op;
        p->workers[t].failed = 0;
    }
    p->pending = stride;
    p->active = true;
    p->generation++;
    pthread_cond_broadcast(&p->work_cv);
    pthread_mutex_unlock(&p->mtx);
    /* NOTE: g_call_lock stays held until wait(). */
    return h;
}

int phxfs_pool_wait(struct phxfs_pool_async *h) {
    if (!h)
        return -EINVAL;

    int failed;
    if (h->inline_done) {
        failed = h->inline_failed;
        free(h);
        return failed;
    }

    struct node_pool *p = &g_pools[h->node];
    pthread_mutex_lock(&p->mtx);
    while (p->pending > 0)
        pthread_cond_wait(&p->done_cv, &p->mtx);
    p->active = false;
    pthread_mutex_unlock(&p->mtx);

    failed = 0;
    for (int t = 0; t < h->stride; t++)
        failed += p->workers[t].failed;

    pthread_mutex_unlock(&g_call_lock);  /* release the pool */
    free(h);
    return failed;
}
