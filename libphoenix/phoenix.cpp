#ifndef _GNU_SOURCE
#define _GNU_SOURCE   /* for getcpu() (P2-5) */
#endif
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <linux/types.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "phoenix.h"
#include "connectors/devconnector.h"
#include "io_engine.h"

#define HUGE_PAGE_SIZE (64 * 1024)
#define SMALL_PAGE_SIZE (4 * 1024)
/*
 * Per-syscall I/O chunk size. The kernel module now supports a single
 * arbitrarily large mmap+ioctl registration (kvmalloc), so we no longer chunk
 * the *mapping*. We only chunk the *I/O* because a single read()/write()/
 * io_uring op transfers at most MAX_RW_COUNT (INT_MAX & PAGE_MASK = 0x7ffff000,
 * ~2GiB) per call. 1GiB is a clean, 64KiB-aligned value well under that cap.
 */
#define PHXFS_IO_CHUNK (1024ULL * 1024 * 1024)  /* 1 GiB */
#define PHXFS_MAX_DEVICES 8
#define QD 128

typedef struct phxfs_mmap_node_s {
    void *vaddr;        // single contiguous host-mapped P2P address
    uint64_t dev_addr;  // registered GPU/device address (lookup key)
    size_t length;      // registered length
    bool has_reg;       // kernel P2P mapping active (ioctl MAP done, not UNMAP)
    bool mapped;        // host VMA active (mmap done, munmap not yet) — kept
                        // separate from has_reg so a failed munmap is retriable
                        // and close still cleans the VMA (P1-1)
    int  user_refs;     // outstanding regmem() references (P2-2 reuse)
    int  refcount;      // in-flight resolutions/I/O referencing this node
    struct phxfs_mmap_node_s *next; // 指向下一个节点的指针
} phxfs_p2p_map_t;

typedef struct phxfs_mmap_buffer_s {
    int device_id;
    int bdev_fd;
    int numa_node;      /* NUMA node of this device's GPU (queried at open) */
    bool init_stat;     /* device is OPEN (fd valid, usable) */
    bool closing;       /* close in progress: reject new dev_get() */
    int  open_count;    /* client open refcount (P1-3) */
    int  active_ops;    /* in-flight device operations (P0-2/P1-4) */
    struct phxfs_mmap_node_s *head;
    struct phxfs_mmap_node_s *last_hit;  /* MRU lookup cache (P2-1) */
    /*
     * lock/drain_cv have *process lifetime*: initialised once by the library
     * constructor and never destroyed, so concurrent dev_get()/map_release()
     * can never touch a destroyed primitive during/after close (P0-2/P1-4).
     */
    pthread_mutex_t lock;
    pthread_cond_t  drain_cv;   /* mapping refcount==0 or active_ops==0 */
} phxfs_mmap_buffer_t;

static int g_device_count = PHXFS_MAX_DEVICES;
static phxfs_mmap_buffer_t mbuffer[PHXFS_MAX_DEVICES];

/* Initialise per-device sync primitives + state once, before any open. */
__attribute__((constructor))
static void phxfs_dev_ctor(void) {
    for (int i = 0; i < PHXFS_MAX_DEVICES; i++) {
        pthread_mutex_init(&mbuffer[i].lock, NULL);
        pthread_cond_init(&mbuffer[i].drain_cv, NULL);
        mbuffer[i].init_stat  = false;
        mbuffer[i].closing    = false;
        mbuffer[i].open_count = 0;
        mbuffer[i].active_ops = 0;
        mbuffer[i].head       = NULL;
        mbuffer[i].last_hit   = NULL;
    }
    /* Honour the connector init contract once at load (P2-7). */
    devconn_init();
}

/*
 * Acquire a device-operation reference. Returns the device buffer if it is
 * OPEN and not closing, else NULL. Holding a ref keeps close() waiting, so
 * an in-flight regmem/dereg/read/write/batch can never race teardown.
 */
static phxfs_mmap_buffer_t *dev_get(int device_id) {
    if (device_id < 0 || device_id >= g_device_count)
        return NULL;
    phxfs_mmap_buffer_t *pb = &mbuffer[device_id];
    pthread_mutex_lock(&pb->lock);
    if (!pb->init_stat || pb->closing) {
        pthread_mutex_unlock(&pb->lock);
        return NULL;
    }
    pb->active_ops++;
    pthread_mutex_unlock(&pb->lock);
    return pb;
}

/* Release a dev_get() reference; wakes a draining close() at zero. */
static void dev_put(phxfs_mmap_buffer_t *pb) {
    if (!pb)
        return;
    pthread_mutex_lock(&pb->lock);
    if (--pb->active_ops == 0 && pb->closing)
        pthread_cond_broadcast(&pb->drain_cv);
    pthread_mutex_unlock(&pb->lock);
}
static std::vector<std::string> phxfs_dev_path = {
    "/dev/phxfs_dev0", "/dev/phxfs_dev1",
    "/dev/phxfs_dev2", "/dev/phxfs_dev3",
    "/dev/phxfs_dev4", "/dev/phxfs_dev5",
    "/dev/phxfs_dev6", "/dev/phxfs_dev7"
};

static std::vector<bool> phxfs_initialized(PHXFS_MAX_DEVICES, false);

/*
 * Serialises open()/close() across devices so concurrent phxfs_open() from
 * multiple threads (e.g. one worker per GPU) and open-vs-close are safe.
 * reg/dereg/batch synchronise per-device on mbuffer[].lock instead.
 */
static pthread_mutex_t g_open_lock = PTHREAD_MUTEX_INITIALIZER;

/* Forward decl: unmap a single registration in the kernel. */
int __phxfs_deregmem(phxfs_mmap_buffer_t *pb, u64 dev_addr, u64 c_addr, size_t len);

/*
 * Tear down every registration owned by `buffer`. Unlike a plain free, this
 * releases the kernel P2P mapping (ioctl UNMAP) and the host VMA (munmap) for
 * each node, so close() does not leak mappings (P1-6). MUST be called while
 * buffer->bdev_fd is still open (the UNMAP ioctl needs it).
 *
 * Concurrency contract: close()/deregmem() must not race with in-flight I/O
 * on the same device — the caller quiesces I/O first. This walk therefore
 * detaches the list under the lock, then unmaps outside it.
 */
void free_phxfs_p2p_map(phxfs_mmap_buffer_t *buffer) {
    pthread_mutex_lock(&buffer->lock);
    struct phxfs_mmap_node_s *current = buffer->head;
    buffer->head = NULL;
    buffer->last_hit = NULL;   /* drop MRU cache (P2-1) */
    pthread_mutex_unlock(&buffer->lock);

    struct phxfs_mmap_node_s *next;
    while (current) {
        next = current->next;
        if (current->has_reg)
            __phxfs_deregmem(buffer, current->dev_addr,
                             (uint64_t)current->vaddr, current->length);
        if (current->mapped && current->vaddr && current->vaddr != MAP_FAILED)
            munmap(current->vaddr, current->length);   /* P1-1: VMA-only ok */
        free(current);
        current = next;
    }
}

/*
 * Tear down an OPEN device after all in-flight operations have drained.
 * Caller must have set closing=true and waited active_ops==0 under pb->lock.
 * The process-lifetime lock/drain_cv are NOT destroyed here.
 */
static void __phxfs_teardown(phxfs_mmap_buffer_t *pb) {
    free_phxfs_p2p_map(pb);              /* UNMAP + munmap each registration */
    if (pb->bdev_fd >= 0)                /* P1-4: fd 0 is valid, must close */
        close(pb->bdev_fd);
    pb->bdev_fd  = -1;
    pb->init_stat = false;
    pb->closing   = false;
}

/*
 * Bring the last client's device down: mark closing (blocks new dev_get()),
 * wait for every in-flight operation to finish, then tear down (P0-2/P1-4).
 * Caller holds g_open_lock; dev ops never take g_open_lock, so waiting here
 * cannot deadlock.
 */
static void __phxfs_close_drain(phxfs_mmap_buffer_t *pb) {
    pthread_mutex_lock(&pb->lock);
    pb->closing = true;
    while (pb->active_ops > 0)
        pthread_cond_wait(&pb->drain_cv, &pb->lock);
    pthread_mutex_unlock(&pb->lock);
    __phxfs_teardown(pb);
}

int phxfs_close(int device_id) {
    if (device_id < 0 || device_id >= g_device_count)
        return -1;
    phxfs_mmap_buffer_t *pb = &mbuffer[device_id];

    pthread_mutex_lock(&g_open_lock);
    if (!pb->init_stat) {
        pthread_mutex_unlock(&g_open_lock);
        return -1;
    }
    /* Only the last client's close actually tears the device down (P1-3). */
    if (pb->open_count > 1) {
        pb->open_count--;
        pthread_mutex_unlock(&g_open_lock);
        return 0;
    }
    pb->open_count = 0;
    phxfs_initialized[device_id] = false;
    __phxfs_close_drain(pb);
    pthread_mutex_unlock(&g_open_lock);
    return 0;
}

/*
 * Query a device's NUMA node from sysfs (done once, at open time):
 *   /sys/class/phxfs-generic/phxfs_devN/pci_bdf
 *     -> /sys/bus/pci/devices/<bdf>/numa_node
 * Returns 0 if it can't be determined.
 */
static int query_dev_numa(int device_id) {
    int node = 0;
    char path[128], bdf[64] = {0};
    snprintf(path, sizeof(path),
             "/sys/class/phxfs-generic/phxfs_dev%d/pci_bdf", device_id);
    FILE *f = fopen(path, "r");
    if (f) {
        if (fgets(bdf, sizeof(bdf), f)) {
            char *nl = strchr(bdf, '\n');
            if (nl) *nl = '\0';
            char np[128];
            snprintf(np, sizeof(np),
                     "/sys/bus/pci/devices/%s/numa_node", bdf);
            FILE *nf = fopen(np, "r");
            if (nf) {
                int v;
                if (fscanf(nf, "%d", &v) == 1 && v >= 0)
                    node = v;
                fclose(nf);
            }
        }
        fclose(f);
    }
    return node;
}

/* Open the char device. lock/drain_cv are process-lifetime (see ctor). */
static int __phxfs_open(const char *dev_path, phxfs_mmap_buffer_t *mbuffer,
                        int device_id) {
    mbuffer->bdev_fd = open(dev_path, O_RDWR);

    if (mbuffer->bdev_fd == -1) {
        printf("failed to open file %s\n", dev_path);
        return -1;
    }
    mbuffer->head = NULL;
    mbuffer->device_id = device_id;
    mbuffer->numa_node = query_dev_numa(device_id);  /* once, at open */
    mbuffer->closing = false;
    mbuffer->active_ops = 0;
    mbuffer->init_stat = true;
    return 0;
}



bool is_phxfs_initialized() {
    /* All accesses to phxfs_initialized are serialised under g_open_lock so
     * the std::vector<bool> is never read/written concurrently (P0-1). */
    pthread_mutex_lock(&g_open_lock);
    bool initialized = false;
    for (int i = 0; i < g_device_count; i++)
        initialized = initialized || phxfs_initialized[i];
    pthread_mutex_unlock(&g_open_lock);
    return initialized;
}

/* Open (or add a client ref to) one device. Caller holds g_open_lock.
 * Sets *newly_opened=true iff this call performed the actual open. */
static int open_one_locked(int id, bool *newly_opened) {
    phxfs_mmap_buffer_t *pb = &mbuffer[id];
    if (pb->init_stat) {          /* already open: just add a client ref */
        pb->open_count++;
        *newly_opened = false;
        return 0;
    }
    int ret = __phxfs_open(phxfs_dev_path[id].c_str(), pb, id);
    if (ret == 0) {
        pb->open_count = 1;
        phxfs_initialized[id] = true;
        *newly_opened = true;
    }
    return ret;
}

/*
 * Open a phxfs device (or add a client reference — P1-3). g_open_lock makes
 * concurrent phxfs_open() from multiple threads safe. deviceID == -1 opens
 * all devices; on failure it rolls back only the refs/opens this call added,
 * leaving devices other clients already opened intact.
 */
int phxfs_open(int deviceID) {
    int ret = 0;
    pthread_mutex_lock(&g_open_lock);

    if (deviceID == -1) {
        bool newly[PHXFS_MAX_DEVICES] = {false};
        bool touched[PHXFS_MAX_DEVICES] = {false};
        for (int id = 0; id < g_device_count; ++id) {
            ret = open_one_locked(id, &newly[id]);
            if (ret < 0) {
                /* Roll back this call's additions only. */
                for (int j = 0; j < id; ++j) {
                    if (!touched[j]) continue;
                    phxfs_mmap_buffer_t *pb = &mbuffer[j];
                    if (newly[j]) {
                        pb->open_count = 0;
                        phxfs_initialized[j] = false;
                        __phxfs_close_drain(pb);
                    } else {
                        pb->open_count--;
                    }
                }
                pthread_mutex_unlock(&g_open_lock);
                return ret;
            }
            touched[id] = true;
        }
    } else if (deviceID >= 0 && deviceID < g_device_count) {
        bool newly;
        ret = open_one_locked(deviceID, &newly);
    } else {
        ret = -1;  /* invalid device id */
    }

    pthread_mutex_unlock(&g_open_lock);
    return ret;
}


int phxfs_find_dev(int device_id) {
    return devconn->find_device(device_id);
}

uint64_t phxfs_get_page_size(void) {
    return devconn->page_size;
}


int insert_phxfs_mmap_node(phxfs_mmap_buffer_t *mbuffer, phxfs_p2p_map_t *new_node) {
    if (!new_node) {
        fprintf(stderr, "%s: new_node is NULL\n", __func__);
        return -1;
    }

    pthread_mutex_lock(&mbuffer->lock);
    new_node->next = mbuffer->head;
    mbuffer->head = new_node;
    pthread_mutex_unlock(&mbuffer->lock);
    return 0;
}

/* Find the registration that contains `dev_addr` (a single byte inside
 * [dev_addr0, dev_addr0+length)). Caller must hold the lock. The precise
 * extent check (offset + length) is done by the resolver, not here.
 * Every dev_addr+length sum here is overflow-safe: phxfs_regmem() rejects a
 * registration whose addr+len overflows before it is ever inserted (P1-2). */
static phxfs_p2p_map_t *find_locked(phxfs_mmap_buffer_t *mbuffer, u64 dev_addr) {
    /* MRU cache: KV/weight workloads hammer one big arena (P2-1). */
    phxfs_p2p_map_t *h = mbuffer->last_hit;
    if (h && h->has_reg && h->dev_addr <= dev_addr &&
        dev_addr < h->dev_addr + h->length)
        return h;
    phxfs_p2p_map_t *current = mbuffer->head;
    while (current) {
        if (current->has_reg && current->dev_addr <= dev_addr &&
            dev_addr < current->dev_addr + current->length) {
            mbuffer->last_hit = current;
            return current;
        }
        current = current->next;
    }
    return NULL;
}

/*
 * Acquire the registration containing `dev_addr` and take a reference so it
 * cannot be munmap()'d until the matching map_release() (P1-2). Returns NULL
 * if `dev_addr` is not inside any registration.
 */
static phxfs_p2p_map_t *map_acquire(phxfs_mmap_buffer_t *mbuffer, u64 dev_addr) {
    pthread_mutex_lock(&mbuffer->lock);
    phxfs_p2p_map_t *m = find_locked(mbuffer, dev_addr);
    if (m)
        m->refcount++;
    pthread_mutex_unlock(&mbuffer->lock);
    return m;
}

/* Drop a reference taken by map_acquire(); wakes a waiting dereg at zero. */
static void map_release(phxfs_mmap_buffer_t *mbuffer, phxfs_p2p_map_t *m) {
    if (!m)
        return;
    pthread_mutex_lock(&mbuffer->lock);
    if (--m->refcount == 0)
        pthread_cond_broadcast(&mbuffer->drain_cv);
    pthread_mutex_unlock(&mbuffer->lock);
}

/* Unlink `node` from the list. Caller must hold the lock. */
static void unlink_locked(phxfs_mmap_buffer_t *mbuffer, phxfs_p2p_map_t *node) {
    if (mbuffer->last_hit == node)
        mbuffer->last_hit = NULL;   /* invalidate MRU cache (P2-1) */
    phxfs_p2p_map_t **pp = &mbuffer->head;
    while (*pp) {
        if (*pp == node) {
            *pp = node->next;
            node->next = NULL;
            return;
        }
        pp = &(*pp)->next;
    }
}

static inline int __phxfs_regmem(phxfs_mmap_buffer_t *mbuffer, u64 dev_addr, u64 c_addr, size_t len) {
    phxfs_ioctl_para_t para;
    int ret;

    if (!mbuffer->init_stat) {
        return -1;
    }

    para.map_param.n_vaddr = (u64)dev_addr;
    para.map_param.c_vaddr = (u64)c_addr;
    para.map_param.c_size = len;
    para.map_param.n_size = len;
    para.map_param.dev.dev_id = mbuffer->device_id;
    ret = ioctl(mbuffer->bdev_fd, PHXFS_IOCTL_MAP, &para);

    return ret;
}

/*
 * Scan for an exact-duplicate registration (same addr+len, still registered)
 * to reuse; set *overlap if a *different* registration overlaps
 * [addr, addr+len). Caller holds pb->lock (P2-2).
 */
static phxfs_p2p_map_t *regmem_scan_locked(phxfs_mmap_buffer_t *pb, u64 addr,
                                           size_t len, bool *overlap) {
    *overlap = false;
    for (phxfs_p2p_map_t *m = pb->head; m; m = m->next) {
        /* A node still occupies its address range while it holds either the
         * kernel P2P mapping (has_reg) or just the host VMA (mapped, e.g. a
         * VMA-only node awaiting a munmap retry). Both must block a new,
         * overlapping registration so a half-torn-down region cannot be reused
         * and later mistaken for it (P1-2). */
        if (!m->has_reg && !m->mapped)
            continue;
        if (m->has_reg && m->dev_addr == addr && m->length == len)
            return m;                       /* exact live duplicate: reuse */
        u64 a0 = m->dev_addr, a1 = m->dev_addr + m->length;
        u64 b0 = addr,        b1 = addr + len;
        if (a0 < b1 && b0 < a1)
            *overlap = true;
    }
    return NULL;
}

int phxfs_regmem(int device_id, const void *addr, size_t len, void **target_addr) {
    /* Parameter validation (P1-5). */
    if (!addr || !target_addr) {
        fprintf(stderr, "%s: NULL addr/target_addr\n", __func__);
        return -EINVAL;
    }
    if (len == 0 || len % HUGE_PAGE_SIZE != 0) {
        fprintf(stderr, "%s: bad len %zu (nonzero, 64KiB-aligned required)\n",
                __func__, len);
        return -EINVAL;
    }
    if ((uint64_t)(uintptr_t)addr > UINT64_MAX - len) {
        fprintf(stderr, "%s: addr+len overflow\n", __func__);
        return -EINVAL;
    }

    phxfs_mmap_buffer_t *pb = dev_get(device_id);
    if (!pb) {
        fprintf(stderr, "%s: device %d not open\n", __func__, device_id);
        return -EINVAL;
    }

    /* Reuse an identical registration; reject a conflicting overlap (P2-2). */
    bool overlap = false;
    pthread_mutex_lock(&pb->lock);
    phxfs_p2p_map_t *dup = regmem_scan_locked(pb, (u64)addr, len, &overlap);
    if (dup) {
        dup->user_refs++;
        *target_addr = dup->vaddr;
        pthread_mutex_unlock(&pb->lock);
        dev_put(pb);
        return 0;
    }
    if (overlap) {
        pthread_mutex_unlock(&pb->lock);
        fprintf(stderr, "%s: region overlaps an existing registration\n", __func__);
        dev_put(pb);
        return -EINVAL;
    }
    pthread_mutex_unlock(&pb->lock);

    phxfs_p2p_map_t *p2p_map = (phxfs_p2p_map_t *)malloc(sizeof(*p2p_map));
    if (!p2p_map) {
        dev_put(pb);
        return -ENOMEM;
    }
    p2p_map->vaddr = NULL;
    p2p_map->length = len;
    p2p_map->dev_addr = (uint64_t)addr;
    p2p_map->has_reg = false;
    p2p_map->mapped = false;
    p2p_map->user_refs = 1;
    p2p_map->refcount = 0;
    p2p_map->next = NULL;

    /* Single contiguous mmap of the whole region (kvmalloc backend: no 2GiB
     * per-mmap limit). */
    p2p_map->vaddr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, pb->bdev_fd, 0);
    if (p2p_map->vaddr == MAP_FAILED) {
        fprintf(stderr, "%s: mmap fail (%s)\n", __func__, strerror(errno));
        free(p2p_map);
        dev_put(pb);
        return -EFAULT;
    }
    p2p_map->mapped = true;

    int ret = __phxfs_regmem(pb, p2p_map->dev_addr, (uint64_t)p2p_map->vaddr, len);
    if (ret) {
        fprintf(stderr, "%s: __phxfs_regmem fail ret=%d (%s)\n", __func__, ret, strerror(errno));
        munmap(p2p_map->vaddr, len);
        free(p2p_map);
        dev_put(pb);
        return -EFAULT;
    }
    p2p_map->has_reg = true;

    /* Re-check under the lock for a concurrent identical creator; if one won,
     * reuse it and drop our now-redundant mapping. */
    pthread_mutex_lock(&pb->lock);
    bool ov2 = false;
    phxfs_p2p_map_t *dup2 = regmem_scan_locked(pb, (u64)addr, len, &ov2);
    if (dup2) {
        dup2->user_refs++;
        *target_addr = dup2->vaddr;
        pthread_mutex_unlock(&pb->lock);
        __phxfs_deregmem(pb, p2p_map->dev_addr, (uint64_t)p2p_map->vaddr, len);
        munmap(p2p_map->vaddr, len);
        free(p2p_map);
        dev_put(pb);
        return 0;
    }
    if (ov2) {
        /* A different, partially-overlapping registration was inserted while
         * we were mmap/ioctl'ing outside the lock. Honour the "overlap is
         * rejected" contract: roll back our own mapping and fail (P0-2). */
        pthread_mutex_unlock(&pb->lock);
        __phxfs_deregmem(pb, p2p_map->dev_addr, (uint64_t)p2p_map->vaddr, len);
        munmap(p2p_map->vaddr, len);
        free(p2p_map);
        fprintf(stderr, "%s: region overlaps a concurrently-created registration\n", __func__);
        dev_put(pb);
        return -EINVAL;
    }
    p2p_map->next = pb->head;   /* insert (lock held) */
    pb->head = p2p_map;
    pthread_mutex_unlock(&pb->lock);

    *target_addr = p2p_map->vaddr;
    dev_put(pb);
    return 0;
}


int __phxfs_deregmem(phxfs_mmap_buffer_t *pb, u64 dev_addr, u64 c_addr, size_t len) {
    phxfs_ioctl_para_t para;
    int ret;

    if(!pb->init_stat) {
        return -1;
    }

    para.map_param.n_vaddr = (u64)dev_addr;
    para.map_param.c_vaddr = (u64)c_addr;
    para.map_param.c_size = len;
    para.map_param.n_size = len;
    para.map_param.dev.dev_id = pb->device_id;

    // print_mem(&para, sizeof(phxfs_ioctl_para_t));
    ret = ioctl(pb->bdev_fd, PHXFS_IOCTL_UNMAP, &para);

    return ret;
}

/*
 * Deregister a mapping (P1-2 / P1-6). Steps, in order:
 *   1. Under the lock: locate by exact dev_addr and validate (length,
 *      has_reg). If it is registered but still has in-flight references
 *      (an I/O batch resolved against it), block on drain_cv until they
 *      drop — dereg is thus safe even if it races a completing batch.
 *   2. Unlink under the lock so no new map_acquire() can find it.
 *   3. Release the lock, then ioctl-UNMAP + munmap.
 *   4. On teardown failure, re-insert the node so the mapping is still
 *      discoverable/retriable rather than silently leaked.
 */
/*
 * Deregister a mapping. The last regmem() reference triggers teardown
 * (P2-2 user_refs). Teardown order (P1-1): drain in-flight I/O refs, unlink,
 * kernel UNMAP (clears has_reg), then munmap (clears mapped). On failure the
 * node is re-inserted with accurate has_reg/mapped so the remaining step can
 * be retried and close() can still finish the cleanup.
 */
int phxfs_deregmem(int device_id, const void *addr, size_t len) {
    phxfs_mmap_buffer_t *pb = dev_get(device_id);
    if (!pb)
        return -1;

    pthread_mutex_lock(&pb->lock);
    phxfs_p2p_map_t *m = pb->head;
    while (m && m->dev_addr != (u64)addr)
        m = m->next;
    if (!m || m->length != len) {
        pthread_mutex_unlock(&pb->lock);
        fprintf(stderr, "%s: p2p_map not found / length mismatch\n", __func__);
        dev_put(pb);
        return -1;
    }
    /* Drop one client reference; only the last one tears down (P2-2). */
    if (m->user_refs > 1) {
        m->user_refs--;
        pthread_mutex_unlock(&pb->lock);
        dev_put(pb);
        return 0;
    }
    /* Last reference: wait for in-flight I/O, then unlink for teardown. */
    while (m->refcount > 0)
        pthread_cond_wait(&pb->drain_cv, &pb->lock);
    unlink_locked(pb, m);
    pthread_mutex_unlock(&pb->lock);

    if (m->has_reg) {
        if (__phxfs_deregmem(pb, m->dev_addr, (uint64_t)m->vaddr, m->length) != 0) {
            fprintf(stderr, "%s: __phxfs_deregmem fail\n", __func__);
            insert_phxfs_mmap_node(pb, m);   /* retriable: has_reg+mapped intact */
            dev_put(pb);
            return -1;
        }
        m->has_reg = false;
    }
    if (m->mapped) {
        if (munmap(m->vaddr, m->length) != 0) {
            fprintf(stderr, "%s: munmap fail (%s)\n", __func__, strerror(errno));
            insert_phxfs_mmap_node(pb, m);   /* retriable: has_reg=false, mapped=true */
            dev_put(pb);
            return -1;
        }
        m->mapped = false;
    }

    free(m);
    dev_put(pb);
    return 0;
}

int phxfs_close(phxfs_fileid_t fid) {
    return close(fid.fd);
}

/*
 * Resolve a registered GPU buffer reference (buf + buf_offset, nbyte) to its
 * host-mapped P2P address and take a reference on the mapping so it cannot be
 * munmap()'d until the caller map_release()s it (P1-2). `buf` may point
 * anywhere inside a registered region; the final host address is
 *     vaddr + (buf - dev_addr) + buf_offset
 * (the (buf - dev_addr) term was missing before — P0-3). All range math is
 * unsigned and overflow-safe; negative buf_offset is rejected (P1-3).
 *
 * Returns:
 *   +1  buf is inside a registration and [buf_offset, nbyte] is valid:
 *       *host = DMA host address, *out_node = referenced node (release later)
 *    0  buf is not inside any registration on this device (caller may treat
 *       it as a plain CPU address); host and out_node are both set to NULL
 *   -1  buf is inside a registration but the extent is invalid; no reference
 *       is held
 */
static int resolve_registered(int device_id, const void *buf, off_t buf_offset,
                              size_t nbyte, void **host,
                              phxfs_p2p_map_t **out_node) {
    *host = NULL;
    *out_node = NULL;
    /* The caller already holds a dev_get() operation reference on this device
     * (read/write/batch acquire it first), so the device is OPEN and cannot be
     * torn down under us — we must NOT re-consult phxfs_initialized here, which
     * a concurrent close() flips to false *before* draining in-flight ops and
     * would spuriously fail an already-admitted operation (P0-1). */
    if (device_id < 0 || device_id >= g_device_count)
        return 0;
    if (buf_offset < 0) {
        fprintf(stderr, "%s: negative buf_offset=%ld\n", __func__, (long)buf_offset);
        return -1;
    }

    phxfs_mmap_buffer_t *pb = &mbuffer[device_id];
    phxfs_p2p_map_t *m = map_acquire(pb, (u64)buf);
    if (!m)
        return 0;  /* not a registered buffer */

    /* find_locked guarantees dev_addr <= buf, so inner is non-negative. */
    uint64_t inner = (uint64_t)buf - m->dev_addr;
    uint64_t off   = (uint64_t)buf_offset;
    uint64_t len   = m->length;

    /* Overflow-safe bound: inner + off + nbyte <= len. */
    if (inner > len || off > len - inner || (uint64_t)nbyte > len - inner - off) {
        fprintf(stderr,
                "%s: out of range: inner=%llu buf_offset=%llu nbyte=%zu length=%llu\n",
                __func__, (unsigned long long)inner, (unsigned long long)off,
                nbyte, (unsigned long long)len);
        map_release(pb, m);
        return -1;
    }

    *host = (void *)((char *)m->vaddr + inner + off);
    *out_node = m;
    return 1;
}

ssize_t phxfs_read(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset){
    phxfs_mmap_buffer_t *pb = dev_get(fid.deviceID);
    if (!pb)
        return -1;
    /* Validate before resolving: a negative nbyte would otherwise convert to a
     * huge size_t and surface as a spurious lookup failure instead of the
     * expected -EINVAL (P1-2). Also reject f_offset+nbyte overflow up front so
     * the I/O loop never computes an overflowed offset (P1-4). */
    if (nbyte < 0 || f_offset < 0 ||
        (uint64_t)nbyte > (uint64_t)INT64_MAX - (uint64_t)f_offset) {
        dev_put(pb);
        return -EINVAL;
    }
    void *host = NULL;
    phxfs_p2p_map_t *node = NULL;
    if (resolve_registered(fid.deviceID, buf, buf_offset, nbyte, &host, &node) != 1) {
        dev_put(pb);
        return -1;  /* single read/write requires a registered GPU buffer */
    }

    char *base = (char *)host;
    ssize_t nbyte_total = 0;
    ssize_t rc = nbyte_total;
    // Single contiguous mapping; chunk only to stay under MAX_RW_COUNT (~2GiB).
    while (nbyte_total < nbyte) {
        size_t this_nbyte = (size_t)(nbyte - nbyte_total);
        if (this_nbyte > PHXFS_IO_CHUNK)
            this_nbyte = PHXFS_IO_CHUNK;
        ssize_t ret = pread(fid.fd, base + nbyte_total, this_nbyte, f_offset + nbyte_total);
        if (ret < 0) {
            if (errno == EINTR)
                continue;                       /* P1-7: retry */
            fprintf(stderr, "%s: pread error: %s\n", __func__, strerror(errno));
            /* Return partial progress if any, else the negative errno. */
            rc = nbyte_total > 0 ? nbyte_total : -errno;
            goto out_read;
        }
        if (ret == 0)
            break; // EOF
        nbyte_total += ret;
    }
    rc = nbyte_total;
out_read:
    map_release(&mbuffer[fid.deviceID], node);
    dev_put(pb);
    return rc;
}

ssize_t phxfs_write(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset){
    phxfs_mmap_buffer_t *pb = dev_get(fid.deviceID);
    if (!pb)
        return -1;
    /* Validate before resolving (see phxfs_read) — P1-2. */
    if (nbyte < 0 || f_offset < 0) {
        dev_put(pb);
        return -EINVAL;
    }
    void *host = NULL;
    phxfs_p2p_map_t *node = NULL;
    if (resolve_registered(fid.deviceID, buf, buf_offset, nbyte, &host, &node) != 1) {
        dev_put(pb);
        return -1;
    }

    char *base = (char *)host;
    ssize_t nbyte_total = 0;
    ssize_t rc = nbyte_total;
    while (nbyte_total < nbyte) {
        size_t this_nbyte = (size_t)(nbyte - nbyte_total);
        if (this_nbyte > PHXFS_IO_CHUNK)
            this_nbyte = PHXFS_IO_CHUNK;
        ssize_t ret = pwrite(fid.fd, base + nbyte_total, this_nbyte, f_offset + nbyte_total);
        if (ret < 0) {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "%s: pwrite error: %s\n", __func__, strerror(errno));
            rc = nbyte_total > 0 ? nbyte_total : -errno;
            goto out_write;
        }
        if (ret == 0)
            break;
        nbyte_total += ret;
    }
    rc = nbyte_total;
out_write:
    map_release(&mbuffer[fid.deviceID], node);
    dev_put(pb);
    return rc;
}

/* ------------------------------------------------------------------ *
 * Batch I/O — engine selection + request resolution
 * ------------------------------------------------------------------ */

/*
 * The batch engine is chosen once at library load (constructor below),
 * not per call, so the hot path just reads a pointer. Preference order:
 *   io_uring -> sync. (libaio slots in here later.)
 * Never NULL: the sync engine always probes successfully.
 */
static const struct phxfs_io_engine *g_engine = nullptr;

static void phxfs_io_engine_select(void) {
    const struct phxfs_io_engine *candidates[] = {
#ifdef PHXFS_HAVE_LIBURING
        &phxfs_io_engine_uring,
#endif
        &phxfs_io_engine_sync,
    };
    for (const auto *e : candidates) {
        if (e->probe && e->probe() == 0) {
            g_engine = e;
            return;
        }
    }
    g_engine = &phxfs_io_engine_sync;  /* guaranteed fallback */
}

__attribute__((constructor))
static void phxfs_io_engine_ctor(void) {
    phxfs_io_engine_select();
}

const struct phxfs_io_engine *phxfs_io_engine_get(void) {
    if (!g_engine)          /* defensive: constructor should have run */
        phxfs_io_engine_select();
    return g_engine;
}

const char *phxfs_io_engine_name(void) {
    return phxfs_io_engine_get()->name;
}

/*
 * Bundled state for one in-flight batch (sync = stack, async = in handle).
 * A batch holds a dev_get() operation ref on every distinct GPU device it
 * touches for its whole duration, so a concurrent close() waits (P0-2/P1-4).
 * CPU-buffer requests use device_id < 0; a request with device_id >= 0 MUST
 * resolve to a registered GPU buffer or it fails — no silent CPU fallthrough,
 * which also closes the closing-device race.
 */
struct batch_ctx {
    struct phxfs_io_op_req *ops;      /* compact resolved requests */
    int                    *map;      /* orig index per compact op */
    phxfs_p2p_map_t        **nodes;   /* mapping ref per op (NULL for CPU) */
    int                    *devs;     /* device per op (-1 for CPU); used to
                                         release without trusting user data
                                         after submit (P1-6) */
    int                     cnt;      /* resolved op count */
    int                     resolve_fail;
    int                     numa;     /* pool routing node */
    bool                    dev_held[PHXFS_MAX_DEVICES]; /* dev_get()'d */
};

static void batch_ctx_free(struct batch_ctx *bc) {
    free(bc->ops);   bc->ops = NULL;
    free(bc->map);   bc->map = NULL;
    free(bc->nodes); bc->nodes = NULL;
    free(bc->devs);  bc->devs = NULL;
}

/* Release mapping refs, then per-device operation refs. Order matters:
 * mapping refs must drop before dev_put lets a draining close() proceed. */
static void batch_ctx_release(struct batch_ctx *bc) {
    for (int k = 0; k < bc->cnt; k++)
        if (bc->nodes[k])
            map_release(&mbuffer[bc->devs[k]], bc->nodes[k]);
    for (int d = 0; d < g_device_count; d++)
        if (bc->dev_held[d])
            dev_put(&mbuffer[d]);
}

/*
 * NUMA node of a phxfs device — pure lookup of the value queried once at
 * phxfs_open() time (see query_dev_numa / __phxfs_open). Routing a batch to
 * the pool on the target GPU's node avoids cross-NUMA DMA.
 */
static int phxfs_dev_numa(int device_id) {
    /* Called only for devices this batch already holds a dev_get() ref on, so
     * no phxfs_initialized re-check (P0-1); the node was queried at open. */
    if (device_id >= 0 && device_id < g_device_count)
        return mbuffer[device_id].numa_node;
    return 0;
}

/*
 * Resolve all requests into a compact op array (P1-4: unresolved requests get
 * result = -EFAULT and are dropped, never handed to the engine). Every
 * distinct GPU device referenced is dev_get()'d up front and tracked in
 * bc->dev_held, so a concurrent close() waits until batch completion.
 * Returns 0, or -ENOMEM.
 */
static int phxfs_batch_prepare(phxfs_io_req_t *reqs, int n, struct batch_ctx *bc) {
    memset(bc, 0, sizeof(*bc));
    bc->ops   = (struct phxfs_io_op_req *)malloc((size_t)n * sizeof(*bc->ops));
    bc->map   = (int *)malloc((size_t)n * sizeof(*bc->map));
    bc->nodes = (phxfs_p2p_map_t **)malloc((size_t)n * sizeof(*bc->nodes));
    bc->devs  = (int *)malloc((size_t)n * sizeof(*bc->devs));
    if (!bc->ops || !bc->map || !bc->nodes || !bc->devs) {
        batch_ctx_free(bc);
        return -ENOMEM;
    }

    /* Hold an operation ref on every distinct GPU device referenced. */
    for (int i = 0; i < n; i++) {
        int d = reqs[i].device_id;
        if (d >= 0 && d < g_device_count && !bc->dev_held[d])
            if (dev_get(d))
                bc->dev_held[d] = true;
    }

    int cnt = 0, fail = 0;
    bool numa_set = false;
    for (int i = 0; i < n; i++) {
        int d = reqs[i].device_id;
        void *host = NULL;
        phxfs_p2p_map_t *node = NULL;

        /* Common validation (P1-2): reject negative file offset and an
         * f_offset+nbytes that would overflow off_t. */
        bool valid = reqs[i].f_offset >= 0 &&
                     (uint64_t)reqs[i].nbytes <=
                         (uint64_t)INT64_MAX - (uint64_t)reqs[i].f_offset;

        if (!valid) {
            host = NULL;
        } else if (d < 0) {
            /* CPU buffer (caller-owned host memory). Guard the pointer +
             * offset + length arithmetic against address overflow (P1-2). */
            if (reqs[i].buf_offset >= 0) {
                uintptr_t b = (uintptr_t)reqs[i].buf;
                uintptr_t o = (uintptr_t)reqs[i].buf_offset;
                if (b <= UINTPTR_MAX - o &&
                    (uint64_t)reqs[i].nbytes <= (uint64_t)(UINTPTR_MAX - (b + o)))
                    host = (void *)(b + o);
            }
        } else if (d < g_device_count && bc->dev_held[d]) {
            /* GPU buffer: must resolve to a registered mapping. */
            if (resolve_registered(d, reqs[i].buf, reqs[i].buf_offset,
                                   reqs[i].nbytes, &host, &node) != 1)
                host = NULL;
        }
        /* else: invalid range or unacquirable (closing) device -> fail. */

        if (!host) {
            reqs[i].result = -EFAULT;
            fail++;
            continue;
        }
        if (!numa_set && d >= 0) {
            bc->numa = phxfs_dev_numa(d);
            numa_set = true;
        }
        bc->ops[cnt].fd = reqs[i].fd;
        bc->ops[cnt].host_addr = host;
        bc->ops[cnt].nbytes = reqs[i].nbytes;
        bc->ops[cnt].f_offset = reqs[i].f_offset;
        bc->ops[cnt].result = -EFAULT;
        bc->map[cnt] = i;
        bc->nodes[cnt] = node;
        bc->devs[cnt] = d;
        cnt++;
    }
    if (!numa_set) {
        /* CPU-only batch: no GPU pins the NUMA node. Route to the caller's
         * current node instead of always node 0 (P2-5), so host buffers on
         * another socket are not all serviced by node 0's workers/slot. */
        unsigned cpu = 0, node = 0;
        /* Use the raw syscall: the getcpu() glibc wrapper only exists on
         * glibc >= 2.29, but SYS_getcpu is always available (P2-5). */
        if (syscall(SYS_getcpu, &cpu, &node, NULL) == 0)
            bc->numa = (int)node;
    }
    bc->cnt = cnt;
    bc->resolve_fail = fail;
    return 0;
}

/* NUMA node an op should run on: its GPU device's node, or the batch's
 * default node for CPU ops. */
static inline int op_numa(const struct batch_ctx *bc, int k) {
    int d = bc->devs[k];
    int node = (d >= 0) ? mbuffer[d].numa_node : bc->numa;
    /* Fold any node outside the pool range onto pool 0 — matching pool_for()'s
     * clamp — so the mixed-batch grouping loop (0..PHXFS_MAX_NUMA_NODES-1) can
     * never silently drop an op whose GPU sits on node >= 8 (P0-3). */
    if (node < 0 || node >= PHXFS_MAX_NUMA_NODES)
        node = 0;
    return node;
}

/*
 * Run all resolved ops (sync), grouped by target NUMA node so each group runs
 * on the pool of its own node — a mixed-device batch no longer forces every
 * request onto the first device's node (P2-3). Fills bc->ops[k].result.
 * Returns the failure count, or the first negative engine-level error.
 */
static int batch_run_grouped(struct batch_ctx *bc, enum phxfs_io_op op) {
    if (bc->cnt <= 0)
        return 0;

    /* Fast path: single NUMA node. */
    bool mixed = false;
    int n0 = op_numa(bc, 0);
    for (int k = 1; k < bc->cnt; k++)
        if (op_numa(bc, k) != n0) { mixed = true; break; }
    if (!mixed)
        return phxfs_pool_run(bc->ops, bc->cnt, op, n0);

    /* Mixed: partition per node into independent arrays, dispatch ALL nodes up
     * front (blocking submit reserves each node's slot and returns once its
     * workers have started), then join — so the NUMA nodes run concurrently
     * instead of node-by-node serially (P2-1). */
    struct phxfs_io_op_req  *nops[PHXFS_MAX_NUMA_NODES] = {0};
    int                     *nidx[PHXFS_MAX_NUMA_NODES] = {0};
    int                      ncnt[PHXFS_MAX_NUMA_NODES] = {0};
    struct phxfs_pool_async *nh[PHXFS_MAX_NUMA_NODES]   = {0};

    for (int k = 0; k < bc->cnt; k++)
        ncnt[op_numa(bc, k)]++;

    bool oom = false;
    for (int node = 0; node < PHXFS_MAX_NUMA_NODES; node++) {
        if (ncnt[node] == 0)
            continue;
        nops[node] = (struct phxfs_io_op_req *)malloc((size_t)ncnt[node] * sizeof(*nops[node]));
        nidx[node] = (int *)malloc((size_t)ncnt[node] * sizeof(*nidx[node]));
        if (!nops[node] || !nidx[node]) { oom = true; break; }
        int m = 0;
        for (int k = 0; k < bc->cnt; k++)
            if (op_numa(bc, k) == node) { nops[node][m] = bc->ops[k]; nidx[node][m] = k; m++; }
    }
    if (oom) {                     /* free partial, fall back to single-node */
        for (int node = 0; node < PHXFS_MAX_NUMA_NODES; node++) { free(nops[node]); free(nidx[node]); }
        return phxfs_pool_run(bc->ops, bc->cnt, op, n0);
    }

    for (int node = 0; node < PHXFS_MAX_NUMA_NODES; node++)
        if (ncnt[node] > 0)
            nh[node] = phxfs_pool_submit(nops[node], ncnt[node], op, node, /*blocking=*/true);

    int total_fail = 0, first_err = 0;
    for (int node = 0; node < PHXFS_MAX_NUMA_NODES; node++) {
        if (ncnt[node] == 0)
            continue;
        int r;
        if (nh[node]) {
            r = phxfs_pool_wait(nh[node]);
        } else {                   /* pool shutting down/unavailable: inline */
            const struct phxfs_io_engine *eng = phxfs_io_engine_get();
            r = eng->submit_batch(nops[node], ncnt[node], op);
        }
        for (int j = 0; j < ncnt[node]; j++)
            bc->ops[nidx[node][j]].result = nops[node][j].result;
        if (r < 0) { if (first_err == 0) first_err = r; }
        else       { total_fail += r; }
        free(nops[node]); free(nidx[node]);
    }
    return first_err ? first_err : total_fail;
}

static int phxfs_batch(phxfs_io_req_t *reqs, int n, enum phxfs_io_op op) {
    if (n <= 0)
        return 0;

    struct batch_ctx bc;
    if (phxfs_batch_prepare(reqs, n, &bc) < 0)
        return -ENOMEM;

    int ret = 0;
    if (bc.cnt > 0) {
        ret = batch_run_grouped(&bc, op);
        for (int k = 0; k < bc.cnt; k++)
            reqs[bc.map[k]].result = bc.ops[k].result;
    }

    int resolve_fail = bc.resolve_fail;
    batch_ctx_release(&bc);
    batch_ctx_free(&bc);

    if (ret < 0)               /* engine-level error dominates */
        return ret;
    return ret + resolve_fail; /* per-request failures + unresolved */
}

int phxfs_read_batch(phxfs_io_req_t *reqs, int n) {
    return phxfs_batch(reqs, n, PHXFS_IO_READ);
}

int phxfs_write_batch(phxfs_io_req_t *reqs, int n) {
    return phxfs_batch(reqs, n, PHXFS_IO_WRITE);
}

/* ------------------------------------------------------------------ *
 * Async batch (compute/I/O overlap)
 *
 * A batch may span several NUMA nodes. We partition the resolved ops by node
 * and submit one non-blocking sub-batch to each node's pool, so every GPU's
 * requests run on the pool local to that GPU instead of all crowding the first
 * device's node (P2-2). wait() joins every sub-batch and aggregates results.
 * ------------------------------------------------------------------ */

/* One per-NUMA-node async sub-submission. */
struct batch_sub {
    struct phxfs_pool_async *pool_h;  /* pool handle for this node */
    struct phxfs_io_op_req  *ops;     /* owned contiguous ops for this node */
    int                     *idx;     /* back-map: sub op -> bc.ops index */
    int                      n;       /* op count for this node */
    int                      node;    /* NUMA node this sub runs on */
};

struct phxfs_batch {
    struct batch_ctx bc;               /* resolved ops + held device/mapping refs */
    phxfs_io_req_t  *reqs;             /* user array, for result copy-back */
    struct batch_sub sub[PHXFS_MAX_NUMA_NODES];
    int              nsub;             /* number of active per-node subs */
};

/* Free every sub's owned arrays (not the pool handles). */
static void batch_subs_free(struct phxfs_batch *h) {
    for (int t = 0; t < h->nsub; t++) {
        free(h->sub[t].ops);  h->sub[t].ops = NULL;
        free(h->sub[t].idx);  h->sub[t].idx = NULL;
    }
    h->nsub = 0;
}

/*
 * Partition h->bc.ops by NUMA node into h->sub[] (one entry per non-empty
 * node), each with a freshly allocated ops/idx array. Does not submit.
 * Returns 0, or -ENOMEM (nothing left allocated on failure).
 */
static int batch_partition(struct phxfs_batch *h) {
    struct batch_ctx *bc = &h->bc;
    h->nsub = 0;

    int cnt_by[PHXFS_MAX_NUMA_NODES] = {0};
    for (int k = 0; k < bc->cnt; k++)
        cnt_by[op_numa(bc, k)]++;

    for (int node = 0; node < PHXFS_MAX_NUMA_NODES; node++) {
        if (cnt_by[node] == 0)
            continue;
        struct batch_sub *s = &h->sub[h->nsub];
        s->pool_h = NULL;
        s->node = node;
        s->n = 0;
        s->ops = (struct phxfs_io_op_req *)malloc((size_t)cnt_by[node] * sizeof(*s->ops));
        s->idx = (int *)malloc((size_t)cnt_by[node] * sizeof(*s->idx));
        if (!s->ops || !s->idx) {
            free(s->ops); free(s->idx);
            batch_subs_free(h);        /* frees the ones built so far */
            return -ENOMEM;
        }
        for (int k = 0; k < bc->cnt; k++) {
            if (op_numa(bc, k) == node) {
                s->ops[s->n] = bc->ops[k];
                s->idx[s->n] = k;
                s->n++;
            }
        }
        h->nsub++;
    }
    return 0;
}

static phxfs_batch_t *phxfs_batch_submit(phxfs_io_req_t *reqs, int n,
                                         enum phxfs_io_op op) {
    if (n <= 0)
        return NULL;

    phxfs_batch_t *h = (phxfs_batch_t *)calloc(1, sizeof(*h));
    if (!h)
        return NULL;
    h->reqs = reqs;

    if (phxfs_batch_prepare(reqs, n, &h->bc) < 0) {
        free(h);
        return NULL;
    }

    if (h->bc.cnt > 0) {
        if (batch_partition(h) < 0) {
            batch_ctx_release(&h->bc);
            batch_ctx_free(&h->bc);
            free(h);
            errno = ENOMEM;
            return NULL;
        }
        /* Submit each node's sub-batch (non-blocking). A per-node submit may
         * fail (EBUSY / pool unavailable). We do NOT drain+discard the ones
         * already running — that would make a "non-blocking" submit block and
         * would hide their side effects (P1-1). Instead:
         *   - if NO sub was submitted, fail cleanly with NULL/errno (nothing
         *     ran, side-effect-free; preserves the single-node EBUSY contract);
         *   - otherwise return a waitable handle. Un-submitted subs' requests
         *     are marked -EBUSY here and reported per-request by wait(), so the
         *     caller can retry exactly those without redoing committed I/O. */
        int submitted = 0, first_err = 0;
        for (int t = 0; t < h->nsub; t++) {
            struct batch_sub *s = &h->sub[t];
            s->pool_h = phxfs_pool_submit(s->ops, s->n, op, s->node, /*blocking=*/false);
            if (s->pool_h) {
                submitted++;
            } else {
                if (first_err == 0) first_err = errno ? errno : EBUSY;
                for (int j = 0; j < s->n; j++)
                    h->bc.ops[s->idx[j]].result = -EBUSY;  /* not run */
            }
        }
        if (submitted == 0) {
            batch_subs_free(h);
            batch_ctx_release(&h->bc);
            batch_ctx_free(&h->bc);
            free(h);
            errno = first_err ? first_err : EBUSY;
            return NULL;
        }
    }
    return h;
}

phxfs_batch_t *phxfs_batch_submit_read(phxfs_io_req_t *reqs, int n) {
    return phxfs_batch_submit(reqs, n, PHXFS_IO_READ);
}

phxfs_batch_t *phxfs_batch_submit_write(phxfs_io_req_t *reqs, int n) {
    return phxfs_batch_submit(reqs, n, PHXFS_IO_WRITE);
}

/* Join every sub-batch, copy per-node results back into bc.ops, then into the
 * user array. Aggregates failures; a negative engine-level error dominates. */
int phxfs_batch_wait(phxfs_batch_t *h) {
    if (!h)
        return -EINVAL;

    int failed = 0, err = 0;
    for (int t = 0; t < h->nsub; t++) {
        struct batch_sub *s = &h->sub[t];
        if (s->pool_h) {
            int r = phxfs_pool_wait(s->pool_h);
            for (int j = 0; j < s->n; j++)
                h->bc.ops[s->idx[j]].result = s->ops[j].result;
            if (r < 0) { if (err == 0) err = r; }
            else       { failed += r; }
        } else {
            /* Sub never submitted (P1-1 partial): its bc.ops results were
             * already set to -EBUSY; count them as failures, don't overwrite. */
            failed += s->n;
        }
    }
    for (int k = 0; k < h->bc.cnt; k++)
        h->reqs[h->bc.map[k]].result = h->bc.ops[k].result;

    int resolve_fail = h->bc.resolve_fail;
    batch_subs_free(h);
    batch_ctx_release(&h->bc);
    batch_ctx_free(&h->bc);
    free(h);
    return (err < 0) ? err : failed + resolve_fail;
}

/*
 * Abandon an async batch: wait for in-flight I/O to quiesce (submitted pool /
 * io_uring ops cannot be cancelled), then release the held device/mapping refs
 * and free the handle WITHOUT copying results back. Gives a caller that drops a
 * batch a defined way to release the NUMA pool(s) and all references instead of
 * leaking them until process exit (P1-1). Returns 0, or -EINVAL for NULL.
 */
int phxfs_batch_destroy(phxfs_batch_t *h) {
    if (!h)
        return -EINVAL;
    for (int t = 0; t < h->nsub; t++)
        if (h->sub[t].pool_h)
            (void)phxfs_pool_wait(h->sub[t].pool_h);
    batch_subs_free(h);
    batch_ctx_release(&h->bc);
    batch_ctx_free(&h->bc);
    free(h);
    return 0;
}

