/*
 * phx_mem.cpp — GPU memory registration, mapping, and buffer resolution.
 */

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "phoenix.h"
#include "phx_internal.h"

/* Forward decl: defined below, called by free_phxfs_p2p_map. */
static int __phxfs_deregmem(phxfs_mmap_buffer_t *pb, u64 dev_addr, u64 c_addr, size_t len);

/* ------------------------------------------------------------------ */
/* Registration list helpers (caller holds pb->lock unless noted)     */
/* ------------------------------------------------------------------ */

/* Find the registration that contains `dev_addr` (a single byte inside
 * [dev_addr0, dev_addr0+length)). Caller must hold the lock. The precise
 * extent check (offset + length) is done by the resolver, not here.
 * Every dev_addr+length sum here is overflow-safe: phxfs_regmem() rejects a
 * registration whose addr+len overflows before it is ever inserted. */
static phxfs_p2p_map_t *find_locked(phxfs_mmap_buffer_t *mbuffer, u64 dev_addr) {
    /* MRU cache: KV/weight workloads hammer one big arena. */
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
 * cannot be munmap()'d until the matching map_release(). Returns NULL
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
void map_release(phxfs_mmap_buffer_t *mbuffer, phxfs_p2p_map_t *m) {
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
        mbuffer->last_hit = NULL;   /* invalidate MRU cache */
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

static int insert_phxfs_mmap_node(phxfs_mmap_buffer_t *mbuffer, phxfs_p2p_map_t *new_node) {
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

/* ------------------------------------------------------------------ */
/* Kernel ioctl wrappers                                              */
/* ------------------------------------------------------------------ */

static int __phxfs_regmem(phxfs_mmap_buffer_t *mbuffer, u64 dev_addr, u64 c_addr, size_t len) {
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

static int __phxfs_deregmem(phxfs_mmap_buffer_t *pb, u64 dev_addr, u64 c_addr, size_t len) {
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

    ret = ioctl(pb->bdev_fd, PHXFS_IOCTL_UNMAP, &para);

    return ret;
}

/* ------------------------------------------------------------------ */
/* Registration scan (caller holds pb->lock)                          */
/* ------------------------------------------------------------------ */

/*
 * Scan for an exact-duplicate registration (same addr+len, still registered)
 * to reuse; set *overlap if a *different* registration overlaps
 * [addr, addr+len). Caller holds pb->lock.
 */
static phxfs_p2p_map_t *regmem_scan_locked(phxfs_mmap_buffer_t *pb, u64 addr,
                                           size_t len, bool *overlap) {
    *overlap = false;
    for (phxfs_p2p_map_t *m = pb->head; m; m = m->next) {
        /* A node still occupies its address range while it holds either the
         * kernel P2P mapping (has_reg) or just the host VMA (mapped, e.g. a
         * VMA-only node awaiting a munmap retry). Both must block a new,
         * overlapping registration so a half-torn-down region cannot be
         * reused and later mistaken for it. */
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

/* ------------------------------------------------------------------ */
/* Bulk teardown (called by __phxfs_teardown in phx_device.cpp)       */
/* ------------------------------------------------------------------ */

/*
 * Tear down every registration owned by `buffer`. Unlike a plain free, this
 * releases the kernel P2P mapping (ioctl UNMAP) and the host VMA (munmap) for
 * each node, so close() does not leak mappings. MUST be called while
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
    buffer->last_hit = NULL;   /* drop MRU cache */
    pthread_mutex_unlock(&buffer->lock);

    struct phxfs_mmap_node_s *next;
    while (current) {
        next = current->next;
        if (current->has_reg)
            __phxfs_deregmem(buffer, current->dev_addr,
                             (uint64_t)current->vaddr, current->length);
        if (current->mapped && current->vaddr && current->vaddr != MAP_FAILED)
            munmap(current->vaddr, current->length);   /* VMA-only node: still ok to unmap */
        free(current);
        current = next;
    }
}

/* ------------------------------------------------------------------ */
/* Public API: regmem / deregmem                                      */
/* ------------------------------------------------------------------ */

int phxfs_regmem(int device_id, const void *addr, size_t len, void **target_addr) {
    if (!addr || !target_addr) {
        fprintf(stderr, "%s: NULL addr/target_addr\n", __func__);
        return -EINVAL;
    }
    if (len == 0 || len % HUGE_PAGE_SIZE != 0) {
        fprintf(stderr, "%s: bad len %zu (nonzero, 64KiB-aligned required)\n",
                __func__, len);
        return -EINVAL;
    }
    if ((uint64_t)(uintptr_t)addr % HUGE_PAGE_SIZE != 0) {
        fprintf(stderr, "%s: addr %p not %d-byte aligned\n",
                __func__, addr, HUGE_PAGE_SIZE);
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

    /* Reuse an identical registration; reject a conflicting overlap. */
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
         * rejected" contract: roll back our own mapping and fail. */
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

/*
 * Deregister a mapping. The last regmem() reference triggers teardown
 * (user_refs). Teardown order: drain in-flight I/O refs, unlink, kernel
 * UNMAP (clears has_reg), then munmap (clears mapped). On failure the node
 * is re-inserted with accurate has_reg/mapped so the remaining step can be
 * retried and close() can still finish the cleanup.
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
    /* Drop one client reference; only the last one tears down. */
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

/* ------------------------------------------------------------------ */
/* Buffer resolution (used by phx_io.cpp)                             */
/* ------------------------------------------------------------------ */

/*
 * Resolve a registered GPU buffer reference (buf + buf_offset, nbyte) to its
 * host-mapped P2P address and take a reference on the mapping so it cannot be
 * munmap()'d until the caller map_release()s it. `buf` may point anywhere
 * inside a registered region; the final host address is
 *     vaddr + (buf - dev_addr) + buf_offset
 * All range math is unsigned and overflow-safe; negative buf_offset is
 * rejected.
 *
 * Returns:
 *   +1  buf is inside a registration and [buf_offset, nbyte] is valid:
 *       *host = DMA host address, *out_node = referenced node (release later)
 *    0  buf is not inside any registration on this device (caller may treat
 *       it as a plain CPU address); host and out_node are both set to NULL
 *   -1  buf is inside a registration but the extent is invalid; no reference
 *       is held
 */
int resolve_registered(int device_id, const void *buf, off_t buf_offset,
                       size_t nbyte, void **host,
                       phxfs_p2p_map_t **out_node) {
    *host = NULL;
    *out_node = NULL;
    /*
     * The caller already holds a dev_get() operation reference on this
     * device (read/write/batch acquire it first), so the device is OPEN and
     * cannot be torn down under us.
     */
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
