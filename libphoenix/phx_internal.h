#ifndef __PHX_INTERNAL_H__
#define __PHX_INTERNAL_H__

/*
 * Internal header shared by phx_device.cpp / phx_mem.cpp / phx_io.cpp.
 * NOT part of the public API — do not include from outside libphoenix.
 */

#include <pthread.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>
#include "common.h"   /* u64 */

#define HUGE_PAGE_SIZE   (64 * 1024)   /* NVIDIA GPU page size */
#define PHXFS_MAX_DEVICES 8

/* ---- Registration node (one per phxfs_regmem call) ---- */
typedef struct phxfs_mmap_node_s {
    void *vaddr;        /* single contiguous host-mapped P2P address */
    uint64_t dev_addr;  /* registered GPU/device address (lookup key) */
    size_t length;      /* registered length */
    bool has_reg;       /* kernel P2P mapping active (ioctl MAP done, not UNMAP) */
    bool mapped;        /* host VMA active (mmap done, munmap not yet) */
    int  user_refs;     /* outstanding regmem() references (dup reuse) */
    int  refcount;      /* in-flight resolutions/I/O referencing this node */
    struct phxfs_mmap_node_s *next;
} phxfs_p2p_map_t;

/* ---- Per-device state ---- */
typedef struct phxfs_mmap_buffer_s {
    int device_id;
    int bdev_fd;
    bool init_stat;     /* device is OPEN (fd valid, usable) */
    bool closing;       /* close in progress: reject new dev_get() */
    int  open_count;    /* client open refcount */
    int  active_ops;    /* in-flight device operations */
    struct phxfs_mmap_node_s *head;
    struct phxfs_mmap_node_s *last_hit;  /* MRU lookup cache */
    pthread_mutex_t lock;
    pthread_cond_t  drain_cv;   /* mapping refcount==0 or active_ops==0 */
} phxfs_mmap_buffer_t;

extern int g_device_count;
extern phxfs_mmap_buffer_t mbuffer[PHXFS_MAX_DEVICES];

/* ---- phx_device.cpp ---- */
phxfs_mmap_buffer_t *dev_get(int device_id);
void dev_put(phxfs_mmap_buffer_t *pb);

/* ---- phx_mem.cpp ---- */
void free_phxfs_p2p_map(phxfs_mmap_buffer_t *buffer);
void map_release(phxfs_mmap_buffer_t *mbuffer, phxfs_p2p_map_t *m);
int resolve_registered(int device_id, const void *buf, off_t buf_offset,
                       size_t nbyte, void **host, phxfs_p2p_map_t **out_node);

/* ---- phx_io.cpp ---- */
/* (no cross-module exports besides the public API in phoenix.h) */

#endif /* __PHX_INTERNAL_H__ */
