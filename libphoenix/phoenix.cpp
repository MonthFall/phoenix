#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <linux/types.h>
#include <pthread.h>
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
    bool has_reg;
    struct phxfs_mmap_node_s *next; // 指向下一个节点的指针
} phxfs_p2p_map_t;

typedef struct phxfs_mmap_buffer_s {
    int device_id;
    int bdev_fd;
    bool init_stat;
    struct phxfs_mmap_node_s *head;
    pthread_mutex_t lock;
} phxfs_mmap_buffer_t;

static int g_device_count = PHXFS_MAX_DEVICES;
static phxfs_mmap_buffer_t mbuffer[PHXFS_MAX_DEVICES];
static std::vector<std::string> phxfs_dev_path = {
    "/dev/phxfs_dev0", "/dev/phxfs_dev1",
    "/dev/phxfs_dev2", "/dev/phxfs_dev3",
    "/dev/phxfs_dev4", "/dev/phxfs_dev5",
    "/dev/phxfs_dev6", "/dev/phxfs_dev7"
};

static std::vector<bool> phxfs_initialized(PHXFS_MAX_DEVICES, false);

void free_phxfs_p2p_map(phxfs_mmap_buffer_t *buffer) {
    struct phxfs_mmap_node_s *current = buffer->head;
    struct phxfs_mmap_node_s *next;
    int count = 0;
    pthread_mutex_lock(&buffer->lock);

    while (current) {
        next = current->next;
        // For nodes that are not pinned, manual release is required
        free(current);
        current = next;
        count++;
    }

    buffer->head = NULL;
    pthread_mutex_unlock(&buffer->lock);

}

static int __phxfs_close(phxfs_mmap_buffer_t *mbuffer) {
    if (!mbuffer->init_stat)
        return -1;
    if (mbuffer->bdev_fd > 0)
        close(mbuffer->bdev_fd);

    free_phxfs_p2p_map(mbuffer);
    pthread_mutex_destroy(&mbuffer->lock);
    return 0;
}

static int phxfs_close_all() {
    for (int i = 0; i < g_device_count; i++) {
        if (phxfs_initialized[i]) {
            __phxfs_close(&mbuffer[i]);
            phxfs_initialized[i] = false;
        }
    }
    return 0;
}

int phxfs_close(int device_id) {
    if (device_id >= 0 && device_id < g_device_count) {
        if (phxfs_initialized[device_id]) {
            __phxfs_close(&mbuffer[device_id]);
            phxfs_initialized[device_id] = false;
        }
        return 0;
    }
    return -1;
}

static int __phxfs_open(const char *dev_path, phxfs_mmap_buffer_t *mbuffer) {
    mbuffer->bdev_fd = open(dev_path, O_RDWR);
    
    if (mbuffer->bdev_fd == -1) {
        printf("failed to open file %s\n", dev_path);
        return -1;
    }
    mbuffer->head = NULL;
    pthread_mutex_init(&mbuffer->lock, NULL);
    mbuffer->init_stat = true;
    return 0;
}



bool is_phxfs_initialized() {
    bool initialized = false;
    for (int i = 0; i < g_device_count; i++) {
        initialized = initialized | phxfs_initialized[i];
    }
    return initialized;
}

/*
 * Open a phxfs device. Each device is initialised independently and only
 * once; a global lock makes concurrent phxfs_open() from multiple threads
 * (e.g. one worker per GPU) safe. deviceID == -1 opens all devices.
 */
static pthread_mutex_t g_open_lock = PTHREAD_MUTEX_INITIALIZER;

int phxfs_open(int deviceID) {
    int ret = 0;

    pthread_mutex_lock(&g_open_lock);

    if (deviceID == -1) {
        for (int id = 0; id < g_device_count; ++id) {
            if (!phxfs_initialized[id]) {
                ret = __phxfs_open(phxfs_dev_path[id].c_str(), &mbuffer[id]);
                if (ret < 0) {
                    phxfs_close_all();
                    pthread_mutex_unlock(&g_open_lock);
                    return ret;
                }
                phxfs_initialized[id] = true;
            }
        }
    } else if (deviceID >= 0 && deviceID < g_device_count) {
        if (!phxfs_initialized[deviceID]) {
            ret = __phxfs_open(phxfs_dev_path[deviceID].c_str(),
                               &mbuffer[deviceID]);
            if (ret == 0)
                phxfs_initialized[deviceID] = true;
        }
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

phxfs_p2p_map_t *find_phxfs_mmap_node(phxfs_mmap_buffer_t *mbuffer, u64 dev_addr, u64 len) {
    phxfs_p2p_map_t *current = mbuffer->head;
    while (current) {
        if (current->dev_addr <= dev_addr && ((current->dev_addr + current->length) >= (dev_addr + len))) {
            return current;
        }
        current = current->next;
    }
    return NULL; // 未找到节点
}

int delete_phxfs_mmap_node(phxfs_mmap_buffer_t *mbuffer, phxfs_p2p_map_t *p2p_map) {
    phxfs_p2p_map_t *current = mbuffer->head;
    phxfs_p2p_map_t *previous = NULL;
    pthread_mutex_lock(&mbuffer->lock);
    while (current) {
        if (current->dev_addr == p2p_map->dev_addr) {
            if (previous) {
                previous->next = current->next;
            } else {
                mbuffer->head = current->next;
            }
            pthread_mutex_unlock(&mbuffer->lock);
            return 0;
        }
        previous = current;
        current = current->next;
    }
    pthread_mutex_unlock(&mbuffer->lock);
    return -1;
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

int phxfs_regmem(int device_id, const void *addr, size_t len, void **target_addr) {
    int ret = 0;

    phxfs_mmap_buffer_t *pb = &mbuffer[device_id];
    phxfs_p2p_map_t *p2p_map = (phxfs_p2p_map_t *)malloc(sizeof(phxfs_p2p_map_t));
    if (!p2p_map) {
        fprintf(stderr, "%s: new_node is NULL\n", __func__);
        return -1;
    }

    if (len % HUGE_PAGE_SIZE != 0) {
        fprintf(stderr, "%s: len is not 64KiB aligned\n", __func__);
        free(p2p_map);
        return -EFAULT;
    }

    p2p_map->vaddr = NULL;
    p2p_map->length = len;
    p2p_map->dev_addr = (uint64_t)addr;
    p2p_map->has_reg = 0;
    p2p_map->next = NULL;
    pb->device_id = device_id;

    /*
     * Single contiguous mmap of the whole region. The phxfs mmap backend is
     * lazy (no size-proportional allocation), and the ioctl insert path now
     * uses kvmalloc, so there is no longer a 2GiB-per-mmap limit and no need
     * to split into chunks.
     */
    p2p_map->vaddr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, pb->bdev_fd, 0);
    if (p2p_map->vaddr == MAP_FAILED) {
        fprintf(stderr, "%s: mmap fail (%s)\n", __func__, strerror(errno));
        free(p2p_map);
        return -EFAULT;
    }

    ret = __phxfs_regmem(pb, p2p_map->dev_addr, (uint64_t)p2p_map->vaddr, len);
    if (ret) {
        fprintf(stderr, "%s: __phxfs_regmem fail ret=%d (%s)\n", __func__, ret, strerror(errno));
        munmap(p2p_map->vaddr, len);
        free(p2p_map);
        return -EFAULT;
    }

    p2p_map->has_reg = 1;
    insert_phxfs_mmap_node(pb, p2p_map);
    *target_addr = p2p_map->vaddr;

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

int phxfs_deregmem(int device_id, const void *addr, size_t len) {
    int ret = 0;
    phxfs_p2p_map_t *p2p_map;
    phxfs_mmap_buffer_t *pb = &mbuffer[device_id];

    p2p_map = find_phxfs_mmap_node(pb, (u64)addr, len);
    if (p2p_map == NULL) {
        fprintf(stderr, "%s: p2p_map is not found fail\n", __func__);
        return -1;
    }

    if (p2p_map->length != len || !p2p_map->has_reg) {
        fprintf(stderr, "%s: p2p_map is not match or has not reg\n", __func__);
        return -1;
    }

    ret = delete_phxfs_mmap_node(pb, p2p_map);
    if (ret) {
        fprintf(stderr, "%s: delete_phxfs_mmap_node fail! not found the map\n", __func__);
        return -1;
    }

    ret = __phxfs_deregmem(pb, p2p_map->dev_addr, (uint64_t)p2p_map->vaddr, p2p_map->length);
    if (ret) {
        fprintf(stderr, "%s: __phxfs_deregmem fail\n", __func__);
        free(p2p_map);
        return -1;
    }

    ret = munmap(p2p_map->vaddr, p2p_map->length);
    if (ret) {
        fprintf(stderr, "%s: munmap fail (%s)\n", __func__, strerror(errno));
        free(p2p_map);
        return -1;
    }

    free(p2p_map);
    return ret;
}

int phxfs_close(phxfs_fileid_t fid) {
    return close(fid.fd);
}

// Resolve a registered GPU buffer to its contiguous host-mapped P2P address.
// Returns vaddr + buf_offset, or NULL if not registered / out of range.
static void *phxfs_resolve_target(int device_id, const void *buf, off_t buf_offset, size_t nbyte) {
    phxfs_mmap_buffer_t *pb = &mbuffer[device_id];
    phxfs_p2p_map_t *p2p_map = find_phxfs_mmap_node(pb, (u64)buf, nbyte);

    if (!p2p_map) {
        fprintf(stderr, "%s: p2p_map not found\n", __func__);
        return NULL;
    }
    if (!p2p_map->has_reg) {
        fprintf(stderr, "%s: p2p_map not registered\n", __func__);
        return NULL;
    }
    if ((size_t)(nbyte + buf_offset) > p2p_map->length) {
        fprintf(stderr, "%s: out of range: nbyte=%zu buf_offset=%ld length=%zu\n",
                __func__, nbyte, (long)buf_offset, p2p_map->length);
        return NULL;
    }
    return (void *)((char *)p2p_map->vaddr + buf_offset);
}

ssize_t phxfs_read(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset){
    char *base = (char *)phxfs_resolve_target(fid.deviceID, buf, buf_offset, nbyte);
    ssize_t nbyte_total = 0;

    if (base == NULL)
        return -1;

    // Single contiguous mapping; chunk only to stay under MAX_RW_COUNT (~2GiB).
    while (nbyte_total < nbyte) {
        size_t this_nbyte = (size_t)(nbyte - nbyte_total);
        if (this_nbyte > PHXFS_IO_CHUNK)
            this_nbyte = PHXFS_IO_CHUNK;
        ssize_t ret = pread(fid.fd, base + nbyte_total, this_nbyte, f_offset + nbyte_total);
        if (ret < 0) {
            fprintf(stderr, "%s: pread error: %s\n", __func__, strerror(errno));
            return -1;
        }
        if (ret == 0)
            break; // EOF
        nbyte_total += ret;
    }
    return nbyte_total;
}

ssize_t phxfs_write(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset){
    char *base = (char *)phxfs_resolve_target(fid.deviceID, buf, buf_offset, nbyte);
    ssize_t nbyte_total = 0;

    if (base == NULL)
        return -1;

    while (nbyte_total < nbyte) {
        size_t this_nbyte = (size_t)(nbyte - nbyte_total);
        if (this_nbyte > PHXFS_IO_CHUNK)
            this_nbyte = PHXFS_IO_CHUNK;
        ssize_t ret = pwrite(fid.fd, base + nbyte_total, this_nbyte, f_offset + nbyte_total);
        if (ret < 0) {
            fprintf(stderr, "%s: pwrite error: %s\n", __func__, strerror(errno));
            return -1;
        }
        if (ret == 0)
            break;
        nbyte_total += ret;
    }
    return nbyte_total;
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
 * Resolve a request's target buffer to a DMA-able host address.
 *   - If (device_id, buf) matches a phxfs_regmem registration, use the
 *     host-mapped P2P VMA (GPU path).
 *   - Otherwise treat buf as an ordinary CPU address (host buffer path).
 * Returns NULL only on an out-of-range registered buffer.
 */
static void *resolve_req_target(const phxfs_io_req_t *r) {
    if (r->device_id >= 0 && r->device_id < g_device_count &&
        phxfs_initialized[r->device_id]) {
        phxfs_mmap_buffer_t *pb = &mbuffer[r->device_id];
        phxfs_p2p_map_t *m = find_phxfs_mmap_node(pb, (u64)r->buf, r->nbytes);
        if (m && m->has_reg) {
            if ((size_t)(r->nbytes + r->buf_offset) > m->length)
                return NULL;  /* out of range */
            return (void *)((char *)m->vaddr + r->buf_offset);
        }
    }
    /* Not a registered GPU buffer: plain CPU address. */
    return (void *)((char *)r->buf + r->buf_offset);
}

/*
 * NUMA node of a phxfs device, cached. Derived from the device's PCI BDF
 * (exported by the kernel module at /sys/class/phxfs-generic/phxfs_devN/
 * pci_bdf) -> /sys/bus/pci/devices/<bdf>/numa_node. Routing a batch to the
 * pool on the target GPU's node avoids cross-NUMA DMA.
 */
static int g_dev_numa[PHXFS_MAX_DEVICES];
static bool g_dev_numa_done[PHXFS_MAX_DEVICES] = {false};

static int phxfs_dev_numa(int device_id) {
    if (device_id < 0 || device_id >= PHXFS_MAX_DEVICES)
        return 0;
    if (g_dev_numa_done[device_id])
        return g_dev_numa[device_id];

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
    g_dev_numa[device_id] = node;
    g_dev_numa_done[device_id] = true;
    return node;
}

static int phxfs_batch(phxfs_io_req_t *reqs, int n, enum phxfs_io_op op) {
    if (n <= 0)
        return 0;

    struct phxfs_io_op_req *ops =
        (struct phxfs_io_op_req *)malloc((size_t)n * sizeof(*ops));
    if (!ops)
        return -ENOMEM;

    /* Route to the NUMA node of the batch's target GPU (first request). */
    int numa = phxfs_dev_numa(reqs[0].device_id);

    int resolve_fail = 0;
    for (int i = 0; i < n; i++) {
        void *host = resolve_req_target(&reqs[i]);
        if (!host) {
            reqs[i].result = -EFAULT;
            resolve_fail++;
        }
        ops[i].fd = reqs[i].fd;
        ops[i].host_addr = host;
        /* Unresolved -> nbytes 0 so the engine never dereferences NULL. */
        ops[i].nbytes = host ? reqs[i].nbytes : 0;
        ops[i].f_offset = reqs[i].f_offset;
        ops[i].result = -EFAULT;
    }

    if (resolve_fail == n) {  /* nothing resolvable */
        free(ops);
        return resolve_fail;
    }

    int ret = phxfs_pool_run(ops, n, op, numa);

    /* Copy engine results back (skip the unresolved ones). */
    for (int i = 0; i < n; i++) {
        if (ops[i].host_addr)
            reqs[i].result = ops[i].result;
    }
    free(ops);

    if (ret < 0)
        return ret;
    return ret + resolve_fail;
}

int phxfs_read_batch(phxfs_io_req_t *reqs, int n) {
    return phxfs_batch(reqs, n, PHXFS_IO_READ);
}

int phxfs_write_batch(phxfs_io_req_t *reqs, int n) {
    return phxfs_batch(reqs, n, PHXFS_IO_WRITE);
}

