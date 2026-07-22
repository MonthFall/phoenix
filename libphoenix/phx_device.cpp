/*
 * phx_device.cpp — device lifecycle: open / close / operation refcounting.
 */

#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "phoenix.h"
#include "phx_internal.h"
#include "connectors/devconnector.h"

int g_device_count = PHXFS_MAX_DEVICES;
phxfs_mmap_buffer_t mbuffer[PHXFS_MAX_DEVICES];

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
    /* Honour the connector init contract once at load. */
    devconn_init();
}

/*
 * Acquire a device-operation reference. Returns the device buffer if it is
 * OPEN and not closing, else NULL. Holding a ref keeps close() waiting, so
 * an in-flight regmem/dereg/read/write/batch can never race teardown.
 */
phxfs_mmap_buffer_t *dev_get(int device_id) {
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
void dev_put(phxfs_mmap_buffer_t *pb) {
    if (!pb)
        return;
    pthread_mutex_lock(&pb->lock);
    if (--pb->active_ops == 0 && pb->closing)
        pthread_cond_broadcast(&pb->drain_cv);
    pthread_mutex_unlock(&pb->lock);
}

/*
 * Tear down an OPEN device after all in-flight operations have drained.
 * Caller must have set closing=true and waited active_ops==0 under pb->lock.
 * The process-lifetime lock/drain_cv are NOT destroyed here.
 */
static void __phxfs_teardown(phxfs_mmap_buffer_t *pb) {
    free_phxfs_p2p_map(pb);              /* UNMAP + munmap each registration */
    if (pb->bdev_fd >= 0)                /* fd 0 is a valid fd and must be closed too */
        close(pb->bdev_fd);
    pb->bdev_fd  = -1;
    pb->init_stat = false;
    pb->closing   = false;
}

/*
 * Drain in-flight operations on a device whose closing=true and init_stat=false
 * have already been set by the caller, then tear down. Does NOT hold
 * g_open_lock, so a drain on one device does not block open/close on another.
 */
static void __phxfs_close_drain(phxfs_mmap_buffer_t *pb) {
    pthread_mutex_lock(&pb->lock);
    while (pb->active_ops > 0)
        pthread_cond_wait(&pb->drain_cv, &pb->lock);
    pthread_mutex_unlock(&pb->lock);
    __phxfs_teardown(pb);
}

int phxfs_close(int device_id) {
    if (device_id < 0 || device_id >= g_device_count)
        return -1;
    phxfs_mmap_buffer_t *pb = &mbuffer[device_id];

    /*
     * g_open_lock protects only the open_count check/decrement and the
     * init_stat/closing flip — NOT the drain. This way a long drain on
     * device A (waiting for in-flight I/O) does not block open/close on
     * device B.
     *
     * Race safety: closing=true is set under pb->lock BEFORE g_open_lock is
     * released, so a concurrent phxfs_open() for this same device that
     * acquires g_open_lock after us sees init_stat==false && closing==true
     * and returns -EBUSY. There is no window where open sees closing==false
     * after close has committed to teardown.
     */
    pthread_mutex_lock(&g_open_lock);
    if (!pb->init_stat) {
        pthread_mutex_unlock(&g_open_lock);
        return -1;
    }
    /* Only the last client's close actually tears the device down. */
    if (pb->open_count > 1) {
        pb->open_count--;
        pthread_mutex_unlock(&g_open_lock);
        return 0;
    }
    pb->open_count = 0;
    phxfs_initialized[device_id] = false;
    /* Set closing under pb->lock before flipping init_stat, so open()'s
     * EBUSY check is watertight (no window where init_stat==false but
     * closing==false). Lock order g_open_lock -> pb->lock is safe: dev ops
     * never take g_open_lock. */
    pthread_mutex_lock(&pb->lock);
    pb->closing = true;
    pb->init_stat = false;
    pthread_mutex_unlock(&pb->lock);
    pthread_mutex_unlock(&g_open_lock);

    __phxfs_close_drain(pb);  /* drain (wait active_ops==0) + teardown, outside g_open_lock */
    return 0;
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
    mbuffer->closing = false;
    mbuffer->active_ops = 0;
    mbuffer->init_stat = true;
    return 0;
}

/*
 * Open a phxfs device, or add a client reference if another caller already
 * has it open. g_open_lock makes concurrent phxfs_open() from multiple
 * threads safe.
 */
int phxfs_open(int deviceID) {
    if (deviceID < 0 || deviceID >= g_device_count)
        return -1;
    phxfs_mmap_buffer_t *pb = &mbuffer[deviceID];

    pthread_mutex_lock(&g_open_lock);
    int ret = 0;
    if (pb->init_stat) {
        pb->open_count++;              /* already open: just add a client ref */
    } else if (pb->closing) {
        /* A previous close() flipped init_stat=false and is still draining
         * in-flight ops outside g_open_lock. Refuse rather than race the
         * teardown; the caller can retry. */
        ret = -EBUSY;
    } else {
        ret = __phxfs_open(phxfs_dev_path[deviceID].c_str(), pb, deviceID);
        if (ret == 0) {
            pb->open_count = 1;
            phxfs_initialized[deviceID] = true;
        }
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
