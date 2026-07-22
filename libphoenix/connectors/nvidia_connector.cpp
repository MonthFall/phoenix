/*
 * NVIDIA DevConnector
 *
 * All NVIDIA/CUDA specific code in the user library lives here.
 * Core libphoenix (phoenix.cpp) never includes CUDA headers or calls
 * CUDA APIs directly.
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <cctype>
#include <cstdio>
#include <fstream>
#include <string>
#include <pthread.h>

#include "devconnector.h"

#define NV_MAX_GPUS 64        /* CUDA-GPU cache size */
/* sysfs scan bound: must match the core's PHXFS_MAX_DEVICES (phoenix.cpp), as
 * find_device returns a phxfs index the core can actually open (P2-3). */
#define PHXFS_DEV_SCAN_MAX 8

/* ------------------------------------------------------------------ */
/* Device discovery: CUDA device ID → phxfs index via PCI BDF match  */
/* ------------------------------------------------------------------ */

/* Cache CUDA-GPU -> phxfs-index so repeated find() calls don't re-query CUDA
 * and re-scan sysfs every time. -2 = unresolved; only successful (>=0) results
 * are cached, so a device that appears later can still be found (P2-3). */
static int             g_dev_cache[NV_MAX_GPUS];
static bool            g_dev_cache_init = false;
static pthread_mutex_t g_dev_cache_lock = PTHREAD_MUTEX_INITIALIZER;

/* Case-insensitive compare of a C string and std::string (BDFs differ only in
 * hex case between CUDA's "0000:65:00.0" and sysfs). */
static bool bdf_equal(const char *a, const std::string &b)
{
    size_t i = 0;
    for (; a[i] && i < b.size(); i++)
        if (tolower((unsigned char)a[i]) != tolower((unsigned char)b[i]))
            return false;
    return a[i] == '\0' && i == b.size();
}

static int nvidia_find_device_uncached(int cuda_gpu_id)
{
    /* Full PCI bus id "domain:bus:device.function" straight from CUDA — no
     * assumption that the function is 0, and no fragile separator parsing. */
    char cuda_bdf[32];
    if (cudaDeviceGetPCIBusId(cuda_bdf, sizeof(cuda_bdf), cuda_gpu_id) != cudaSuccess) {
        fprintf(stderr, "nvidia_find_device: cudaDeviceGetPCIBusId failed for GPU %d\n",
                cuda_gpu_id);
        return -1;
    }

    /* Match against sysfs pci_bdf entries. A missing phxfs_devN does not stop
     * the scan (indices need not be contiguous). */
    for (int i = 0; i < PHXFS_DEV_SCAN_MAX; i++) {
        std::string sysfs_path = "/sys/class/phxfs-generic/phxfs_dev"
                                 + std::to_string(i) + "/pci_bdf";
        std::ifstream ifs(sysfs_path);
        if (!ifs.is_open())
            continue;
        std::string sysfs_bdf;
        if (!std::getline(ifs, sysfs_bdf))
            continue;
        while (!sysfs_bdf.empty() &&
               (sysfs_bdf.back() == '\n' || sysfs_bdf.back() == '\r' ||
                sysfs_bdf.back() == ' '  || sysfs_bdf.back() == '\t'))
            sysfs_bdf.pop_back();
        if (bdf_equal(cuda_bdf, sysfs_bdf))
            return i;
    }
    return -1;
}

static int nvidia_find_device(int cuda_gpu_id)
{
    if (cuda_gpu_id < 0 || cuda_gpu_id >= NV_MAX_GPUS)
        return nvidia_find_device_uncached(cuda_gpu_id);

    pthread_mutex_lock(&g_dev_cache_lock);
    if (!g_dev_cache_init) {
        for (int i = 0; i < NV_MAX_GPUS; i++)
            g_dev_cache[i] = -2;   /* unresolved */
        g_dev_cache_init = true;
    }
    int cached = g_dev_cache[cuda_gpu_id];
    pthread_mutex_unlock(&g_dev_cache_lock);
    if (cached >= 0)
        return cached;             /* only successful lookups are cached */

    int idx = nvidia_find_device_uncached(cuda_gpu_id);
    if (idx >= 0) {
        /* Cache success only. A negative result is deliberately NOT cached, so
         * a device that appears after the first query (module inserted / sysfs
         * node created later) is found on a later call without restarting the
         * process (P2-3). */
        pthread_mutex_lock(&g_dev_cache_lock);
        g_dev_cache[cuda_gpu_id] = idx;
        pthread_mutex_unlock(&g_dev_cache_lock);
    }
    return idx;
}

/* ------------------------------------------------------------------ */
/* Connector registration                                             */
/* ------------------------------------------------------------------ */

static int nvidia_init(void)
{
    /* CUDA runtime auto-initializes on first API call — nothing to do */
    return 0;
}

static struct devconn_ops nvidia_devconn = {
    .name         = "nvidia",
    .page_size    = 64 * 1024,
    .init         = nvidia_init,
    .find_device  = nvidia_find_device,
};

/* The global connector — referenced by core code via extern */
struct devconn_ops *devconn = &nvidia_devconn;

int devconn_init(void)
{
    if (devconn && devconn->init)
        return devconn->init();
    return 0;
}
