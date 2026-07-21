/*
 * NVIDIA DevConnector
 *
 * All NVIDIA/CUDA specific code in the user library lives here.
 * Core libphoenix (phoenix.cc, integration.cc) never includes
 * CUDA headers or calls CUDA APIs directly.
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "devconnector.h"

/* ------------------------------------------------------------------ */
/* Device discovery: CUDA device ID → phxfs index via PCI BDF match  */
/* ------------------------------------------------------------------ */

static int nvidia_find_device(int cuda_gpu_id)
{
    int domain, bus, device;
    cudaError_t err;

    err = cudaDeviceGetAttribute(&domain, cudaDevAttrPciDomainId, cuda_gpu_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "nvidia_find_device: PciDomainId failed for GPU %d\n",
                cuda_gpu_id);
        return -1;
    }
    err = cudaDeviceGetAttribute(&bus, cudaDevAttrPciBusId, cuda_gpu_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "nvidia_find_device: PciBusId failed for GPU %d\n",
                cuda_gpu_id);
        return -1;
    }
    err = cudaDeviceGetAttribute(&device, cudaDevAttrPciDeviceId, cuda_gpu_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "nvidia_find_device: PciDeviceId failed for GPU %d\n",
                cuda_gpu_id);
        return -1;
    }
    /* GPUs are typically on PCI function 0 */
    int function = 0;

    /* Match against sysfs pci_bdf entries */
    for (int i = 0; i < 64; i++) {
        std::string sysfs_path = "/sys/class/phxfs-generic/phxfs_dev"
                                 + std::to_string(i) + "/pci_bdf";
        std::ifstream ifs(sysfs_path);
        if (!ifs.is_open())
            break;  /* no more phxfs devices */

        unsigned int sdomain, sbus, sdevice, sfunction;
        char sep1, sep2, sep3;
        ifs >> std::hex >> sdomain >> sep1 >> sbus >> sep2 >> sdevice >> sep3 >> sfunction;

        if ((int)sdomain == domain &&
            (int)sbus == bus &&
            (int)sdevice == device &&
            (int)sfunction == function) {
            return i;
        }
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* Async I/O: launch host callback on CUDA stream                     */
/* ------------------------------------------------------------------ */

static int nvidia_launch_async(void *stream, void (*callback)(void *), void *data)
{
    return (int)cudaLaunchHostFunc((cudaStream_t)stream,
                                   reinterpret_cast<cudaHostFn_t>(callback),
                                   data);
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
    .launch_async = nvidia_launch_async,
};

/* The global connector — referenced by core code via extern */
struct devconn_ops *devconn = &nvidia_devconn;

int devconn_init(void)
{
    if (devconn && devconn->init)
        return devconn->init();
    return 0;
}
