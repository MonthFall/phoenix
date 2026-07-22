#ifndef __DEVCONNECTOR_H__
#define __DEVCONNECTOR_H__

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * DevConnector — vendor-specific device operations.
 *
 * Each vendor provides one implementation (e.g. nvidia_connector.cc),
 * selected at build time by PHXFS_VENDOR.  Core libphoenix code calls
 * through the global *devconn pointer and never touches vendor APIs
 * (CUDA, HIP, CANN, ...) directly.
 *
 * Adding a new vendor:
 *   1. Create <vendor>_connector.cc implementing devconn_ops
 *   2. Add the source file to CMakeLists.txt for that vendor
 *   3. No changes to phoenix.h / phoenix.cpp
 */
struct devconn_ops {
    const char  *name;                  /* "nvidia", "amd", "huawei", ... */
    uint64_t     page_size;             /* device page size in bytes */

    /*
     * Initialize vendor runtime (optional, may be NULL).
     * Called once at library load.
     * Returns 0 on success.
     */
    int   (*init)(void);

    /*
     * Map a vendor-specific device ID to a phxfs device index.
     *   NVIDIA: device_id is a CUDA device ID
     *   AMD:    device_id is a HIP device ID
     *   Huawei: device_id is an NPU ID
     * Returns phxfs device index (>=0) or -1 on failure.
     */
    int   (*find_device)(int device_id);
};

/*
 * Global active connector — initialized at link time by the
 * compiled-in connector file.  Never NULL in a valid build.
 */
extern struct devconn_ops *devconn;

/*
 * Initialize the compiled-in connector.
 * Returns 0 on success.  Safe to call multiple times.
 */
int devconn_init(void);

#ifdef __cplusplus
}
#endif

#endif /* __DEVCONNECTOR_H__ */
