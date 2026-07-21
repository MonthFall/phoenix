#ifndef __PHOENIX_H__
#define __PHOENIX_H__
#include <cstddef>
#include <cstdint>
#include <stdint.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct phxfs_fileid {
    int fd;
    int deviceID;
} phxfs_fileid_t;

int phxfs_open(int device_id);
int phxfs_close(int device_id);

/* Returns the device page size in bytes (NVIDIA=64KB, etc.). */
uint64_t phxfs_get_page_size(void);

/*
 * Map a vendor-specific device ID to a phxfs device index.
 * The vendor is selected at build time (PHXFS_VENDOR).
 *   NVIDIA: device_id is a CUDA device ID
 *   AMD:    device_id is a HIP device ID
 *   Huawei: device_id is an NPU ID
 */
int phxfs_find_dev(int device_id);

/* Backward-compatible wrapper for existing NVIDIA callers */
static inline int phxfs_find_dev_for_cuda_gpu(int cuda_gpu_id) {
    return phxfs_find_dev(cuda_gpu_id);
}

ssize_t phxfs_read(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);
ssize_t phxfs_write(phxfs_fileid_t fid, void *buf, off_t buf_offset, ssize_t nbyte, off_t f_offset);

int phxfs_regmem(int device_id, const void *addr, size_t len, void **target_addr);
int phxfs_deregmem(int device, const void *addr, size_t len);

/*
 * Async I/O — vendor-agnostic.
 *   stream: vendor stream handle (CUstream / hipStream_t / ...)
 *   Returns 0 on success.
 */
int phxfs_read_async(phxfs_fileid_t fid,
                     void *buf,
                     size_t nbytes, off_t offset,
                     ssize_t *bytes_done,
                     void *stream);

int phxfs_write_async(phxfs_fileid_t fid,
                      void *buf,
                      size_t nbytes, off_t offset,
                      ssize_t *bytes_done,
                      void *stream);

#ifdef __cplusplus
}
#endif

#endif
