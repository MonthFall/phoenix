#include <cstddef>
#include <cstdint>
#ifdef PHXFS_HAVE_LIBURING
#include <liburing.h>
#endif
#include <sys/types.h>
#include <unistd.h>

#include "phoenix.h"
#include "connectors/devconnector.h"


enum phxfs_op {
    PHXFS_OP_READ = 0,
    PHXFS_OP_WRITE = 1,
};

struct phxfs_data {
    phxfs_fileid_t fid;
    int op;
    void *buf;
    size_t nbytes;
    off_t file_offset;
    ssize_t *bytes_done;
};

/* Callback executed on the stream's host-callback thread. */
static void phxfs_callback(void *user_data) {
    auto* data = static_cast<phxfs_data*>(user_data);
    if (data->op == PHXFS_OP_READ)
        *data->bytes_done = phxfs_read(data->fid, data->buf, 0, (ssize_t)data->nbytes, data->file_offset);
    else
        *data->bytes_done = phxfs_write(data->fid, data->buf, 0, (ssize_t)data->nbytes, data->file_offset);
    delete data;
}

int phxfs_async(phxfs_fileid_t fid, enum phxfs_op op,
                void *buf,
                size_t nbytes, off_t offset,
                ssize_t *bytes_done,
                void *stream) {
    auto* data = new phxfs_data{
        .fid = fid, .op = op, .buf = buf,
        .nbytes = nbytes, .file_offset = offset,
        .bytes_done = bytes_done
    };
    return devconn->launch_async(stream, phxfs_callback, data);
}

int phxfs_read_async(phxfs_fileid_t fid,
                     void *buf,
                     size_t nbytes, off_t offset,
                     ssize_t *bytes_done,
                     void *stream) {
    return phxfs_async(fid, PHXFS_OP_READ, buf, nbytes, offset, bytes_done, stream);
}

int phxfs_write_async(phxfs_fileid_t fid,
                      void *buf,
                      size_t nbytes, off_t offset,
                      ssize_t *bytes_done,
                      void *stream) {
    return phxfs_async(fid, PHXFS_OP_WRITE, buf, nbytes, offset, bytes_done, stream);
}
