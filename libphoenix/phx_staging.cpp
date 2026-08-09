/*
 * phx_staging.cpp — staging-mode I/O path (PHX_MAP_MODE_STAGING).
 *
 * In staging mode the kernel does NOT remap the whole GPU BAR, so user GPU
 * buffers carry no struct pages and stay registerable by RDMA/peermem. The
 * only remapped region is a small Phoenix-owned staging pool. Data therefore
 * flows in two hops that never touch host RAM:
 *
 *     SSD --(P2P DMA, io_uring/pool)--> staging pool --(D2D copy)--> user GPU
 *
 * The read path double-buffers the staging pool (two slots) so the next
 * chunk's SSD DMA overlaps the current chunk's device-to-device copy. All
 * vendor (CUDA/HIP/...) calls go through the devconn connector, so this file
 * stays vendor-agnostic.
 */

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

#include "phoenix.h"
#include "phx_internal.h"
#include "connectors/devconnector.h"
#include "io_engine/io_engine.h"

/* Two slots => double buffering (overlap SSD DMA with D2D copy). */
#define PHX_STAGING_SLOTS       2
#define PHX_STAGING_DEFAULT_MB  256

/* Staging pool size: env override PHX_STAGING_SIZE_MB, else the default,
 * rounded down to a multiple of (SLOTS * 64KiB) and never below one unit. */
static size_t staging_size_bytes(void) {
    size_t mb = PHX_STAGING_DEFAULT_MB;
    const char *env = getenv("PHX_STAGING_SIZE_MB");
    if (env) {
        long v = atol(env);
        if (v > 0)
            mb = (size_t)v;
    }
    size_t unit = (size_t)PHX_STAGING_SLOTS * HUGE_PAGE_SIZE;
    size_t sz = (mb * 1024 * 1024 / unit) * unit;
    if (sz < unit)
        sz = unit;
    return sz;
}

/* ------------------------------------------------------------------ */
/* Setup / teardown                                                   */
/* ------------------------------------------------------------------ */

int phx_staging_setup(int device_id) {
    PHX_RANGE("phx.staging.setup");
    if (!devconn || !devconn->mem_alloc || !devconn->mem_free ||
        !devconn->memcpy_dtod) {
        fprintf(stderr, "phx_staging: connector '%s' lacks device-memory ops; "
                        "staging mode unsupported\n",
                (devconn && devconn->name) ? devconn->name : "?");
        return -ENOTSUP;
    }

    phxfs_mmap_buffer_t *pb = dev_get(device_id);
    if (!pb)
        return -EINVAL;

    size_t sz = staging_size_bytes();
    void *dptr = NULL;
    int rc = devconn->mem_alloc(device_id, sz, &dptr);
    if (rc != 0) {
        dev_put(pb);
        return rc;
    }
    /* The kernel pins/inserts staging pages by their 64KiB device page, so the
     * pool base must be 64KiB-aligned (device allocations of this size are in
     * practice, but check rather than mis-pin silently). */
    if ((uintptr_t)dptr % HUGE_PAGE_SIZE != 0) {
        fprintf(stderr, "phx_staging: staging pool %p not %d-byte aligned\n",
                dptr, HUGE_PAGE_SIZE);
        devconn->mem_free(dptr);
        dev_put(pb);
        return -EINVAL;
    }

    /* Real (non-no-op) registration of the staging pool; this is the single
     * buffer that triggers the kernel's bounded staging remap. */
    void *host = NULL;
    rc = phx_regmem_internal(pb, dptr, sz, &host);
    if (rc != 0) {
        devconn->mem_free(dptr);
        dev_put(pb);
        return rc;
    }

    pb->staging_dptr = dptr;
    pb->staging_host = host;
    pb->staging_size = sz;
    dev_put(pb);
    return 0;
}

void phx_staging_teardown(phxfs_mmap_buffer_t *pb) {
    /* The staging registration node itself is torn down by
     * free_phxfs_p2p_map() before this runs; here we only release the device
     * allocation. */
    if (pb->staging_dptr && devconn && devconn->mem_free)
        devconn->mem_free(pb->staging_dptr);
    pb->staging_dptr = NULL;
    pb->staging_host = NULL;
    pb->staging_size = 0;
}

/* ------------------------------------------------------------------ */
/* Per-tile I/O helpers                                               */
/* ------------------------------------------------------------------ */

namespace {

/* One unit of work: a <= slot_sz slice of one request. */
struct Tile {
    int    fd;
    off_t  f_offset;
    size_t len;
    void  *udev;      /* device address within the user buffer */
    int    req_idx;   /* owning request (for result aggregation) */
};

/* Reject a request whose offsets/length are out of range before any I/O. */
static bool req_params_valid(const phxfs_io_req_t &r) {
    if (r.f_offset < 0 || r.buf_offset < 0)
        return false;
    if ((uint64_t)r.nbytes > (uint64_t)INT64_MAX - (uint64_t)r.f_offset)
        return false;
    return true;
}

/* Plain chunked pread/pwrite for CPU-buffer (device_id < 0) requests, so a
 * staging-mode batch can still carry ordinary host-memory requests. */
static ssize_t cpu_xfer(const phxfs_io_req_t &r, int is_write) {
    if (!req_params_valid(r))
        return -EINVAL;
    char *base = (char *)r.buf + r.buf_offset;
    ssize_t done = 0;
    ssize_t nbyte = (ssize_t)r.nbytes;
    while (done < nbyte) {
        size_t chunk = (size_t)(nbyte - done);
        if (chunk > PHXFS_IO_CHUNK)
            chunk = PHXFS_IO_CHUNK;
        ssize_t ret = is_write
            ? pwrite(r.fd, base + done, chunk, r.f_offset + done)
            : pread(r.fd, base + done, chunk, r.f_offset + done);
        if (ret < 0) {
            if (errno == EINTR)
                continue;
            return done > 0 ? done : -errno;
        }
        if (ret == 0)
            break;
        done += ret;
    }
    return done;
}

/*
 * Run all device_id==d requests through the staging pool. Sets reqs[i].result
 * for those requests (nbytes on success, negative on failure). The device
 * operation reference is already held by the caller.
 */
static void run_device(int d, phxfs_io_req_t *reqs, int n, int is_write) {
    phxfs_mmap_buffer_t *pb = &mbuffer[d];
    if (!pb->staging_host || pb->staging_size == 0)
        return;   /* results stay at their -EFAULT default */

    const size_t slot_sz = pb->staging_size / PHX_STAGING_SLOTS;
    char *host[PHX_STAGING_SLOTS];
    char *dev[PHX_STAGING_SLOTS];
    for (int s = 0; s < PHX_STAGING_SLOTS; s++) {
        host[s] = (char *)pb->staging_host + (size_t)s * slot_sz;
        dev[s]  = (char *)pb->staging_dptr + (size_t)s * slot_sz;
    }

    /* Build the flat tile list and mark tentative per-request success. */
    std::vector<Tile> tiles;
    for (int i = 0; i < n; i++) {
        if (reqs[i].device_id != d)
            continue;
        if (!req_params_valid(reqs[i]))
            continue;                       /* leave result = -EFAULT */
        reqs[i].result = (ssize_t)reqs[i].nbytes;   /* tentative (0 if empty) */
        char *ubase = (char *)reqs[i].buf + reqs[i].buf_offset;
        size_t off = 0;
        while (off < reqs[i].nbytes) {
            size_t len = reqs[i].nbytes - off;
            if (len > slot_sz)
                len = slot_sz;
            Tile t;
            t.fd       = reqs[i].fd;
            t.f_offset = reqs[i].f_offset + (off_t)off;
            t.len      = len;
            t.udev     = ubase + off;
            t.req_idx  = i;
            tiles.push_back(t);
            off += len;
        }
    }

    const int T = (int)tiles.size();
    if (T == 0)
        return;

    /*
     * Pack the flat tile list into slot-sized groups. One pool_submit fills a
     * whole staging slot (up to slot_sz bytes of requests, fanned out over
     * io_uring), so the SSD side stays batched: only ~total/slot_sz submits
     * happen instead of one per request. slot_off[t] is tile t's byte offset
     * within its slot.
     */
    std::vector<std::vector<int>> groups;
    std::vector<size_t> slot_off(T);
    {
        std::vector<int> cur;
        size_t used = 0;
        for (int t = 0; t < T; t++) {
            if (used + tiles[t].len > slot_sz && !cur.empty()) {
                groups.push_back(cur);
                cur.clear();
                used = 0;
            }
            slot_off[t] = used;
            cur.push_back(t);
            used += tiles[t].len;
        }
        if (!cur.empty())
            groups.push_back(cur);
    }
    const int G = (int)groups.size();

    std::vector<struct phxfs_io_op_req> ops[PHX_STAGING_SLOTS];
    struct phxfs_pool_async *hs[PHX_STAGING_SLOTS] = { NULL, NULL };

    /*
     * Hand one whole group to the I/O pool as a single batched job: the slot's
     * host addresses against the requests' file offsets. `op` picks the
     * direction (SSD->staging for a read, staging->SSD for a write), so only
     * ~total/slot_sz jobs are submitted instead of one per request.
     */
    auto submit_group = [&](int g, int slot, enum phxfs_io_op op) {
        phx_range_push(op == PHXFS_IO_WRITE ? "phx.staging.ssd.submit.write"
                                            : "phx.staging.ssd.submit");
        const std::vector<int> &grp = groups[g];
        ops[slot].resize(grp.size());
        for (size_t i = 0; i < grp.size(); i++) {
            const Tile &t = tiles[grp[i]];
            struct phxfs_io_op_req &o = ops[slot][i];
            o.fd        = t.fd;
            o.host_addr = host[slot] + slot_off[grp[i]];
            o.nbytes    = t.len;
            o.f_offset  = t.f_offset;
            o.result    = -EFAULT;
        }
        hs[slot] = phxfs_pool_submit(ops[slot].data(), (int)ops[slot].size(),
                                     op, /*blocking=*/true);
        phx_range_pop();
    };

    if (!is_write) {
        /*
         * Read pipeline over slot-groups: prefetch group g+1's batched DMA into
         * the free slot, then copy group g out to the user buffer while that
         * DMA runs, so NVMe and the copy engine overlap.
         *
         * Two things keep the copy leg cheap:
         *   - Tiles whose source AND destination are both contiguous are
         *     merged into a single copy. Within a slot the sources always are
         *     (slot_off is cumulative), so a batch reading consecutive file
         *     ranges into one arena collapses a whole group into one copy
         *     instead of one per request. Per-copy launch overhead (~5us)
         *     otherwise dominates: a 128KiB device-to-device copy itself takes
         *     well under 1us.
         *   - The copies are issued asynchronously on a per-slot queue, so this
         *     thread goes straight back to feeding the I/O pool. A slot is only
         *     refilled after its queue has drained (its copies have finished
         *     reading it), and every queue is drained before returning, so the
         *     user buffer is complete when the batch call returns.
         */
        const bool async = devconn->memcpy_dtod_async && devconn->queue_sync;
        /* Requests whose copies are in flight on each slot's queue, so a
         * queue_sync() failure can be attributed to them. */
        std::vector<int> slot_reqs[PHX_STAGING_SLOTS];

        auto sync_slot = [&](int s) {
            if (!async || slot_reqs[s].empty())
                return;
            phx_range_push("phx.staging.d2d.sync");
            int rc = devconn->queue_sync(d, s);
            phx_range_pop();
            if (rc != 0)
                for (size_t k = 0; k < slot_reqs[s].size(); k++)
                    reqs[slot_reqs[s][k]].result = -EIO;
            slot_reqs[s].clear();
        };

        auto copy_group = [&](int g, int s, bool got) {
            phx_range_push("phx.staging.d2d");
            const std::vector<int> &grp = groups[g];
            size_t i = 0;
            while (i < grp.size()) {
                int ti = grp[i];
                const Tile &t = tiles[ti];
                if (!got || ops[s][i].result < 0) {
                    /* io_uring submission or completion failed */
                    reqs[t.req_idx].result = -EIO;
                    i++;
                    continue;
                }
                /* Accept short reads (result < t.len): this happens when
                 * the read reaches EOF on a file whose size is not aligned
                 * to the DMA tile size.  The bytes that were read are valid
                 * and must be copied to the user buffer. */
                size_t bytes = (size_t)ops[s][i].result;
                if (bytes == 0) {
                    reqs[t.req_idx].result = 0;
                    i++;
                    continue;
                }
                /* Set short-read result so the caller knows how many bytes
                 * were actually transferred. */
                if (bytes < t.len)
                    reqs[t.req_idx].result = (ssize_t)bytes;
                /* Extend the run while both sides stay contiguous and every
                 * tile's DMA returned a full read (short read = EOF, stop). */
                size_t j = i + 1;
                while (j < grp.size()) {
                    int tj = grp[j];
                    const Tile &u = tiles[tj];
                    if (!(got && ops[s][j].result == (ssize_t)u.len))
                        break;
                    if ((char *)u.udev != (char *)t.udev + bytes)
                        break;
                    if (slot_off[tj] != slot_off[ti] + bytes)
                        break;
                    bytes += u.len;
                    j++;
                }
                char *src = dev[s] + slot_off[ti];
                int rc = async
                    ? devconn->memcpy_dtod_async(d, s, t.udev, src, bytes)
                    : devconn->memcpy_dtod(t.udev, src, bytes);
                if (rc != 0) {
                    for (size_t k = i; k < j; k++)
                        reqs[tiles[grp[k]].req_idx].result = -EIO;
                } else if (async) {
                    for (size_t k = i; k < j; k++)
                        slot_reqs[s].push_back(tiles[grp[k]].req_idx);
                }
                i = j;
            }
            phx_range_pop();
        };

        submit_group(0, 0, PHXFS_IO_READ);
        for (int g = 0; g < G; g++) {
            int s = g & 1;
            if (g + 1 < G) {
                /* The slot about to be refilled still holds group g-1's data;
                 * its copies must be done reading it first. */
                sync_slot((g + 1) & 1);
                submit_group(g + 1, (g + 1) & 1, PHXFS_IO_READ);
            }
            bool got = false;
            if (hs[s]) {
                phx_range_push("phx.staging.ssd.wait");
                phxfs_pool_wait(hs[s]);
                phx_range_pop();
                hs[s] = NULL;
                got = true;
            }
            copy_group(g, s, got);
        }
        for (int s = 0; s < PHX_STAGING_SLOTS; s++)
            sync_slot(s);
    } else {
        /*
         * Write pipeline, the mirror of the read path: upload group g into slot
         * s with merged asynchronous copies, then hand the whole slot to the
         * I/O pool as one batched job. Both slots can have a job in flight, so
         * one group's staging->SSD DMA overlaps the next group's upload.
         *
         * As on the read side, merging contiguous runs is what keeps the copy
         * leg cheap (one copy per group instead of one per request), and
         * batching the SSD leg is what gives the device a deep queue instead of
         * one 128KiB write at a time.
         *
         * Ordering rules that make the result correct:
         *   - A slot is not refilled until the job reading it has completed,
         *     otherwise the upload would overwrite data still being written out.
         *   - The slot's copies are drained before its job is submitted, so the
         *     bytes are in staging before the SSD reads them.
         *   - Every job is drained before returning, so on return each request
         *     has been issued to the device (same contract as the sync path).
         */
        const bool async = devconn->memcpy_dtod_async && devconn->queue_sync;
        int slot_group[PHX_STAGING_SLOTS];
        for (int s = 0; s < PHX_STAGING_SLOTS; s++)
            slot_group[s] = -1;

        /* Mark every request of group g failed (its data never reached disk). */
        auto fail_group = [&](int g) {
            const std::vector<int> &grp = groups[g];
            for (size_t k = 0; k < grp.size(); k++)
                reqs[tiles[grp[k]].req_idx].result = -EIO;
        };

        /* Wait for slot s's write job, then record its per-request results. */
        auto drain_slot = [&](int s) {
            if (!hs[s])
                return;
            phx_range_push("phx.staging.ssd.write.wait");
            phxfs_pool_wait(hs[s]);
            phx_range_pop();
            hs[s] = NULL;
            const std::vector<int> &grp = groups[slot_group[s]];
            for (size_t i = 0; i < grp.size(); i++)
                if (ops[s][i].result != (ssize_t)tiles[grp[i]].len)
                    reqs[tiles[grp[i]].req_idx].result = -EIO;
            slot_group[s] = -1;
        };

        /*
         * Copy group g's tiles user->staging, merging runs that are contiguous
         * on both sides. Returns false if any copy failed; the group is then
         * not written out at all, since writing a partially populated slot
         * would put stale bytes on disk.
         */
        auto upload_group = [&](int g, int s) {
            phx_range_push("phx.staging.d2d.up");
            const std::vector<int> &grp = groups[g];
            bool ok = true;
            size_t i = 0;
            while (i < grp.size()) {
                int ti = grp[i];
                const Tile &t = tiles[ti];
                size_t j = i + 1;
                size_t bytes = t.len;
                while (j < grp.size()) {
                    int tj = grp[j];
                    const Tile &u = tiles[tj];
                    if ((char *)u.udev != (char *)t.udev + bytes)
                        break;
                    if (slot_off[tj] != slot_off[ti] + bytes)
                        break;
                    bytes += u.len;
                    j++;
                }
                char *dst = dev[s] + slot_off[ti];
                int rc = async
                    ? devconn->memcpy_dtod_async(d, s, dst, t.udev, bytes)
                    : devconn->memcpy_dtod(dst, t.udev, bytes);
                if (rc != 0)
                    ok = false;
                i = j;
            }
            /* The SSD reads this slot next, so the copies must have landed. */
            if (async && devconn->queue_sync(d, s) != 0)
                ok = false;
            phx_range_pop();
            if (!ok)
                fail_group(g);
            return ok;
        };

        for (int g = 0; g < G; g++) {
            int s = g & 1;
            drain_slot(s);              /* slot still busy with group g-2 */
            if (!upload_group(g, s))
                continue;
            slot_group[s] = g;
            submit_group(g, s, PHXFS_IO_WRITE);
            if (!hs[s]) {               /* submit failed: nobody writes it */
                fail_group(g);
                slot_group[s] = -1;
            }
        }
        for (int s = 0; s < PHX_STAGING_SLOTS; s++)
            drain_slot(s);
    }
}

}  // namespace

/* ------------------------------------------------------------------ */
/* Public (internal) entry point                                      */
/* ------------------------------------------------------------------ */

int phx_staging_batch(phxfs_io_req_t *reqs, int n, int is_write) {
    if (n <= 0)
        return 0;

    /* Name carries shape so a timeline row identifies its batch directly. */
    size_t total = 0;
    for (int i = 0; i < n; i++)
        total += reqs[i].nbytes;
    char rname[64];
    snprintf(rname, sizeof(rname), "phx.staging.%s n=%d %zuKiB",
             is_write ? "write" : "read", n, total / 1024);
    PHX_RANGE(rname);

    /* Hold an operation ref on every distinct GPU device referenced, so a
     * concurrent close() waits until this batch completes. */
    bool held[PHXFS_MAX_DEVICES] = { false };
    for (int i = 0; i < n; i++) {
        int d = reqs[i].device_id;
        if (d >= 0 && d < g_device_count && !held[d] && dev_get(d))
            held[d] = true;
    }

    for (int i = 0; i < n; i++)
        reqs[i].result = -EFAULT;   /* default: unresolved/failed */

    /* CPU-buffer requests take the plain host path. */
    for (int i = 0; i < n; i++) {
        if (reqs[i].device_id < 0)
            reqs[i].result = cpu_xfer(reqs[i], is_write);
    }

    /* GPU requests run per device (each device has its own staging pool). */
    for (int d = 0; d < g_device_count; d++) {
        if (held[d])
            run_device(d, reqs, n, is_write);
    }

    for (int d = 0; d < g_device_count; d++) {
        if (held[d])
            dev_put(&mbuffer[d]);
    }

    int fails = 0;
    for (int i = 0; i < n; i++)
        if (reqs[i].result != (ssize_t)reqs[i].nbytes)
            fails++;
    return fails;
}
