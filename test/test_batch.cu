// test_batch.cu
//
// phxfs_read_batch (io_uring) correctness + bandwidth.
//   Part A — correctness on GPU 0 (contiguous + scattered offsets).
//   Part B — concurrent bandwidth: NGPU worker threads, each on its own
//            GPU with its own io_uring ring, submit BATCHES batches of
//            REQS x CHUNK reads. Reports aggregate throughput.
//
// Build: via CMake (make test_batch)
// Run:   ./test_batch [ngpu]        (default: all visible GPUs, capped 8)
//        File: TEST_FILE + ".batch.<gpu>" per GPU (default dir /mnt/nvme0)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include "phoenix.h"

#define KiB (1024ULL)
#define MiB (1024ULL * 1024)
#define GiB (1024ULL * 1024 * 1024)

#ifndef TEST_FILE
#define TEST_FILE "/mnt/nvme0/phxfs_test.bin"
#endif

#define REQS      512          // requests per batch
#define CHUNK     (128 * KiB)  // bytes per request
#define BATCHES   32           // batches per GPU worker

static int tests_run = 0, tests_passed = 0, tests_failed = 0;
#define CHECK(cond, msg, ...) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  [PASS] " msg "\n", ##__VA_ARGS__); } \
    else { tests_failed++; printf("  [FAIL] " msg "\n", ##__VA_ARGS__); } \
} while (0)

static double now_sec() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// byte at file offset o == (o & 0xff)
static int ensure_pattern_file(const char *path, size_t size) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { printf("open(write) %s: %s\n", path, strerror(errno)); return -1; }
    const size_t BUF = 8 * MiB;
    uint8_t *tmp = (uint8_t *)malloc(BUF);
    size_t done = 0;
    while (done < size) {
        size_t w = (size - done < BUF) ? (size - done) : BUF;
        for (size_t i = 0; i < w; i++) tmp[i] = (uint8_t)((done + i) & 0xff);
        if (pwrite(fd, tmp, w, done) != (ssize_t)w) {
            printf("pwrite: %s\n", strerror(errno)); free(tmp); close(fd); return -1;
        }
        done += w;
    }
    free(tmp); fsync(fd);
    posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED);
    close(fd);
    return 0;
}

// ---- correctness on GPU 0 ----
static void correctness(const char *base_file) {
    printf("--- Part A: correctness (GPU 0) ---\n");
    std::string fp = std::string(base_file) + ".batch.0";
    const size_t buf_bytes = (size_t)REQS * CHUNK;
    if (ensure_pattern_file(fp.c_str(), buf_bytes) != 0) return;

    cudaSetDevice(0);
    int dev = phxfs_find_dev(0);
    if (dev < 0 || phxfs_open(dev) != 0) { printf("phxfs open failed\n"); return; }

    void *gbuf = nullptr, *target = nullptr;
    cudaMalloc(&gbuf, buf_bytes); cudaMemset(gbuf, 0, buf_bytes); cudaStreamSynchronize(0);
    if (phxfs_regmem(dev, gbuf, buf_bytes, &target) != 0) { printf("regmem failed\n"); return; }

    int dfd = open(fp.c_str(), O_RDONLY | O_DIRECT);
    std::vector<phxfs_io_req_t> reqs(REQS);
    for (int i = 0; i < REQS; i++) {
        reqs[i] = phxfs_io_req_t{};
        reqs[i].fd = dfd; reqs[i].device_id = dev; reqs[i].buf = gbuf;
        reqs[i].buf_offset = (off_t)i * CHUNK; reqs[i].nbytes = CHUNK;
        reqs[i].f_offset = (off_t)i * CHUNK;
    }
    int ret = phxfs_read_batch(reqs.data(), REQS);
    CHECK(ret == 0, "phxfs_read_batch returned %d (0 = all ok)", ret);

    uint8_t *hbuf = (uint8_t *)malloc(buf_bytes);
    cudaMemcpy(hbuf, gbuf, buf_bytes, cudaMemcpyDeviceToHost); cudaStreamSynchronize(0);
    size_t mism = 0; long long first_bad = -1;
    for (size_t o = 0; o < buf_bytes; o++)
        if (hbuf[o] != (uint8_t)(o & 0xff)) { if (first_bad < 0) first_bad = (long long)o; mism++; }
    CHECK(mism == 0, "per-byte verify: %zu mismatch(es) (first @0x%llx)",
          mism, (unsigned long long)(first_bad < 0 ? 0 : first_bad));

    cudaMemset(gbuf, 0, buf_bytes); cudaStreamSynchronize(0);
    for (int i = 0; i < REQS; i++) {
        reqs[i].buf_offset = (off_t)i * CHUNK;
        reqs[i].f_offset   = (off_t)(REQS - 1 - i) * CHUNK;
    }
    ret = phxfs_read_batch(reqs.data(), REQS);
    cudaMemcpy(hbuf, gbuf, buf_bytes, cudaMemcpyDeviceToHost); cudaStreamSynchronize(0);
    size_t scat_bad = 0;
    for (int i = 0; i < REQS; i++) {
        size_t fbase = (size_t)(REQS - 1 - i) * CHUNK;
        for (size_t j = 0; j < CHUNK; j++)
            if (hbuf[(size_t)i * CHUNK + j] != (uint8_t)((fbase + j) & 0xff)) { scat_bad++; break; }
    }
    CHECK(ret == 0 && scat_bad == 0, "scattered-offset batch correct (%zu bad chunks)", scat_bad);

    // async submit/wait: submit, do (dummy) compute, then wait + verify.
    cudaMemset(gbuf, 0, buf_bytes); cudaStreamSynchronize(0);
    for (int i = 0; i < REQS; i++) {
        reqs[i].buf_offset = (off_t)i * CHUNK;
        reqs[i].f_offset   = (off_t)i * CHUNK;
    }
    phxfs_batch_t *h = phxfs_batch_submit_read(reqs.data(), REQS);
    CHECK(h != nullptr, "phxfs_batch_submit_read returned a handle");
    volatile double spin = 0;              // stand-in for overlapping compute
    for (int k = 0; k < 1000000; k++) spin += k * 0.5;
    int aret = h ? phxfs_batch_wait(h) : -1;
    CHECK(aret == 0, "phxfs_batch_wait returned %d (0 = all ok)", aret);
    cudaMemcpy(hbuf, gbuf, buf_bytes, cudaMemcpyDeviceToHost); cudaStreamSynchronize(0);
    size_t amis = 0;
    for (size_t o = 0; o < buf_bytes; o++)
        if (hbuf[o] != (uint8_t)(o & 0xff)) { amis++; }
    CHECK(amis == 0, "async batch per-byte verify: %zu mismatch(es)", amis);

    close(dfd); phxfs_deregmem(dev, gbuf, buf_bytes); phxfs_close(dev);
    cudaFree(gbuf); free(hbuf);
}

// ---- per-GPU worker: own GPU, own file, own ring (thread_local) ----
struct worker_result { double secs; size_t bytes; bool ok; };

// Prepare each GPU's file up front (NOT timed).
static bool prepare_files(int ngpu, const char *base_file) {
    const size_t file_bytes = (size_t)BATCHES * (size_t)REQS * CHUNK;
    for (int g = 0; g < ngpu; g++) {
        std::string fp = std::string(base_file) + ".batch." + std::to_string(g);
        if (ensure_pattern_file(fp.c_str(), file_bytes) != 0) {
            printf("prepare file for gpu%d failed\n", g);
            return false;
        }
    }
    return true;
}

static void gpu_worker(int gpu, const char *base_file, worker_result *out) {
    out->ok = false; out->bytes = 0; out->secs = 0;

    if (cudaSetDevice(gpu) != cudaSuccess) { printf("gpu%d: cudaSetDevice failed\n", gpu); return; }
    int dev = phxfs_find_dev(gpu);
    if (dev < 0) { printf("gpu%d: phxfs_find_dev=%d\n", gpu, dev); return; }
    if (phxfs_open(dev) != 0) { printf("gpu%d: phxfs_open(%d) failed\n", gpu, dev); return; }

    const size_t buf_bytes = (size_t)REQS * CHUNK;
    const size_t file_bytes = (size_t)BATCHES * buf_bytes;

    std::string fp = std::string(base_file) + ".batch." + std::to_string(gpu);

    void *gbuf = nullptr, *target = nullptr;
    if (cudaMalloc(&gbuf, buf_bytes) != cudaSuccess) { printf("gpu%d: cudaMalloc failed\n", gpu); return; }
    cudaMemset(gbuf, 0, buf_bytes); cudaStreamSynchronize(0);
    if (phxfs_regmem(dev, gbuf, buf_bytes, &target) != 0) {
        printf("gpu%d: phxfs_regmem(dev=%d) failed\n", gpu, dev); cudaFree(gbuf); return;
    }

    int dfd = open(fp.c_str(), O_RDONLY | O_DIRECT);
    if (dfd < 0) { printf("gpu%d: open %s: %s\n", gpu, fp.c_str(), strerror(errno));
                   phxfs_deregmem(dev, gbuf, buf_bytes); cudaFree(gbuf); return; }

    // One large batch of REQS*BATCHES requests reading distinct file
    // offsets into cyclic GPU buffer offsets. This exercises the internal
    // NUMA pool: a single blocking call should saturate the array.
    const int NREQ = REQS * BATCHES;
    std::vector<phxfs_io_req_t> reqs(NREQ);
    for (int i = 0; i < NREQ; i++) {
        reqs[i] = phxfs_io_req_t{};
        reqs[i].fd = dfd; reqs[i].device_id = dev; reqs[i].buf = gbuf;
        reqs[i].nbytes = CHUNK;
        reqs[i].buf_offset = (off_t)(i % REQS) * CHUNK;   // cyclic within buffer
        reqs[i].f_offset   = (off_t)i * CHUNK;            // distinct file region
    }

    posix_fadvise(dfd, 0, file_bytes, POSIX_FADV_DONTNEED);
    double t0 = now_sec();
    int rc = phxfs_read_batch(reqs.data(), NREQ);
    out->secs = now_sec() - t0;
    if (rc != 0) { printf("gpu%d big batch failed rc=%d\n", gpu, rc); }
    out->bytes = (size_t)NREQ * CHUNK;
    out->ok = (rc == 0);

    close(dfd); phxfs_deregmem(dev, gbuf, buf_bytes); phxfs_close(dev); cudaFree(gbuf);
}

// Single-GPU, multi-thread worker: N threads each with its own ring read
// distinct file regions into distinct sub-ranges of one shared buffer.
// Isolates "multiple rings scale bandwidth" from NUMA/GPU-placement.
static void mt_worker(int gpu, void *gbuf, size_t sub_off, const char *fp,
                      int tid, worker_result *out) {
    out->ok = false; out->bytes = 0; out->secs = 0;
    cudaSetDevice(gpu);
    int dev = phxfs_find_dev(gpu);
    const size_t buf_bytes = (size_t)REQS * CHUNK;

    int dfd = open(fp, O_RDONLY | O_DIRECT);
    if (dfd < 0) { printf("t%d: open %s: %s\n", tid, fp, strerror(errno)); return; }

    std::vector<phxfs_io_req_t> reqs(REQS);
    for (int i = 0; i < REQS; i++) {
        reqs[i] = phxfs_io_req_t{};
        reqs[i].fd = dfd; reqs[i].device_id = dev; reqs[i].buf = gbuf;
        reqs[i].nbytes = CHUNK;
    }
    posix_fadvise(dfd, 0, (off_t)BATCHES * buf_bytes, POSIX_FADV_DONTNEED);
    double t0 = now_sec();
    for (int b = 0; b < BATCHES; b++) {
        off_t base = (off_t)b * buf_bytes;
        for (int i = 0; i < REQS; i++) {
            reqs[i].buf_offset = (off_t)sub_off + (off_t)i * CHUNK;
            reqs[i].f_offset   = base + (off_t)i * CHUNK;
        }
        if (phxfs_read_batch(reqs.data(), REQS) != 0) { printf("t%d batch %d failed\n", tid, b); break; }
        out->bytes += buf_bytes;
    }
    out->secs = now_sec() - t0;
    out->ok = true;
    close(dfd);
}

int main(int argc, char **argv) {
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    int ngpu = (argc > 1) ? atoi(argv[1]) : ndev;
    int mt   = (argc > 2) ? atoi(argv[2]) : 0;   // >0: single-GPU, mt threads
    if (ngpu > ndev) ngpu = ndev;
    if (ngpu > 8) ngpu = 8;
    if (ngpu < 1) ngpu = 1;

    printf("=== test_batch ===\n");
    printf("I/O engine: %s | shape: %d batches x %d reqs x %zu KiB\n",
           phxfs_io_engine_name(), BATCHES, REQS, CHUNK / KiB);

    correctness(TEST_FILE);

    double t0, wall;
    size_t total_bytes = 0;
    bool all_ok = true;

    if (mt > 0) {
        // Single GPU (0), mt threads, mt independent rings, mt distinct files.
        if (mt > 16) mt = 16;
        printf("\n--- Part B: single-GPU multi-ring (GPU 0, %d threads/rings) ---\n", mt);
        if (!prepare_files(mt, TEST_FILE)) { printf("file prep failed\n"); return 1; }

        cudaSetDevice(0);
        int dev = phxfs_find_dev(0);
        if (dev < 0 || phxfs_open(dev) != 0) { printf("phxfs open failed\n"); return 1; }
        // One buffer with mt sub-ranges so rings don't overlap.
        const size_t sub = (size_t)REQS * CHUNK;
        void *gbuf = nullptr, *target = nullptr;
        cudaMalloc(&gbuf, sub * mt); cudaMemset(gbuf, 0, sub * mt); cudaStreamSynchronize(0);
        if (phxfs_regmem(dev, gbuf, sub * mt, &target) != 0) { printf("regmem failed\n"); return 1; }

        std::vector<std::thread> ts;
        std::vector<worker_result> res(mt);
        t0 = now_sec();
        for (int t = 0; t < mt; t++) {
            std::string fp = std::string(TEST_FILE) + ".batch." + std::to_string(t);
            ts.emplace_back(mt_worker, 0, gbuf, sub * t, strdup(fp.c_str()), t, &res[t]);
        }
        for (auto &t : ts) t.join();
        wall = now_sec() - t0;

        for (int t = 0; t < mt; t++) {
            if (!res[t].ok) { all_ok = false; printf("  t%d: FAILED\n", t); continue; }
            double gib = (double)res[t].bytes / GiB;
            printf("  t%d: %6.2f GiB in %7.3f ms -> %6.2f GiB/s\n",
                   t, gib, res[t].secs * 1e3, gib / res[t].secs);
            total_bytes += res[t].bytes;
        }
        phxfs_deregmem(dev, gbuf, sub * mt); phxfs_close(dev); cudaFree(gbuf);
    } else {
        printf("\n--- Part B: concurrent bandwidth (%d GPU worker(s), independent rings) ---\n", ngpu);
        if (!prepare_files(ngpu, TEST_FILE)) { printf("file prep failed\n"); return 1; }

        std::vector<std::thread> ts;
        std::vector<worker_result> res(ngpu);
        t0 = now_sec();
        for (int g = 0; g < ngpu; g++)
            ts.emplace_back(gpu_worker, g, TEST_FILE, &res[g]);
        for (auto &t : ts) t.join();
        wall = now_sec() - t0;

        for (int g = 0; g < ngpu; g++) {
            if (!res[g].ok) { all_ok = false; printf("  gpu%d: FAILED\n", g); continue; }
            double gib = (double)res[g].bytes / GiB;
            printf("  gpu%d: %6.2f GiB in %7.3f ms -> %6.2f GiB/s\n",
                   g, gib, res[g].secs * 1e3, gib / res[g].secs);
            total_bytes += res[g].bytes;
        }
    }

    double tgib = (double)total_bytes / GiB;
    printf("  ----\n");
    printf("  aggregate: %.2f GiB in %.3f ms -> %.2f GiB/s (%.2f GB/s)\n",
           tgib, wall * 1e3, tgib / wall, (double)total_bytes / 1e9 / wall);
    CHECK(all_ok, "all workers completed");

    printf("\n=== %d run, %d passed, %d failed ===\n", tests_run, tests_passed, tests_failed);
    return tests_failed ? 1 : 0;
}
