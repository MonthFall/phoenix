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

static int tests_run = 0, tests_passed = 0, tests_failed = 0, tests_skipped = 0;
#define CHECK(cond, msg, ...) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  [PASS] " msg "\n", ##__VA_ARGS__); } \
    else { tests_failed++; printf("  [FAIL] " msg "\n", ##__VA_ARGS__); } \
} while (0)
/* Environment-not-available (no GPU / module / O_DIRECT): counted, not a pass
 * or fail — distinct from a genuine setup error, which uses CHECK(false). */
#define SKIP(msg, ...) do { \
    tests_skipped++; \
    printf("  [SKIP] " msg "\n", ##__VA_ARGS__); \
} while (0)

static double now_sec() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Unique per-chunk pattern: every 8-byte word at file offset `o` holds a
// value that encodes BOTH the chunk index and the word's position:
//     word(o) = (chunk_index << 40) | (o & 0xffffffffff)
// where chunk_index = o / CHUNK. This makes swapped, duplicated, missing, or
// mis-routed chunks detectable (unlike a uniform byte pattern that repeats
// every 256 bytes).
static inline uint64_t pattern_word(uint64_t off) {
    uint64_t chunk = off / CHUNK;
    return (chunk << 40) | (off & 0xffffffffffULL);
}

static int ensure_pattern_file(const char *path, size_t size) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { printf("open(write) %s: %s\n", path, strerror(errno)); return -1; }
    const size_t BUF = 8 * MiB;
    uint8_t *tmp = (uint8_t *)malloc(BUF);
    size_t done = 0;
    while (done < size) {
        size_t w = (size - done < BUF) ? (size - done) : BUF;
        uint64_t *wp = (uint64_t *)tmp;
        for (size_t j = 0; j < w / 8; j++) wp[j] = pattern_word(done + j * 8);
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

// Verify GPU buffer region [buf_base, +len) equals the file pattern starting
// at file offset f_base. Returns mismatch count.
static size_t verify_region(const uint8_t *hbuf, size_t buf_base, size_t len,
                            uint64_t f_base) {
    const uint64_t *wp = (const uint64_t *)(hbuf + buf_base);
    size_t bad = 0;
    for (size_t j = 0; j < len / 8; j++)
        if (wp[j] != pattern_word(f_base + j * 8)) bad++;
    return bad;
}

// ---- CPU-buffer correctness: runs WITHOUT the kernel module or a GPU ----
// Exercises the batch/async engine + pool + lifecycle on plain host memory
// (device_id < 0), covering cases the GPU path can't easily reach in CI.
static void cpu_correctness(const char *base_file) {
    printf("--- Part 0: CPU-buffer correctness (no GPU/module) ---\n");
    const int    N = 64;
    const size_t total = (size_t)N * CHUNK;
    std::string fp = std::string(base_file) + ".cpu";
    if (ensure_pattern_file(fp.c_str(), total) != 0) return;

    uint8_t *buf = nullptr;
    if (posix_memalign((void **)&buf, 4096, total) != 0) { printf("memalign fail\n"); return; }
    memset(buf, 0, total);
    int fd = open(fp.c_str(), O_RDONLY);
    if (fd < 0) { printf("open fail\n"); free(buf); return; }

    auto mk = [&](std::vector<phxfs_io_req_t> &r, int n) {
        r.assign(n, phxfs_io_req_t{});
        for (int i = 0; i < n; i++) {
            r[i].fd = fd; r[i].device_id = -1; r[i].buf = buf;
            r[i].buf_offset = (off_t)i * CHUNK; r[i].nbytes = CHUNK;
            r[i].f_offset = (off_t)i * CHUNK;
        }
    };

    // read batch + verify
    { std::vector<phxfs_io_req_t> r; mk(r, N);
      int rc = phxfs_read_batch(r.data(), N);
      CHECK(rc == 0 && verify_region(buf, 0, total, 0) == 0, "CPU read batch + verify (rc=%d)", rc); }

    // async small batches n=1..3 + cross-thread wait
    { bool ok = true;
      for (int n = 1; n <= 3 && ok; n++) {
          memset(buf, 0, total);
          std::vector<phxfs_io_req_t> r; mk(r, n);
          phxfs_batch_t *h = phxfs_batch_submit_read(r.data(), n);
          if (!h) { ok = false; break; }
          // wait on a different thread than submit
          int wret = 0; std::thread t([&]{ wret = phxfs_batch_wait(h); }); t.join();
          if (wret != 0 || verify_region(buf, 0, (size_t)n * CHUNK, 0) != 0) ok = false;
      }
      CHECK(ok, "CPU async small n=1..3 + cross-thread wait"); }

    // pipeline (P2-2): several concurrent same-node submits all succeed (bounded
    // FIFO queue, no spurious EBUSY) and each completes with correct data.
    { const int P = 3;
      uint8_t *pbuf[P] = {nullptr};
      phxfs_batch_t *ph[P] = {nullptr};
      std::vector<phxfs_io_req_t> pr[P];
      bool ok = true;
      for (int q = 0; q < P; q++) {
          if (posix_memalign((void **)&pbuf[q], 4096, total) != 0) { ok = false; break; }
          memset(pbuf[q], 0, total);
          pr[q].assign(N, phxfs_io_req_t{});
          for (int i = 0; i < N; i++) {
              pr[q][i].fd = fd; pr[q][i].device_id = -1; pr[q][i].buf = pbuf[q];
              pr[q][i].buf_offset = (off_t)i * CHUNK; pr[q][i].nbytes = CHUNK;
              pr[q][i].f_offset = (off_t)i * CHUNK;
          }
          ph[q] = phxfs_batch_submit_read(pr[q].data(), N);
          if (!ph[q]) ok = false;
      }
      for (int q = 0; q < P; q++) {
          if (ph[q]) {
              int w = phxfs_batch_wait(ph[q]);
              if (w != 0 || verify_region(pbuf[q], 0, total, 0) != 0) ok = false;
          }
          free(pbuf[q]);
      }
      CHECK(ok, "pipeline: %d concurrent same-node submits succeed + verify", P); }

    // invalid device id and negative offset -> per-request -EFAULT, no crash
    { std::vector<phxfs_io_req_t> r; mk(r, 4);
      r[1].device_id = 999;      // out of range GPU id
      r[2].buf_offset = -CHUNK;  // negative offset
      int rc = phxfs_read_batch(r.data(), 4);
      CHECK(rc == 2, "invalid-device + negative-offset -> 2 failures (rc=%d)", rc);
      CHECK(r[1].result == -EFAULT && r[2].result == -EFAULT, "bad reqs marked -EFAULT"); }

    // CPU write batch: write buf -> a fresh file, read back and verify (T6)
    { std::string wp = std::string(base_file) + ".cpuw";
      int wfd = open(wp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
      if (wfd >= 0) {
          uint8_t *src = (uint8_t *)malloc(total);
          for (size_t j = 0; j < total / 8; j++) ((uint64_t *)src)[j] = pattern_word(j * 8);
          std::vector<phxfs_io_req_t> r(N);
          for (int i = 0; i < N; i++) { r[i] = phxfs_io_req_t{}; r[i].fd = wfd; r[i].device_id = -1;
              r[i].buf = src; r[i].buf_offset = (off_t)i * CHUNK; r[i].nbytes = CHUNK; r[i].f_offset = (off_t)i * CHUNK; }
          int wrc = phxfs_write_batch(r.data(), N);
          fsync(wfd);
          uint8_t *chk = (uint8_t *)malloc(total);
          ssize_t rd = pread(wfd, chk, total, 0);
          CHECK(wrc == 0 && rd == (ssize_t)total && verify_region(chk, 0, total, 0) == 0,
                "CPU write batch + read-back verify (wrc=%d)", wrc);
          free(src); free(chk); close(wfd); unlink(wp.c_str());
      } }

    close(fd); free(buf); unlink(fp.c_str());
}

// ---- correctness on GPU 0 ----
static void correctness(const char *base_file) {
    printf("--- Part A: correctness (GPU 0) ---\n");
    int ndev = 0; cudaGetDeviceCount(&ndev);
    if (ndev < 1) { SKIP("Part A: no CUDA device"); return; }
    std::string fp = std::string(base_file) + ".batch.0";
    const size_t buf_bytes = (size_t)REQS * CHUNK;
    if (ensure_pattern_file(fp.c_str(), buf_bytes) != 0) { CHECK(false, "Part A setup: pattern file"); return; }

    cudaSetDevice(0);
    int dev = phxfs_find_dev(0);
    if (dev < 0 || phxfs_open(dev) != 0) { SKIP("Part A: no phxfs device / module not loaded"); return; }

    void *gbuf = nullptr, *target = nullptr;
    cudaMalloc(&gbuf, buf_bytes); cudaMemset(gbuf, 0, buf_bytes); cudaStreamSynchronize(0);
    if (phxfs_regmem(dev, gbuf, buf_bytes, &target) != 0) {
        CHECK(false, "Part A setup: regmem"); phxfs_close(dev); cudaFree(gbuf); return; }

    int dfd = open(fp.c_str(), O_RDONLY | O_DIRECT);
    if (dfd < 0) { SKIP("Part A: O_DIRECT open unsupported");
        phxfs_deregmem(dev, gbuf, buf_bytes); phxfs_close(dev); cudaFree(gbuf); return; }
    std::vector<phxfs_io_req_t> reqs(REQS);
    for (int i = 0; i < REQS; i++) {
        reqs[i] = phxfs_io_req_t{};
        reqs[i].fd = dfd; reqs[i].device_id = dev; reqs[i].buf = gbuf;
        reqs[i].buf_offset = (off_t)i * CHUNK; reqs[i].nbytes = CHUNK;
        reqs[i].f_offset = (off_t)i * CHUNK;
    }
    int ret = phxfs_read_batch(reqs.data(), REQS);
    CHECK(ret == 0, "full batch (%d reqs) returned %d (0 = all ok)", REQS, ret);

    uint8_t *hbuf = (uint8_t *)malloc(buf_bytes);
    cudaMemcpy(hbuf, gbuf, buf_bytes, cudaMemcpyDeviceToHost); cudaStreamSynchronize(0);
    size_t mism = verify_region(hbuf, 0, buf_bytes, 0);
    CHECK(mism == 0, "full batch per-word verify: %zu mismatch(es)", mism);

    // scattered offsets: chunk i <- file chunk (REQS-1-i). Unique pattern
    // detects mis-routing (uniform patterns could not).
    cudaMemset(gbuf, 0, buf_bytes); cudaStreamSynchronize(0);
    for (int i = 0; i < REQS; i++) {
        reqs[i].buf_offset = (off_t)i * CHUNK;
        reqs[i].f_offset   = (off_t)(REQS - 1 - i) * CHUNK;
    }
    ret = phxfs_read_batch(reqs.data(), REQS);
    cudaMemcpy(hbuf, gbuf, buf_bytes, cudaMemcpyDeviceToHost); cudaStreamSynchronize(0);
    size_t scat_bad = 0;
    for (int i = 0; i < REQS; i++)
        scat_bad += verify_region(hbuf, (size_t)i * CHUNK, CHUNK,
                                  (uint64_t)(REQS - 1 - i) * CHUNK) ? 1 : 0;
    CHECK(ret == 0 && scat_bad == 0, "scattered-offset batch correct (%zu bad chunks)", scat_bad);

    // small batches n = 0..5 (regression for the worker-participation bug).
    // Each n repeated a few times to shake out generation/pending races.
    bool small_ok = true;
    for (int n = 0; n <= 5; n++) {
        for (int rep = 0; rep < 20; rep++) {
            cudaMemset(gbuf, 0, buf_bytes); cudaStreamSynchronize(0);
            for (int i = 0; i < n; i++) {
                reqs[i].buf_offset = (off_t)i * CHUNK;
                reqs[i].f_offset   = (off_t)i * CHUNK;
            }
            int r = phxfs_read_batch(reqs.data(), n);
            if (r != 0) { small_ok = false; break; }
            if (n > 0) {
                cudaMemcpy(hbuf, gbuf, (size_t)n * CHUNK, cudaMemcpyDeviceToHost);
                cudaStreamSynchronize(0);
                if (verify_region(hbuf, 0, (size_t)n * CHUNK, 0) != 0) { small_ok = false; break; }
            }
        }
        if (!small_ok) { printf("  small-batch failed at n=%d\n", n); break; }
    }
    CHECK(small_ok, "small batches n=0..5 (x20 each) correct");

    // alternate large and small batches (reuse of pool/ring state).
    bool alt_ok = true;
    for (int rep = 0; rep < 10 && alt_ok; rep++) {
        int sizes[] = {REQS, 1, REQS, 2, 3};
        for (int s : sizes) {
            cudaMemset(gbuf, 0, buf_bytes); cudaStreamSynchronize(0);
            for (int i = 0; i < s; i++) {
                reqs[i].buf_offset = (off_t)i * CHUNK;
                reqs[i].f_offset   = (off_t)i * CHUNK;
            }
            if (phxfs_read_batch(reqs.data(), s) != 0) { alt_ok = false; break; }
            cudaMemcpy(hbuf, gbuf, (size_t)s * CHUNK, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(0);
            if (verify_region(hbuf, 0, (size_t)s * CHUNK, 0) != 0) { alt_ok = false; break; }
        }
    }
    CHECK(alt_ok, "alternating large/small batches correct");

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
    CHECK(verify_region(hbuf, 0, buf_bytes, 0) == 0, "async batch per-word verify");

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

    const int    NREQ = REQS * BATCHES;
    const size_t file_bytes = (size_t)NREQ * CHUNK;
    // T2: one distinct GPU target region per in-flight request (no overlap),
    // so the result can be verified and mis-routing/duplication is detectable.
    const size_t buf_bytes = file_bytes;

    std::string fp = std::string(base_file) + ".batch." + std::to_string(gpu);

    void *gbuf = nullptr, *target = nullptr;
    if (cudaMalloc(&gbuf, buf_bytes) != cudaSuccess) { printf("gpu%d: cudaMalloc(%zu) failed\n", gpu, buf_bytes); return; }
    cudaMemset(gbuf, 0, buf_bytes); cudaStreamSynchronize(0);
    if (phxfs_regmem(dev, gbuf, buf_bytes, &target) != 0) {
        printf("gpu%d: phxfs_regmem(dev=%d) failed\n", gpu, dev); cudaFree(gbuf); return;
    }

    int dfd = open(fp.c_str(), O_RDONLY | O_DIRECT);
    if (dfd < 0) { printf("gpu%d: open %s: %s\n", gpu, fp.c_str(), strerror(errno));
                   phxfs_deregmem(dev, gbuf, buf_bytes); cudaFree(gbuf); return; }

    // One large batch: request i reads file chunk i into GPU chunk i — every
    // target region is distinct. A single blocking call should saturate the
    // array while exercising the internal NUMA pool.
    std::vector<phxfs_io_req_t> reqs(NREQ);
    for (int i = 0; i < NREQ; i++) {
        reqs[i] = phxfs_io_req_t{};
        reqs[i].fd = dfd; reqs[i].device_id = dev; reqs[i].buf = gbuf;
        reqs[i].nbytes = CHUNK;
        reqs[i].buf_offset = (off_t)i * CHUNK;   // distinct GPU target
        reqs[i].f_offset   = (off_t)i * CHUNK;   // distinct file region
    }

    posix_fadvise(dfd, 0, file_bytes, POSIX_FADV_DONTNEED);
    double t0 = now_sec();
    int rc = phxfs_read_batch(reqs.data(), NREQ);
    out->secs = now_sec() - t0;
    out->bytes = (size_t)NREQ * CHUNK;

    bool ok = (rc == 0);
    if (rc != 0) printf("gpu%d big batch failed rc=%d\n", gpu, rc);
    // Verify the whole buffer against the pattern (not timed).
    if (ok) {
        uint8_t *hbuf = (uint8_t *)malloc(buf_bytes);
        cudaMemcpy(hbuf, gbuf, buf_bytes, cudaMemcpyDeviceToHost); cudaStreamSynchronize(0);
        size_t bad = verify_region(hbuf, 0, buf_bytes, 0);
        if (bad) { printf("gpu%d: DATA MISMATCH %zu words\n", gpu, bad); ok = false; }
        free(hbuf);
    }
    out->ok = ok;

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
    bool ok = true;
    double t0 = now_sec();
    for (int b = 0; b < BATCHES; b++) {
        off_t base = (off_t)b * buf_bytes;
        for (int i = 0; i < REQS; i++) {
            reqs[i].buf_offset = (off_t)sub_off + (off_t)i * CHUNK;
            reqs[i].f_offset   = base + (off_t)i * CHUNK;
        }
        if (phxfs_read_batch(reqs.data(), REQS) != 0) {
            printf("t%d batch %d failed\n", tid, b);
            ok = false;      // T1: do NOT report success after a failure
            break;
        }
        out->bytes += buf_bytes;
    }
    out->secs = now_sec() - t0;
    out->ok = ok;
    // T2: verify the last batch actually landed in this thread's sub-range with
    // the right file data — a return-code-only check can't detect mis-routing,
    // duplication or a late/dropped write. (Not timed.)
    if (ok) {
        uint8_t *hbuf = (uint8_t *)malloc(buf_bytes);
        cudaMemcpy(hbuf, (uint8_t *)gbuf + sub_off, buf_bytes, cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(0);
        uint64_t f_base = (uint64_t)(BATCHES - 1) * buf_bytes;   // last batch's file base
        size_t bad = verify_region(hbuf, 0, buf_bytes, f_base);
        if (bad) { printf("t%d: DATA MISMATCH %zu words\n", tid, bad); out->ok = false; }
        free(hbuf);
    }
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

    const char *base_file = getenv("PHXFS_TEST_FILE");
    if (!base_file) base_file = TEST_FILE;
    printf("=== test_batch ===\n");
    printf("I/O engine: %s | shape: %d batches x %d reqs x %zu KiB\n",
           phxfs_io_engine_name(), BATCHES, REQS, CHUNK / KiB);

    cpu_correctness(base_file);   // module-free CPU-buffer coverage
    correctness(base_file);

    double t0, wall;
    size_t total_bytes = 0;
    bool all_ok = true;

    if (mt > 0) {
        // Single GPU (0), mt threads, mt independent rings, mt distinct files.
        if (mt > 16) mt = 16;
        printf("\n--- Part B: single-GPU multi-ring (GPU 0, %d threads/rings) ---\n", mt);
        if (!prepare_files(mt, base_file)) { printf("file prep failed\n"); return 1; }

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
            std::string fp = std::string(base_file) + ".batch." + std::to_string(t);
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
        if (!prepare_files(ngpu, base_file)) { printf("file prep failed\n"); return 1; }

        std::vector<std::thread> ts;
        std::vector<worker_result> res(ngpu);
        t0 = now_sec();
        for (int g = 0; g < ngpu; g++)
            ts.emplace_back(gpu_worker, g, base_file, &res[g]);
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

    printf("\n=== %d run, %d passed, %d failed, %d skipped ===\n",
           tests_run, tests_passed, tests_failed, tests_skipped);
    return tests_failed ? 1 : 0;
}
