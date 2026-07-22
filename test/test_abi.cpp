// test_abi.cpp — standalone ABI boundary check (T6).
//
// Verifies libphoenix.so can be dlopen'd with RTLD_NOW (every dependency is
// resolvable, no missing symbols) and that its dynamic export table is exactly
// the public C API — public symbols present, internal helpers hidden.
//
// Run: test_abi /path/to/libphoenix.so   (path defaults to "libphoenix.so")
#include <cstdio>
#include <dlfcn.h>

static const char *kPublic[] = {
    "phxfs_open", "phxfs_close", "phxfs_get_page_size", "phxfs_find_dev",
    "phxfs_read", "phxfs_write", "phxfs_regmem", "phxfs_deregmem",
    "phxfs_read_batch", "phxfs_write_batch", "phxfs_batch_submit_read",
    "phxfs_batch_submit_write", "phxfs_batch_wait", "phxfs_batch_destroy",
    "phxfs_io_engine_name",
};

// A few internal symbols that MUST NOT leak into the dynamic export table.
static const char *kInternal[] = {
    "__phxfs_deregmem", "phxfs_pool_run", "phxfs_pool_submit",
    "phxfs_io_engine_get", "devconn_init",
};

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "libphoenix.so";
    int failed = 0;

    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) { printf("[FAIL] dlopen(%s, RTLD_NOW): %s\n", path, dlerror()); return 1; }
    printf("[PASS] dlopen RTLD_NOW (%s)\n", path);

    int miss = 0;
    for (size_t i = 0; i < sizeof(kPublic) / sizeof(kPublic[0]); i++) {
        if (!dlsym(h, kPublic[i])) { printf("[FAIL] missing public symbol: %s\n", kPublic[i]); miss++; }
    }
    if (miss) failed += miss;
    else printf("[PASS] all %zu public symbols resolvable\n", sizeof(kPublic) / sizeof(kPublic[0]));

    int leak = 0;
    for (size_t i = 0; i < sizeof(kInternal) / sizeof(kInternal[0]); i++) {
        if (dlsym(h, kInternal[i])) { printf("[FAIL] internal symbol exported: %s\n", kInternal[i]); leak++; }
    }
    if (leak) failed += leak;
    else printf("[PASS] internal symbols hidden\n");

    dlclose(h);
    printf("=== test_abi: %s ===\n", failed ? "FAIL" : "OK");
    return failed ? 1 : 0;
}
