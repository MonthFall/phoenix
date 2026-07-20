cmake_minimum_required (VERSION 3.18)
project (phoenix-test)

find_package (CUDA 12.4 REQUIRED)

set (phoenix_root "${PROJECT_SOURCE_DIR}/..")
set (phoenix_include "${phoenix_root}/libphoenix/include")

# Test file path (override with -DTEST_FILE=/path/to/file)
set (TEST_FILE "/mnt/nvme0/phxfs_test.bin" CACHE STRING "Path for test I/O file")

# --- test_regmem: memory registration lifecycle ---
add_executable (test_regmem test_regmem.cu)
add_dependencies (test_regmem libphoenix)
target_include_directories (test_regmem PRIVATE "${phoenix_include}")
target_link_libraries (test_regmem libphoenix CUDA::cudart)
target_compile_definitions (test_regmem PRIVATE TEST_FILE="${TEST_FILE}")

# --- test_io: read/write correctness + performance ---
add_executable (test_io test_io.cu)
add_dependencies (test_io libphoenix)
target_include_directories (test_io PRIVATE "${phoenix_include}")
target_link_libraries (test_io libphoenix CUDA::cudart)
target_compile_definitions (test_io PRIVATE TEST_FILE="${TEST_FILE}")
