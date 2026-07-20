cmake_minimum_required (VERSION 3.18)

set (CMAKE_CUDA_ARCHITECTURES 90)

project (libphoenix LANGUAGES CUDA C CXX)

set (CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_STANDARD 20)

find_package (Threads REQUIRED)
find_package (CUDA 12.4 REQUIRED)
find_package (CUDAToolkit REQUIRED)
find_library (CUFILE_LIBRARIES cufile ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library (DL_LIB dl REQUIRED)

find_package(PkgConfig REQUIRED)

set (no_liburing false CACHE BOOL "Do not use liburing (io_uring async I/O)")

if (NOT no_liburing)
    pkg_check_modules(LIBURING liburing)
    if (LIBURING_FOUND)
        message(STATUS "liburing found: ${LIBURING_INCLUDE_DIRS}")
    else ()
        message(STATUS "liburing NOT found - io_uring async disabled")
    endif ()
else ()
    message(STATUS "liburing disabled by no_liburing=ON")
endif ()

if (LIBURING_FOUND)
    add_compile_definitions(PHXFS_HAVE_LIBURING)
endif ()

set (NVIDIA "" CACHE PATH "Path to Nvidia driver source")
set (KERNEL "/lib/modules/${CMAKE_SYSTEM_VERSION}/build" CACHE PATH "Path to kernel source, or module directory")

set (nvidia_archs "90" CACHE STRING "NVIDIA compute architectures")
set (no_module false CACHE BOOL "Do not build kernel module")
set (no_cuda false CACHE BOOL "Do not build library with CUDA support")
set (BUILD_SHARED_LIBS true CACHE BOOL "Build shared library, set to false for static")

# Kernel module: target GPU vendor (single-vendor-per-machine).
# This selects which P2P backend is compiled into the module.
set (PHXFS_VENDOR "NVIDIA" CACHE STRING "Target GPU vendor for kernel module: NVIDIA, AMD, HUAWEI")
set_property (CACHE PHXFS_VENDOR PROPERTY STRINGS NVIDIA AMD HUAWEI)

if (no_smartio)
    set (no_smartio_samples true)
    set (no_smartio_benchmarks true)
endif ()

# Set global compiler flags
set (GPU_ARCHS "" CACHE INTERNAL "" FORCE)
foreach (arch ${nvidia_archs})
    string (CONCAT GPU_ARCHS ${GPU_ARCHS} " -gencode arch=compute_${arch},code=sm_${arch}")
endforeach ()

set (convert_warnings implicit-function-declaration incompatible-pointer-types int-conversion return-type unused-function unused-variable)
set (flags "")
foreach (flag ${convert_warnings})
    string (APPEND flags " -Werror=${flag}")
endforeach()

# C_FLAGS
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra ${flags}" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} -g -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)
# CXX_FLAGS
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG " CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)

# GCC 8 requires linking stdc++fs for std::filesystem support
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    set(FS_LIBRARY stdc++fs)
else()
    set(FS_LIBRARY "")
endif()


if (no_cc)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CCUDA_NVCC_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
endif ()

include_directories ("${PROJECT_SOURCE_DIR}/libphoenix/include"
                    "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/")

# Set vendor-specific backend object file and compile define
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    set (VENDOR_BACKEND "nvidia-backend.o")
    set (VENDOR_DEFINE "CONFIG_PHXFS_VENDOR_NVIDIA")
elseif (PHXFS_VENDOR STREQUAL "AMD")
    set (VENDOR_BACKEND "amd-backend.o")
    set (VENDOR_DEFINE "CONFIG_PHXFS_VENDOR_AMD")
elseif (PHXFS_VENDOR STREQUAL "HUAWEI")
    set (VENDOR_BACKEND "huawei-backend.o")
    set (VENDOR_DEFINE "CONFIG_PHXFS_VENDOR_HUAWEI")
else ()
    message (FATAL_ERROR "Unsupported PHXFS_VENDOR: ${PHXFS_VENDOR}. Use NVIDIA, AMD, or HUAWEI.")
endif ()

# Try to find Nvidia driver headers (only needed for NVIDIA backend)
set (driver_include "")
if (PHXFS_VENDOR STREQUAL "NVIDIA")
    # Directly glob for nv-p2p.h in common locations
    file (GLOB nv_p2p_files
        "/usr/src/nvidia-*/nvidia-peermem/nv-p2p.h"
        "/usr/src/nvidia-*/nv-p2p.h"
        "${NVIDIA}/nv-p2p.h"
        "${NVIDIA}/nvidia-peermem/nv-p2p.h")

    if (nv_p2p_files)
        list (REVERSE nv_p2p_files)
        list (GET nv_p2p_files 0 first_match)
        get_filename_component (driver_include "${first_match}" DIRECTORY)
    endif ()

    if (driver_include)
        message(STATUS "Found NVIDIA driver header: ${driver_include}/nv-p2p.h")
    else ()
        message(WARNING
            "nv-p2p.h not found! Searched:\n"
            "  - /usr/src/nvidia-*/nvidia-peermem/nv-p2p.h\n"
            "  - /usr/src/nvidia-*/nv-p2p.h\n"
            "  - NVIDIA cache var: '${NVIDIA}'\n"
            "Pass -DNVIDIA=/path/to/nvidia/driver/src to cmake.")
    endif ()
endif ()

# # Module source files
set (module_root "${PROJECT_SOURCE_DIR}/module")
set (module_output "${CMAKE_BINARY_DIR}/module")
# for Makefile.in, do not delete
if (driver_include)
    set (module_ccflags "-I${driver_include} -D${VENDOR_DEFINE}")
else ()
    set (module_ccflags "-D${VENDOR_DEFINE}")
endif ()
configure_file ("${module_root}/Makefile.in" "${module_output}/Makefile" @ONLY)
message (STATUS "Configuring module build in ${module_output}")

add_custom_target(
    modules
    ALL
    COMMAND make
    WORKING_DIRECTORY ${module_output}
    COMMENT "Building kernel module"
)

add_custom_target(
    clean_modules
    COMMAND make clean
    WORKING_DIRECTORY ${module_output}
    COMMENT "Cleaning kernel module"
)

add_custom_target(
    insmod
    COMMAND sudo make insmod
    WORKING_DIRECTORY ${module_output}
    COMMENT "Inserting kernel module"
)

add_custom_target(
    rmmod
    COMMAND sudo make rmmod
    WORKING_DIRECTORY ${module_output}
    COMMENT "Removing kernel module"
)


# Library source files
set(libphoenix_root "${PROJECT_SOURCE_DIR}/libphoenix")
file(GLOB libphoenix_source "${libphoenix_root}/*.cc")
file(GLOB libphoenix_include "${libphoenix_root}/include/*.h")

add_library(libphoenix ${libphoenix_source} ${libphoenix_include})
set_target_properties (libphoenix PROPERTIES OUTPUT_NAME "phoenix")
target_include_directories(libphoenix PRIVATE "${libphoenix_root}" CUDA::cudart)
# Set install locations
install (FILES ${libphoenix_include} DESTINATION "include")
install (TARGETS libphoenix DESTINATION "lib")


# Make target depend on pthreads
macro (set_multithread target)
    target_link_libraries (${target} Threads::Threads)
endmacro ()

# Output directories (shared by remaining targets, e.g. example/)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Benchmarks are not built (the directory is scheduled for removal).

add_subdirectory ("${PROJECT_SOURCE_DIR}/example/")
add_subdirectory ("${PROJECT_SOURCE_DIR}/test/")

