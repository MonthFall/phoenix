cmake_minimum_required (VERSION 3.18)
project (libphoenix-example)
find_package (CUDA 12.4 REQUIRED)

add_executable (example example.cc)
add_dependencies (example libphoenix)
target_include_directories (example PRIVATE "${libphoenix_root}/include")
target_link_libraries (example libphoenix CUDA::cudart)
