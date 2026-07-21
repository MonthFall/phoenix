# Install & Quick Start

## Prerequisites (tested environment)

- **OS**: Ubuntu 22.04
- **Kernel**: Linux 6.1 (a kernel source / module tree matching your running kernel is required to build `phxfs`)
- **NVIDIA driver**: 550.54 with the open driver and `nvidia-fs` (GPU Direct Storage) enabled
- **OFED**: MLNX_OFED 24.10 (for NVMe-of / NFS RDMA paths)
- **CUDA**: 12.4
- **Build tools**: CMake ≥ 3.18, a CUDA-capable toolchain, `liburing`

## 1. NVIDIA GDS

```shell
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo bash cuda_12.4.0_550.54.14_linux.run
# select the nvidia-fs option and choose the open driver
```

## 2. MLNX_OFED

```shell
sudo ./mlnxofedinstall --with-nvmf --with-nfsrdma --enable-gds --add-kernel-support --dkms --skip-unsupported-devices-check
sudo update-initramfs -u -k `uname -r`
sudo reboot
```

## 3. Storage backend

### NVMe-of
```shell
cd scripts
sudo bash nvme_of.sh <target|initiator> <setup|cleanup>
```
### NFS
```shell
cd scripts
sudo bash nfs.sh <server|client>
```

## 4. Build Phoenix

```shell
mkdir -p build && cd build
cmake ../
make -j
```
This compiles the user library, the `phxfs` kernel module, the example, and the tests.

To skip the kernel module: `cmake -Dno_module=true ../`.

To target a different accelerator vendor: `cmake -DPHXFS_VENDOR=AMD ../` (default is `NVIDIA`).

## 5. Install the kernel module

```shell
cd build && sudo make insmod
```
Run `nvidia-smi` first to `modprobe` the NVIDIA driver. If installation fails, see [troubleshooting.md](troubleshooting.md).

## 6. Quick demo

A minimal end-to-end example lives in `example/example.cpp`. Build it via the top-level CMake (target `example`) and run:

```shell
cd build && sudo ./bin/example <file_path> <io_size> <mode>
```

## 7. Run tests

```shell
cd build
./bin/test_regmem 0   # memory registration lifecycle
./bin/test_io 0        # I/O correctness + performance
```

For vLLM weight loading, install the adapter and set `--load-format phxsafetensors` (see [adapters.md](adapters.md)).
