#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "phx_cache.h"

namespace py = pybind11;

PYBIND11_MODULE(_phxcache, m) {
    m.doc() = "Phoenix KV cache adapter for LMCache (phxfs DMA)";

    py::class_<AsyncReadResult, std::shared_ptr<AsyncReadResult>>(m, "AsyncReadResult")
        .def_property_readonly("bytes_done", &AsyncReadResult::bytes_done,
                               "Bytes transferred (valid only after stream "
                               "synchronize)");

    py::class_<PhxCache>(m, "PhxCache")
        .def(py::init<int>(), py::arg("cuda_gpu_id"),
             "Initialize Phoenix device connection for the given CUDA GPU.")
        .def("regmem",
             [](PhxCache &self, uintptr_t gpu_addr, size_t size) -> uintptr_t {
                 return self.regmem(gpu_addr, size);
             },
             py::arg("gpu_addr"), py::arg("size"),
             "Register GPU memory for Phoenix DMA. Returns target_addr.")
        .def("deregmem",
             [](PhxCache &self, uintptr_t gpu_addr, size_t size) {
                 self.deregmem(gpu_addr, size);
             },
             py::arg("gpu_addr"), py::arg("size"),
             "Deregister GPU memory previously registered with regmem().")
        .def_property_readonly("device_id", &PhxCache::device_id,
                               "Phoenix device ID.")
        .def("close", &PhxCache::close, "Close the phxfs device.");

    py::class_<PhxFile>(m, "PhxFile")
        .def(py::init<const PhxCache &, const std::string &, int>(),
             py::arg("cache"), py::arg("path"), py::arg("flags"),
             "Open a file for Phoenix I/O.")
        .def("__enter__",
             [](PhxFile &self) -> PhxFile & { return self; })
        .def("__exit__",
             [](PhxFile &self, py::handle, py::handle, py::handle) {
                 self.close();
                 return false;
             })
        .def("read",
             [](PhxFile &self, uintptr_t buf, off_t buf_offset,
                ssize_t nbyte, off_t f_offset) -> ssize_t {
                 return self.read(buf, buf_offset, nbyte, f_offset);
             },
             py::call_guard<py::gil_scoped_release>(),
             py::arg("buf"), py::arg("buf_offset"), py::arg("nbyte"),
             py::arg("f_offset"),
             "Synchronous read via phxfs_read.")
        .def("read_async",
             [](PhxFile &self, uintptr_t buf, size_t nbytes, off_t offset,
                uintptr_t stream) -> std::shared_ptr<AsyncReadResult> {
                 return self.read_async(buf, nbytes, offset, stream);
             },
             py::arg("buf"), py::arg("nbytes"), py::arg("offset"),
             py::arg("stream"),
             "Asynchronous read via phxfs_read_async. Submit to CUDA "
             "stream, synchronize, then check result.bytes_done.")
        .def("write",
             [](PhxFile &self, uintptr_t buf, off_t buf_offset,
                ssize_t nbyte, off_t f_offset) -> ssize_t {
                 return self.write(buf, buf_offset, nbyte, f_offset);
             },
             py::call_guard<py::gil_scoped_release>(),
             py::arg("buf"), py::arg("buf_offset"), py::arg("nbyte"),
             py::arg("f_offset"),
             "Synchronous write via phxfs_write.")
        .def("write_async",
             [](PhxFile &self, uintptr_t buf, size_t nbytes, off_t offset,
                uintptr_t stream) -> std::shared_ptr<AsyncReadResult> {
                 return self.write_async(buf, nbytes, offset, stream);
             },
             py::arg("buf"), py::arg("nbytes"), py::arg("offset"),
             py::arg("stream"),
             "Asynchronous write via phxfs_write_async.")
        .def("close", &PhxFile::close, "Close the file descriptor.");
}
