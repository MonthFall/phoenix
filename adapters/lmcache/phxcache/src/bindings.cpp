#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "phx_cache.h"

namespace py = pybind11;

PYBIND11_MODULE(_phxcache, m) {
    m.doc() = "Phoenix KV cache adapter for LMCache (phxfs DMA)";

    py::class_<PhxCache>(m, "PhxCache")
        .def(py::init<int>(), py::arg("device_id"),
             "Initialize Phoenix device connection for the given device.")
        .def("regmem",
             [](PhxCache &self, uintptr_t dev_addr, size_t size) -> uintptr_t {
                 return self.regmem(dev_addr, size);
             },
             py::arg("dev_addr"), py::arg("size"),
             "Register device memory for Phoenix DMA. Returns target_addr.")
        .def("deregmem",
             [](PhxCache &self, uintptr_t dev_addr, size_t size) {
                 self.deregmem(dev_addr, size);
             },
             py::arg("dev_addr"), py::arg("size"),
             "Deregister device memory previously registered with regmem().")
        .def_property_readonly("device_id", &PhxCache::device_id,
                               "Phoenix device ID.")
        .def_property_readonly("page_size", &PhxCache::page_size,
                               "Device page size in bytes (vendor-specific).")
        .def("close", &PhxCache::close, "Close the phxfs device.")
        .def("read_batch",
             [](PhxCache &self, uintptr_t buf_base,
                const std::vector<std::tuple<int, off_t, size_t, off_t>> &reqs)
                -> std::vector<ssize_t> {
                 return self.read_batch(buf_base, reqs);
             },
             py::call_guard<py::gil_scoped_release>(),
             py::arg("buf_base"), py::arg("reqs"),
             "Batch read via phxfs_read_batch. "
             "reqs=[(fd, buf_offset, nbytes, f_offset), ...]")
        .def("write_batch",
             [](PhxCache &self,
                const std::vector<std::tuple<int, uintptr_t, off_t, size_t, off_t>> &reqs)
                -> std::vector<ssize_t> {
                 return self.write_batch(reqs);
             },
             py::call_guard<py::gil_scoped_release>(),
             py::arg("reqs"),
             "Batch write via phxfs_write_batch (CPU buffers). "
             "reqs=[(fd, buf_ptr, buf_offset, nbytes, f_offset), ...]");

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
        .def("write",
             [](PhxFile &self, uintptr_t buf, off_t buf_offset,
                ssize_t nbyte, off_t f_offset) -> ssize_t {
                 return self.write(buf, buf_offset, nbyte, f_offset);
             },
             py::call_guard<py::gil_scoped_release>(),
             py::arg("buf"), py::arg("buf_offset"), py::arg("nbyte"),
             py::arg("f_offset"),
             "Synchronous write via phxfs_write.")
        .def("close", &PhxFile::close, "Close the file descriptor.");
}
