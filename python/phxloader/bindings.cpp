#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "phx_loader.h"

namespace py = pybind11;

PYBIND11_MODULE(_phxloader, m) {
    m.doc() = "Phoenix DMA loader for GPU Direct Storage";

    py::class_<PhxLoader>(m, "PhxLoader")
        .def(py::init<int>(),
             py::arg("cuda_device_id"))
        .def("regmem",
             [](PhxLoader &self, uintptr_t gpu_ptr, size_t size) -> uintptr_t {
                 void *ptr = reinterpret_cast<void *>(gpu_ptr);
                 return self.regmem(ptr, size);
             },
             py::arg("gpu_ptr"), py::arg("size"),
             "Register GPU memory for DMA. Returns CPU mapped address.")
        .def("deregmem",
             [](PhxLoader &self, uintptr_t gpu_ptr, size_t size) {
                 void *ptr = reinterpret_cast<void *>(gpu_ptr);
                 self.deregmem(ptr, size);
             },
             py::arg("gpu_ptr"), py::arg("size"),
             "Deregister GPU memory previously registered with regmem().")
        .def("read_data_section",
             [](PhxLoader &self, const std::string &path, uintptr_t gpu_ptr,
                off_t data_offset, size_t data_size) -> off_t {
                 return self.read_data_section(path, gpu_ptr, data_offset,
                                               data_size);
             },
             py::arg("path"), py::arg("gpu_ptr"),
             py::arg("data_offset"), py::arg("data_size"),
             "Read entire data section in one DMA. Returns pre_padding offset.")
        .def("close", &PhxLoader::close, "Close the phxfs device.");
}
