#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "phx_loader_v1.h"

namespace py = pybind11;

PYBIND11_MODULE(_phxloader_v1, m) {
    m.doc() = "Phoenix DMA loader V1 for GPU Direct Storage";

    py::class_<PhxLoaderV1>(m, "PhxLoaderV1")
        .def(py::init<int>(),
             py::arg("cuda_device_id"))
        .def("regmem",
             [](PhxLoaderV1 &self, uintptr_t gpu_ptr, size_t size) -> uintptr_t {
                 void *ptr = reinterpret_cast<void *>(gpu_ptr);
                 return self.regmem(ptr, size);
             },
             py::arg("gpu_ptr"), py::arg("size"),
             "Register GPU memory for DMA. Returns CPU mapped address.")
        .def("deregmem",
             [](PhxLoaderV1 &self, uintptr_t gpu_ptr, size_t size) {
                 void *ptr = reinterpret_cast<void *>(gpu_ptr);
                 self.deregmem(ptr, size);
             },
             py::arg("gpu_ptr"), py::arg("size"),
             "Deregister GPU memory previously registered with regmem().")
        .def("read_data_section",
             [](PhxLoaderV1 &self, const std::string &path, uintptr_t gpu_ptr,
                off_t data_offset, size_t data_size) -> off_t {
                 return self.read_data_section(path, gpu_ptr, data_offset,
                                               data_size);
             },
             py::arg("path"), py::arg("gpu_ptr"),
             py::arg("data_offset"), py::arg("data_size"),
             "Read entire data section in one DMA. Returns pre_padding offset.")
        .def("close", &PhxLoaderV1::close, "Close the phxfs device.");
}
