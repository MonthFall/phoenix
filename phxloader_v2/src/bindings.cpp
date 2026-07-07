#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "phx_loader_v2.h"

namespace py = pybind11;

PYBIND11_MODULE(_phxloader_v2, m) {
    m.doc() = "Phoenix DMA loader V2 for GPU Direct Storage";

    py::class_<PhxLoaderV2>(m, "PhxLoaderV2")
        .def(py::init<int>(),
             py::arg("cuda_device_id"))
        .def("regmem",
             [](PhxLoaderV2 &self, uintptr_t gpu_ptr, size_t size) -> uintptr_t {
                 void *ptr = reinterpret_cast<void *>(gpu_ptr);
                 return self.regmem(ptr, size);
             },
             py::arg("gpu_ptr"), py::arg("size"),
             "Register GPU memory for DMA. Returns CPU mapped address.")
        .def("deregmem",
             [](PhxLoaderV2 &self, uintptr_t gpu_ptr, size_t size) {
                 void *ptr = reinterpret_cast<void *>(gpu_ptr);
                 self.deregmem(ptr, size);
             },
             py::arg("gpu_ptr"), py::arg("size"),
             "Deregister GPU memory previously registered with regmem().")
        .def("read_into_registered",
             [](PhxLoaderV2 &self, const std::string &path,
                uintptr_t gpu_ptr,
                const std::vector<std::tuple<off_t, off_t, size_t>> &batch) {
                 self.read_into_registered(path, gpu_ptr, batch);
             },
             py::call_guard<py::gil_scoped_release>(),
             py::arg("path"), py::arg("gpu_ptr"), py::arg("batch"),
             "Batch read multiple (buf_offset, file_offset, nbytes) entries "
             "from a single file into the registered GPU buffer.")
        .def("read_into_registered_async",
             [](PhxLoaderV2 &self, const std::string &path,
                uintptr_t gpu_ptr,
                const std::vector<std::tuple<off_t, off_t, size_t>> &batch) {
                 self.read_into_registered_async(path, gpu_ptr, batch);
             },
             py::arg("path"), py::arg("gpu_ptr"), py::arg("batch"),
             "Async batch read: launches DMA in a C++ thread, returns "
             "immediately. Use wait_dma() to join.")
        .def("wait_dma",
             [](PhxLoaderV2 &self) { self.wait_dma(); },
             py::call_guard<py::gil_scoped_release>(),
             "Wait for the most recent read_into_registered_async to "
             "complete.")
        .def("get_dma_seconds",
             [](PhxLoaderV2 &self) { return self.get_dma_seconds(); },
             "Get accumulated pure DMA time in seconds (C++ steady_clock).")
        .def("reset_dma_timer",
             [](PhxLoaderV2 &self) { self.reset_dma_timer(); },
             "Reset the DMA timer to zero.")
        .def("close", &PhxLoaderV2::close, "Close the phxfs device.");
}
