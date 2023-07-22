#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "raw_image.cpp"
#include "better_raw_image.cpp"
#include "image.cpp"

namespace py = pybind11;
using ri = core::RawImage;
using bri = better::BetterRawImage;


PYBIND11_MODULE(core, m) {
  /* ############################################################
   *                               CURRENTLY USED
   * ############################################################
   */
  py::class_<ri>(m, "current_raw_image", py::buffer_protocol())
    .def_buffer([](ri &m) -> py::buffer_info {
      return py::buffer_info(m.getDataRef(), sizeof(float), py::format_descriptor<float>::format(),
                             2, {m.getHeight(), m.getWidth()},
                             {sizeof(float) * m.getWidth(), sizeof(float)});
    })
    .def(py::init<py::array_t<float>>())
    .def("get_height", &ri::getHeight, "Returns the image's height in pixels.")
    .def("get_width", &ri::getWidth, "Returns the image's width in pixels.")
    .def("get_ppi", &ri::getPPI, "Returns the image's total number of pixels.")
    .def("get_all_pixels", &ri::getPixels, "Returns a list of the images pixels.");


  /* ############################################################
   *                               BETTER TO USE
   * ############################################################
   */
  py::class_<bri>(m, "BetterRawImage", py::buffer_protocol())
    .def_buffer([](bri &m) -> py::buffer_info {
      return py::buffer_info(
         m.pixels,
         m.itemsize,
         m.format,
         m.ndim,
         {m.height, m.width},
         m.strides);
    })
    .def(py::init<py::array_t<double>>())
    .def("getHeight", &bri::getHeight, "Returns the image's height in pixels.")
    .def("getWidth", &bri::getWidth, "Returns the image's width in pixels.")
    .def("nPixels", &bri::nPixels, "Returns the image's total number of pixels.")
    .def("getPixels", &bri::getPixels, "Returns a list of the images pixels.")
    .def("copyAddOne", &bri::copyAddOne, "Add 1.0 to every value.")
    .def("inplaceAddOne", &bri::inplaceAddOne, "Add 1.0 to every value.");


  /* ############################################################
   *                               MDLEL EXAMPLE
   * ############################################################
   */
  model::image_type_bindings_factory<double>(m, "Double");
  model::image_type_bindings_factory<float>(m, "Float");
  model::image_type_bindings_factory<long int>(m, "Int32"); // at least 32
  model::image_type_bindings_factory<long long int>(m, "Int64"); // at least 64

}
