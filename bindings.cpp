#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "image.cpp"


PYBIND11_MODULE(core, m) {

  model::layered_type_bindings_factory<double>(m, "Double");
  //py::class_<Image<double> "LayeredImage"
  //model::image_type_bindings_factory<double>(m, "Double");

}
