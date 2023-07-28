#include <pybind11/pybind11.h>
// Including Eigen here will create Python bindings automatically
#include <pybind11/eigen.h>

#include "image.cpp"


PYBIND11_MODULE(core, m) {
  model::layered_type_bindings_factory<double, long int>(m, "DI");   // Double and 64 bit Int
  model::layered_type_bindings_factory<double, int>(m, "DI32"); // Double and 32 bit Int
}
