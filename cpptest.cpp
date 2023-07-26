#include <iostream>
#include <vector>

#include "image.cpp"

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


int main(void){
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive

  int a[2][3] {{1, 2, 3}, {4, 5, 6}};
  // Why not, just tab it into the display of the neighbouring office monitor
  auto arr = py::array_t<int>(py::buffer_info(a,
                                               sizeof(int),
                                               py::format_descriptor<int>::format(),
                                               2,
                                               {2, 3},
                                               {2*sizeof(int), sizeof(int)}));

  auto img = model::Image<int>(arr);
  std::cout << img.data << std::endl;
  return 0;
}
