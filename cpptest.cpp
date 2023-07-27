#include <iostream>
#include <vector>

#include "image.cpp"

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


using DoubleImage = model::Image<double>;
using IntImage = model::Image<double>;


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

  //auto img = model::Image<int>(arr);
  auto img = IntImage();
  std::cout << img << std::endl;

  DoubleImage img2 {
    {2, 3, 4},
    {5, 6, 7},
  };
  std::cout << img2 << std::endl;
  std::cout << std::endl;

  IntImage img3(3, 3);
  img3 << 1, 2, 3,
    4, 5, 6,
    7, 8, 9;
  std::cout << img3 << std::endl;

  return 0;
}
