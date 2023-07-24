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

  std::cout << a[0][0] << std::endl;
  py::print(arr);

  auto test = model::Image<int>(arr);
  py::print(test.asArray());
  py::print(test(0, 0));
  std::cout << test.addOne(10.0) << std::endl;

  auto row = test(1);
  for(size_t i=0; i<test.width; i++)
    std::cout << "    " << row[i] << std::endl;

  auto test2 = model::Image<int>(a);
  std::cout << test2 << std::endl;
  std::cout << test2.addOne(1) << std::endl;

  return 0;
}
