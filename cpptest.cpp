// Note this is pure C++ - no pybinds
#include <iostream>
#include "image.cpp"

namespace py = pybind11;
namespace mo = model;


int main(void){
  int a[2][3] {{1, 2, 3}, {4, 5, 6}};

  // These are all Eigen constructors, plenty more too. See their documentation.
  mo::Image<double> img2 {
    {2, 3, 4},
    {5, 6, 7},
  };
  std::cout << img2 << std::endl;
  std::cout << std::endl;

  mo::Image<int> img3(3, 3);
  img3 << 1, 2, 3,
    4, 5, 6,
    7, 8, 9;
  std::cout << img3 << std::endl;
  std::cout << std::endl;

  // This is our LayeredImage 
  mo::LayeredImage<double, int> layered1 (img2, img2, img3);
  std::cout << layered1.sci <<  std::endl;
  std::cout << std::endl;

  // The extended attributes exist in C++. Note that to
  // expose these in Python requires writing the binding
  // interface and this breaks Pybinds default bindings.
  auto img = mo::Image<int>(5, "m",  5, "c");
  std::cout << img << std::endl;
  std::cout << img.width << ", " << img.height << std::endl;
  std::cout << std::endl;

  return 0;
}
