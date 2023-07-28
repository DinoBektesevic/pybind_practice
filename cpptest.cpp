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

  // Check out the important class, in C++ I feel like this approach opens up
  // so many possibilities to really go ham-wild with squeezing the masks into
  // tiny formats, not to mention all the linear and general algebra operations
  // we don't have to implement anymore
  mo::LayeredImage<double, int> layered1 (img2, img2, img3);
  std::cout << layered1.sci <<  std::endl;
  std::cout << std::endl;

  // The added attributes exist in C++. They are unavailible in Python though -
  // we didn't provide out bindings and leveraged pybind's templating engine
  // to sort that out for us instead. Pybind will just cast it to an Eigen::Ref
  // or Eigen::Map and then use that objects data accessor to access the array
  // and array only. It is then re-casted to numpy array. So these classes are,
  // in Python-land, numpy arrays and nothing more. To make them more, we would
  // have to provide our own bindings, and that would include having to provide
  // the array getters and setters. They are very very hard to implement, and
  // really hard to get to via any kind of a reference.
  // This principle, that pybind11 is not as much of an interfacing library as
  // a casting tool, at least when it comes to numpy, is something that will
  // be a primary guider of the design of the codebase. The friction is
  // otherwise too great.
  // In some ways, this is perhaps disappointing? Briefly, for context, the
  // other way were the raw pointers to arrays (see main branch) in which case
  // we also don't get the indexing, getters or setters without implementing
  // them ourselves either is imho far more disappointing.
  // In other ways it's actually pretty good - it makes it clean and simple.
  // We typedef the Matrix classes we need and just use them as if they were
  // numpy arrays in C++. Lick out all additional functionality into a class,
  // or a set of namespaced functions even, which keep these types as
  // attributes, or whose methods (functions in case of functions) take in these
  // types as parameters and return the same types - leveraging pybind11 to
  // autocast to numpy arrays without copies.
  // All other syntactic sugars (short-cutting __getitem__ methods, making things
  // look like attributes and so on) will have to be done python-side.
  // I don't know why I wrote all of this, guess as a note to myself.
  auto img = mo::Image<int>(5, "m",  5, "c");
  std::cout << img << std::endl;
  std::cout << img.width << ", " << img.height << std::endl;
  std::cout << std::endl;

  return 0;
}
