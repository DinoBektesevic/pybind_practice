#include "better_raw_image.h"

/* ############################################################
 *           Better names (because C++ matches Python)
 *           Better buffer protocol (because no copy is made)
 * ############################################################
 */
namespace better {
#ifdef Py_PYTHON_H
  BetterRawImage::BetterRawImage(pybind11::array_t<double> arr) { setArray(arr); }

  void BetterRawImage::setArray(pybind11::array_t<double>& arr) {
    pybind11::buffer_info info = arr.request();

    if (info.ndim != 2) throw std::runtime_error("Array must have 2 dimensions.");

    // if (info.format != py::format_descriptor<Scalar>::format())
    //  throw std::runtime_error("Incompatible format: expected a double array!");

    width = info.shape[1];
    height = info.shape[0];
    strides = info.strides;
    itemsize = info.itemsize;
    format = info.format;
    ndim = info.ndim;
    size = info.size;
    // wouldn't call a constructor or create a copy by performing type
    // conversion only if the given arr is also double type. This is the default
    // in Python, but ints are also very often and we would certainly benefit by
    // falling back to floats in terms of memory footprint. Constructors for
    // this class would then be neccessary then if we wanted to avoid the copy.
    pixels = static_cast<double*>(info.ptr);
  }
#endif
} /* namespace better */
