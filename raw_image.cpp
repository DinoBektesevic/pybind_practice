#include "raw_image.h"

namespace core {
#ifdef Py_PYTHON_H
  RawImage::RawImage(pybind11::array_t<float> arr) { setArray(arr); }

  void RawImage::setArray(pybind11::array_t<float>& arr) {
    pybind11::buffer_info info = arr.request();

    if (info.ndim != 2) throw std::runtime_error("Array must have 2 dimensions.");

    width = info.shape[1];
    height = info.shape[0];
    float* pix = static_cast<float*>(info.ptr);

    pixels = std::vector<float>(pix, pix + getPPI());
  }
#endif

} /* namespace core */
