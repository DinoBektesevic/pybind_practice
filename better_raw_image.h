#ifndef BETTER_RAWIMAGE_H_
#define BETTER_RAWIMAGE_H_

#include <assert.h>
#include <iostream>

#ifdef Py_PYTHON_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#endif

/* ############################################################
 *           Better names (because C++ matches Python)
 *           Better buffer protocol (because no copy is made)
 * ############################################################
 */
namespace better {
  class BetterRawImage {
  public:
    virtual ~BetterRawImage(){};

#ifdef Py_PYTHON_H
    explicit BetterRawImage(pybind11::array_t<double> arr);
    void setArray(pybind11::array_t<double>& arr);
#endif
    unsigned getWidth() const { return width; }
    unsigned getHeight() const { return height; }
    unsigned nPixels() const { return width * height; }
    py::array_t<double> getPixels() const {
      // hopefully this isn't a memory leak?
      return py::array_t<double>(
         {height, width},
         strides,
         pixels);
    }
    py::array_t<double> copyAddOne() {
      auto res = py::array_t<double>(size);
      py::buffer_info buff_info = res.request();
      // this code is rife for implicit conversion to create a copy in an
      // completely unnoticeable, unobvious way
      double* ptr = static_cast<double*>(buff_info.ptr);

      for (size_t i=0; i<width; i++)
        for (size_t j=0; j<height; j++){
          auto idx = i*height + j;
          ptr[idx] = pixels[idx] + 1.0;
        }

      res.resize({width, height});
      return res;
    }
    void inplaceAddOne() {
      for (size_t i=0; i<width; i++)
        for (size_t j=0; j<height; j++)
          pixels[i*height + j] += 1.0;
    }


    unsigned width;
    unsigned height;
    py::ssize_t itemsize;
    py::ssize_t size;
    std::string format;
    py::ssize_t ndim;
    std::vector<py::ssize_t> strides;
    double* pixels;

    //private:
    // I'm lazy, but I do wonder if this should just be a struct anyhow
  }; // BetterRawImage
}; // namespace better
#endif /* BETTER_RAWIMAGE_H_ */

