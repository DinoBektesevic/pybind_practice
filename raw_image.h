#ifndef RAWIMAGE_H_
#define RAWIMAGE_H_

#include <array>
#include <vector>
#include <assert.h>

#ifdef Py_PYTHON_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#endif

constexpr float NO_DATA = -9999.0;


/* ############################################################
 *                               CURRENTLY USED
 * ############################################################
 */
namespace core{
  class RawImage {
  public:
#ifdef Py_PYTHON_H
    explicit RawImage(pybind11::array_t<float> arr);
    void setArray(pybind11::array_t<float>& arr);
#endif
    unsigned getWidth() const { return width; }
    unsigned getHeight() const { return height; }
    unsigned getPPI() const { return width * height; }
    float* getDataRef() { return pixels.data(); }
    const std::vector<float>& getPixels() const { return pixels; }
    float getPixel(int x, int y) const {
      return (x >= 0 && x < width && y >= 0 && y < height) ? pixels[y * width + x] : NO_DATA;
    }

    virtual ~RawImage(){};

  private:
    unsigned width;
    unsigned height;
    std::vector<float> pixels;
  }; // RawImage
}; // namespace core
#endif /* RAWIMAGE_H_ */
