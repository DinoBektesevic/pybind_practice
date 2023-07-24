#ifndef kbmod_IMAGE_H_
#define kbmod_IMAGE_H_

// I'm not sure there's any sense in protecting against not having python shared
// libraries availible in this implementation.
#include <assert.h>
#include <string>
#include <vector>
#include <iostream>
#include <csignal>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


namespace model {

  template <typename T>
  struct Image {
    // I see no reasons why anything should be private. All the methods and
    // attributes exist for NumPy arrays, so the user would expect to see
    // them if we are already pretending to be a numpy array equivalent
    unsigned width;
    unsigned height;
    py::ssize_t itemsize;
    py::ssize_t size;
    std::vector<py::ssize_t> shape;
    std::string format;
    py::ssize_t ndim;
    std::vector<py::ssize_t> strides;
    // I really wish I could declare this as an array, because of the [][] and
    // normalcy, but I don't understand how to cast a void* (see below)
    T* pixels;

    /*  ###########################################################
     *                    Constructors and destructors
     *  ########################################################### */
    Image(const py::array_t<T> arr){
      py::buffer_info info = arr.request();

      if (info.ndim != 2) throw std::runtime_error("Image must have 2 dimensions.");

      // this doesn't exist for np arrays, but we add it because it's guaranteed
      // we only work with 2D data.
      this->width = info.shape[1];
      this->height = info.shape[0];

      // All of this exists for a regular np.array and is general-purpose
      // except maybe the format, which is the numpy name for the datatypes
      // underlying its formats usually only refered to through a dtype
      this->shape = info.shape;
      this->strides = info.strides;
      this->itemsize = info.itemsize;
      this->size = info.size;
      this->format = info.format;
      this->ndim = info.ndim;
      // I wish I knew how to cast this into an array?
      this->pixels = static_cast<T*>(info.ptr);
    }


    // I do enjoy how I'm constantly forced to strip and then recreate context
    template <size_t rows, size_t cols>
    Image (int (&arr)[rows][cols]) {
      this->width = cols; // sizeof(arr)/sizeof(T);
      this->height = rows; // sizeof(arr)/sizeof(arr[0]);
      this->itemsize = sizeof(T);
      this->size  = width*height;
      this->shape = std::vector<py::ssize_t>({width, height});
      this->format = py::format_descriptor<T>::format();
      this->ndim = 2;
      this->pixels = &arr[0][0];
    }


    /*  ###########################################################
     *                    Python and C++ operators
     * We can adopt an underscore as the secret methods bound to python operators
     *  ########################################################### */
    std::string _repr(){
      auto arr = this->asArray();
      auto npstr = py::str(arr.attr("__repr__")());
      auto imgstr = npstr.attr("replace")("array", "Image");
      return py::str(imgstr);
    }

    // it would probably be better (more general) if this were based on
    // iterators? Then we wouldn't have to cast lists and buffers?
    py::array_t<bool> _compare(const T* other_ptr, std::vector<py::ssize_t> shape) const{
      if (this->shape != shape)
        // dear brother in christ how does anyone format strings in C++ and not lose it
        throw std::invalid_argument("operands could not be broadcast together "
                                    "with shapes (" + std::to_string(this->shape[0]) +
                                    ", " + std::to_string(this->shape[1]) + " and (" +
                                    std::to_string(shape[0]) + ", " +
                                    std::to_string(shape[1]));

      auto equal = py::array_t<bool>(this->size);
      auto equal_ptr = static_cast<bool*>(equal.request().ptr);
      for (size_t i=0; i < this->size; i++){
        if (this->pixels[i] != other_ptr[i])
          equal_ptr[i] = false;
        else
          equal_ptr[i] = true;
      }

      equal.resize({this->width, this->height});
      return equal;
    }

    py::array_t<bool> operator==(const py::buffer other) const{
      auto buff_info = other.request();
      // is static cast here a safe option here?
      return this->_compare(static_cast<T*>(buff_info.ptr), buff_info.shape);
    }

    py::array_t<bool> operator==(const Image other) const{
      return this->_compare(other.pixels, other.shape);
    }

    // This is another neat example of how implicit cast makes a copy - if this
    // were a vector, the list wouldn't be passed by reference. It would be a
    // natural thing to expect a vector - because that's what Pybind11 says you
    // should do, but it would also be equally reasonable to expect a buffer
    py::array_t<bool> operator==(const py::list other) const{
      size_t i=0, j=0;
      auto col_stride = this->strides[0];
      auto row_stride = this->strides[1];
      auto height = other.size();
      auto width = other[0].cast<py::list>().size();

      auto equal = py::array_t<bool>(size);
      auto equal_ptr = static_cast<bool*>(equal.request().ptr);
      for (auto row : other){
        for (auto elem : row){
          auto idx = i*this->width + j;
          if (this->pixels[idx] != elem.cast<T>())
            equal_ptr[idx] = false;
          else
            equal_ptr[idx] = true;
          j += 1;
        }
        i += 1;
        j = 0;
      }
      equal.resize({this->width, this->height});
      return equal;
    }

    py::object _getitem(const py::object key){
      return this->asArray().attr("__getitem__")(key);
    }

    // While we can intercept the getitem operator in python by binding it to
    // a method that exits back to python - we still need reasonable ways of
    // accessing items in C++
    T operator()(const unsigned int i, const unsigned int j){
      if (i*j > this->size)
        throw std::out_of_range("Image index out or range.");
      return this->pixels[j*width + i];
    }

    T* operator()(const unsigned int i){
      if (i > this->width)
        throw std::out_of_range("Image index out or range.");
      // I really am not sure if this is reasonable?
      return reinterpret_cast<T (&)[width]> (this->pixels[i*width]);
    }

    // Ooh, I just love this line of logic
    friend std::ostream& operator<<(std::ostream &s, const Image<T> &img) {
      return s << img.array2string();
    }

    /*  ###########################################################
     *                    Methods and whatever else we need
     *  ########################################################### */
    std::string array2string() const{
      auto arr = this->asArray();
      // this is an actual reference to a method in numpy which I think is amazing
      py::object npstr = arr.attr("__str__");
      return py::str(npstr());
    }

    py::array_t<T> asArray() const {
      // A memory leak? No! it's a copy I think.
      return py::array_t<T>({height, width}, strides, pixels);
    }

    Image& addOne(const int x) {
      for (size_t i=0; i<width; i++)
        for (size_t j=0; j<height; j++)
          pixels[i*height + j] += 1.0;
      return *this;
    }

    Image& addOne(const double x) {
      for (size_t i=0; i<width; i++)
        for (size_t j=0; j<height; j++)
          pixels[i*height + j] += 1.0;
      return *this;
    }


  }; // Image


  /*  ###########################################################
   *                    Binding Factory
   *  ########################################################### */
  template<typename T>
  void image_type_bindings_factory(py::module &m, const std::string &typestr) {
    using Class = Image<T>;
    std::string pyclass_name = typestr + std::string("Image");
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol())
      .def_buffer([](Class &m) -> py::buffer_info {
          return py::buffer_info(
                                 m.pixels,
                                 m.itemsize,
                                 m.format,
                                 m.ndim,
                                 {m.height, m.width},
                                 m.strides);
        })
      .def(py::init<py::array_t<T>>())
      .def_readonly("size", &Class::size)
      .def_readonly("width", &Class::width)
      .def_readonly("height", &Class::height)
      .def_readonly("format", &Class::format)
      .def_readonly("itemsize", &Class::itemsize)
      .def_readonly("size", &Class::size)
      .def_readonly("shape", &Class::shape)
      .def_readonly("strides", &Class::strides)
      .def("__str__", &Class::array2string)
      .def("__repr__", &Class::_repr)
      .def("__len__", [](){return &Class::size;})
      .def("__getitem__", &Class::_getitem)
// It does make sense to protect against defining bindings that rely on the Py
// shared libraries to be loaded in cases when the file is compiled as a self
// standing C++. In that case the CPython is never loaded.
// https://github.com/python/cpython/blob/3aeffc0d8f28655186f99d013ee9653c65b92f84/Include/Python.h#L5
#ifdef py_PYTHON_H
      .def(py::self == py::self)
      .def(py::self == py::array())
      .def(py::self == py::list())
      .def(py::self == py::buffer())
#endif
      // https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
      //static_cast<void (Pet::*)(const std::string &)>
      // my understanding of this mess is static_cast to a function pointer, i.e.
      // static_cast< return_type member_pointer input_type > (method)
      // In C++ 14 and higher there is a shorthand py::overload_cast<T>
      //clever syntactic sugar for equality operator made this workaround
      // unnecessary
      .def("addOne", static_cast<Class& (Class::*)(int)>(&Class::addOne))
      .def("addOne", static_cast<Class& (Class::*)(float)>(&Class::addOne));
  } // image_type_factory
}; // namespace model
#endif /* kbmod_IMAGE_H_ */

