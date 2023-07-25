#ifndef kbmod_IMAGE_H_
#define kbmod_IMAGE_H_

// I'm not sure there's any sense in protecting against not having python shared
// libraries availible in this implementation.
#include <assert.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm> // copy
#include <utility> // swap
#include <csignal>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

namespace arrays {
  // cleverrer design?
  // https://stackoverflow.com/questions/994488/what-is-proxy-class-in-c

  // Forward declarations for arr2d proxies.
  template<typename T>
  class proxy;


  template<typename T>
  class const_proxy;


  template <typename T>
  class arr2d {
  public:
    arr2d() {}

    arr2d(std::size_t n, std::size_t m)
      : storage(new T[n*m]), n(n), m(m) {}

    arr2d(std::size_t n, std::size_t m, T* ptr)
      : storage(ptr), n(n), m(n) {}

    arr2d(arr2d const& that)
      : storage(new T[that.n*that.m]), n(that.n), m(that.m) {
      std::copy(that.begin(), that.end(), this->begin());
    }

    arr2d(arr2d&& that)
      : storage(std::move(that.storage)), n(that.n), m(that.m) {
      that.n = that.m = 0;
    }

    arr2d& operator=(arr2d that) {
      swap(*this, that);
      return *this;
    }

    std::size_t rows() const { return n; }
    std::size_t columns() const { return m; }

    proxy<T> operator[](std::ptrdiff_t index) {
      assert(index >= 0 && index < n);
      return proxy<T>(storage.get(), index, m);
    }

    const_proxy<T> operator[](std::ptrdiff_t index) const {
      assert(index >= 0 && index < n);
      return const_proxy<T>(storage.get(), index, m);
    }

    T* begin() { return &storage[0]; }
    T* end() { return &storage[0] + n*m; }
    T* data() { return &storage[0]; }
    T const* begin() const { return &storage[0]; }
    T const* end() const { return &storage[0] + n*m; }
    T const* data() const { return &storage[0]; }

    friend void swap(arr2d& x, arr2d& y) {
      std::swap(x.storage, y.storage);
      std::swap(x.n, y.n);
      std::swap(x.m, y.m);
    }

  private:
    friend class proxy<T>;
    friend class const_proxy<T>;

    //T* storage;
    std::unique_ptr<T[]> storage;
    std::size_t n;
    std::size_t m;
    // I am worried Python can not rescind it's ownership in all circumstances
    // and a free will be called or a double free will be called if we use:
    //std::unique_ptr<T[]> storage;
    //std::shared_ptr<T[]> storage;
    // but that would be so cool to be able to do. I need to read
    // https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#std-shared-ptr
    // more carefully again and perhaps check out some source code
    //T* storage;
  };

  template<typename T>
  class proxy {
  public:
    T& operator[](std::ptrdiff_t index) {
      assert(index >= 0 && index < m);
      return storage[row * m + index];
    }
    T* begin() { return &storage[row * m]; }
    T* end() { return &storage[row * m] + m; }
    T* data() { return &storage[row * m]; }

  private:
    template<typename type>
    void operator=(type const&);

    friend class arr2d<T>;

    proxy(T* storage, std::ptrdiff_t row, std::size_t m)
      : storage(storage), row(row), m(m) {}

    T* storage;
    //std::unique_ptr<T*> storage;
    std::ptrdiff_t row;
    std::size_t m;
  };


  template<typename T>
  class const_proxy {
  public:
    T const& operator[](std::ptrdiff_t index) {
      assert(index >= 0 && index < m);
      return storage[row * m + index];
    }
    T const* begin() const { return &storage[row * m]; }
    T const* end() const { return &storage[row * m] + m; }
    T const* data() const { return &storage[row * m]; }

  private:
    template<typename type>
    void operator=(type const&);

    friend class arr2d<T>;

    const_proxy(T const* storage, std::ptrdiff_t row, std::size_t m)
      : storage(storage), row(row), m(m) {}

    T const* storage;
    //std::unique_ptr<T const*> storage;
    std::ptrdiff_t row;
    std::size_t m;
  };
} // arrays


namespace model {

  template <typename T>
  struct Image {
    // I see no reasons why anything should be private. All the methods and
    // attributes exist for NumPy arrays, so the user would expect to see
    // them if we are already pretending to be a numpy array equivalent. We also
    // don't really need to carry all this stuff with us either, or we could put
    // it in the bindings and carry the whole buffer along.

    // this doesn't exist for np arrays, but we add it because it's guaranteed
    // we only work with 2D data.
    std::size_t width;
    std::size_t height;

    // internal numpy str name for the datatypes underlying its formats
    // usually only refered to through a dtype. Also not accessible from Python
    std::string format;

    // all of this exists in Python interface for an np array
    // py:ssize_t is signed, std::size_t is unsigned
    std::size_t itemsize;
    std::size_t size;
    std::vector<py::ssize_t> shape;
    std::size_t ndim;
    std::vector<py::ssize_t> strides;
    arrays::arr2d<T> pixels;

    /*  ###########################################################
     *                    Constructors and destructors
     *  ########################################################### */
    Image(py::array_t<T> arr){
      py::buffer_info info = arr.request();

      if (info.ndim != 2) throw std::runtime_error("Image must have 2 dimensions.");

      this->width = info.shape[1];
      this->height = info.shape[0];
      this->shape = info.shape;
      this->strides = info.strides;
      this->itemsize = info.itemsize;
      this->size = info.size;
      this->format = info.format;
      this->ndim = info.ndim;
      this->pixels = arrays::arr2d<T>(width, height, static_cast<T*>(info.ptr));
    }

    // I do enjoy how I'm constantly forced to strip and then recreate context
    template <std::size_t rows, std::size_t cols>
    Image (int (&arr)[rows][cols]) {
      this->width = cols;
      this->height = rows;
      this->itemsize = sizeof(T);
      this->size  = width*height;
      this->shape = std::vector<py::ssize_t>({width, height});
      this->format = py::format_descriptor<T>::format();
      this->ndim = 2;
      this->pixels = arrays::arr2d<T>(width, height, &arr[0][0]);
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
      for (std::size_t i=0; i < this->size; i++){
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
      std::size_t i=0, j=0;
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



//    // While we can intercept the getitem operator in python by binding it to
//    // a method that exits back to python - we still need reasonable ways of
//    // accessing items in C++
//    T operator()(const unsigned int i, const unsigned int j){
//      if (i*j > this->size)
//        throw std::out_of_range("Image index out or range.");
//      return this->pixels[j*width + i];
//    }
//
//    T* operator()(const unsigned int i){
//      if (i > this->width)
//        throw std::out_of_range("Image index out or range.");
//      // I really am not sure if this is reasonable?
//      return reinterpret_cast<T (&)[width]> (this->pixels[i*width]);
//    }

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
      return py::array_t<T>({height, width}, strides, pixels.data());
    }

    Image& addOne(const int x) {
      // we can iterate over the raw pointer
      auto elems = this->pixels.data();
      for (std::size_t i=0; i<width; i++)
        for (std::size_t j=0; j<height; j++)
          elems[i*height + j] += 1;
      return *this;
    }

    Image& addOne(const double x) {
      // we can iterate over the raw pointer
      auto elems = this->pixels.data();
      for (std::size_t i=0; i<width; i++)
        for (std::size_t j=0; j<height; j++)
          elems[i*height + j] += 1.0;
      return *this;
    }

    //Image& addOne(const double x) {
    //  // but it's probably better to follow idiomatic modern C++
    //  for (auto row : this->pixels)
    //    for (auto elem : row)
    //      elem += 1.0;
    //  return *this;
    //}
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
                                 m.pixels.data(),
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
      .def("addOne", static_cast<Class& (Class::*)(double)>(&Class::addOne));
  } // image_type_factory
}; // namespace model
#endif /* kbmod_IMAGE_H_ */

