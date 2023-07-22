#ifndef IMAGE_H_
#define IMAGE_H_

// no point in IF-DEF PYTHON_H because everything assumes Python and NumPy?
// Unless we are thinking of providing multiple interfaces? In which case, happy
// to put the checks back.
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
      T* pixels;
      // bool _owner = true; // think about destructors and stuff? Instantiated
                             // from an array<T> pointer - not an owner so don't
                             // cleanup to prevent segfault, otherwise we clean
                             // up? But I don't actually understand how default
                             // destructors work in C++ and neither does the
                             // internet...


      /*  ###########################################################
       *                    Constructors and destructors
       *  ########################################################### */
      Image(py::array_t<T> arr){
        py::buffer_info info = arr.request();

        if (info.ndim != 2) throw std::runtime_error("Image must have 2 dimensions.");

        // this doesn't exist for np arrays, but we add it because it's guaranteed
        // we only work with 2D data.
        this->width = info.shape[1];
        this->height = info.shape[0];

        // All of this exists for a regular np.array and is general-purpose
        // except maybe the format
        this->shape = info.shape;
        this->strides = info.strides;
        this->itemsize = info.itemsize;
        this->format = info.format;
        this->ndim = info.ndim;
        this->size = info.size;
        this->pixels = static_cast<T*>(info.ptr);
      }


    /*  ###########################################################
     *                    Python and C++ operators
     *  ########################################################### */
    std::string _repr(){
      auto arr = this->asArray();
      auto npstr = py::str(arr.attr("__repr__")());
      auto imgstr = npstr.attr("replace")("array", "Image");
      return py::str(imgstr);
    }

    py::array_t<bool> _compare(const T* other_ptr, std::vector<py::ssize_t> shape) const{
      // it would probably be better (more general) if this were based on
      // iterators? Then we wouldn't have to cast lists and buffers?
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

    py::object _getitem(py::object key){
      return this->asArray().attr("__getitem__")(key);
    }

    py::object _setitem(py::object key, py::object val){
      return this->asArray().attr("__setitem__")(key, val);
    }

    // While we can intercept the getitem operator in python by binding it to
    // a method that exits back to python - we still need reasonable ways of
    // accessing items in C++
    T operator[](py::tuple idx){
      auto len = idx.attr("__len__")();
      if (len.cast<int>() > 2)
        throw std::out_of_range("Too many indices: Image is strictly 2 dimensional.");

      auto y = idx[0].cast<int>();
      auto x = idx[1].cast<int>();
      if (x*y > this->size)
        throw std::out_of_range("Image index out or range.");
      return this->pixels[x*width + y];
    }

    py::array_t<T> operator[](py::int_ idx){
      return this->asArray().attr("__getitem__")(idx);
    }

    py::array_t<T> operator[](py::slice idx){
      return this->asArray().attr("__getitem__")(idx);
    }

    /*  ###########################################################
     *                    Methods and whatever else we need
     *  We can adopt an underscore as the secret methods bound to python operators
     *  ########################################################### */
    std::string array2string(){
      auto arr = this->asArray();
      // this is an actual reference to a method in numpy which I think is amazing
      py::object npstr = arr.attr("__str__");
      return py::str(npstr());
    }

    py::array_t<T> asArray() const {
      // A memory leak? No! it's a copy I think - _setitem doen
      return py::array_t<T>({height, width}, strides, pixels);
    }

    Image& addOne() {
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
      // https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
      //static_cast<void (Pet::*)(const std::string &)>
      // my understanding of this mess is static_cast to a function pointer, i.e.
      // static_cast< return_type member_pointer input_type > (method)
      // C++ 14 and higher py::overload_cast<T>
      // Note also, here we only define these for example reasons - the first
      // definition is not natural.
      //.def("__getitem__", static_cast<py::array_t<T> (Class::*)(py::int_)>(&Class::operator[]))
      //.def("__getitem__", static_cast<py::array_t<T> (Class::*)(py::slice)>(&Class::operator[]))
      //.def("__getitem__", static_cast<T (Class::*)(py::tuple)>(&Class::operator[]))
      .def("__getitem__", &Class::_getitem)
      .def("__setitem__", &Class::_setitem);
      //cleverly syntactic sugar here makes overloading unnecessary
//      .def(py::self == py::self)
//      .def(py::self == py::array())
//      .def(py::self == py::list())
//      .def(py::self == py::buffer());
  } // image_type_factory
}; // namespace model
#endif /* IMAGE_H_ */

