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
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// This is a lot more general than is allowed by the typedef in the class
// but what the heck - let's live a little
//constexpr bool rowMajor = Matrix::Flags & Eigen::RowMajorBit;

namespace model {

  template <typename T>
  struct Image {
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
    typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;

    friend class Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    unsigned width;
    unsigned height;
    Matrix data;

    /*  ###########################################################
     *                    Constructors and destructors
     *  ########################################################### */
    Image (){
      this->width = 3;
      this->height = 3;
      this->data = Matrix::Zero(3, 3);
    }

    Image(const py::array_t<T> b){
      py::buffer_info info = b.request();

      // Why have a strongly typed language when you're going to get silently
      // implicitly casted anyhow...
      //py::print("Type short int      " + py::format_descriptor<short int>::format() );
      //py::print("Type int            " + py::format_descriptor<int>::format() );
      //py::print("Type long int       " + py::format_descriptor<long int>::format() );
      //py::print("Type long long int  " + py::format_descriptor<long long int>::format() );
      //py::print("Type float          " + py::format_descriptor<float>::format() );
      //py::print("Type double         " + py::format_descriptor<double>::format() );

      auto pytype = py::format_descriptor<T>::format();
      //py::print("Type T is " + pytype + " array type is " + info.format);
      if (info.format != pytype)
        throw std::runtime_error("Incompatible format: expected a " + pytype +
                                 " array, got " + info.format + " instead.");

      if (info.ndim != 2)
        throw std::runtime_error("Incompatible buffer dimension!");

      auto strides = Strides(info.strides[0]/(py::ssize_t)sizeof(T),
                             info.strides[1]/(py::ssize_t)sizeof(T));

      //auto strides = Strides(info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(T),
      //                       info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(T));

      // so annoying this style where the broken lines don't de-dent inwards
      auto map = Eigen::Map<Matrix, 0, Strides>(static_cast<T*>(info.ptr),
                                                info.shape[0],
                                                info.shape[1],
                                                strides);
      this->data = Matrix(map);
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
        return py::buffer_info(m.data.data(),
                               sizeof(T),
                               py::format_descriptor<T>::format(),
                               2,
                               {m.height, m.width},
                               {
                                 sizeof(T) * m.data.cols(),
                                 sizeof(T)
                               });
        })
      .def(py::init())
      .def(py::init<py::array_t<T>>())
      .def_readonly("width", &Class::width)
      .def_readonly("height", &Class::height)
      .def_readonly("data", &Class::data, py::return_value_policy::reference_internal);
  } // image_type_factory
}; // namespace model
#endif /* kbmod_IMAGE_H_ */

