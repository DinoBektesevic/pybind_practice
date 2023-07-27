#ifndef kbmod_IMAGE_H_
#define kbmod_IMAGE_H_

#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// ABSOLUTELY DO NOT INCLUDE pybind11/eigen.h in this file
#include <pybind11/eigen.h>
//#include <Eigen/Core>


namespace py = pybind11;

namespace model {

  // For argument sake let's just say that we know the PSF size in advave
  // like we would for a 3x3 affine transform matrix for example, to better
  // fit in with showcasing features here
  using PSF = Eigen::Matrix<float, 10, 10, Eigen::RowMajor>;


  // This too could easily be a typedef also, but for the sake of argument
  // lets say we wanted to extend the Matrix interface and add "convolve"
  // as an method for some reason. In this case we are just adding width and height
  template <typename T>
  struct Image : public Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> {
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Matrix::Matrix;

    unsigned width;
    unsigned height;

    Image(Matrix arr) :
      Matrix(arr),
      width(arr.cols()),
      height(arr.rows()) {}

  };


  template <typename T>
  struct LayeredImage {

    //using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    //using MatrixRef = Eigen::Ref<Matrix>;
    //Matrix sci;
    //Matrix var;
    //Matrix mask;
    //LayeredImage<T>(Matrix science, Matrix variance, Matrix mask) :
    //  sci(science),
    //  var(variance),
    //  mask(mask){}

    using TImage = Image<T>;
    TImage sci;
    TImage var;
    Image<int> mask;
    LayeredImage<T>(TImage science, TImage variance, Image<int> mask) :
      sci(science),
      var(variance),
      mask(mask){}
  };

  /*  ###########################################################
   *                    Binding Factory
   *  ########################################################### */
  template<typename T>
  void layered_type_bindings_factory(py::module &m, const std::string &typestr) {

    // There will be a lot of typedefs and convenience renamings happening it semems
    using Class = LayeredImage<T>;
    //using TImage = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    //using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using TMatrix = Image<T>;
    using TMRef = Eigen::Ref<TMatrix>;
    using IMRef = Eigen::Ref<Image<int>>;

    // Binds the class name dynamically to the specified type
    std::string img_class_name = typestr + std::string("Image");
    std::string layered_class_name = typestr + std::string("LayeredImage");

    py::class_<Class>(m, layered_class_name.c_str())
      .def(py::init<TMatrix, TMatrix, Image<int>>(),
           py::arg("science").noconvert(true),
           py::arg("variance").noconvert(true),
           py::arg("mask").noconvert(true))
      .def("sci", [](Class& c) {return TMRef(c.sci);})
      .def("var", [](Class& c) {return TMRef(c.var);})
      .def("mask", [](Class& c) {return IMRef(c.mask);});




//    std::string img_class_name = typestr + std::string("Image");
//    std::string matrix_class_name = typestr + std::string("Matrix");
//
//    py::class_<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(m, matrix_class_name.c_str());
//
//    py::class_<Class, Matrix>(m, img_class_name.c_str(), py::buffer_protocol())
//      .def_buffer([](Class &m) -> py::buffer_info {
//        return py::buffer_info(m.data(),
//                               sizeof(T),
//                               py::format_descriptor<T>::format(),
//                               2,
//                               {m.height, m.width},
//                               {
//                                 sizeof(T) * m.cols(),
//                                 sizeof(T)
//                               });
//        })
//      .def(py::init<>())
//      //.def(py::init<T>(x))
//      .def(py::init<py::array>(), py::arg("arr").noconvert(true))
//      .def_readonly("width", &Class::width)
//      .def_readonly("height", &Class::height)
//      .def("data", [](Class& m){return m.data();});
      //.def("__getitem__", &Class::operator());
    //.def_readonly("data", &Class::data, py::return_value_policy::reference_internal);
  } // image_type_factory
}; // namespace model



namespace Eigen{
  namespace internal{
    /* Tell Eigen's expression template system about these new types so that it can
     * optimize them in its lazy evaluation.
     */

    template <typename T>
    struct traits<model::Image<T>>
      : public Eigen::internal::traits<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {};

    template <typename T>
    struct evaluator<model::Image<T>>
      : public Eigen::internal::evaluator<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {};
  }
}






#endif /* kbmod_IMAGE_H_ */


