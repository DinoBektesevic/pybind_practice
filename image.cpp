#ifndef kbmod_IMAGE_H_
#define kbmod_IMAGE_H_

#include <pybind11/pybind11.h>
#include <Eigen/Core>


namespace py = pybind11;


namespace model {

  // This could have been just a typedef, but for the sake of
  // learning, lets say we wanted to extend the Matrix. In
  // this case we are just adding width and height
  template <typename T>
  struct Image : public Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> {
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Matrix::Matrix;

    int width;
    int height;

    Image(int use, const char my[2], int constructor, const char please[2]) :
      Matrix(Matrix::Zero(use, constructor)),
      width(use),
      height(constructor) {}
  };

  // A class that will use our extended Matrix class and
  // cast to np.array in Python using already existing
  // Eigen bindings in Pybind11.
  template <typename DataType, typename MaskType>
  struct LayeredImage {
    using Data = Image<DataType>;
    using Mask = Image<MaskType>;
    using DataRef = Eigen::Ref<Data>;
    using MaskRef = Eigen::Ref<Mask>;

    Data sci;
    Data var;
    Mask mask;

    LayeredImage<DataType, MaskType>(Data science, Data variance, Mask mask) :
      sci(science),
      var(variance),
      mask(mask){}
  };

  /*  ###########################################################
   *                    Binding Factory
   *  ########################################################### */
  // It would not be possible to directly expose our extension
  // to Python without short-circuiting Pybind's Eigen
  // auto-casting and breaking it. Testing performance against
  // LayeredImage would be unfair, so I've bound the same
  // underlying Eigen object to a light-weight class. More
  // work, than just the conversion to array, but ultimately
  // should not have a large impact.
  template<typename T>
  void image_bindings_factory(py::module &m, const std::string &name_prefix) {
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MRef = Eigen::Ref<Matrix>;
    struct MatrixWrap {
      Matrix data;
      MatrixWrap(Matrix d): data(d) {}
    };
    
    std::string class_name = name_prefix + std::string("Image");
    py::class_<MatrixWrap>(m, class_name.c_str())
      .def(py::init<Matrix>(), py::arg("array").noconvert(true))
      .def_property("data",
                    /*getter=*/ [](MatrixWrap& cls) { return MRef(cls.data);},
                    /*setter=*/ [](MatrixWrap& cls, MRef value) { cls.data = value; });
  }

  template<typename DataType, typename MaskType>
  void layered_type_bindings_factory(py::module &m, const std::string &name_prefix) {
    using Class = LayeredImage<DataType, MaskType>;
    using Data = Image<DataType>;
    using Mask = Image<MaskType>;
    using DataRef = Eigen::Ref<Data>;
    using MaskRef = Eigen::Ref<Mask>;

    std::string layered_class_name = name_prefix + std::string("Layered");
    py::class_<Class>(m, layered_class_name.c_str())
      .def(py::init<Data, Data, Mask>(),
           py::arg("science").noconvert(true),
           py::arg("variance").noconvert(true),
           py::arg("mask").noconvert(true))
      .def_property("sci",
                    /*getter=*/ [](Class& cls) { return DataRef(cls.sci);},
                    /*setter=*/ [](Class& cls, DataRef value) { cls.sci = value; })
      .def_property("var",
                    /*getter=*/ [](Class& cls) { return DataRef(cls.var);},
                    /*setter=*/ [](Class& cls, DataRef value) { cls.var = value; })
      .def_property("mask",
                    /*getter=*/ [](Class& cls) { return MaskRef(cls.mask);},
                    /*setter=*/ [](Class& cls, MaskRef value) { cls.mask = value; });
  }
}; // namespace model


namespace Eigen{
  namespace internal{
    template <typename T>
    struct traits<model::Image<T>>
      : public Eigen::internal::traits<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {};

    template <typename T>
    struct evaluator<model::Image<T>>
      : public Eigen::internal::evaluator<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {};
  }
}
#endif /* kbmod_IMAGE_H_ */
