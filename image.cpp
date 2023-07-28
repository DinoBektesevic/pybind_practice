#ifndef kbmod_IMAGE_H_
#define kbmod_IMAGE_H_

#include <pybind11/pybind11.h>
//#include <pybind11/eigen.h>
#include <Eigen/Core>


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
    // I have a feeling this is hiding a lot of weeeird details
    //http://eigen.tuxfamily.org/dox-3.2/TopicCustomizingEigen.html
    using Matrix::Matrix;

    int width;
    int height;

    // bloody hell it's tough to guess a constructor that isn't already taken
    Image(int use, const char my[2], int constructor, const char please[2]) :
      Matrix(Matrix::Zero(use, constructor)),
      width(use),
      height(constructor) {}
  };


  // The LayeredImage is basically supposed to be a transparent collection of
  // out Image<T> objects. I guess a case could be made for a denser packing of
  // the images into an <Height, Width, 3> array - but we have to move them to GPU
  // and then it gets hard to unpack, and also the types are, at the moment,
  // different

  //template<typename T>
  //using Image = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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

  struct test_attr {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Matrix sci;
    test_attr(): sci(Matrix::Zero(3, 3)) {}
  };


  /*  ###########################################################
   *                    Binding Factory
   *  ########################################################### */
  void test_arr_bindings_factory(py::module &m) {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    py::class_<test_attr>(m, "test_attr")
      .def(py::init<>())
      .def_readonly("readonly", &test_attr::sci)
      .def_readwrite("readwrite", &test_attr::sci)
      //.def("defSci", &test_attr::sci)
      .def("lambdaSci", [](test_attr& c) {return c.sci;})
      .def("lambdaRefSci", [](test_attr& c) {return Eigen::Ref<Matrix>(c.sci);});
  }


  template<typename T>
  void image_bindings_factory(py::module &m, const std::string &name_prefix) {
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Class = Image<T>;
    std::string class_name = name_prefix + std::string("Image");
    py::class_<Class>(m, class_name.c_str());
  }


  template<typename DataType, typename MaskType>
  void layered_type_bindings_factory(py::module &m, const std::string &name_prefix) {

    // There will be a lot of typedefs and convenience re-namings it seems
    using Class = LayeredImage<DataType, MaskType>;
    using Data = Image<DataType>;
    using Mask = Image<MaskType>;
    using DataRef = Eigen::Ref<Data>;
    using MaskRef = Eigen::Ref<Mask>;

    // Binds the class name dynamically to the specified type
    std::string data_class_name = name_prefix + std::string("Data");
    std::string mask_class_name = name_prefix + std::string("Mask");
    std::string layered_class_name = name_prefix + std::string("Layered");

    // We can't register these here because we might accidentally bind the same
    // type to two different names, but separating this at the moment felt like
    // a whole lot of work for nothing much so I saved state in this commit before
    // I forget how I got here.
    //py::class_<Data>(m, data_class_name.c_str());
    //py::class_<Mask>(m, mask_class_name.c_str());

    py::class_<Class>(m, layered_class_name.c_str())
      .def(py::init<Data, Data, Mask>(),
           py::arg("science").noconvert(true),
           py::arg("variance").noconvert(true),
           py::arg("mask").noconvert(true))
      // haha, super funny, pybind.h l. 1676 sets `const type &c` which sets the
      // returned array.flags.WRITEABLE to false (because we inherited):
      // https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#returning-values-to-python
      // forcing us to define our own getters and setters directly, of course
      // it's not possible to overload these, so we need to go and check the
      // type of the provided object and then cast it and then assign. Of course
      // this is not trivial because things need to be castable before assignement
      // operation and there are many different data types one can assign to an
      // array, depending on whether or not it's referenced via a slice and what
      // kind of slice.
      .def_property("sci",
                    /*getter=*/ [](Class& cls) { return DataRef(cls.sci);},
                    /*setter=*/ [](Class& cls, DataRef value) { cls.sci = value; })
      .def_property("var",
                    /*getter=*/ [](Class& cls) { return DataRef(cls.var);},
                    /*setter=*/ [](Class& cls, DataRef value) { cls.var = value; })
      .def_property("mask",
                    /*getter=*/ [](Class& cls) { return MaskRef(cls.mask);},
                    /*setter=*/ [](Class& cls, MaskRef value) { cls.mask = value; })
      .def("get_sci", [](Class& c) {return DataRef(c.sci);})
      .def("get_var", [](Class& c) {return DataRef(c.var);})
      .def("get_mask", [](Class& c) {return MaskRef(c.mask);});
  } // image_type_factory
}; // namespace model


namespace Eigen{
  namespace internal{
    /* Tell Eigen's expression template system about these new types so that it
     * can optimize them in its lazy evaluation and so that Eigen templating
     * can figure out what it can cast it to, and subsequently also pybind.
     * For us this is super simple - we just point it to the same traits and
     * evaluators that the Eigen::Matrix of an appropriate type would use -
     * however, this means we are dropping all the attributes and methods we
     * return, because we are casting not into our class, but into Matrix.
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
