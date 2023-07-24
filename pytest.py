import timeit
import numpy as np

import core


n_instantiation, n_access = 500, 100000
arrint = np.array([[1, 2], [3, 4]])
arrdouble = np.array([[1.0, 2.0], [3.0, 4.0]])
arrfloat = np.array([[1, 2], [3, 4]], dtype=np.float32)
list = [[1, 2], [3, 4]]
bigArr = np.zeros((2000, 4000))
big32Arr = np.zeros((4000, 2000), dtype=np.int32)


#print("\n")
#print(" "*30 + "Current RawImage")
#print()
#print(f"    Timing instantiation for {n_instantiation} repetitions.")
#bigArrT = timeit.timeit(stmt="core.current_raw_image(bigArr)", globals=globals(), number=n_instantiation)
#big32ArrT = timeit.timeit(stmt="core.current_raw_image(big32Arr)", globals=globals(), number=n_instantiation)
#print(f"From double: {bigArrT/n_instantiation:>10.7} seconds per iteration; {bigArrT} seconds total.")
#print(f"From  float: {big32ArrT/n_instantiation:>10.7} seconds per iteration; {big32ArrT} seconds total.")
#print()
#print("    Pixel access timings and behaviour:")
#curRawImgArr = core.current_raw_image(arr)
#curRawImgList = core.current_raw_image(list)
#print(f"fromNpArray.get_all_pixels() \n {curRawImgArr.get_all_pixels()}")
#print(f"fromList.current_raw_image(list) \n {curRawImgList.get_all_pixels()}")
#bigArrT = timeit.timeit(stmt="curRawImgArr.get_all_pixels()", globals=globals(), number=n_access)
#print(f"getPixels: {bigArrT/n_access:>10.7} seconds per iteration; {bigArrT} seconds total.")
#del curRawImgArr
#del curRawImgList
#
#
#print("\n")
#
#
#print(" "*30 + "Better RawImage")
#print()
#print(f"    Timing instantiation for {n_instantiation} repetitions.")
#bigArrT = timeit.timeit(stmt="core.BetterRawImage(bigArr)", globals=globals(), number=n_instantiation)
#big32ArrT = timeit.timeit(stmt="core.BetterRawImage(big32Arr)", globals=globals(), number=n_instantiation)
#print(f"From double: {bigArrT/n_instantiation:>10.7} seconds per iteration; {bigArrT} seconds total.")
#print(f"From  float: {big32ArrT/n_instantiation:>10.7} seconds per iteration; {big32ArrT} seconds total.")
#print()
#print("    Pixel access timings and behaviour:")
#curRawImgArr = core.BetterRawImage(arr)
#curRawImgList = core.BetterRawImage(list)
#print(f"Casting in Python: np.array(array, copy=False) \n {np.array(curRawImgArr, copy=False)}")
#print(f"Casting from  C++: array.getPixels() \n {curRawImgArr.getPixels()}")
#getPixT = timeit.timeit(stmt="curRawImgArr.getPixels()", globals=globals(), number=n_access)
#getArrT = timeit.timeit(stmt="np.array(curRawImgArr, copy=False)", globals=globals(), number=n_access)
#print(f"Casting in Python: {getArrT/n_access:>10.7} seconds per iteration; {getArrT} seconds total.")
#print(f"Casting from  C++: {getPixT/n_access:>10.7} seconds per iteration; {getPixT} seconds total.")
#print()
#print("    Just to quickly model how to work with this object (see code):")
#print(f"Add 1.0 to every element and return a copy     `array.copyAddOne()` = \n {curRawImgArr.copyAddOne()}")
#curRawImgArr.inplaceAddOne()
#print(f"Add 1.0 to every element without a copy `array.inplacecopyAddOne()` = \n {curRawImgArr.getPixels()}")
#del curRawImgArr
#del curRawImgList


print("\n")


print(" "*30 + "Model Raw Image, aka 'Image'")
print("""
The following demos my idea of what it means to properly wrap and integrate a
C++ class with Python. The distinguishing features are that it retains the
performance of the second approach, while elevating the interface provided by
either the first or second approach to one that more closely follows numpy
arrays. Two critical aspects here are:
1) to the user it must be interchangeable by a simple numpy array
2) it has to extend the numpy array interface to enable us to process it on GPU
   as an unravelled array.
3) Minimize copy operations wherever possible.

In practice this means  copious overrides of various operators,
self-referentiality which enables method, operator and function call chaining,
and clear to read, easy to maintained, documentation that integrates seamlessly
with that of Python docs standards.
""")


print(" "*15 + "As pure C++ implemntation as possible")


# We would have a delegator that would instantiate the correct C++ class for
# convenience. Multiple constructors could be declared here as factory class
# methods.
class Image:
    @classmethod
    def __get_cpp_class(cls, dtype):
        # It is critical here to understand the numpy and C++ standards
        # https://en.cppreference.com/w/cpp/language/types
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#scalars
        if dtype == "float64":
            return core.DoubleImage
        elif dtype == "float32":
            return core.FloatImage
        elif dtype == "int32":
            return core.Int32Image
        elif dtype == "int64":
            return core.Int64Image

    @classmethod
    def fromArray(cls, arr):
        cls = cls.__get_cpp_class(arr.dtype.name)
        return cls(arr)


iimg = Image.fromArray(arrint)  # good for masks?
fimg = Image.fromArray(arrfloat)
dimg = Image.fromArray(arrdouble)
print(iimg)
print(fimg)
print(dimg)
print()
print(repr(dimg))
print()

print("shape", dimg.shape)
print("width", dimg.width)
print("height", dimg.height)
print("size", dimg.size)
print("itemsize", dimg.itemsize)
print("format", dimg.format)
print("strides", dimg.strides)
print()

print("Item access is a bit rough:")
print(""" The following is allowed by NumPy. Half of the accessor operator is
figured out in Python, half in C as far as I can read their source code. Pybind
offers seamless integration with the slice objects - but only a slice objects.
All of the following is allowed by numpy

array[0, 0]      --> scalar, detected as tuple of py::int_
array[0][0]      --> scalar, detected as py::int_
array[0]         --> array, 1D for us, detected as py::int_
array[:2]        --> array, 1 or 2D, detected as a slice
array[1:2, 1:2]  --> array, 2D, 1D, scalar , detected as tuple of slices
array[[0, 0]]    --> array, N-D each element of idx is treated as py::int_ so
                     it's a stack of selected rows

This is super hard to override in a consistent way so I didn't. I punted it all
to python. I don't know how big of a penatly that incurs, if any at all, but it
does prevent __setitem__ from working because we return a copy. This might not
be the worst thing though?
""")
print(iimg[0, 0], iimg[0:1])
print(iimg[:0])
print(iimg[0], iimg[[0]], iimg[:, 0])
print(iimg[[0, 1]], type(iimg[[0, 1]]), type(iimg[0]))
print()

print(f"    Timing instantiation for {n_instantiation} repetitions.")
bigArrT = timeit.timeit(stmt="core.DoubleImage(bigArr)", globals=globals(), number=n_instantiation)
big32ArrT = timeit.timeit(stmt="core.Int32Image(big32Arr)", globals=globals(), number=n_instantiation)
print(f"From double: {bigArrT/n_instantiation:>10.7} seconds per iteration; {bigArrT} seconds total.")
print(f"From  float: {big32ArrT/n_instantiation:>10.7} seconds per iteration; {big32ArrT} seconds total.")
print()


print("Comparisons")
print(iimg == iimg)
print(list == iimg)
print(iimg == list)
print(iimg == dimg)
print(dimg == arrdouble)  # Interestingly, this happens inside Image, but this:
print(arrdouble == dimg)  # goes to NumPy. Put a print in operator method to see
print()                   # I guess if we want to target it we need to define .def(array == self)


print("Set array[0, 0] = 8, check changes to all:")
arrdouble[0, 0] = 8
print(arrdouble)
print(arrdouble)
print()

print("Method overloading (arguments don't do anything in this example)")
print(fimg.addOne(1))
print(fimg.addOne(1.0))
print()


# I'll be honest I'm not sure why this works given all the online discussions on
# what are the appropriate ways to steal a pointer, see for example:
# https://github.com/pybind/pybind11/issues/3126
# https://github.com/pybind/pybind11/issues/3126
# https://stackoverflow.com/questions/54876346/pybind11-and-stdvector-how-to-free-data-using-capsules
# Which are all discussions on various into and outof Py/C++ ownership transfers
# I must have lucked out with one of the
# https://github.com/pybind/pybind11/blob/master/include/pybind11/numpy.h#L1020
# constructors or something. I think I'm calling the L1059 - but unclear how a
# default "handle"  prevents deletion (I guess it automatically increments a PyRef?)
print("Delete array, then access image. No error expected.")
 # we have to garbage collect in case the object remains alive, just unnamed
import gc
del arrdouble
print(gc.collect(), gc.get_count())
print(dimg)
print(gc.is_tracked(dimg)) # doesn't mean what you think it does
print(gc.garbage)
print()

