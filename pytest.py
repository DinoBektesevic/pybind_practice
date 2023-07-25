import timeit
import numpy as np

import core


n_instantiation, n_access = 500, 100000
arrint = np.array([[1, 2], [3, 4]])
arrdouble = np.array([[1.0, 2.0], [3.0, 4.0]])
arrfloat = np.array([[1, 2], [3, 4]], dtype=np.float32)
listint = [[1, 2], [3, 4]]
bigArr = np.zeros((2000, 4000))
big32Arr = np.zeros((2000, 4000), dtype=np.int32)


print("\n")
print(" "*30 + "Current RawImage")
print()
print(f"    Instantiation timing ({n_instantiation} repetitions).")
bigArrT = timeit.timeit(stmt="core.current_raw_image(bigArr)", globals=globals(), number=n_instantiation)
big32ArrT = timeit.timeit(stmt="core.current_raw_image(big32Arr)", globals=globals(), number=n_instantiation)
print(f"From double: {bigArrT/n_instantiation:>10.7} seconds per iteration; {bigArrT} seconds total.")
print(f"From  float: {big32ArrT/n_instantiation:>10.7} seconds per iteration; {big32ArrT} seconds total.")
print()
print("    Pixel access timings and behaviour:")
curRawImgArr = core.current_raw_image(arrdouble)
curRawImgList = core.current_raw_image(listint)
print(f"fromNpArray.get_all_pixels() \n {curRawImgArr.get_all_pixels()}")
print(f"fromList.current_raw_image(list) \n {curRawImgList.get_all_pixels()}")
bigArrT = timeit.timeit(stmt="curRawImgArr.get_all_pixels()", globals=globals(), number=n_access)
print(f"getPixels: {bigArrT/n_access:>10.7} seconds per iteration; {bigArrT} seconds total.")
del curRawImgArr
del curRawImgList


print("\n")


print(" "*30 + "Better RawImage")
print()
print(f"    Instantiation timing ({n_instantiation} repetitions)")
bigArrT = timeit.timeit(stmt="core.BetterRawImage(bigArr)", globals=globals(), number=n_instantiation)
big32ArrT = timeit.timeit(stmt="core.BetterRawImage(big32Arr)", globals=globals(), number=n_instantiation)
print(f"From double: {bigArrT/n_instantiation:>10.7} seconds per iteration; {bigArrT} seconds total.")
print(f"From  float: {big32ArrT/n_instantiation:>10.7} seconds per iteration; {big32ArrT} seconds total.")
print()
print("    Pixel access timings and behaviour:")
curRawImgArr = core.BetterRawImage(arrfloat)
curRawImgList = core.BetterRawImage(listint)
print(f"Casting in Python: np.array(array, copy=False) \n {np.array(curRawImgArr, copy=False)}")
print(f"Casting from  C++: array.getPixels() \n {curRawImgArr.getPixels()}")
getPixT = timeit.timeit(stmt="curRawImgArr.getPixels()", globals=globals(), number=n_access)
getArrT = timeit.timeit(stmt="np.array(curRawImgArr, copy=False)[:]", globals=globals(), number=n_access)
print(f"Casting in Python: {getArrT/n_access:>10.7} seconds per iteration; {getArrT} seconds total.")
print(f"Casting from  C++: {getPixT/n_access:>10.7} seconds per iteration; {getPixT} seconds total.")
print()
print("    Just to quickly model how to work with this object (see code):")
print(f"Add 1.0 to every element and return a copy     `array.copyAddOne()` = \n {curRawImgArr.copyAddOne()}")
curRawImgArr.inplaceAddOne()
print(f"Add 1.0 to every element without a copy `array.inplacecopyAddOne()` = \n {curRawImgArr.getPixels()}")
del curRawImgArr
del curRawImgList


print("\n")


print(" "*30 + "Experimental")
print("""
The following demos different ideas of wrapping and integrating a C++ class
with Python. Two features we want to have are the performance, i.e. minimize
copy operations, and more idiomatic interface both in C++ and Python. Two
critical aspects that we need to have are:
1) be interchangeable by a simple numpy array
2) it has to extend the numpy array interface to enable us to process it on GPU
   as an unravelled array.
In practice this means  copious overrides of various operators and self
referential-ity.
""")


print(" "*15 + "Home-brewed")
print("Check out branch idiomatic for a better C++ API, but it doesn't work.")
print("\n")

# We would have to have a delegator that would instantiate the correct C++
# class depending on the given type of the container. The way to check for the
# type differs based on the container and if the types are not well matched the
# CPP code will make a copy (the invoked constructor will be a copy, not a move
# constructor). Not the worst, because templating requires us to specify a
# whole bunch of type-bound classes so having them all under one roof for
# convenience is neat.
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

print(f"    Instantiation timing ({n_instantiation} repetitions).")
bigArrT = timeit.timeit(stmt="core.DoubleImage(bigArr)", globals=globals(), number=n_instantiation)
big32ArrT = timeit.timeit(stmt="core.Int32Image(big32Arr)", globals=globals(), number=n_instantiation)
print(f"From double: {bigArrT/n_instantiation:>10.7} seconds per iteration; {bigArrT} seconds total.")
print(f"From  float: {big32ArrT/n_instantiation:>10.7} seconds per iteration; {big32ArrT} seconds total.")
print()
print("    Pixel access timings and behaviour:")
getPixT = timeit.timeit(stmt="dimg[:]", globals=globals(), number=n_access)
getPixTi = timeit.timeit(stmt="dimg.get(0)", globals=globals(), number=n_access)
getPixTij = timeit.timeit(stmt="dimg.get(0, 0)", globals=globals(), number=n_access)
print(f"__getitem__[:]: {getPixT/n_access:>10.7} seconds per iteration; {getPixT} seconds total.")
print(f"img.get(0)    : {getPixTi/n_access:>10.7} seconds per iteration; {getPixTi} seconds total.")
print(f"img.get(0, 0) : {getPixTij/n_access:>10.7} seconds per iteration; {getPixTij} seconds total.")
print()

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

print("Item access.")
print("""Half of the access operator in Np is figured out in Python, half in C
(not really, most is C but there's some in Py) as far as I can read the code.
All of the following is allowed by numpy:

array[0, 0]      --> scalar, detected as tuple of py::int_
array[0][0]      --> scalar, detected as py::int_
array[0]         --> array, 1D for us, detected as py::int_
array[:2]        --> array, 1 or 2D, detected as a slice
array[1:2, 1:2]  --> array, 2D, 1D, scalar , detected as tuple of slices
array[[0, 0]]    --> array, N-D each element of idx is treated as py::int_ so
                     it's a stack of selected rows

This is super hard to override in a consistent way so I didn't. I punted it all
to python. So item access is pretty good, but it does prevent __setitem__ from
working because we return a copy. This might not be the worst thing though?
""")
print(iimg[0, 0], iimg[0:1])
print(iimg[:0])
print(iimg[0], iimg[[0]], iimg[:, 0])
print(iimg[[0, 1]], type(iimg[[0, 1]]), type(iimg[0]))
print()


print("Comparisons")
print(iimg == iimg)
print(listint == iimg)
print(iimg == listint)
print(iimg == dimg)
print(dimg == arrdouble)  # Interestingly, this happens inside Image, but this:
print(arrdouble == dimg)  # goes to NumPy. Put a print in operator method to see
print()                   # I guess if we want to target it we need to define .def(array == self)


print("Set array[0, 0] = 8, check changes to all:")
arrdouble[0, 0] = 8
print(arrdouble)
print(arrdouble)
print()

print("Usage (see code)")
print(fimg.addOne())
print()
# I'll be honest I'm not sure why this works given all the online discussions on
# what are the appropriate ways to steal a pointer, see for example:
# https://github.com/pybind/pybind11/issues/3126
# https://github.com/pybind/pybind11/issues/3126
# https://stackoverflow.com/questions/54876346/pybind11-and-stdvector-how-to-free-data-using-capsules
# I must have lucked out with one of the
# https://github.com/pybind/pybind11/blob/master/include/pybind11/numpy.h#L1020
# constructors or something. I think I'm calling the L1059 - but unclear how a
# default "handle"  prevents deletion (I guess it auto-increments a PyRef?)
# Yep, I think it's this:
# https://pybind11.readthedocs.io/en/stable/upgrade.html?highlight=pybind11%3A%3Aself#stricter-compile-time-error-checking
print("Delete array, then access image. No error expected.")
 # we have to garbage collect in case the object remains alive
import gc
del arrdouble
print(gc.collect(), gc.get_count())
print(dimg)
print(gc.is_tracked(dimg)) # doesn't mean what you think it does
print(gc.garbage)
print()

