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
        # because this changes system to system
        # https://en.cppreference.com/w/cpp/language/types
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#scalars
        if dtype == "float64":
            return core.DoubleImage
        elif dtype == "float32":
            return core.FloatImage
        elif dtype == "int64": # This is wrong but nvm
            return core.IntImage
        #elif dtype == "int64":
        #    return core.LongIntImage

    @classmethod
    def fromArray(cls, arr):
        cls = cls.__get_cpp_class(arr.dtype.name)
        return cls(arr)


iimg = Image.fromArray(arrint)  # good for masks?
fimg = Image.fromArray(arrfloat)
dimg = Image.fromArray(arrdouble)


print(f"    Instantiation timing ({n_instantiation} repetitions).")
bigArrT = timeit.timeit(stmt="core.DoubleImage(bigArr)", globals=globals(), number=n_instantiation)
big32ArrT = timeit.timeit(stmt="core.IntImage(big32Arr)", globals=globals(), number=n_instantiation)
print(f"From double: {bigArrT/n_instantiation:>10.7} seconds per iteration; {bigArrT} seconds total.")
print(f"From  float: {big32ArrT/n_instantiation:>10.7} seconds per iteration; {big32ArrT} seconds total.")
print()
print("    Pixel access timings and behaviour:")
getPixT = timeit.timeit(stmt="dimg.data[:]", globals=globals(), number=n_access)
print(f"__getitem__[:]: {getPixT/n_access:>10.7} seconds per iteration; {getPixT} seconds total.")
print()
