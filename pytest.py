import timeit
import numpy as np

import core

n_instantiation, n_access = 500, 100000
arrint = np.array([[1, 2], [3, 4]])
arrdouble = np.array([[1.0, 2.0], [3.0, 4.0]])
arrfloat = np.array([[1, 2], [3, 4]], dtype=np.float32)
listint = [[1, 2], [3, 4]]
listdouble = [[1.0, 2.0], [3.0, 4.0]]
bigArr = np.zeros((2000, 4000), dtype=float)
big32Arr = np.zeros((2000, 4000), dtype=np.int32)


print(" "*30 + "Eigen MatrixWrapper")
print()
print(f"    Instantiation timing ({n_instantiation} repetitions)")
bigArrT = timeit.timeit(stmt="core.DoubleImage(arrdouble)", globals=globals(), number=n_instantiation)
print(f"From double: {bigArrT/n_instantiation:>10.7} seconds per iteration; {bigArrT} seconds total.")
print()
print("    Pixel access timings and behaviour:")
curRawImgArr = core.DoubleImage(arrdouble)
print(f"Casting in Python: np.array(array.data, copy=False) \n {np.array(curRawImgArr.data, copy=False)}")
print(f"Casting from  C++: array.data \n {curRawImgArr.data}")
getPixT = timeit.timeit(stmt="curRawImgArr.data[:]", globals=globals(), number=n_access)
print(f"__getitem__[:]: {getArrT/n_access:>10.7} seconds per iteration; {getArrT} seconds total.")
print()


# So this would have to be hidden by some lookup because certainly
# we wouldn't want the users to have to figure out what type they need
test = core.DILayered(arrdouble, arrdouble, arrint)
print(test)
print(test.sci)
print()


# LOOK AT THIS! ALL NUMPY INDEXING WORKS FLAWLESSLY!
print("Mask (int) before")
print(test.mask)
print("and after setting arr[:, 0]=10")
test.mask[:, 0] = 10
print(test.mask)
print()

# look, it even force-casts correctly!
print("Science (dobule) before")
print(test.sci)
print("and after setting arr[0] = [27, 28]")
test.sci[0] = [27, 28]
print(test.sci)
