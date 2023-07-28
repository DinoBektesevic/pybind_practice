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


# So this would have to be hidden by some lookup because certainly
# we wouldn't want the users to have to figure out what type they need
#test = core.DI32Layered(arrdouble, arrdouble, arrint.astype(np.int32))
test = core.DILayered(arrdouble, arrdouble, arrint)
print(test)
print(repr(test))
print()

print("readwrite", test.sci)
print()


# LOOK AT THIS! ALL NUMPY INDEXING WORKS FLAWLESSLY!
# too bad it's a method - at least untill I figure out
# how to bind it to an more of an attribute/property of a class.
# The issue, however, lies that to do that I need to be able to
# use .def_property(name, attr, getter, setter) and it's not
# trivial to access the getters and setters from Eigen without
# having to bind the whole class and all its overrides
print(test.mask)
test.mask[:, 0] = 10
print(test.mask)
print()

# look, it even force-casts correctly!
test.sci[0] = [27, 28]
print(test.sci)
