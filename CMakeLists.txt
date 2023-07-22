cmake_minimum_required(VERSION 3.9...3.17)
project(core LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
find_package(Python ${PYTHON_VERSION} COMPONENTS Interpreter Development NumPy REQUIRED)
add_subdirectory(lib/pybind11)

pybind11_add_module(core SHARED
    bindings.cpp
)

set_target_properties(core PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_compile_options(core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:
    -g3
>)

pybind11_add_module(test SHARED cpptest.cpp bindings.cpp image.cpp)
#target_link_libraries(test PUBLIC pybind11::module pybind11::pybind11 pybind11::embed)
#add_executable(test.exe cpptest.cpp)

# hecking heck, I can't link against a pybind11 module
# so basically I have to declare an independent library here
# and compile it all again?
#include_directories(lib/pybind11)
#add_library(cpptest MODULE cpptest.cpp)
#target_include_directories(cpptest PUBLIC lib/pybind11 lib/pybind11/include/pybind11)
#target_link_directories(cpptest PUBLIC lib/pybind11 lib/pybind11/include/pybind11)
#target_link_libraries(cpptest PUBLIC pybind11::module pybind11::pybind11 lib/pybind11 lib/pybind11/include/pybind11)
#add_executable(cpptestexe cpptest.cpp)
