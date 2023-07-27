cmake_minimum_required(VERSION 3.9...3.17)
project(core LANGUAGES CXX)
#set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

find_package(Python ${PYTHON_VERSION} COMPONENTS Interpreter Development NumPy REQUIRED)

add_subdirectory(lib/pybind11)

pybind11_add_module(core SHARED
    bindings.cpp
)

set_target_properties(core PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_include_directories(core PRIVATE
    lib/eigen #3.4.1 release
)

#target_compile_options(core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:
#    -flto=auto
#    -O0
#    -g
#>)

# hold on to your hat!
# This will make the casting to wrong types faster than "better"
# raw image instantiation is when type-matched correctly
# set(CMAKE_CXX_STANDARD 17)
target_compile_options(core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:
    -flto=auto
    -O3
    -march=native
>)

