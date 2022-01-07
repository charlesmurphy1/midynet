cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(SamplableSet)
set(CMAKE_CXX_STANDARD 11)

add_library(samplableset
    BinaryTree.cpp
    HashPropensity.cpp
    SamplableSet.cpp
)

set_target_properties(samplableset PROPERTIES POSITION_INDEPENDENT_CODE TRUE)  # Required for pybind11 build with CMake
