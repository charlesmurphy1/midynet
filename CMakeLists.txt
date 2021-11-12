cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(FastMIDyNet)
set(CMAKE_CXX_STANDARD 11)

option(BUILD_TESTS "build gtest unit tests" ON)

include_directories(include)
include_directories(base_graph/include)
set(BASEGRAPH ${CURRENT_CMAKE_SRC_DIRECTORY}/base_graph/build/libBaseGraph.a)

add_subdirectory(src)


if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
