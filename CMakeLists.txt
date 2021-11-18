cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(FastMIDyNet)

option(BUILD_TESTS "build gtest unit tests" ON)

include_directories(include)
include_directories(base_graph/include)
include_directories(SamplableSet/src)
set(BASEGRAPH "${CMAKE_CURRENT_SOURCE_DIR}/base_graph/build/libBaseGraph.a")
set(SAMPLABLESET "${CMAKE_CURRENT_SOURCE_DIR}/SamplableSet/src/build/libsamplableset.a")

add_subdirectory(src)


if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

set(CMAKE_CXX_STANDARD 11)