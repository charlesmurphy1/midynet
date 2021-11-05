cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(FastMIDyNet)
set(CMAKE_CXX_STANDARD 11)

option(BUILD_TESTS "build gtest unit tests" ON)

include_directories(include)
add_subdirectory(src)

if (BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})

    add_subdirectory(test)
endif()
