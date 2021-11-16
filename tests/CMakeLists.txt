find_package(GTest)

if (NOT ${GTEST_FOUND})
    include(FetchContent)
    FetchContent_Declare(
            googletest
            # Specify the commit you depend on and update it regularly.
            URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif ()

file(GLOB_RECURSE DYNAMICS_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_dynamics/test_*.cpp)
file(GLOB_RECURSE GENERATORS_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_generators/test_*.cpp)
file(GLOB_RECURSE UTILITY_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_utility/test_*.cpp)

add_executable(test_dynamics ${DYNAMICS_TEST_SRC})
add_executable(test_generators ${GENERATORS_TEST_SRC})
add_executable(test_utility ${UTILITY_TEST_SRC})

set(allGTests
    test_dynamics
    test_generators
    test_utility
    )


foreach(UnitTest ${allGTests})
    target_link_libraries(${UnitTest} gtest gtest_main ${BASEGRAPH} midynet)
    add_test(${UnitTest} ${UnitTest})
endforeach()
