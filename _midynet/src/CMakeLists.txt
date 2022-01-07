file(GLOB_RECURSE MIDYNET_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(midynet ${MIDYNET_SRC})


target_link_libraries(midynet ${BASEGRAPH} ${SAMPLABLESET})
set_target_properties(midynet PROPERTIES
    LINKER_LANGUAGE CXX
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
