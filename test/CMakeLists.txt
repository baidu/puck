include_directories(../)
file(
    GLOB GTEST_CPP_FILES
    "test_params.cpp"
    "test_index.cpp"
    "test_index_type.cpp"
    "test_index_recall.cpp"
    "test_main.cpp"
    "test_tinker.cpp"
)

add_executable(runtest)

target_sources(runtest PUBLIC ${GTEST_CPP_FILES})
target_sources(runtest PRIVATE "${PROJECT_SOURCE_DIR}/pyapi_wrapper/py_api_wrapper.cpp")

set(PUCK_TEST_DATA_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/data/siftsmall.tar.gz")
if(NOT EXISTS "${PUCK_TEST_DATA_ARCHIVE}")
    message(FATAL_ERROR "Puck test data archive not found: ${PUCK_TEST_DATA_ARCHIVE}")
endif()
set(PUCK_TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(REMOVE_RECURSE "${PUCK_TEST_DATA_DIR}")
file(MAKE_DIRECTORY "${PUCK_TEST_DATA_DIR}")
file(ARCHIVE_EXTRACT
    INPUT "${PUCK_TEST_DATA_ARCHIVE}"
    DESTINATION "${PUCK_TEST_DATA_DIR}"
)
target_compile_definitions(runtest PRIVATE
    PUCK_TEST_DATA_ARCHIVE="${PUCK_TEST_DATA_ARCHIVE}"
    PUCK_TEST_DATA_DIR="${PUCK_TEST_DATA_DIR}/siftsmall"
)

include(FetchContent)
FetchContent_Declare(googletest
  URL "https://github.com/google/googletest/archive/release-1.10.0.tar.gz")
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

target_link_libraries(runtest PUBLIC
    puck
    gtest
)

add_test(NAME runtest COMMAND runtest)
set_tests_properties(runtest PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

set(GTEST_TOOLS_PREFIX ${CMAKE_INSTALL_PREFIX}/test)
install(TARGETS runtest  DESTINATION ${GTEST_TOOLS_PREFIX})
