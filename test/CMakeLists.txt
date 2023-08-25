include_directories(../)
file(
    GLOB GTEST_CPP_FILES
    "test_params.cpp"
    "test_index.cpp"
    "test_index_recall.cpp"
)

add_executable(runtest)

target_sources(runtest PUBLIC ${GTEST_CPP_FILES})

include(FetchContent)
FetchContent_Declare(googletest
  URL "https://github.com/google/googletest/archive/release-1.10.0.tar.gz")
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

target_link_libraries(runtest PUBLIC
    puck
    gtest_main
)

gtest_discover_tests(runtest)

set(GTEST_TOOLS_PREFIX ${CMAKE_INSTALL_PREFIX}/test)
install(TARGETS runtest  DESTINATION ${GTEST_TOOLS_PREFIX})
