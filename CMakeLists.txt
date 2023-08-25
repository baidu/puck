cmake_minimum_required(VERSION 3.21)

project(puck)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_BINARY_DIR}/third_libs)

# config GIT_URL with github mirrors to speed up dependent repos clone
option(GIT_URL "Git URL to clone dependent repos" ${GIT_URL})
if(NOT GIT_URL)
    set(GIT_URL "https://github.com")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11 -w -g0 -Ofast -fPIC")
set(CMAKE_C_FLAGS "-std=c++11 -w -g0 -Ofast -fPIC")
#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-DFINTEGER=int)

add_compile_options(-fopenmp -lpthread -lcrypto -lrt -lgfortran -ldl -lz -fPIC -rdynamic)
add_compile_options(-Ofast -g -pipe -fPIC -march=native -w)

include(${PROJECT_SOURCE_DIR}/cmake/gflags.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/glog.cmake)

find_package(OpenMP REQUIRED)
find_package(MKL REQUIRED)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/output)
add_subdirectory(puck)
add_subdirectory(demo)
add_subdirectory(tools)

# 是否编译gtest
option (WITH_TESTING "Use gtest" OFF)
if (WITH_TESTING)
    add_subdirectory(test)
endif (WITH_TESTING)

# 是否编译python接口
option (USE_PYTHON "Use python of puck" OFF)
if (USE_PYTHON)
    # minimum requirements
    set(PYTHON_MINIMUM_VERSION 3.6)

    # find python3
    find_package(Python3 ${PYTHON_MINIMUM_VERSION} REQUIRED COMPONENTS Interpreter Development)
    message(STATUS "Found Python: ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}.${Python3_VERSION_PATCH}")
    if (NOT Python3_SITELIB)
        message(FATAL_ERROR "site-packages not found. ")
    else ()
        message(STATUS "site-packages: ${Python3_SITELIB}")
    endif ()
    add_subdirectory(pyapi_wrapper)
endif (USE_PYTHON)
#


