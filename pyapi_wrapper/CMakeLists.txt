# SWIG compile for Python(3)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}" EXACT REQUIRED)
message(STATUS "Found PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Found PYTHON_LIBRARY: ${PYTHON_LIBRARY}")
message(STATUS "Found CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_SWIG_FLAGS "-relativeimport")

set(SWIG_INTERFACE "./py_puck_api.i")
set(SWIG_USED_CPP "./py_api_wrapper.cpp")

set_source_files_properties(${SWIG_INTERFACE} PROPERTIES CPLUSPLUS ON)
set_source_files_properties(${SWIG_INTERFACE} PROPERTIES SWIG_FLAGS "-includeall")

include_directories(./
                    ../
                    ${MKLROOT}
                    ${PYTHON_INCLUDE_DIR})

# add     py_puck_api.cpp ->  py_puck_api.py _PyPuck.so    in a row
swig_add_library(   PyPuck
                    LANGUAGE python
                    SOURCES ${SWIG_INTERFACE} ${SWIG_USED_CPP}
                    OUTFILE_DIR ./py_puck_api_cpp
                    OUTPUT_DIR ./)

# link all the libraries
swig_link_libraries(PyPuck puck ${PYTHON_LIBRARY})
swig_link_libraries(PyPuck  OpenMP::OpenMP_CXX)
swig_link_libraries(PyPuck  ${MKL_LIBRARIES})

message(STATUS "Found CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
message(STATUS "Found CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
add_custom_command(TARGET PyPuck POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_BINARY_DIR}/py_puck_api.py"
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/py_puck_api.py")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/py_puck_api.py"
    DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

install(FILES
    "${PROJECT_SOURCE_DIR}/baidu_puck/__init__.py"
    "${PROJECT_SOURCE_DIR}/baidu_puck/_index.py"
    "${PROJECT_SOURCE_DIR}/baidu_puck/hierarchical_cluster.py"
    "${PROJECT_SOURCE_DIR}/baidu_puck/puck.py"
    "${PROJECT_SOURCE_DIR}/baidu_puck/tinker.py"
    DESTINATION ${CMAKE_INSTALL_PREFIX}/baidu_puck)

install(TARGETS PyPuck
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/baidu_puck
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/baidu_puck
    )
