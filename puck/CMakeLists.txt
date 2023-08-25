include_directories(../)
file(
    GLOB PUCK_CPP_FILES
    "./base/*.cpp"
    "./gflags/*.cpp"
    "./hierarchical_cluster/*.cpp"
    "./puck/*.cpp"
    "./tinker/*.cpp"
    "./tinker/*/*.cc"
    "./*.cpp"
)
file(
    GLOB PUCK_HEAD_FILES
    "./base/*.h"
    "./gflags/*.h"
    "./hierarchical_cluster/*.h"
    "./puck/*.h"
    "./tinker/*.h"
    "./tinker/*/*.h"
    "./*.h"
)

add_library(puck ${PUCK_CPP_FILES})

target_include_directories(puck PUBLIC
$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

set_target_properties(puck PROPERTIES
POSITION_INDEPENDENT_CODE ON
WINDOWS_EXPORT_ALL_SYMBOLS ON
)
target_compile_definitions(puck PUBLIC FINTEGER=int)

target_link_libraries(puck PUBLIC gflags)
target_link_libraries(puck PUBLIC glog)
    
target_link_libraries(puck PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(puck PUBLIC ${MKL_LIBRARIES})


set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include)

install(TARGETS puck
ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

foreach(header ${PUCK_HEAD_FILES})
    get_filename_component(dir ${header} DIRECTORY)
    file(RELATIVE_PATH sub_dir ${PROJECT_SOURCE_DIR} ${dir})
    install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${sub_dir}
)
endforeach()
