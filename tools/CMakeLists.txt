include_directories(../)
file(
    GLOB PUCK_CPP_FILES
    "../puck/base/*.cpp"
    "../puck/gflags/*.cpp"
    "../puck/hierarchical_cluster/*.cpp"
    "../puck/puck/*.cpp"
    "../puck/tinker/*.cpp"
    "../puck/tinker/*/*.cc"
    "../puck/*.cpp"
)

add_executable(train)
target_sources(train PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/train.cpp)
target_sources(train PUBLIC ${PUCK_CPP_FILES})
target_compile_definitions(train PUBLIC)

target_link_libraries(train gflags)
target_link_libraries(train glog)

target_link_libraries(train  OpenMP::OpenMP_CXX)
target_link_libraries(train  ${MKL_LIBRARIES})

add_executable(build)
target_sources(build PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/build.cpp)
target_sources(build PUBLIC ${PUCK_CPP_FILES})
target_compile_definitions(build PUBLIC)
target_link_libraries(build  OpenMP::OpenMP_CXX)
target_link_libraries(build  ${MKL_LIBRARIES})

target_link_libraries(build gflags)
target_link_libraries(build glog) 

set(BUILD_TOOLS_PREFIX ${CMAKE_INSTALL_PREFIX}/build_tools)
install(TARGETS train build 
RUNTIME DESTINATION ${BUILD_TOOLS_PREFIX}/bin
)

install(TARGETS train build  DESTINATION ${BUILD_TOOLS_PREFIX}/bin
)
install(DIRECTORY "script"  DESTINATION ${BUILD_TOOLS_PREFIX}/
)
install(DIRECTORY "demo/conf"  DESTINATION ${BUILD_TOOLS_PREFIX}/
)
