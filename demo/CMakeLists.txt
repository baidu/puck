include_directories(..)
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

add_executable(search_client)
target_sources(search_client PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/search_client.cpp)
target_sources(search_client PUBLIC ${PUCK_CPP_FILES})
target_compile_definitions(search_client PUBLIC)


target_link_libraries(search_client  OpenMP::OpenMP_CXX)
target_link_libraries(search_client  ${MKL_LIBRARIES})
target_link_libraries(search_client gflags)
target_link_libraries(search_client glog)

add_executable(insert_demo)
target_sources(insert_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/insert_demo.cpp)
target_sources(insert_demo PUBLIC ${PUCK_CPP_FILES})
target_compile_definitions(insert_demo PUBLIC)

target_link_libraries(insert_demo  OpenMP::OpenMP_CXX)
target_link_libraries(insert_demo  ${MKL_LIBRARIES})
target_link_libraries(insert_demo gflags)
target_link_libraries(insert_demo glog)

set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX}/bin)

install(TARGETS search_client insert_demo 
RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY "conf"  DESTINATION ${CMAKE_INSTALL_PREFIX}/
)
