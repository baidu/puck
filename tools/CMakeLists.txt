include_directories(../)

add_executable(train ${CMAKE_CURRENT_SOURCE_DIR}/train.cpp)
target_compile_definitions(train PUBLIC)
target_link_libraries(train puck)

add_executable(build ${CMAKE_CURRENT_SOURCE_DIR}/build.cpp)
target_compile_definitions(build PUBLIC)
target_link_libraries(build puck)

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
