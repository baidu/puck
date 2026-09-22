include_directories(..)

add_executable(search_client ${CMAKE_CURRENT_SOURCE_DIR}/search_client.cpp)
target_compile_definitions(search_client PUBLIC)
target_link_libraries(search_client puck)

add_executable(insert_demo ${CMAKE_CURRENT_SOURCE_DIR}/insert_demo.cpp)
target_compile_definitions(insert_demo PUBLIC)
target_link_libraries(insert_demo puck)

set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX}/bin)

install(TARGETS search_client insert_demo 
RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY "conf"  DESTINATION ${CMAKE_INSTALL_PREFIX}/
)
