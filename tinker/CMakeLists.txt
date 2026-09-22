# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(TINKER_SOURCES
    internal/distances.cpp
    tinker_index.cpp
)

set(TINKER_PUBLIC_HEADERS
    internal/distances.h
    internal/utils.h
    internal/visited_list.h
    tinker_index.h
)

add_library(tinker OBJECT ${TINKER_SOURCES})

target_include_directories(tinker PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(tinker PRIVATE
    gflags
    glog
    OpenMP::OpenMP_CXX
    Threads::Threads
)

set_target_properties(tinker PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(FILES tinker_index.h DESTINATION include/tinker)
install(FILES
    internal/distances.h
    internal/utils.h
    internal/visited_list.h
    DESTINATION include/tinker/internal
)
