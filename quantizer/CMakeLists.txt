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

set(QUANTIZER_SOURCES
    FLAT.cpp
    LVQ.cpp
    PQ.cpp
)

set(QUANTIZER_PUBLIC_HEADERS
    FLAT.h
    LVQ.h
    PQ.h
    factory.h
    quantizer.h
)

add_library(quantizer OBJECT ${QUANTIZER_SOURCES})

target_include_directories(quantizer PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(quantizer PRIVATE
    glog
    OpenMP::OpenMP_CXX
    Threads::Threads
    ${MKL_LIBRARIES}
)

set_target_properties(quantizer PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(FILES ${QUANTIZER_PUBLIC_HEADERS}
    DESTINATION include/quantizer
)
