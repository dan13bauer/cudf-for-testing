# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(timestamp_reader)

project(
  timestamp_reader
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

add_executable(timestamp_reader timestamp_reader.cpp)
target_link_libraries(timestamp_reader PRIVATE cudf::cudf)
target_compile_features(timestamp_reader PRIVATE cxx_std_20)
target_compile_definitions(
  timestamp_reader PRIVATE TIMESTAMP_READER_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
install(TARGETS timestamp_reader DESTINATION bin/examples/libcudf/timestamp_reader)
