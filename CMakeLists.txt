# Whitematter: lightweight C++ neural network framework
# FetchContent-friendly: add_subdirectory(whitematter) then target_link_libraries(your_app whitematter)

cmake_minimum_required(VERSION 3.14)
project(whitematter VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options (OFF by default for portability)
option(WHITEMATTER_METAL "Build with Metal backend (macOS only)" OFF)
option(WHITEMATTER_CUDA "Build with CUDA backend" OFF)

# ------------------------------------------------------------------------------
# Core library sources (match Makefile)
# ------------------------------------------------------------------------------
set(CORE_SRCS
  core/memory_pool.cpp
  core/autograd.cpp
  core/broadcast.cpp
  core/tensor.cpp
  core/layers/linear.cpp
  core/layers/activations.cpp
  core/layers/conv.cpp
  core/layers/normalization.cpp
  core/layers/embedding.cpp
  core/layers/recurrent.cpp
  core/layers/attention.cpp
  core/layers/sequential.cpp
  core/loss.cpp
  core/optimizer.cpp
  core/serialization/serialize.cpp
  core/dataloader.cpp
  core/model_zoo.cpp
  core/serialization/onnx_export.cpp
  core/serialization/onnx_import.cpp
  core/device.cpp
  core/ops/simd_ops_avx.cpp
  core/ops/simd_ops_neon.cpp
  core/ops/simd_ops_fallback.cpp
  core/ops/matmul_cpu.cpp
  core/ops/im2col.cpp
  core/ops/conv_ops.cpp
  core/ops/augmentation.cpp
)

# Metal: stub (default) or backend on macOS
if(WHITEMATTER_METAL AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  enable_language(OBJCXX)
  list(APPEND CORE_SRCS core/metal/metal_backend.mm)
  add_compile_definitions(WHITEMATTER_METAL)
  set(CMAKE_FIND_FRAMEWORK LAST)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  set(EXTRA_LIBS ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
else()
  list(APPEND CORE_SRCS core/metal/metal_stub.cpp)
endif()

# CUDA: stub (default) or backend
if(WHITEMATTER_CUDA)
  enable_language(CUDA)
  list(APPEND CORE_SRCS core/cuda/cuda_backend.cu)
  add_compile_definitions(WHITEMATTER_CUDA)
  set(EXTRA_LIBS ${EXTRA_LIBS} cudart cublas)
else()
  list(APPEND CORE_SRCS core/cuda/cuda_stub.cpp)
endif()

# ------------------------------------------------------------------------------
# Dataset sources
# ------------------------------------------------------------------------------
set(DATASET_SRCS
  datasets/mnist.cpp
  datasets/cifar10.cpp
)

# ------------------------------------------------------------------------------
# Compiler flags (portable)
# ------------------------------------------------------------------------------
add_compile_options(-O3 -Wall -Wextra -ffast-math -funroll-loops)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  if(APPLE)
    add_compile_options(-mcpu=apple-m1)
  else()
    add_compile_options(-march=armv8-a+simd)
  endif()
elseif(NOT APPLE)
  add_compile_options(-mavx -mfma)
endif()

# OpenMP (optional so Linux CI works without extra config)
set(EXTRA_LIBS "")
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set(OPENMP_FOUND TRUE)
else()
  set(OPENMP_FOUND FALSE)
endif()

# ------------------------------------------------------------------------------
# Library target (consumers link this)
# ------------------------------------------------------------------------------
add_library(whitematter STATIC
  ${CORE_SRCS}
  ${DATASET_SRCS}
)
target_include_directories(whitematter PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/core
  ${CMAKE_CURRENT_SOURCE_DIR}/core/ops
  ${CMAKE_CURRENT_SOURCE_DIR}/datasets
)
if(WHITEMATTER_METAL AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  target_include_directories(whitematter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/core/metal)
endif()
if(WHITEMATTER_CUDA)
  target_include_directories(whitematter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/core/cuda)
endif()
if(OPENMP_FOUND)
  target_link_libraries(whitematter PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(whitematter PUBLIC ${EXTRA_LIBS})

# Export for FetchContent consumers
include(GNUInstallDirs)
install(TARGETS whitematter
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY core/ core/metal core/cuda
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/whitematter
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
install(DIRECTORY datasets/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/whitematter
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
