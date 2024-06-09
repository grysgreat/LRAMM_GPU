# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(cublas_gemm_benchmark LANGUAGES CXX CUDA)

# Use C++11
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -arch=sm_75")
add_compile_options(-arch=sm_75)
add_compile_options(-O3)
add_compile_options(-w)
add_compile_options(-lcusolver)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(RCXIGEMM_BIN_NAME rcxigemm)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/cutlass/tools/util/include")
    message(FATAL_ERROR [[
        CUTLASS library not found in third-party/cutlass.
        Please fetch with git submodule update --init --recursive
        ]])
endif()

include_directories(${SOURCE_DIR})
include_directories(${SOURCE_DIR}/matrix)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
include_directories(${SOURCE_DIR}/lib/cutlass/include)
include_directories(${SOURCE_DIR}/lib/cutlass/tools/util/include)
include_directories(${SOURCE_DIR}/lib/cutlass/examples/common)


add_subdirectory(matrix)

# cublas_gemm_benchmark
add_executable(${RCXIGEMM_BIN_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/run_matrix.cu
)



target_link_libraries(${RCXIGEMM_BIN_NAME}
    matrix
)

target_link_libraries(${RCXIGEMM_BIN_NAME} -lcublas -lcurand -lcudart -lcusolver)



