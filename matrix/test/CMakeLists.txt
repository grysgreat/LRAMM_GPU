# cublas_gemm_benchmark
add_executable(operator_matrix_test
    ${CMAKE_CURRENT_SOURCE_DIR}/operator_matrix_test.cu
)
add_executable(cublas_lib_test
    ${CMAKE_CURRENT_SOURCE_DIR}/cublas_lib_test.cu
)
add_executable(cutlass_gemm_test
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass_gemm_test.cu
)

add_executable(cusolver_test
    ${CMAKE_CURRENT_SOURCE_DIR}/cusolver_test.cu
)

add_executable(lrxigemm_test
    ${CMAKE_CURRENT_SOURCE_DIR}/lrxigemm_test.cu
)
add_executable(sketch_op_test
    ${CMAKE_CURRENT_SOURCE_DIR}/sketch_op_test.cu
)
target_link_libraries(operator_matrix_test -lcublas -lcurand -fopenmp)
target_link_libraries(cublas_lib_test -lcublas -lcurand -fopenmp)
target_link_libraries(cutlass_gemm_test -lcublas -lcurand -fopenmp)
target_link_libraries(cusolver_test -lcusolver -lcudart -lcurand -lcublas -fopenmp)
target_link_libraries(lrxigemm_test -lcusolver -lcudart -lcurand -lcublas -fopenmp)
target_link_libraries(sketch_op_test -lcusolver -lcudart -lcurand -lcublas -fopenmp)

include_directories(${SOURCE_DIR}/matrix)

target_link_libraries(operator_matrix_test
    matrix
)

target_link_libraries(cublas_lib_test
    matrix
)

target_link_libraries(cutlass_gemm_test
    matrix
)
target_link_libraries(cusolver_test
    matrix
)
target_link_libraries(lrxigemm_test
    matrix
)
target_link_libraries(sketch_op_test
    matrix
)