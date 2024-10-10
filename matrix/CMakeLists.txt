list(APPEND objects
gen_matrix.cu
operator_matrix.cu
cutlass_gemm_op.cu
get_opsPerf.cu
)


add_library(
    matrix
    OBJECT
    ${objects}
)

add_subdirectory(test)