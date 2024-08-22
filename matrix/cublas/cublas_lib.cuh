#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"
using data_type = float;




#ifndef cub_axpy
#define cub_axpy
void cublas_saxpy(float *d_A,float *d_B,float alpha, int size,cublasHandle_t cublasH,cudaStream_t stream = NULL){
    const int incx = 1;
    const int incy = 1;

    /* step 3: compute */
    CUBLAS_CHECK(cublasSaxpy(cublasH, size, &alpha, d_A, incx, d_B, incy));
    cudaDeviceSynchronize();

    return;

}


void cublas_gemm_rowmajor(
    cublasHandle_t *cublashandler, float *d_A, float *d_B, float *d_C, int rowA, int colA,
    int rowB, int colB, float alpha, float beta){
    cublasSgemm(*cublashandler, CUBLAS_OP_N, CUBLAS_OP_N, colB, rowA, rowB,
                &alpha, d_B, colB, d_A, rowB, &beta, d_C, colB);  
    cudaDeviceSynchronize();
}

void cublas_gemv_rowmajor(
    cublasHandle_t *cublashandler, float *d_A, float *d_x,
     float *d_y, int rowA, int colA, float alpha, float beta){
    cublasSgemm(*cublashandler, CUBLAS_OP_N, CUBLAS_OP_N, 1, rowA, colA,
                    &alpha, d_x, 1, d_A, colA, &beta, d_y, 1);  
    cudaDeviceSynchronize();

}

void cublas_gemv_rowmajor_trans(
    cublasHandle_t *cublashandler, float *d_A, float *d_x,
     float *d_y, int rowA, int colA, float alpha, float beta){
    cublasSgemv(*cublashandler, CUBLAS_OP_N, colA, rowA, 
                &alpha, d_A, colA, d_x, 1, &beta, d_y, 1);  
    cudaDeviceSynchronize();
}

#endif