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

void cublas_gemm_rowmajor(
    cublasHandle_t *cublashandler, half *d_A, half *d_B, half *d_C, int rowA, int colA,
    int rowB, int colB, float alpha, float beta){

    CUBLAS_CHECK(cublasGemmEx(*cublashandler, CUBLAS_OP_N, CUBLAS_OP_N, 
                    colB, rowA, rowB, &alpha,
                    d_B,CUDA_R_16F, colB, 
                    d_A, CUDA_R_16F, rowB, &beta, 
                    d_C, CUDA_R_16F, colB, 
                    CUBLAS_COMPUTE_32F  , 
                    CUBLAS_GEMM_ALGO0_TENSOR_OP 
                ));


    cudaDeviceSynchronize();
}

void cublas_gemm_rowmajor(
    cublasHandle_t *cublashandler, half *d_A, half *d_B, half *d_C, int rowA, int colA,
    int rowB, int colB, half alpha, half beta){



    CUBLAS_CHECK(cublasGemmEx(*cublashandler, CUBLAS_OP_N, CUBLAS_OP_N, 
                    colB, rowA, rowB, &alpha,
                    d_B,CUDA_R_16F, colB, 
                    d_A, CUDA_R_16F, rowB, &beta, 
                    d_C, CUDA_R_16F, colB, 
                    CUBLAS_COMPUTE_16F , 
                    CUBLAS_GEMM_ALGO0_TENSOR_OP 
                ));


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

void cublas_gemv_rowmajor_ld(
    cublasHandle_t *cublashandler, float *d_A, float *d_x,
     float *d_y, int rowA, int colA, float alpha, float beta,
     int lda, int ldx, int ldy){
    cublasSgemm(*cublashandler, CUBLAS_OP_N, CUBLAS_OP_N, 1, rowA, colA,
                    &alpha, d_x, ldx, d_A, lda, &beta, d_y, ldy);  
    cudaDeviceSynchronize();

}

void cublas_gemv_rowmajor_trans_ld(
    cublasHandle_t *cublashandler, float *d_A, float *d_x,
     float *d_y, int rowA, int colA, float alpha, float beta,
     int lda, int ldx, int ldy){
    cublasSgemv(*cublashandler, CUBLAS_OP_N, colA, rowA, 
                &alpha, d_A, lda, d_x, ldx, &beta, d_y, ldy);  
    cudaDeviceSynchronize();
}

void cublas_gemv_rowmajor_i(
    cublasHandle_t *cublashandler, float *d_A, float *d_x,
     float *d_y, int rowA, int colA, float alpha, float beta, cudaStream_t *pstream ){
    cublasSetStream(*cublashandler, *pstream);
    cublasSgemm(*cublashandler, CUBLAS_OP_N, CUBLAS_OP_N, 1, rowA, colA,
                    &alpha, d_x, 1, d_A, colA, &beta, d_y, 1);  
    cudaStreamSynchronize (*pstream);
}

void cublas_gemv_rowmajor_trans_i(
    cublasHandle_t *cublashandler, float *d_A, float *d_x,
     float *d_y, int rowA, int colA, float alpha, float beta, cudaStream_t *pstream){
    cublasSetStream(*cublashandler, *pstream);
    cublasSgemv(*cublashandler, CUBLAS_OP_N, colA, rowA, 
                &alpha, d_A, colA, d_x, 1, &beta, d_y, 1);  
    cudaStreamSynchronize (*pstream);
}

float cublas_absmax(
    cublasHandle_t *cublashandler, float *d_x, int size){
    int index;float maxnum;
    cublasIsamax(*cublashandler, size, d_x, 1, &index);
    cudaDeviceSynchronize();
    cudaMemcpy( &maxnum, &d_x[index-1], sizeof(float) , cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return maxnum;
}

float cublas_norm2(cublasHandle_t *cublashandler, float *d_x, int size){
    float res;
    cublasSnrm2(*cublashandler, size, d_x, 1, &res);
    cudaDeviceSynchronize();
    return res;
}

void cublas_sscal(cublasHandle_t *cublashandler, float *d_x, int size,float alpha){
    cublasSscal(*cublashandler, size, &alpha,d_x, 1);
    cudaDeviceSynchronize();
    return ;
}

float cublas_norm2_i(cublasHandle_t *cublashandler, float *d_x, int size){
    float res;
    cublasSnrm2(*cublashandler, size, d_x, 1, &res);
    return res;
}

void cublas_sscal_i(cublasHandle_t *cublashandler, float *d_x, int size,float alpha){
    cublasSscal(*cublashandler, size, &alpha,d_x, 1);
    return ;
}


void cublas_strans(cublasHandle_t *cublashandler, float *d_in, float *d_out, int row, int col){
    float alpha = 1.0;
    float beta = 0.0;
    cublasSgeam(*cublashandler, CUBLAS_OP_T, CUBLAS_OP_N, col, row, &alpha, d_in, row, &beta, d_out, col, d_out, col);
    cudaDeviceSynchronize();
    return ;
}

void cublas_scopy(cublasHandle_t *cublashandler, float *d_in, float *d_out, int size){
    float alpha = 1.0;
    float beta = 0.0;
    cublasScopy(*cublashandler, size, d_in, 1, d_out, 1);   
    cudaDeviceSynchronize();
    return ;
}

#endif