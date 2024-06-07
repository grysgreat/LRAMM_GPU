#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"
using data_type = float;


void cublas_saxpy(float *d_A,float *d_B,float alpha, int size,cublasHandle_t cublasH,cudaStream_t stream = NULL){


    const int incx = 1;
    const int incy = 1;

    /* step 3: compute */
    CUBLAS_CHECK(cublasSaxpy(cublasH, size, &alpha, d_A, incx, d_B, incy));
    cudaDeviceSynchronize();


    return;

}