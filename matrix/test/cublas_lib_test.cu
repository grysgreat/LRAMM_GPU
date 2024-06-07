#include "../cublas/cublas_lib.cuh"
#include "../operator_matrix.cuh"
#include "../gen_matrix.cuh"
#include "../print_matrix.cuh"

#include <iomanip>
#include "stdio.h"
#include <chrono>




int axpy_perf_test(){
    int num = 8192;
    int N=num,M=num,K=num;

    float *matrixA = (float *)malloc(sizeof(float) * M*N);
    int8_t *matrixA8 = (int8_t *)malloc(sizeof(int8_t) * M*N);
    float *matrixB = (float *)malloc(sizeof(float) * M*N);

    float *vec_row = (float *)malloc(sizeof(float) * M*N);
    float *vec_col = (float *)malloc(sizeof(float) * M*N);
    float *work = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,N,'u');
    generate_matrix<float>(matrixB,M,N,'u');

    float *matrixA_dev;
    float *matrixB_dev;
    float *work_dev;
    cudaMalloc((void**)&matrixA_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixB_dev, sizeof(float) * M*N);

    // start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(matrixA_dev, matrixA, sizeof(float) * M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_dev, matrixA, sizeof(float) * M*N, cudaMemcpyHostToDevice);

    float alpha = 1.0;

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));


    auto start = std::chrono::high_resolution_clock::now();
    cublas_saxpy(matrixA_dev,matrixB_dev,alpha, num*num,cublasH,stream);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    int time  = diff.count()*1000*1000;
    std::cout<<"size="<<M<<"//axpy - gpu time:" << std::fixed << std::setprecision(6) << time << std::endl;


    cudaMemcpy(matrixB, matrixB_dev, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
}




int main(){
    axpy_perf_test();
}