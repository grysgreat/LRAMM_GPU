#include <random>
#include <chrono>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "cuda_runtime.h"
#include "../lrxigemm.cuh"
#include <chrono>
#include "../gen_matrix.cuh"
#include "../print_matrix.cuh"


template <typename T>
void xgemm(const T A[], const T B[], T C[], int rowsA, int colsA, int rowsB, int colsB) {


    T* tmp =(T *)malloc(sizeof(T) * rowsA*colsB);

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            tmp[i * colsB + j] = 0;
            for (int k = 0; k < colsA; ++k) {
                tmp[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
    
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i * colsB + j]  = tmp[i * colsB + j];
		}
	}
    
}

template <typename T>
T get_Ferror(T matrix_ref[],T matrix_cmp[],int rows,int cols){

    T sumR=0,sum=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            sumR+=(matrix_ref[i*cols+j] - matrix_cmp[i*cols+j])*(matrix_ref[i*cols+j] - matrix_cmp[i*cols+j]);
            sum+=(matrix_ref[i*cols+j])*(matrix_ref[i*cols+j]);
        }
    }

    T ans = sqrt(sumR)/sqrt(sum);
    return ans;

}

void xigemm_acc(){
    int max = 1024;
    float *matrixA = (float *)malloc(sizeof(float) * max*max);
    float *matrixB = (float *)malloc(sizeof(float) * max*max);
    float *matrixC = (float *)malloc(sizeof(float) * max*max);
    float *matrixCQ = (float *)malloc(sizeof(float) * max*max);
    float *matrixR = (float *)malloc(sizeof(float) * max*max);

    int M=max , N=max, K = max;
    generate_matrix<float>(matrixA,M,K,'k');
    generate_matrix<float>(matrixB,K,N,'k');    

    xgemm(matrixA,matrixB,matrixC,M,K,K,N);

    float *A_d, *B_d, *C_d;
    cudaMalloc((float **)&A_d, sizeof(float) * M*K);
    cudaMalloc((float **)&B_d, sizeof(float) * K*N);
    cudaMalloc((float **)&C_d, sizeof(float) * M*N);
    cudaMemcpy(A_d, matrixA, sizeof(float) * M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, matrixB, sizeof(float) * K*N, cudaMemcpyHostToDevice);



    xigemm<float,8>(A_d,B_d,C_d,M,K,K,N);
    cudaMemcpy( matrixCQ,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // printMatrix(matrixC,M,N);
    // printf("\n\n");
    // printMatrix(matrixCQ,M,N);
    // printf("\n\n");
    float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 

    printf("%.7f\n",R3);


    return ;
}

void xigemm_perf(){
    int max = 8192;
    float *matrixA = (float *)malloc(sizeof(float) * max*max);
    float *matrixB = (float *)malloc(sizeof(float) * max*max);
    float *matrixC = (float *)malloc(sizeof(float) * max*max);
    float *matrixCQ = (float *)malloc(sizeof(float) * max*max);
    float *matrixR = (float *)malloc(sizeof(float) * max*max);

    int M=max , N=max, K = max;
    generate_matrix<float>(matrixA,M,K,'k');
    generate_matrix<float>(matrixB,K,N,'k');    

    float* d_work;
    cudaMalloc((float **)&d_work, sizeof(float) * 10);

    float *A_d, *B_d, *C_d;
    cudaMalloc((float **)&A_d, sizeof(float) * M*K);
    cudaMalloc((float **)&B_d, sizeof(float) * K*N);
    cudaMalloc((float **)&C_d, sizeof(float) * M*N);
    cudaMemcpy(A_d, matrixA, sizeof(float) * M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, matrixB, sizeof(float) * K*N, cudaMemcpyHostToDevice);




    auto start = std::chrono::high_resolution_clock::now();
    xigemm<float,8>(A_d,B_d,C_d,M,K,K,N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    double time  = diff.count();
    std::cout<< std::fixed << std::setprecision(6) << time << "\n";
 

    cudaMemcpy( matrixC,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    return ;
}

void lrxigemm_acc(){
    int max = 1024;
    float *matrixA = (float *)malloc(sizeof(float) * max*max);
    float *matrixB = (float *)malloc(sizeof(float) * max*max);
    float *matrixC = (float *)malloc(sizeof(float) * max*max);
    float *matrixCQ = (float *)malloc(sizeof(float) * max*max);
    float *matrixR = (float *)malloc(sizeof(float) * max*max);

    int M=max , N=max, K = max;

    generate_matrix<float>(matrixA,M,K,'k');
    generate_matrix<float>(matrixB,K,N,'k');    

    xgemm(matrixA,matrixB,matrixC,M,K,K,N);

    float *A_d, *B_d, *C_d;
    cudaMalloc((float **)&A_d, sizeof(float) * M*K);
    cudaMalloc((float **)&B_d, sizeof(float) * K*N);
    cudaMalloc((float **)&C_d, sizeof(float) * M*N);
    cudaMemcpy(A_d, matrixA, sizeof(float) * M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, matrixB, sizeof(float) * K*N, cudaMemcpyHostToDevice);

    xigemm<float,8>(A_d,B_d,C_d,M,K,K,N);
    cudaMemcpy( matrixCQ,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    float R2 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
    printf("%.7f\n",R2);

    cublasHandle_t cublasH = NULL;
    cusolverDnHandle_t cusolverH = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    lrxigemm<float,8>(A_d,B_d,C_d,M,K,K,N,10, &cusolverH, &cublasH);
    cudaMemcpy( matrixCQ,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 

    printf("%.7f\n",R3);


    return ;
}

void lrxigemm_perf(){
    int max = 8192;//1024*16;
    float *matrixA = (float *)malloc(sizeof(float) * max*max);
    float *matrixB = (float *)malloc(sizeof(float) * max*max);
    float *matrixC = (float *)malloc(sizeof(float) * max*max);
    float *matrixCQ = (float *)malloc(sizeof(float) * max*max);
    float *matrixR = (float *)malloc(sizeof(float) * max*max);

    int M=max , N=max, K = max;
    generate_matrix<float>(matrixA,M,K,'l');
    generate_matrix<float>(matrixB,K,N,'r');    

    float* d_work;
    cudaMalloc((float **)&d_work, sizeof(float) * 10);

    float *A_d, *B_d, *C_d;
    cudaMalloc((float **)&A_d, sizeof(float) * M*K);
    cudaMalloc((float **)&B_d, sizeof(float) * K*N);
    cudaMalloc((float **)&C_d, sizeof(float) * M*N);
    cudaMemcpy(A_d, matrixA, sizeof(float) * M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, matrixB, sizeof(float) * K*N, cudaMemcpyHostToDevice);

    cublasHandle_t cublasH = NULL;
    cusolverDnHandle_t cusolverH = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    lrxigemm_speed<float,8>(A_d,B_d,C_d,M,K,K,N,2, &cusolverH, &cublasH);
    cudaDeviceSynchronize();


    int iter = 1;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<iter;i++) {
        lrxigemm_speed<float,8>(A_d,B_d,C_d,M,K,K,N,2, &cusolverH, &cublasH);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    double time  = diff.count()/((double)iter);
    std::cout<< std::fixed << std::setprecision(6) << time << "\n";
 


    cudaMemcpy( matrixC,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    return ;
}

int main(){
    lrxigemm_perf();
    // xigemm_perf();
    

    //lrxigemm_acc();
}