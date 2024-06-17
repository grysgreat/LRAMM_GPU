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


void gemm_acc_test(){
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    // 定义数组的大小
    int M=32,N=32,K=32;
    // 创建一个使用float类型的数组
    std::vector<float> int4b_arrayA(M*K);
    std::vector<float> int4b_arrayB(K*N);
    std::vector<float> int32b_arrayC(K*N);


    // 初始化数组
    for (int i = 0; i < M; ++i) {

        for(int j=0;j<K;j++){
            // 将每个元素初始化为它的索引值，注意这里只是示例，实际值可能需要根据量化规则来确定
            if(i==0||i==1)int4b_arrayB[i*K+j] = static_cast<float>(i+1);
            if(j==0||j==1)int4b_arrayA[i*K+j] = static_cast<float>(j+1);
            int32b_arrayC[i*K+j] = 0;
        }

    }

    for (int i = 0; i < M; ++i) {
        for(int j=0;j<N;j++){
            printf("%d,",static_cast<int>(int4b_arrayA[i*M+j]));
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < M; ++i) {
        for(int j=0;j<N;j++){
            printf("%d,",static_cast<int>(int4b_arrayB[i*M+j]));
        }
        printf("\n");
    }
    printf("\n");

    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * M*N);
    cudaMalloc((void**)&d_B, sizeof(float) * M*N);
    cudaMalloc((void**)&d_C, sizeof(float) * M*N);
    cudaMemcpy(d_A, int4b_arrayA.data(), sizeof(float) * M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, int4b_arrayB.data(), sizeof(float) * M*N, cudaMemcpyHostToDevice);

    float  beta = 0.0, alpha = 1.0;
    K=2;
    cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K,
                &alpha, d_A, M, d_B, M, &beta, d_C, M);  


    cudaMemcpy( int32b_arrayC.data(),d_C, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
  
    for (int i = 0; i < M; ++i) {

        for(int j=0;j<N;j++){
            printf("%d,",static_cast<int>(int32b_arrayC[i*M+j]));
        }
        printf("\n");
    }

}

int main(){
    //axpy_perf_test();

    gemm_acc_test();
}