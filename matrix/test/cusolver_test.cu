#include <random>
#include <chrono>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "../cusolver_connector.cuh"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "../operator_matrix.cuh"
#include <chrono>
template <typename T>
void generate_matrix(std::vector<T> &matrix,int rows,int cols,char type ){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    if(type == 'u'){
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                matrix[i*cols+j] = dis(gen);
                if(i==j&&i!=rows-1) matrix[i*cols+j] = i*cols+j;// (matrix[i*cols+j]);
                else  matrix[i*cols+j]=i*cols+j;//(matrix[i*cols+j]);
            }
        }     
    }
}
void print_Matrix(float matrix[], int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            printf("%.1f ", matrix[index]);
        }
        std::cout << std::endl;
    }
}


int main(int argc, char *argv[]) {



    using data_type = float;
    /* Input matrix dimensions */
    const int64_t m = 32;
    const int64_t n = 32;
    const int64_t lda = m;;
    const int64_t ldu = m;
    const int64_t ldv = n;
    /* rank of matrix A */
    const int64_t min_mn = std::min(m, n);

    std::vector<data_type> A(m*n);

    generate_matrix<float>(A,m,n,'u');

    int M=m,N=n;
    print_Matrix(A.data(),M,N);

    data_type *d_A = nullptr;
    data_type *d_U = nullptr;
    data_type *d_S = nullptr;
    data_type *d_V = nullptr;
    data_type *d_AO = nullptr;

    data_type *U = (data_type *)malloc(sizeof(data_type) * m*n);
    data_type *V = (data_type *)malloc(sizeof(data_type) * m*n);
    data_type *S = (data_type *)malloc(sizeof(data_type) * m*n);

    int rank = 32;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_AO), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(data_type) * ldu * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(data_type) * ldv * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(data_type) * min_mn));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(data_type) * lda * n, cudaMemcpyHostToDevice));


    cusolverDnHandle_t cusolverH = NULL;

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    cusolver_rsvd(m, n, d_A, d_U,d_V,d_S, rank, &cusolverH);

    cudaMemcpy( U,d_U, sizeof(data_type) * M*N, cudaMemcpyDeviceToHost);
    cudaMemcpy( V,d_V, sizeof(data_type) * M*N, cudaMemcpyDeviceToHost);
    cudaMemcpy( S,d_S, sizeof(data_type) * min_mn, cudaMemcpyDeviceToHost);


    printf("\n");
    print_Matrix(U,M,N);
    printf("\n");
    print_Matrix(V,M,N);
    printf("\n");
    print_Matrix(S,M,N);

    diag_matmul(d_V, d_S, rank, n);

    cudaMemcpy( V,d_V, sizeof(data_type) * M*N, cudaMemcpyDeviceToHost);
    printf("\n");
    print_Matrix(V,M,N);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0, beta = 0.0;



    strans(d_U,d_U,rank,m);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, rank,
                &alpha, d_U, m, d_V, n, &beta, d_AO, n);    

    cudaMemcpy( U,d_AO, sizeof(data_type) * M*N, cudaMemcpyDeviceToHost);
    printf("\n");
    print_Matrix(U,M,N);

}