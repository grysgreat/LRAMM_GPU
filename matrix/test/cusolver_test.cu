#include <random>
#include <chrono>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "../cusolver_connector.cuh"

template <typename T>
void generate_matrix(std::vector<T> &matrix,int rows,int cols,char type ){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    if(type == 'u'){
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                matrix[i*cols+j] = dis(gen);
                if(i==j&&i!=rows-1) matrix[i*cols+j] = (matrix[i*cols+j]);
                else  matrix[i*cols+j]=(matrix[i*cols+j]);
            }
        }     
    }
}

int main(int argc, char *argv[]) {

    using data_type = float;
    /* Input matrix dimensions */
    const int64_t m = 1024;
    const int64_t n = 1024;
    const int64_t lda = m;
    const int64_t ldu = m;
    const int64_t ldv = n;
    /* rank of matrix A */
    const int64_t min_mn = std::min(m, n);

    std::vector<data_type> A(m*n);

    generate_matrix<float>(A,m,n,'u');

    data_type *d_A = nullptr;
    data_type *d_U = nullptr;
    data_type *d_S = nullptr;
    data_type *d_V = nullptr;

    int rank = 10;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(data_type) * ldu * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(data_type) * ldv * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(data_type) * min_mn));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(data_type) * lda * n, cudaMemcpyHostToDevice));

    cusolver_rsvd(m, n, d_A, d_U,d_V,d_S, 10);

    

}