
#include <cuda_runtime.h>
#include "cusolverDn.h"
#include "cusolver_utils.h"
#include "operator_matrix.cuh"


void cusolver_rsvd(
    int rows, int cols, float *d_A,
    float *d_U,float *d_V,float *d_S,int rank, cusolverDnHandle_t *cusolverhandler){

    cusolverDnHandle_t cusolverH = *cusolverhandler;
    cudaStream_t stream = NULL;
    cusolverDnParams_t params_gesvdr = NULL;

    using data_type = float;

    /* Input matrix dimensions */
    const int64_t m = rows;
    const int64_t n = cols;
    const int64_t lda = m;
    const int64_t ldu = m;
    const int64_t ldv = n;
    /* rank of matrix A */
    const int64_t min_mn = std::min(m, n);

    /* Compute left/right eigenvectors */
    signed char jobu = 'S';
    signed char jobv = 'S';

    /* Number of iterations */
    const int64_t iters = 0;
    const int64_t p = std::min(2, static_cast<int>(n - rank));

    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void *d_work = nullptr;              /* device workspace for getrf */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void *h_work = nullptr;              /* host workspace for getrf */
    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));




    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    CUSOLVER_CHECK(cusolverDnCreateParams(&params_gesvdr));

    /* step 3: query working space of SVD */
    CUSOLVER_CHECK(cusolverDnXgesvdr_bufferSize(
        cusolverH, params_gesvdr, jobu, jobv, m, n, rank, p, iters,
        traits<data_type>::cuda_data_type, d_A, lda, traits<data_type>::cuda_data_type, d_S,
        traits<data_type>::cuda_data_type, d_U, ldu, traits<data_type>::cuda_data_type, d_V, ldv,
        traits<data_type>::cuda_data_type, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));
    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }


    CUSOLVER_CHECK(cusolverDnXgesvdr(
        cusolverH, params_gesvdr, jobu, jobv, m, n, rank, p, iters,
        traits<data_type>::cuda_data_type, d_A, lda, traits<data_type>::cuda_data_type, d_S,
        traits<data_type>::cuda_data_type, d_U, ldu, traits<data_type>::cuda_data_type, d_V, ldv,
        traits<data_type>::cuda_data_type, d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info));


}

void cusolver_rsvd_LR(
    int rows, int cols, float *d_A,
    float *d_U,float *d_V,int rank, cusolverDnHandle_t *cusolverhandler){

    cusolverDnHandle_t cusolverH = *cusolverhandler;
    cudaStream_t stream = NULL;
    cusolverDnParams_t params_gesvdr = NULL;

    using data_type = float;

    /* Input matrix dimensions */
    const int64_t m = rows;
    const int64_t n = cols;
    const int64_t lda = m;
    const int64_t ldu = m;
    const int64_t ldv = n;
    /* rank of matrix A */
    const int64_t min_mn = std::min(m, n);

    /* Compute left/right eigenvectors */
    signed char jobu = 'S';
    signed char jobv = 'S';

    /* Number of iterations */
    const int64_t iters = 0;
    const int64_t p = std::min(2, static_cast<int>(n - rank));

    size_t workspaceInBytesOnDevice = 1e5; /* size of workspace */
    void *d_work = nullptr;              /* device workspace for getrf */
    size_t workspaceInBytesOnHost = 1e5;   /* size of workspace */
    void *h_work = nullptr;              /* host workspace for getrf */
    int *d_info = nullptr;
    int *h_info = nullptr;

    data_type *d_S;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int))); 
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), min_mn*sizeof(data_type)));




    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    CUSOLVER_CHECK(cusolverDnCreateParams(&params_gesvdr));

    /* step 3: query working space of SVD */
    CUSOLVER_CHECK(cusolverDnXgesvdr_bufferSize(
        cusolverH, params_gesvdr, jobu, jobv, m, n, rank, p, iters,
        traits<data_type>::cuda_data_type, d_A, lda, traits<data_type>::cuda_data_type, d_S,
        traits<data_type>::cuda_data_type, d_U, ldu, traits<data_type>::cuda_data_type, d_V, ldv,
        traits<data_type>::cuda_data_type, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));
    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }


    CUSOLVER_CHECK(cusolverDnXgesvdr(
        cusolverH, params_gesvdr, jobu, jobv, m, n, rank, p, iters,
        traits<data_type>::cuda_data_type, d_A, lda, traits<data_type>::cuda_data_type, d_S,
        traits<data_type>::cuda_data_type, d_U, ldu, traits<data_type>::cuda_data_type, d_V, ldv,
        traits<data_type>::cuda_data_type, d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info));


    diag_matmul(d_V, d_S, rank, n);

}