#include "operator_matrix.cuh"
#include "cutlass_gemm_op.cuh"
#include "cusolver_connector.cuh"
#include "./cublas/cublas_lib.cuh"


template <typename T,int digit>
void lrxigemm(T *A, T *B, T *C, int rowsA, int colsA, int rowsB, int colsB, int rank) {

    using lowPtype = int8_t;

    /*Step 0. prepare Handle and stream*/
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));


    /*Step 1. prepare work space*/
    int threadsPerBlock = 1024; 
    int max_work_size = (max(colsA*rowsA, colsB*rowsB)+threadsPerBlock-1)/threadsPerBlock;

    T* c_work = (T *)malloc(sizeof(T) * max_work_size);
    T* d_work;
    cudaMalloc((T **)&d_work, sizeof(T) * max_work_size);

    T *A_d, *B_d, *C_d, *PA_d, *PB_d;
    lowPtype *AI_d, *BI_d;
    int32_t *CI_d;
    cudaMalloc((T **)&A_d, sizeof(T) * colsA*rowsA);
    cudaMalloc((T **)&B_d, sizeof(T) * colsB*rowsB);

    cudaMalloc((T **)&PA_d, sizeof(T) * colsA*rowsA);
    cudaMalloc((T **)&PB_d, sizeof(T) * colsB*rowsB);

    cudaMalloc((T **)&C_d, sizeof(T) * rowsA*colsB);
    cudaMalloc((lowPtype **)&AI_d, sizeof(lowPtype) * colsA*rowsA);
    cudaMalloc((lowPtype **)&BI_d, sizeof(lowPtype) * colsB*rowsB);
    cudaMalloc((int32_t **)&CI_d, sizeof(int32_t) * rowsA*colsB);

    cudaMemcpy(A_d, A, sizeof(float) * colsA*rowsA, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(float) * colsB*rowsB, cudaMemcpyHostToDevice);
    

    /*Step 2. Perform a direct quantization algorithm*/
    const int max_int = (1<<(digit-1)) - 1;
    T max_mA = max_abs(A_d, d_work, c_work, colsA*rowsA);
    T max_mB = max_abs(B_d, d_work, c_work, colsB*rowsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;
    T lambdaC = lambdaA*lambdaB;

    quantitize_int8(A_d, AI_d, rowsA, colsA, lambdaA);
    quantitize_int8(B_d, BI_d, rowsB, colsB, lambdaB);


    cut_gemm(AI_d, BI_d, CI_d, rowsA, colsA, rowsB, colsB);

    dequantitize_int8(CI_d, C_d, rowsA, colsB, lambdaC);


    /*Step 2. Calculate the residual part*/
    dequantitize_int8(AI_d, PA_d, rowsA, colsA, lambdaA);
    dequantitize_int8(BI_d, PB_d, rowsB, colsB, lambdaB);

    // here A_d and B_d represent RA and RB.
    T alpha = -1.0;
    cublas_saxpy(PA_d, A_d ,alpha, rowsA*colsA, cublasH,stream);
    cublas_saxpy(PB_d, B_d ,alpha, rowsB*colsB, cublasH,stream);

    T *AL_d, *AR_d, *BL_d, *BR_d;

}