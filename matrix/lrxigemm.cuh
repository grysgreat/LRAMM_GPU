#include "operator_matrix.cuh"
#include "cutlass_gemm_op.cuh"
#include "cusolver_connector.cuh"
#include "./cublas/cublas_lib.cuh"
#include "sketch_op.cuh"

template <typename T>
T get_Ferror1(T matrix_ref[],T matrix_cmp[],int rows,int cols){

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

void print_Matrix(float matrix[], int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            printf("%.2f ", matrix[index]);
        }
        std::cout << std::endl;
    }
}


template <typename T,int digit>
void lrxigemm(
    T *A_d, T *B_d, T *C_d, 
    int rowsA, int colsA, int rowsB, int colsB, int rank, 
    cusolverDnHandle_t *cusolverhandler, cublasHandle_t *cublashandler) {


    using lowPtype = int8_t;

    /*Step 0. prepare Handle and stream*/
    cusolverDnHandle_t cusolverH = *cusolverhandler;
    cublasHandle_t cublasH = *cublashandler;
    cudaStream_t stream = NULL;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));


    /*Step 1. prepare work space*/
    int threadsPerBlock = 1024; 
    int max_work_size = (max(colsA*rowsA, colsB*rowsB)+threadsPerBlock-1)/threadsPerBlock;

    T* c_work = (T *)malloc(sizeof(T) * max_work_size);
    T* d_work;
    cudaMalloc((T **)&d_work, sizeof(T) * max_work_size);

    T  *PA_d, *PB_d, *AR, *BR;
    lowPtype *AI_d, *BI_d, *Itmp_d;
    int32_t *CI_d;

    cudaMalloc((T **)&PA_d, sizeof(T) * colsA*rowsA);
    cudaMalloc((T **)&PB_d, sizeof(T) * colsB*rowsB);
    cudaMalloc((T **)&AR, sizeof(T) * colsA*rowsA);
    cudaMalloc((T **)&BR, sizeof(T) * colsB*rowsB);

    cudaMalloc((lowPtype **)&AI_d, sizeof(lowPtype) * colsA*rowsA);
    cudaMalloc((lowPtype **)&BI_d, sizeof(lowPtype) * colsB*rowsB);
    cudaMalloc((int32_t **)&CI_d, sizeof(int32_t) * rowsA*colsB);
    cudaMalloc((lowPtype **)&Itmp_d, sizeof(lowPtype) * colsB*rowsB);


    cudaMemcpy(AR, A_d, colsA*rowsA * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(BR, B_d, colsB*rowsB * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    /*Step 2. Perform a direct quantization algorithm*/
    const int max_int = (1<<(digit-1)) - 1;
    T max_mA = max_abs(A_d, d_work, c_work, colsA*rowsA);
    T max_mB = max_abs(B_d, d_work, c_work, colsB*rowsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;
    T lambdaC = lambdaA*lambdaB;

    quantitize_int8(A_d, AI_d, rowsA, colsA, lambdaA);
    quantitize_int8(B_d, BI_d, rowsB, colsB, lambdaB);

    I8trans(Itmp_d,BI_d,rowsB,colsB);
    cut_gemm(AI_d, Itmp_d, CI_d, rowsA, colsA, rowsB, colsB);


    dequantitize_int32(CI_d, C_d, rowsA, colsB, lambdaC);


    // /*Step 3. Calculate the residual part*/
    dequantitize_int8(AI_d, PA_d, rowsA, colsA, lambdaA);
    dequantitize_int8(BI_d, PB_d, rowsB, colsB, lambdaB);
    // here A_d and B_d represent RA and RB.
    T alpha = -1.0;
    cublas_saxpy(PA_d, AR ,alpha, rowsA*colsA, cublasH,stream);
    cublas_saxpy(PB_d, BR ,alpha, rowsB*colsB, cublasH,stream);
    T *AL_d, *AR_d, *BL_d, *BR_d, *tmp_d, *tmp_d2;
    cudaMalloc((T **)&AL_d, sizeof(T) * rowsA * colsA);
    cudaMalloc((T **)&AR_d, sizeof(T) * rowsA * colsA);
    cudaMalloc((T **)&BL_d, sizeof(T) * rowsB * colsB);
    cudaMalloc((T **)&BR_d, sizeof(T) * rowsB * colsB);
    cudaMalloc((T **)&tmp_d, sizeof(T) * rowsA * colsB);
    cudaMalloc((T **)&tmp_d2, sizeof(T) * rowsA * colsB);

    cudaFree(AI_d);
    cudaFree(BI_d);
    cudaFree(CI_d);
    cudaFree(Itmp_d);

    cusolver_rsvd_LR(rowsA, colsA, AR, AL_d, AR_d, rank, &cusolverH);
    cusolver_rsvd_LR(rowsB, colsB, BR, BL_d, BR_d, rank, &cusolverH);
    


    float  beta = 0.0;
    alpha = 1.0;

    //下面�?先算出残�?部分的测试，�?前�??一步�?�算有问�?
//begin full size correct


    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rowsA, colsA, rank,
                &alpha, AL_d, rowsA, AR_d, colsA, &beta, tmp_d, rowsA); 
    cudaDeviceSynchronize();
    cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, rowsA, colsB, colsA,
            &alpha, tmp_d, colsA, PB_d, colsB, &beta, tmp_d2, colsB);  
    cudaDeviceSynchronize();
    strans(tmp_d,tmp_d2,rowsA,colsB);
    cublas_saxpy(tmp_d, C_d ,alpha, rowsA*colsB, cublasH,stream);

    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rowsA, colsA, rank,
                &alpha, BL_d, rowsA, BR_d, colsA, &beta, tmp_d, rowsA); 
    cudaDeviceSynchronize();
    cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, rowsA, colsB, colsA,
            &alpha, PA_d, colsA, tmp_d, colsB, &beta, tmp_d2, colsB);  
    cudaDeviceSynchronize();
    strans(tmp_d,tmp_d2,rowsA,colsB);
    cublas_saxpy(tmp_d, C_d ,alpha, rowsA*colsB, cublasH,stream);



    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rowsA, colsA, rank,
                &alpha, BL_d, rowsA, BR_d, colsA, &beta, tmp_d, rowsA); 
    cudaDeviceSynchronize();
    cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, rowsA, colsB, colsA,
            &alpha, AR, colsA, tmp_d, colsB, &beta, tmp_d2, colsB);  
    cudaDeviceSynchronize();
    strans(tmp_d,tmp_d2,rowsA,colsB);
    cublas_saxpy(tmp_d, C_d ,alpha, rowsA*colsB, cublasH,stream);

//end full size correct
    
    return;
}



template <typename T,int digit>
void lrxigemm_speed(
    T *A_d, T *B_d, T *C_d, 
    int rowsA, int colsA, int rowsB, int colsB, int rank, 
    cusolverDnHandle_t *cusolverhandler, cublasHandle_t *cublashandler) {

    // auto start = std::chrono::high_resolution_clock::now();
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> diff = end - start;
    // double time  = diff.count();

    using lowPtype = int8_t;

    /*Step 0. prepare Handle and stream*/
    cusolverDnHandle_t cusolverH = *cusolverhandler;
    cublasHandle_t cublasH = *cublashandler;


    /*Step 1. prepare work space*/
    int threadsPerBlock = 1024; 
    int max_work_size = (max(colsA*rowsA, colsB*rowsB)+threadsPerBlock-1)/threadsPerBlock;

    T* c_work = (T *)malloc(sizeof(T) * max_work_size);
    T* d_work;
    cudaMalloc((T **)&d_work, sizeof(T) * max_work_size);

    // void* i_d;
    // cudaMalloc(reinterpret_cast<void **>(&i_d), sizeof(lowPtype) * colsA*rowsA *3+ sizeof(int32_t) * rowsA*colsB);


    T  *PA_d, *PB_d;
    lowPtype *AI_d, *BI_d, *Itmp_d;
    int32_t *CI_d;
    // lowPtype *AI_d = &((reinterpret_cast<lowPtype *>(i_d))[0]);
    // lowPtype *BI_d = &((reinterpret_cast<lowPtype *>(i_d))[colsA*rowsA]);
    // lowPtype *Itmp_d = &((reinterpret_cast<lowPtype *>(i_d))[2*colsA*rowsA]);
    // int32_t *CI_d = reinterpret_cast<int32_t *>(&((reinterpret_cast<lowPtype *>(i_d))[3*colsA*rowsA]));

    cudaMalloc((T **)&PA_d, sizeof(T) * colsA*rowsA);
    cudaMalloc((T **)&PB_d, sizeof(T) * colsB*rowsB);

    cudaMalloc((lowPtype **)&AI_d, sizeof(lowPtype) * colsA*rowsA);
    cudaMalloc((lowPtype **)&BI_d, sizeof(lowPtype) * colsB*rowsB);
    cudaMalloc((int32_t **)&CI_d, sizeof(int32_t) * rowsA*colsB);
    cudaMalloc((lowPtype **)&Itmp_d, sizeof(lowPtype) * colsB*rowsB);

    /*Step 2. Perform a direct quantization algorithm*/
    const int max_int = (1<<(digit-1)) - 1;
    T max_mA = max_abs(A_d, d_work, c_work, colsA*rowsA);
    T max_mB = max_abs(B_d, d_work, c_work, colsB*rowsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;
    T lambdaC = lambdaA*lambdaB;

    quantitize_int8(A_d, AI_d, rowsA, colsA, lambdaA);
    quantitize_int8(B_d, BI_d, rowsB, colsB, lambdaB);

    I8trans(Itmp_d,BI_d,rowsB,colsB);
    cut_gemm(AI_d, Itmp_d, CI_d, rowsA, colsA, rowsB, colsB);



    dequantitize_int32(CI_d, C_d, rowsA, colsB, lambdaC);


    // // // /*Step 3. Calculate the residual part*/
    dequantitize_int8(AI_d, PA_d, rowsA, colsA, lambdaA);
    dequantitize_int8(BI_d, PB_d, rowsB, colsB, lambdaB);

    // cudaFree(AI_d);
    // cudaFree(BI_d);
    // cudaFree(CI_d);
    // cudaFree(Itmp_d);
 


    // // // // here A_d and B_d represent RA and RB.
    T alpha = -1.0;
    cublas_saxpy(PA_d, A_d ,alpha, rowsA*colsA, cublasH);
    cublas_saxpy(PB_d, B_d ,alpha, rowsB*colsB, cublasH);
    T *L_d, *R_d, *tmp_d, *tmp_d2;
    cudaMalloc((T **)&L_d, sizeof(T) * rowsA * colsA);
    cudaMalloc((T **)&R_d, sizeof(T) * rowsA * colsA);
    cudaMalloc((T **)&tmp_d, sizeof(T) * rowsA * colsB);
    cudaMalloc((T **)&tmp_d2, sizeof(T) * rowsA * colsB);


    
    
 
    // // // printf("size  = %d\n",sizeof(T) * rowsB * colsB);

    float  beta = 0.0;
    alpha = 1.0;
    cusolver_rsvd_LR(rowsA, colsA, A_d, L_d, R_d, rank, &cusolverH);
    cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, rank, colsB, colsA,
                &alpha, R_d, colsA, PB_d, colsB, &beta, tmp_d, rank);    
    cudaDeviceSynchronize();
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rowsA, colsB, rank,
                &alpha, L_d, rowsA, tmp_d, rank, &beta, tmp_d2, colsB);  
    cudaDeviceSynchronize();  
    cublas_saxpy(tmp_d2, C_d ,alpha, rowsA*colsB, cublasH);



    cusolver_rsvd_LR(rowsB, colsB, B_d, L_d, R_d, rank, &cusolverH);
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rowsA, rank, colsA,
                &alpha, PA_d, colsA, L_d, colsA, &beta, tmp_d, rowsA); 
    cudaDeviceSynchronize();  
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rowsA, colsB, rank,
                &alpha, tmp_d, rowsA, R_d, colsB, &beta, tmp_d2, rowsA); 
     cudaDeviceSynchronize(); 
     cublas_saxpy(tmp_d2, C_d ,alpha, rowsA*colsB, cublasH);


    // cudaFree(L_d);
    // cudaFree(R_d);
    // cudaFree(PA_d);
    // cudaFree(PB_d);
    // cudaFree(tmp_d);
    // cudaFree(tmp_d2);
    
    return;
}

template <typename T,int digit>
void xigemm(T *A_d, T *B_d, T *C_d, int rowsA, int colsA, int rowsB, int colsB) {

    using lowPtype = int8_t;


    /*Step 1. prepare work space*/
    int threadsPerBlock = 1024; 
    const int max_work_size = (max(colsA*rowsA, colsB*rowsB)+threadsPerBlock-1)/threadsPerBlock;

    T* c_work = (T *)malloc(sizeof(T) * max_work_size);
    T* d_work;
    cudaMalloc((T **)&d_work, sizeof(T) * max_work_size);

    lowPtype *AI_d, *BI_d, *Itmp_d;
    int32_t *CI_d;


    int maxRC1 = max(rowsA,rowsB);
    int maxRC2 = max(colsA,colsB);
    cudaMalloc((lowPtype **)&AI_d, sizeof(lowPtype) * colsA*rowsA);
    cudaMalloc((lowPtype **)&BI_d, sizeof(lowPtype) * colsB*rowsB);
    cudaMalloc((int32_t **)&CI_d, sizeof(int32_t) * maxRC1*maxRC2);
    cudaMalloc((lowPtype **)&Itmp_d, sizeof(lowPtype) * maxRC1*maxRC2);


    /*Step 2. Perform a direct quantization algorithm*/
    const int max_int = (1<<(digit-1)) - 1;
    T max_mA = max_abs(A_d, d_work, c_work, colsA*rowsA);
    T max_mB = max_abs(B_d, d_work, c_work, colsB*rowsB);
    cudaDeviceSynchronize();
    // printf("max_mA = %.7f\n",max_mA);

    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;
    T lambdaC = lambdaA*lambdaB;

    quantitize_int8(A_d, AI_d, rowsA, colsA, lambdaA);
    quantitize_int8(B_d, BI_d, rowsB, colsB, lambdaB);

    I8trans(Itmp_d,BI_d,rowsB,colsB);
    cut_gemm(AI_d, Itmp_d, CI_d, rowsA, colsA, rowsB, colsB);
    cudaDeviceSynchronize();


    dequantitize_int32(CI_d, C_d, rowsA, colsB, lambdaC);
    cudaDeviceSynchronize();

    cudaFree(AI_d);cudaFree(BI_d);cudaFree(CI_d);cudaFree(Itmp_d);cudaFree(d_work);
}




void shgemm(float *A_d, float *B_d, float *C_d, int rowsA, int colsA, int rowsB, int colsB, cublasHandle_t *cublashandler) {
    /*Step 1. prepare work space*/
    half *Ah_d, *Bh_d, *Ch_d ;
    cublasHandle_t cublasH = *cublashandler;
    int maxRC1 = max(rowsA,rowsB);
    int maxRC2 = max(colsA,colsB);
    cudaMalloc((half **)&Ah_d, sizeof(half) * colsA*rowsA);
    cudaMalloc((half **)&Bh_d, sizeof(half) * colsB*rowsB);
    cudaMalloc((half **)&Ch_d, sizeof(half) * colsB*rowsA);

    float2half(A_d, Ah_d, rowsA, colsA);
    float2half(B_d, Bh_d, rowsB, colsB);

    half alpha = 1.0, beta = 0.0;
    cublas_gemm_rowmajor(
        &cublasH, Ah_d, Bh_d, Ch_d,  rowsA,  colsA,
        rowsB,  colsB, alpha,  beta);

    // std::vector<float> tmp(rowsA*colsB);

    half2float(Ch_d, C_d, rowsA, colsB);
    cudaDeviceSynchronize();
    // cudaMemcpy( tmp.data(),C_d, sizeof(float) * rowsA*colsB, cudaMemcpyDeviceToHost);
    // for(int i=0;i<10;i++){
    //     printf("%f, ",(float)tmp[i]);
    // }

}


template <typename T,int digit>
void xigemm_mem(T *A_d, T *B_d, T *C_d, char *work_dev, int rowsA, int colsA, int rowsB, int colsB) {

    using lowPtype = int8_t;


    /*Step 1. prepare work space*/
    int threadsPerBlock = 1024; 
    const int max_work_size = (max(colsA*rowsA, colsB*rowsB)+threadsPerBlock-1)/threadsPerBlock;

    T* c_work = (T *)malloc(sizeof(T) * max_work_size);
    T* d_work;
    cudaMalloc((T **)&d_work, sizeof(T) * max_work_size);

    lowPtype *AI_d, *BI_d, *Itmp_d;
    int32_t *CI_d;


    int maxRC1 = max(rowsA,rowsB);
    int maxRC2 = max(colsA,colsB);

    AI_d = (lowPtype *)work_dev;
    BI_d = ((lowPtype *)(&work_dev[colsA*rowsA*sizeof(lowPtype)]));
    Itmp_d = ((lowPtype *)(&work_dev[colsA*rowsA*sizeof(lowPtype)+colsB*rowsB*sizeof(lowPtype)]));
    CI_d = ((int32_t *)(&work_dev[colsA*rowsA*sizeof(lowPtype)+2*colsB*rowsB*sizeof(lowPtype)]));


    /*Step 2. Perform a direct quantization algorithm*/
    const int max_int = (1<<(digit-1)) - 1;
    T max_mA = max_abs(A_d, d_work, c_work, colsA*rowsA);
    T max_mB = max_abs(B_d, d_work, c_work, colsB*rowsB);
    cudaDeviceSynchronize();
    // printf("max_mA = %.7f\n",max_mA);

    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;
    T lambdaC = lambdaA*lambdaB;

    quantitize_int8(A_d, AI_d, rowsA, colsA, lambdaA);
    quantitize_int8(B_d, BI_d, rowsB, colsB, lambdaB);

    I8trans(Itmp_d,BI_d,rowsB,colsB);
    cut_gemm(AI_d, Itmp_d, CI_d, rowsA, colsA, rowsB, colsB);
    cudaDeviceSynchronize();


    dequantitize_int32(CI_d, C_d, rowsA, colsB, lambdaC);
    cudaDeviceSynchronize();

    cudaFree(d_work);
}
template <typename T,int digit>
void rxigemm(
    T *A_d, T *B_d, T *C_d, 
    int rowsA, int colsA, int rowsB, int colsB, int rank, 
    cusolverDnHandle_t *cusolverhandler, cublasHandle_t *cublashandler) {

    using lowPtype = int8_t;

    /*Step 0. prepare Handle and stream*/
    cusolverDnHandle_t cusolverH = *cusolverhandler;
    cublasHandle_t cublasH = *cublashandler;
    cudaStream_t stream = NULL;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));


    /*Step 1. prepare work space*/
    int threadsPerBlock = 1024; 
    int max_work_size = (max(colsA*rowsA, colsB*rowsB)+threadsPerBlock-1)/threadsPerBlock;

    T* c_work = (T *)malloc(sizeof(T) * max_work_size);
    T* d_work;
    cudaMalloc((T **)&d_work, sizeof(T) * max_work_size);

    T  *PA_d, *PB_d;
    lowPtype *AI_d, *BI_d, *Itmp_d;
    int32_t *CI_d;

    cudaMalloc((T **)&PA_d, sizeof(T) * colsA*rowsA);
    cudaMalloc((T **)&PB_d, sizeof(T) * colsB*rowsB);

    cudaMalloc((lowPtype **)&AI_d, sizeof(lowPtype) * colsA*rowsA);
    cudaMalloc((lowPtype **)&BI_d, sizeof(lowPtype) * colsB*rowsB);
    cudaMalloc((lowPtype **)&Itmp_d, sizeof(lowPtype) * colsB*rowsB);
    cudaMalloc((int32_t **)&CI_d, sizeof(int32_t) * rowsA*colsB);


    /*Step 2. Perform a direct quantization algorithm*/
    const int max_int = (1<<(digit-1)) - 1;
    T max_mA = max_abs(A_d, d_work, c_work, colsA*rowsA);
    T max_mB = max_abs(B_d, d_work, c_work, colsB*rowsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;
    T lambdaC = lambdaA*lambdaB;

    quantitize_int8(A_d, AI_d, rowsA, colsA, lambdaA);
    quantitize_int8(B_d, BI_d, rowsB, colsB, lambdaB);

    I8trans(Itmp_d,BI_d,rowsB,colsB);
    cut_gemm(AI_d, Itmp_d, CI_d, rowsA, colsA, rowsB, colsB);
    dequantitize_int32(CI_d, C_d, rowsA, colsB, lambdaC);


    // /*Step 3. Calculate the residual part*/
    dequantitize_int8(AI_d, PA_d, rowsA, colsA, lambdaA);
    dequantitize_int8(BI_d, PB_d, rowsB, colsB, lambdaB);

    // here A_d and B_d represent RA and RB.
    T alpha = -1.0;
    cublas_saxpy(PA_d, A_d ,alpha, rowsA*colsA, cublasH,stream);
    cublas_saxpy(PB_d, B_d ,alpha, rowsB*colsB, cublasH,stream);


    T *tmp_d, *tmp_d2;
    cudaMalloc((T **)&tmp_d, sizeof(T) * rowsA * colsB);
    cudaMalloc((T **)&tmp_d2, sizeof(T) * rowsA * colsB);
    
    T *host = (T *)malloc(sizeof(data_type) * rowsA*colsA);
    T *host_ACC = (T *)malloc(sizeof(data_type) * rowsA*colsA);

    float  beta = 0.0;
    alpha = 1.0;

    cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, rowsA, colsB, colsA,
            &alpha, A_d, colsA, PB_d, colsB, &beta, tmp_d2, colsB);  
    cudaDeviceSynchronize();
    strans(tmp_d,tmp_d2,rowsA,colsB);
    cublas_saxpy(tmp_d, C_d ,alpha, rowsA*colsB, cublasH,stream);

    CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, rowsA, colsB, colsA,
            &alpha, PA_d, colsA, B_d, colsB, &beta, tmp_d2, colsB));  
    cudaDeviceSynchronize();
    strans(tmp_d,tmp_d2,rowsA,colsB);
    cublas_saxpy(tmp_d, C_d ,alpha, rowsA*colsB, cublasH,stream);

    // cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, rowsA, colsB, colsA,
    //         &alpha, A_d, colsA, B_d, colsB, &beta, tmp_d2, colsB);  
    // cudaDeviceSynchronize();
    // strans(tmp_d,tmp_d2,rowsA,colsB);
    // cublas_saxpy(tmp_d, C_d ,alpha, rowsA*colsB, cublasH,stream);


    return;
}


template <typename T,int digit>
void skxigemm(
    T *A_d, T *B_d, T *C_d, 
    int rowsA, int colsA, int rowsB, int colsB, int rank, 
    cusolverDnHandle_t *cusolverhandler, cublasHandle_t *cublashandler) {

    rank = 1;
    using lowPtype = int8_t;

    /*Step 0. prepare Handle and stream*/
    cublasHandle_t cublasH = *cublashandler;
    cudaStream_t stream = NULL;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /*Step 1. prepare work space*/
    T  *PA_d, *PB_d, *AR, *BR, *B_tmp;
    lowPtype *AI_d, *BI_d, *Itmp_d;
    int32_t *CI_d;


    cudaMalloc((T **)&PA_d, sizeof(T) * colsA*rowsA);
    cudaMalloc((T **)&PB_d, sizeof(T) * colsB*rowsB);
    cudaMalloc((T **)&AR, sizeof(T) * colsA*rowsA);
    cudaMalloc((T **)&BR, sizeof(T) * colsB*rowsB);

    cudaMalloc((lowPtype **)&AI_d, sizeof(lowPtype) * colsA*rowsA);
    cudaMalloc((lowPtype **)&BI_d, sizeof(lowPtype) * colsB*rowsB);
    cudaMalloc((int32_t **)&CI_d, sizeof(int32_t) * rowsA*colsB);
    cudaMalloc((lowPtype **)&Itmp_d, sizeof(lowPtype) * colsB*rowsB);


    /*Step 2. Perform a direct quantization algorithm*/
    const int max_int = (1<<(digit-1)) - 1;
    T max_mA = cublas_absmax(&cublasH, A_d, colsA*rowsA);//max_mA2;// max_abs(A_d, d_work, c_work, colsA*rowsA);
    T max_mB = cublas_absmax(&cublasH, B_d, colsB*rowsB);//max_abs(B_d, d_work, c_work, colsB*rowsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;
    T lambdaC = lambdaA*lambdaB;


    quantitize_int8(A_d, AI_d, rowsA, colsA, lambdaA);
    quantitize_int8(B_d, BI_d, rowsB, colsB, lambdaB);
    I8trans(Itmp_d,BI_d,rowsB,colsB);
    cut_gemm(AI_d, Itmp_d, CI_d, rowsA, colsA, rowsB, colsB);
    dequantitize_int32(CI_d, C_d, rowsA, colsB, lambdaC);


    cudaMemcpy(AR, A_d, colsA*rowsA * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(BR, B_d, colsB*rowsB * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    // /*Step 3. Calculate the residual part*/
    dequantitize_int8(AI_d, PA_d, rowsA, colsA, lambdaA);
    dequantitize_int8(BI_d, PB_d, rowsB, colsB, lambdaB);
    // here A_d and B_d represent RA and RB.
    T alpha = -1.0;
    cublas_saxpy(PA_d, AR ,alpha, rowsA*colsA, cublasH,stream);
    cublas_saxpy(PB_d, BR ,alpha, rowsB*colsB, cublasH,stream);
    T *AL_d, *AR_d, *BL_d, *BR_d, *tmp_d;
    int maxlen = max(max(rowsB,max(rowsA,colsA)),colsB);
    cudaMalloc((T **)&AL_d, sizeof(T) * rowsA );
    cudaMalloc((T **)&AR_d, sizeof(T) * colsA);
    cudaMalloc((T **)&BL_d, sizeof(T) * rowsB);
    cudaMalloc((T **)&BR_d, sizeof(T) * colsB);
    cudaMalloc((T **)&tmp_d, sizeof(T) * maxlen);

    curandGenerator_t gen;
    sketch_r1( AR, AL_d, AR_d,rowsA, colsA, &gen,cublashandler);
    sketch_r1( BR, BL_d, BR_d,rowsB, colsB, &gen,cublashandler);

    float  beta = 0.0;
    alpha = 1.0;

//begin full size correct
    cublas_gemm_rowmajor(
        &cublasH, AR_d, PB_d, tmp_d,  rank,  colsA,
        rowsB,  colsB, alpha,  beta);
    beta = 1.0;
    cublas_gemm_rowmajor(
        &cublasH, AL_d, tmp_d, C_d,  rowsA,  rank,
        rank,  colsB, alpha,  beta);
    beta = 0.0;
    cublas_gemm_rowmajor(
        &cublasH, PA_d, BL_d, tmp_d,  rowsA,  colsA,
        rowsB,  rank, alpha,  beta);
    beta = 1.0;
    cublas_gemm_rowmajor(
        &cublasH, tmp_d, BR_d, C_d,  rowsA,  rank,
        rank,  colsB, alpha,  beta);
    beta = 0.0;
    cublas_gemm_rowmajor(
        &cublasH, AR, BL_d, tmp_d,  rowsA,  colsA,
        rowsB,  rank, alpha,  beta);
    beta = 1.0;
    cublas_gemm_rowmajor(
        &cublasH, tmp_d, BR_d, C_d,  rowsA,  rank,
        rank,  colsB, alpha,  beta);

    return;
}


template <typename T,int digit>
void skxigemm_mem(
    T *A_d, T *B_d, T *C_d, char *work_dev,
    int rowsA, int colsA, int rowsB, int colsB, int rank, 
    cusolverDnHandle_t *cusolverhandler, cublasHandle_t *cublashandler) {

    rank = 1;
    using lowPtype = int8_t;

    /*Step 0. prepare Handle and stream*/
    cublasHandle_t cublasH = *cublashandler;


    /*Step 1. prepare work space*/
    T  *PA_d, *PB_d, *AR, *BR, *B_tmp;
    lowPtype *AI_d, *BI_d, *Itmp_d;
    int32_t *CI_d;


    AI_d = (lowPtype *)work_dev;
    BI_d = ((lowPtype *)(&work_dev[colsA*rowsA*sizeof(lowPtype)]));
    Itmp_d = ((lowPtype *)(&work_dev[colsA*rowsA*sizeof(lowPtype)+colsB*rowsB*sizeof(lowPtype)]));
    CI_d = ((int32_t *)(&work_dev[colsA*rowsA*sizeof(lowPtype)+2*colsB*rowsB*sizeof(lowPtype)]));


    /*Step 2. Perform a direct quantization algorithm*/
    const int max_int = (1<<(digit-1)) - 1;
    T max_mA = cublas_absmax(&cublasH, A_d, colsA*rowsA);//max_mA2;// max_abs(A_d, d_work, c_work, colsA*rowsA);
    T max_mB = cublas_absmax(&cublasH, B_d, colsB*rowsB);//max_abs(B_d, d_work, c_work, colsB*rowsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;
    T lambdaC = lambdaA*lambdaB;


    quantitize_int8(A_d, AI_d, rowsA, colsA, lambdaA);
    quantitize_int8(B_d, BI_d, rowsB, colsB, lambdaB);
    I8trans(Itmp_d,BI_d,rowsB,colsB);
    cut_gemm(AI_d, Itmp_d, CI_d, rowsA, colsA, rowsB, colsB);
    dequantitize_int32(CI_d, C_d, rowsA, colsB, lambdaC);


    PA_d = (T *)Itmp_d;
    PB_d = (T *)(&PA_d[colsA*rowsA]);
    AR = (T *)(&PB_d[colsB*rowsB]);
    BR = (T *)(&AR[colsA*rowsA]);


    cublas_scopy(&cublasH,A_d,AR,colsA*rowsA);
    cublas_scopy(&cublasH,B_d,BR,colsB*rowsB);
    // cudaMemcpy(AR, A_d, colsA*rowsA * sizeof(T), cudaMemcpyDeviceToDevice);
    // cudaMemcpy(BR, B_d, colsB*rowsB * sizeof(T), cudaMemcpyDeviceToDevice);
    // cudaDeviceSynchronize();


    // /*Step 3. Calculate the residual part*/
    dequantitize_int8(AI_d, PA_d, rowsA, colsA, lambdaA);
    dequantitize_int8(BI_d, PB_d, rowsB, colsB, lambdaB);
    // here A_d and B_d represent RA and RB.
    T alpha = -1.0;
    cublas_saxpy(PA_d, AR ,alpha, rowsA*colsA, cublasH);
    cublas_saxpy(PB_d, BR ,alpha, rowsB*colsB, cublasH);
    T *AL_d, *AR_d, *BL_d, *BR_d, *tmp_d;
    int maxlen = max(max(rowsB,max(rowsA,colsA)),colsB);

    
    AL_d = (T *)(&work_dev[colsB*rowsB]);
    AR_d = (T *)(&AL_d[rowsA]);
    BL_d = (T *)(&AR_d[colsA]);
    BR_d = (T *)(&BL_d[rowsB]);
    tmp_d = (T *)(&BR_d[colsB]);

    curandGenerator_t gen;
    sketch_r1_re( AR, AL_d, AR_d,rowsA, colsA, &gen,cublashandler);
    sketch_r1_re( BR, BL_d, BR_d,rowsB, colsB, &gen,cublashandler);

    float  beta = 0.0;
    alpha = 1.0;

//begin full size correct
    cublas_gemm_rowmajor(
        &cublasH, AR_d, B_d, tmp_d,  rank,  colsA,
        rowsB,  colsB, alpha,  beta);
    beta = 1.0;
    cublas_gemm_rowmajor(
        &cublasH, AL_d, tmp_d, C_d,  rowsA,  rank,
        rank,  colsB, alpha,  beta);
    beta = 0.0;
    cublas_gemm_rowmajor(
        &cublasH, PA_d, BL_d, tmp_d,  rowsA,  colsA,
        rowsB,  rank, alpha,  beta);
    beta = 1.0;
    cublas_gemm_rowmajor(
        &cublasH, tmp_d, BR_d, C_d,  rowsA,  rank,
        rank,  colsB, alpha,  beta);

    return;
}




template <typename T,int digit>
void skxigemm_mem_fusion(
    T *A_d, T *B_d, T *C_d, char *work_dev,
    int rowsA, int colsA, int rowsB, int colsB, int rank, 
    cublasHandle_t *cublashandler) {

    rank = 1;
    using lowPtype = int8_t;

    /*Step 0. prepare Handle and stream*/
    cublasHandle_t cublasH = *cublashandler;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(cublasH, stream);

    /*Step 1. prepare work space*/
    T  *PA_d, *PB_d, *AR, *BR, *B_tmp;
    lowPtype *AI_d, *BI_d, *Itmp_d;
    int32_t *CI_d;
    // here A_d and B_d represent RA and RB.
    T alpha = -1.0;
    T *AL_d, *AR_d, *BL_d, *BR_d, *tmp_d;
    int maxlen = max(max(rowsB,max(rowsA,colsA)),colsB);

    

    AI_d = (lowPtype *)work_dev;
    BI_d = ((lowPtype *)(&work_dev[colsA*rowsA*sizeof(lowPtype)]));
    Itmp_d = ((lowPtype *)(&work_dev[colsA*rowsA*sizeof(lowPtype)+colsB*rowsB*sizeof(lowPtype)]));
    CI_d = ((int32_t *)(&work_dev[colsA*rowsA*sizeof(lowPtype)+2*colsB*rowsB*sizeof(lowPtype)]));

    PA_d = ((T *)(&work_dev[colsA*rowsA*sizeof(lowPtype)+2*colsB*rowsB*sizeof(lowPtype)+colsB*rowsA*sizeof(int32_t)]));
    PB_d = (T *)(&PA_d[colsA*rowsA]);
    AR = (T *)(&PB_d[colsB*rowsB]);
    BR = (T *)(&AR[colsA*rowsA]);


    curandGenerator_t gen;

    /*Step 2. Perform a direct quantization algorithm*/
    const int max_int = (1<<(digit-1)) - 1;
    T max_mA = cublas_absmax(&cublasH, A_d, colsA*rowsA);//max_mA2;// max_abs(A_d, d_work, c_work, colsA*rowsA);
    T max_mB = cublas_absmax(&cublasH, B_d, colsB*rowsB);//max_abs(B_d, d_work, c_work, colsB*rowsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;
    T lambdaC = lambdaA*lambdaB;


    quantitize_getR_int8(A_d, AI_d, PA_d, AR, rowsA, colsA, lambdaA);
    quantitize_getR_int8(B_d, BI_d, PB_d, BR, rowsB, colsB, lambdaB);
    I8trans(Itmp_d,BI_d,rowsB,colsB);

    AL_d = (T *)(&work_dev[colsA*rowsA*sizeof(lowPtype)]);
    AR_d = (T *)(&AL_d[rowsA]);
    BL_d = (T *)(&AR_d[colsA]);
    BR_d = (T *)(&BL_d[rowsB]);
    tmp_d = (T *)(&BR_d[colsB]);

    sketch_r1_stream( AR, AL_d, AR_d, &tmp_d[colsA],rowsA, colsA, &gen,cublashandler, &stream);
    sketch_r1_stream( BR, BL_d, BR_d, &tmp_d[4*colsA+3*rowsA],rowsB, colsB, &gen,cublashandler, &stream);


    cut_gemm_workspace(AI_d, Itmp_d, CI_d, (lowPtype *)&tmp_d[8*colsA+8*rowsA], rowsA, colsA, rowsB, colsB);

    dequantitize_int32(CI_d, C_d, rowsA, colsB, lambdaC);


    float  beta = 0.0;
    alpha = 1.0;

    //begin full size correct
    cublas_gemm_rowmajor(
        &cublasH, AR_d, B_d, tmp_d,  rank,  colsA,
        rowsB,  colsB, alpha,  beta);
    beta = 1.0;
    cublas_gemm_rowmajor(
        &cublasH, AL_d, tmp_d, C_d,  rowsA,  rank,
        rank,  colsB, alpha,  beta);
    beta = 0.0;
    cublas_gemm_rowmajor(
        &cublasH, PA_d, BL_d, tmp_d,  rowsA,  colsA,
        rowsB,  rank, alpha,  beta);
    beta = 1.0;
    cublas_gemm_rowmajor(
        &cublasH, tmp_d, BR_d, C_d,  rowsA,  rank,
        rank,  colsB, alpha,  beta);

    return;
}



void skxhgemm(
    half *A_d, half *B_d, float *C_d, float *RA, float *RB,  float *PA_d, float *PB_d, 
    int rowsA, int colsA, int rowsB, int colsB, int rank, cublasHandle_t *cublashandler) {

    rank = 1;

    /*Step 0. prepare Handle and stream*/
    cublasHandle_t cublasH = *cublashandler;
    cudaStream_t stream = NULL;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    float  beta = 0.0, alpha = 1.0;
    cublas_gemm_rowmajor(
        &cublasH, A_d, B_d, C_d, rowsA, colsA,
        rowsB, colsB, alpha, beta);    

    float *AL_d, *AR_d, *BL_d, *BR_d, *tmp_d;
    int maxlen = max(max(rowsB,max(rowsA,colsA)),colsB);
    cudaMalloc((float **)&AL_d, sizeof(float) * rowsA );
    cudaMalloc((float **)&AR_d, sizeof(float) * colsA);
    cudaMalloc((float **)&BL_d, sizeof(float) * rowsB);
    cudaMalloc((float **)&BR_d, sizeof(float) * colsB);
    cudaMalloc((float **)&tmp_d, sizeof(float) * maxlen);

    curandGenerator_t gen;
    sketch_r1( RA, AL_d, AR_d,rowsA, colsA, &gen,cublashandler);
    sketch_r1( RB, BL_d, BR_d,rowsB, colsB, &gen,cublashandler);

    beta = 0.0;
    alpha = 1.0;

//begin full size correct
    // cublas_gemm_rowmajor(
    //     &cublasH, AR_d, PB_d, tmp_d,  rank,  colsA,
    //     rowsB,  colsB, alpha,  beta);
    // beta = 1.0;
    // cublas_gemm_rowmajor(
    //     &cublasH, AL_d, tmp_d, C_d,  rowsA,  rank,
    //     rank,  colsB, alpha,  beta);
    // beta = 0.0;
    // cublas_gemm_rowmajor(
    //     &cublasH, PA_d, BL_d, tmp_d,  rowsA,  colsA,
    //     rowsB,  rank, alpha,  beta);
    // beta = 1.0;
    // cublas_gemm_rowmajor(
    //     &cublasH, tmp_d, BR_d, C_d,  rowsA,  rank,
    //     rank,  colsB, alpha,  beta);
    // beta = 0.0;
    // cublas_gemm_rowmajor(
    //     &cublasH, RA, BL_d, tmp_d,  rowsA,  colsA,
    //     rowsB,  rank, alpha,  beta);
    // beta = 1.0;
    // cublas_gemm_rowmajor(
    //     &cublasH, tmp_d, BR_d, C_d,  rowsA,  rank,
    //     rank,  colsB, alpha,  beta);

    beta = 1.0;
    cublas_gemm_rowmajor(
        &cublasH, RA, PB_d, C_d,  rowsA,  colsA,
        rowsB,  colsB, alpha,  beta);
    cublas_gemm_rowmajor(
        &cublasH, PA_d, RB, C_d,  rowsA,  colsA,
        rowsB,  colsB, alpha,  beta);
    cublas_gemm_rowmajor(
        &cublasH, RA, RB, C_d,  rowsA,  colsA,
        rowsB,  colsB, alpha,  beta);
    return;
}


template <typename T,int digit>
void skxigemm_before(
    T *A_d, T *B_d, T *C_d, 
    int rowsA, int colsA, int rowsB, int colsB, int rank, 
    cusolverDnHandle_t *cusolverhandler, cublasHandle_t *cublashandler) {

    rank = 1;
    using lowPtype = int8_t;

    /*Step 0. prepare Handle and stream*/
    cublasHandle_t cublasH = *cublashandler;
    cudaStream_t stream = NULL;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /*Step 1. prepare work space*/
    T  *PA_d, *PB_d, *AR, *BR, *B_tmp, *A_dbef;
    lowPtype *AI_d, *BI_d, *Itmp_d;
    int32_t *CI_d;

    cudaMalloc((T **)&A_dbef, sizeof(T) * colsA*rowsA);
    cudaMalloc((T **)&PA_d, sizeof(T) * colsA*rowsA);
    cudaMalloc((T **)&PB_d, sizeof(T) * colsB*rowsB);
    cudaMalloc((T **)&AR, sizeof(T) * colsA*rowsA);
    cudaMalloc((T **)&BR, sizeof(T) * colsB*rowsB);

    cudaMalloc((lowPtype **)&AI_d, sizeof(lowPtype) * colsA*rowsA);
    cudaMalloc((lowPtype **)&BI_d, sizeof(lowPtype) * colsB*rowsB);
    cudaMalloc((int32_t **)&CI_d, sizeof(int32_t) * rowsA*colsB);
    cudaMalloc((lowPtype **)&Itmp_d, sizeof(lowPtype) * colsB*rowsB);


    T alpha = 1.0;
    T *AL_d, *AR_d, *BL_d, *BR_d, *tmp_d;
    int maxlen = max(max(rowsB,max(rowsA,colsA)),colsB);
    cudaMalloc((T **)&AL_d, sizeof(T) * rowsA );
    cudaMalloc((T **)&AR_d, sizeof(T) * colsA);
    cudaMalloc((T **)&BL_d, sizeof(T) * rowsB);
    cudaMalloc((T **)&BR_d, sizeof(T) * colsB);
    cudaMalloc((T **)&tmp_d, sizeof(T) * maxlen);

    curandGenerator_t gen;
    sketch_r1( A_d, AL_d, AR_d,rowsA, colsA, &gen,cublashandler);

    float beta = 0.0;
    cublas_gemm_rowmajor(
        &cublasH, AL_d, AR_d, AR,  rowsA,  1,
        1,  colsA, alpha,  beta);


    beta = 0.0;
    alpha = 1.0;

//begin full size correct
    cublas_gemm_rowmajor(
        &cublasH, AR_d, B_d, tmp_d,  rank,  colsA,
        rowsB,  colsB, alpha,  beta);
    beta = 0.0;
    cublas_gemm_rowmajor(
        &cublasH, AL_d, tmp_d, C_d,  rowsA,  rank,
        rank,  colsB, alpha,  beta);



    cudaMemcpy(A_dbef, A_d, colsA*rowsA * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    alpha = -1.0;
    cublas_saxpy(AR, A_dbef ,alpha, rowsA*colsA, cublasH,stream);
    cudaDeviceSynchronize();


    /*Step 2. Perform a direct quantization algorithm*/
    const int max_int = (1<<(digit-1)) - 1;
    T max_mA = cublas_absmax(&cublasH, A_dbef, colsA*rowsA);//max_mA2;// max_abs(A_d, d_work, c_work, colsA*rowsA);
    T max_mB = cublas_absmax(&cublasH, B_d, colsB*rowsB);//max_abs(B_d, d_work, c_work, colsB*rowsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;
    T lambdaC = lambdaA*lambdaB;

    quantitize_int8(A_dbef, AI_d, rowsA, colsA, lambdaA);
    quantitize_int8(B_d, BI_d, rowsB, colsB, lambdaB);
    I8trans(Itmp_d,BI_d,rowsB,colsB);
    cut_gemm(AI_d, Itmp_d, CI_d, rowsA, colsA, rowsB, colsB);
    dequantitize_int32(CI_d, AR, rowsA, colsB, lambdaC);


    alpha = 1.0;
    cublas_saxpy(AR, C_d ,alpha, rowsA*colsB, cublasH,stream);


    cudaMemcpy(AR, A_dbef, colsA*rowsA * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(BR, B_d, colsB*rowsB * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();


    // /*Step 3. Calculate the residual part*/
    dequantitize_int8(AI_d, PA_d, rowsA, colsA, lambdaA);
    dequantitize_int8(BI_d, PB_d, rowsB, colsB, lambdaB);
    // here A_d and B_d represent RA and RB.
    alpha = -1.0;
    cublas_saxpy(PA_d, AR ,alpha, rowsA*colsA, cublasH,stream);
    cublas_saxpy(PB_d, BR ,alpha, rowsB*colsB, cublasH,stream);

    sketch_r1( AR, AL_d, AR_d,rowsA, colsA, &gen,cublashandler);
    sketch_r1( BR, BL_d, BR_d,rowsB, colsB, &gen,cublashandler);

    beta = 0.0;
    alpha = 1.0;

//begin full size correct
    cublas_gemm_rowmajor(
        &cublasH, AR_d, PB_d, tmp_d,  rank,  colsA,
        rowsB,  colsB, alpha,  beta);
    beta = 1.0;
    cublas_gemm_rowmajor(
        &cublasH, AL_d, tmp_d, C_d,  rowsA,  rank,
        rank,  colsB, alpha,  beta);
    beta = 0.0;
    cublas_gemm_rowmajor(
        &cublasH, PA_d, BL_d, tmp_d,  rowsA,  colsA,
        rowsB,  rank, alpha,  beta);
    beta = 1.0;
    cublas_gemm_rowmajor(
        &cublasH, tmp_d, BR_d, C_d,  rowsA,  rank,
        rank,  colsB, alpha,  beta);
    beta = 0.0;
    cublas_gemm_rowmajor(
        &cublasH, AR, BL_d, tmp_d,  rowsA,  colsA,
        rowsB,  rank, alpha,  beta);
    beta = 1.0;
    cublas_gemm_rowmajor(
        &cublasH, tmp_d, BR_d, C_d,  rowsA,  rank,
        rank,  colsB, alpha,  beta);


    return;
}

