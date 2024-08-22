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

    cusolver_rsvd_LR(rowsA, colsA, AR, AL_d, AR_d, rank, &cusolverH);
    cusolver_rsvd_LR(rowsB, colsB, BR, BL_d, BR_d, rank, &cusolverH);
    


    float  beta = 0.0;
    alpha = 1.0;

    //下面是先算出残差部分的测试，目前第一步计算有问题
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

    T *PA_d, *PB_d;
    lowPtype *AI_d, *BI_d, *Itmp_d;
    int32_t *CI_d;

    cudaMalloc((T **)&PA_d, sizeof(T) * colsA*rowsA);
    cudaMalloc((T **)&PB_d, sizeof(T) * colsB*rowsB);

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

    // lowPtype *matrixTmp = (lowPtype *)malloc(sizeof(lowPtype) * rowsA*colsA);
    // cudaMemcpy(matrixTmp, BI_d, sizeof(lowPtype) * 1024, cudaMemcpyDeviceToHost);

    // float *mf = (float *)malloc(sizeof(float) * rowsB*colsB);
    // cudaMemcpy(mf, B_d, sizeof(float) * 1024, cudaMemcpyDeviceToHost);
    
    // for(int i=0;i<1024;i++){
    //     printf("%f,",mf[i]);
    // }


    // for(int i=0;i<1024;i++){
    //     printf("%d,",(int)matrixTmp[i]);
    // }


    I8trans(Itmp_d,BI_d,rowsB,colsB);
    cut_gemm(AI_d, Itmp_d, CI_d, rowsA, colsA, rowsB, colsB);
    cudaDeviceSynchronize();


    dequantitize_int32(CI_d, C_d, rowsA, colsB, lambdaC);
    cudaDeviceSynchronize();

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

    curandGenerator_t gen;
    sketch_r1( AR, AL_d, AR_d,rowsA, colsA, &gen,cublashandler,d_work,c_work);
    sketch_r1( BR, BL_d, BR_d,rowsB, colsB, &gen,cublashandler,d_work,c_work);


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
