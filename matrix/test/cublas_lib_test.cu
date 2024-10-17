#include "../cublas/cublas_lib.cuh"
#include "../operator_matrix.cuh"
#include "../gen_matrix.cuh"

#include <iomanip>
#include "stdio.h"
#include <chrono>


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
    int M=16,N=14,K=2;
    // 创建一个使用float类型的数组
    std::vector<float> int4b_arrayA(M*K);
    std::vector<float> int4b_arrayB(K*N);
    std::vector<float> int32b_arrayC(M*N);


    // 初始化数组
    for (int i = 0; i < M; ++i) {
        for(int j=0;j<K;j++){
            // 将每个元素初始化为它的索引值，注意这里只是示例，实际值可能需要根据量化规则来确定
            int4b_arrayA[i*K+j] = static_cast<float>(i*K+j);
        }
    }
    for (int i = 0; i < K; ++i) {
        for(int j=0;j<N;j++){
            // 将每个元素初始化为它的索引值，注意这里只是示例，实际值可能需要根据量化规则来确定
            int4b_arrayB[i*N+j] = static_cast<float>(j);
        }
    }


    for (int i = 0; i < M; ++i) {
        for(int j=0;j<K;j++){
            printf("%d,",static_cast<int>(int4b_arrayA[i*K+j]));
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < K; ++i) {
        for(int j=0;j<N;j++){
            printf("%d,",static_cast<int>(int4b_arrayB[i*N+j]));
        }
        printf("\n");
    }
    printf("\n");

    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * M*K);
    cudaMalloc((void**)&d_B, sizeof(float) * K*N);
    cudaMalloc((void**)&d_C, sizeof(float) * M*N);
    cudaMemcpy(d_A, int4b_arrayA.data(), sizeof(float) * M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, int4b_arrayB.data(), sizeof(float) * K*N, cudaMemcpyHostToDevice);

    float  beta = 0.0, alpha = 1.0;
    // cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K,
    //             &alpha, d_A, M, d_B, M, &beta, d_C, M);  

    // cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
    //             &alpha, d_B, N, d_A, K, &beta, d_C, N);  

    cublas_gemm_rowmajor(
        &cublasH, d_A, d_B, d_C, M, K,
        K, N, alpha, beta);

    cudaMemcpy( int32b_arrayC.data(),d_C, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
  
    for (int i = 0; i < M; ++i) {
        for(int j=0;j<N;j++){
            printf("%d,",static_cast<int>(int32b_arrayC[i*N+j]));
        }
        printf("\n");
    }

}


void gemm_acc_test2(){
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    // 定义数组的大小
    int M=16,N=14,K=2;
    // 创建一个使用float类型的数组
    std::vector<float> int4b_arrayA(M*K);
    std::vector<float> int4b_arrayB(K*N);
    std::vector<float> int32b_arrayC(M*N);


    // 初始化数组
    for (int i = 0; i < M; ++i) {
        for(int j=0;j<K;j++){
            // 将每个元素初始化为它的索引值，注意这里只是示例，实际值可能需要根据量化规则来确定
            int4b_arrayA[i*K+j] = static_cast<float>(i*K+j);
        }
    }
    for (int i = 0; i < K; ++i) {
        for(int j=0;j<N;j++){
            // 将每个元素初始化为它的索引值，注意这里只是示例，实际值可能需要根据量化规则来确定
            int4b_arrayB[i*N+j] = static_cast<float>(j);
        }
    }


    for (int i = 0; i < M; ++i) {
        for(int j=0;j<K;j++){
            printf("%d,",static_cast<int>(int4b_arrayA[i*K+j]));
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < K; ++i) {
        for(int j=0;j<N;j++){
            printf("%d,",static_cast<int>(int4b_arrayB[i*N+j]));
        }
        printf("\n");
    }
    printf("\n");

    float* d_A;
    float* d_B;
    float* d_C;
    float* d_C_TMP;
    cudaMalloc((void**)&d_A, sizeof(float) * M*K);
    cudaMalloc((void**)&d_B, sizeof(float) * K*N);
    cudaMalloc((void**)&d_C, sizeof(float) * M*N);
    cudaMalloc((void**)&d_C_TMP, sizeof(float) * M*N);
    cudaMemcpy(d_A, int4b_arrayA.data(), sizeof(float) * M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, int4b_arrayB.data(), sizeof(float) * K*N, cudaMemcpyHostToDevice);

    float  beta = 0.0, alpha = 1.0;
    // cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K,
    //             &alpha, d_A, M, d_B, M, &beta, d_C, M);  

    // cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
    //             &alpha, d_B, N, d_A, K, &beta, d_C, N);  


    cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, d_A, K, d_B, K, &beta, d_C_TMP, M);
    cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N, M, &alpha, d_C_TMP, M, &beta, d_C, N, d_C, N);
    cudaDeviceSynchronize();


    cudaMemcpy( int32b_arrayC.data(),d_C, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
  
    for (int i = 0; i < M; ++i) {
        for(int j=0;j<N;j++){
            printf("%d,",static_cast<int>(int32b_arrayC[i*N+j]));
        }
        printf("\n");
    }

}


void gemv_acc_test(){
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    // 定义数组的大小
    int M=32,N=16;
    // 创建一个使用float类型的数组
    std::vector<float> int4b_arrayA(M*N);
    std::vector<float> int4b_arrayB(N);
    std::vector<float> int32b_arrayC(M*4);


    // 初始化数组
    for (int i = 0; i < M; ++i) {
        for(int j=0;j<N;j++){
            // 将每个元素初始化为它的索引值，注意这里只是示例，实际值可能需要根据量化规则来确定
            int4b_arrayA[i*N+j] = static_cast<float>(i+1);
        }
    }
    for (int i = 0; i < N; ++i) int4b_arrayB[i] = i;
    for (int i = 0; i < M; ++i) int32b_arrayC[i] = 0;

    



    for (int i = 0; i < M; ++i) {
        for(int j=0;j<N;j++){
            printf("%d,",static_cast<int>(int4b_arrayA[i*N+j]));
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < N; ++i) printf("%d,",static_cast<int>(int4b_arrayB[i]));

    printf("\n");

    float* d_A;
    float* d_B;
    float* d_C;
    float* d_A_tmp;
    cudaMalloc((void**)&d_A, sizeof(float) * M*N);
    cudaMalloc((void**)&d_A_tmp, sizeof(float) * M*N);
    cudaMalloc((void**)&d_B, sizeof(float) * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M);
    cudaMemcpy(d_A, int4b_arrayA.data(), sizeof(float) * M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, int4b_arrayB.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

    float  beta = 0.0, alpha = 1.0;
    
    // strans(d_A_tmp,d_A,M,N);
    // cublasSgemv(cublasH, CUBLAS_OP_N, M, N, 
    //             &alpha, d_A_tmp, M, d_B, 1, &beta, d_C, 1);  

    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 1, 4, 4,
                    &alpha, d_B, 1, d_A, N, &beta, d_C, 4);  
    cudaDeviceSynchronize();

    cudaMemcpy( int32b_arrayC.data(),d_C, sizeof(float) * M, cudaMemcpyDeviceToHost);
    printf("\n");

    for (int i = 0; i < M; ++i) {
        for(int j=0;j<4;j++){
            printf("%d,",static_cast<int>(int32b_arrayC[i*4+j]));
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
}



void gemv_perf_test(){
    int test_para[2048][5] = {
        {1024,1024},
        {2048,2048},
        {3072,3072},
        {4096,4096},
        {5120,5120},
        {6144,6144},
        {7168,7168},
        {8192,8192},
        {9216,9216},
        {10240,10240},
        {11264,11264},
        {12288,12288},
        {13312,13312},
        {14336,14336},
        {15360,15360},
        {16384,16384},
        {17408,17408},
        {18432,18432},
        {19456,19456},
        {20480,20480},
        {21504,21504},
        {22528,22528},
        {23552,23552},
        {24576,24576},
        {25600,25600},
        {26624,26624},
        {27648,27648},
        {28672,28672},
        {29696,29696},
        {30720,30720},
        {31744,31744},
        {32768,32768},
        }; 
  
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));

    int max = 4096*8;
    float *matrixA = (float *)malloc(sizeof(float) * max*max);
    float *matrixB = (float *)malloc(sizeof(float) * max*max);
    float *matrixC = (float *)malloc(sizeof(float) * max*max);

    char * work;
    cudaMalloc((char **)&work, sizeof(float) * (max*max*6+max*5));

    std::cout<<"M\tN\tSGEMV\n";
    const int digit = 8;
    
    float *A_d, *B_d, *C_d;
    generate_matrix<float>(matrixA,max,max,'u');
    generate_matrix<float>(matrixB,max,max,'u');   
    cudaMalloc((float **)&A_d, sizeof(float) * max*max);
    cudaMalloc((float **)&B_d, sizeof(float) * max);
    cudaMalloc((float **)&C_d, sizeof(float) * max);
    cudaMemcpy(A_d, matrixA, sizeof(float) * max*max, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, matrixB, sizeof(float) * max, cudaMemcpyHostToDevice);    
    for(int i=0;i<32;i++){

        int N=test_para[i][0],M=test_para[i][1];
        
        float alpha = 1.0, beta = 0.0;
        if(M==0) return;
        cudaDeviceSynchronize();
        std::cout<<M<<"\t"<<N<<"\t";

        //计算float和int矩阵乘法得到结果矩阵

        // xgemm(matrixA,matrixB,matrixC,M,K,K,N);
          beta = 0.0, alpha = 1.0;
        {
            auto start = std::chrono::high_resolution_clock::now();
            cublas_gemv_rowmajor( &cublasH,A_d, B_d, C_d, M, N, alpha, beta);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> diff = end - start;
            double time  = diff.count();
            printf("%.7lf\n",time);
            cudaDeviceSynchronize();
        }

    }
    return;        
}


void hgemm_perf_test(){

    int test_para[2048][3] = {
        {1024,1024,1024},
        {2048,2048,2048},
        {3072,3072,3072},
        {4096,4096,4096},
        {5120,5120,5120},
        {6144,6144,6144},
        {7168,7168,7168},
        {8192,8192,8192},
        {9216,9216,9216},
        {10240,10240,10240},
        {11264,11264,11264},
        {12288,12288,12288},
        {13312,13312,13312},
        {14336,14336,14336},
        {15360,15360,15360},
        {16384,16384,16384},
        {17408,17408,17408},
        {18432,18432,18432},
        {19456,19456,19456},
        {20480,20480,20480},
        {21504,21504,21504},
        {22528,22528,22528},
        {23552,23552,23552},
        {24576,24576,24576},
        {25600,25600,25600},
        {26624,26624,26624},
        {27648,27648,27648},
        {28672,28672,28672},
        {29696,29696,29696},
        {30720,30720,30720},
        {31744,31744,31744},
        {32768,32768,32768},
    }; 

    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    // 定义数组的大小

    int max = 4096*8;
    // 创建一个使用float类型的数组
    std::vector<half> int4b_arrayA(max*max);
    std::vector<half> int4b_arrayB(max*max);
    std::vector<half> int32b_arrayC(max*max);

    half* d_A;
    half* d_B;
    half* d_C;
    float* d_C32;
    cudaMalloc((void**)&d_A, sizeof(half) * max*max);
    cudaMalloc((void**)&d_B, sizeof(half) * max*max);
    cudaMalloc((void**)&d_C, sizeof(half) * max*max);
    cudaMalloc((void**)&d_C32, sizeof(float) * max*max);
    cudaMemcpy(d_A, int4b_arrayA.data(), sizeof(half) * max*max, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, int4b_arrayB.data(), sizeof(half) * max*max, cudaMemcpyHostToDevice);

    float beta = 0.0, alpha = 1.0;

    std::cout<<"M\tN\tK\th16-16_16\th16-16_32\th16-32_32\n";

    for(int i=0;i<32;i++){
        

        int N=test_para[i][0],M=test_para[i][1],K=test_para[i][2];
        half alpha = 1.0, beta = 0.0;
        if(M==0) return;

        std::cout<<M<<"\t"<<N<<"\t"<<K<<"\t";

        //计算float和int矩阵乘法得到结果矩阵
        cublas_gemm_rowmajor(
            &cublasH, d_A, d_B, d_C, M, K,
            K, N, alpha, beta);
        {
            auto start = std::chrono::high_resolution_clock::now();
            cublas_gemm_rowmajor(
                &cublasH, d_A, d_B, d_C, M, K,
                K, N, alpha, beta);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> diff = end - start;
            double time  = diff.count();
            printf("%.7lf\t",time);
            cudaDeviceSynchronize();
        }
        float alpha2 = 1.0, beta2 = 0.0;
        {
            auto start = std::chrono::high_resolution_clock::now();
            cublas_gemm_rowmajor(
                &cublasH, d_A, d_B, d_C, M, K,
                K, N, alpha2, beta2);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> diff = end - start;
            double time  = diff.count();
            printf("%.7lf\t",time);
            cudaDeviceSynchronize();
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            cublas_gemm_rowmajor(
                &cublasH, d_A, d_B, d_C32, M, K,
                K, N, alpha2, beta2);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> diff = end - start;
            double time  = diff.count();
            printf("%.7lf\n",time);
            cudaDeviceSynchronize();
        }


    }
    return;        

}

void mslag2d(float *in, half *out,int size){
    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0; i<size; i++){
        out[i] = (half)in[i];
    }
}
void mdlag2s(half *in, float *out,int size){
    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0; i<size; i++){
        out[i] = (float)in[i];
    }
}

void mslag2h_withR(float *in, half *out, float *res, float *pin,int size){
    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0; i<size; i++){
        out[i] = (half)in[i];
        res[i] = in[i] - (float)out[i];
        pin[i] = (float)out[i];
    }
}


void hgemm1616_acc_test(){
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    // 定义数组的大小
    int M=1024,N=1024,K=1024;
    // 创建一个使用float类型的数组
    std::vector<float> arrayA(M*K);
    std::vector<float> arrayB(K*N);
    std::vector<float> arrayC(M*N);
    std::vector<float> arrayhfC(M*N);


    generate_matrix<float>(arrayA.data(),M,K,'k');
    generate_matrix<float>(arrayB.data(),K,N,'k');    


    float* d_A;
    float* d_B;
    float* d_C;
    float* d_C_TMP;
    cudaMalloc((void**)&d_A, sizeof(float) * M*K);
    cudaMalloc((void**)&d_B, sizeof(float) * K*N);
    cudaMalloc((void**)&d_C, sizeof(float) * M*N);
    cudaMalloc((void**)&d_C_TMP, sizeof(float) * M*N);
    cudaMemcpy(d_A, arrayA.data(), sizeof(float) * M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, arrayB.data(), sizeof(float) * K*N, cudaMemcpyHostToDevice);

    float  beta = 0.0, alpha = 1.0;




    cublas_gemm_rowmajor(
        &cublasH, d_A, d_B, d_C, M, K,
        K, N, alpha, beta);

    cudaMemcpy( arrayC.data(),d_C, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
  

    // 创建一个使用half类型的数组
    std::vector<half> arrayhA(M*K);
    std::vector<half> arrayhB(K*N);
    std::vector<half> arrayhC(M*N);

    mslag2d(arrayA.data(),arrayhA.data(),M*K);
    mslag2d(arrayB.data(),arrayhB.data(),N*K);
    // mslag2d(arrayC.data(),arrayhC.data(),M*N);


    half* d_hA;
    half* d_hB;
    half* d_hC;
    cudaMalloc((void**)&d_hA, sizeof(half) * M*K);
    cudaMalloc((void**)&d_hB, sizeof(half) * K*N);
    cudaMalloc((void**)&d_hC, sizeof(half) * M*N);
    cudaMemcpy(d_hA, arrayhA.data(), sizeof(half) * M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hB, arrayhB.data(), sizeof(half) * K*N, cudaMemcpyHostToDevice);
    

    half  beta2 = 0.0, alpha2 = 1.0;    
    cublas_gemm_rowmajor(
        &cublasH, d_hA, d_hB, d_hC, M, K,
        K, N, alpha2, beta2);
    cudaDeviceSynchronize();

    cudaMemcpy( arrayhC.data(),d_hC, sizeof(half) * M*N, cudaMemcpyDeviceToHost);
  
    // for (int i = 0; i < M; ++i) {
    //     for(int j=0;j<N;j++){
    //         printf("%f,",(float)(arrayhB[i*N+j]));
    //     }
    //     printf("\n");
    // }

    mdlag2s(arrayhC.data(),arrayhfC.data(),M*N);
    float R3 = get_Ferror<float>(arrayC.data(),arrayhfC.data(),M,N); 

    printf("%.7f\n",R3);
}
void hgemm1632_acc_test(){
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    // 定义数组的大小
    int M=1024,N=1024,K=1024;
    // 创建一个使用float类型的数组
    std::vector<float> arrayA(M*K);
    std::vector<float> arrayB(K*N);
    std::vector<float> arrayC(M*N);
    std::vector<float> arrayhfC(M*N);


    generate_matrix<float>(arrayA.data(),M,K,'k');
    generate_matrix<float>(arrayB.data(),K,N,'k');    


    float* d_A;
    float* d_B;
    float* d_C;
    float* d_C_TMP;
    cudaMalloc((void**)&d_A, sizeof(float) * M*K);
    cudaMalloc((void**)&d_B, sizeof(float) * K*N);
    cudaMalloc((void**)&d_C, sizeof(float) * M*N);
    cudaMalloc((void**)&d_C_TMP, sizeof(float) * M*N);
    cudaMemcpy(d_A, arrayA.data(), sizeof(float) * M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, arrayB.data(), sizeof(float) * K*N, cudaMemcpyHostToDevice);

    float  beta = 0.0, alpha = 1.0;




    cublas_gemm_rowmajor(
        &cublasH, d_A, d_B, d_C, M, K,
        K, N, alpha, beta);

    cudaMemcpy( arrayC.data(),d_C, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
  

    // 创建一个使用half类型的数组
    std::vector<half> arrayhA(M*K);
    std::vector<half> arrayhB(K*N);
    std::vector<float> arrayhC(M*N);

    mslag2d(arrayA.data(),arrayhA.data(),M*K);
    mslag2d(arrayB.data(),arrayhB.data(),N*K);
    // mslag2d(arrayC.data(),arrayhC.data(),M*N);


    half* d_hA;
    half* d_hB;
    float* d_hC;
    cudaMalloc((void**)&d_hA, sizeof(half) * M*K);
    cudaMalloc((void**)&d_hB, sizeof(half) * K*N);
    cudaMalloc((void**)&d_hC, sizeof(float) * M*N);
    cudaMemcpy(d_hA, arrayhA.data(), sizeof(half) * M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hB, arrayhB.data(), sizeof(half) * K*N, cudaMemcpyHostToDevice);
    

    float  beta2 = 0.0, alpha2 = 1.0;    
    cublas_gemm_rowmajor(
        &cublasH, d_hA, d_hB, d_hC, M, K,
        K, N, alpha2, beta2);
    cudaDeviceSynchronize();

    cudaMemcpy( arrayhfC.data(),d_hC, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
  
    // for (int i = 0; i < M; ++i) {
    //     for(int j=0;j<N;j++){
    //         printf("%f,",(float)(arrayhB[i*N+j]));
    //     }
    //     printf("\n");
    // }
    float R3 = get_Ferror<float>(arrayC.data(),arrayhfC.data(),M,N); 

    printf("%.7f\n",R3);
}

int main(){
    //axpy_perf_test();
    //gemv_acc_test();
    //gemm_acc_test();
    //gemm_acc_test2();
    //hgemm_acc_test();

    // gemv_perf_test();
    // hgemm_perf_test();
    hgemm_perf_test();

    hgemm1616_acc_test();
    hgemm1632_acc_test();
    return 0;
}