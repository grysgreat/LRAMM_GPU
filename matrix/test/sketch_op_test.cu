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



void  curand_test(){
    int len = 32;
    float *Sketch;
    curandGenerator_t gen;
    curandGenerator_t *gen_p=&gen;
    cudaMalloc((void**)&Sketch, sizeof(float) * len);
    curandCreateGenerator(gen_p, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateNormal(*gen_p, Sketch, len, 0.0f, 1.0f);

    float *Sketch_h = (float *)malloc(sizeof(float) * len);

    cudaMemcpy(Sketch_h, Sketch, sizeof(float) * len, cudaMemcpyDeviceToHost);
    for(int j=0;j<len;j++){
        printf("%.4f, ",Sketch_h[j]);
    }    
    printf("\n\n");
    return;

}

void sketch_acc_test(){
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    int M=32,N=16;

    std::vector<float> int4b_arrayA(M*N);
    std::vector<float> int4b_arrayB(M);
    std::vector<float> int32b_arrayC(N);


    for (int i = 0; i < M; ++i) {
        for(int j=0;j<N;j++){
            int4b_arrayA[i*N+j] = static_cast<float>(i+1);
        }
    }




    for (int i = 0; i < M; ++i) {
        for(int j=0;j<N;j++){
            printf("%d,",static_cast<int>(int4b_arrayA[i*N+j]));
        }
        printf("\n");
    }

    printf("\n");

    float* d_A;
    float* d_B;
    float* d_C;
    float* d_A_tmp;
    cudaMalloc((void**)&d_A, sizeof(float) * M*N);
    cudaMalloc((void**)&d_A_tmp, sizeof(float) * M*N);
    cudaMalloc((void**)&d_B, sizeof(float) * M);
    cudaMalloc((void**)&d_C, sizeof(float) * N);
    cudaMemcpy(d_A, int4b_arrayA.data(), sizeof(float) * M*N, cudaMemcpyHostToDevice);

    curandGenerator_t gen;

    /*prepare work space*/
    int threadsPerBlock = 2048; 
    int max_work_size = (max(M*N, 1)+threadsPerBlock-1)/threadsPerBlock;

    float* c_work = (float *)malloc(sizeof(float) * max_work_size);
    float* d_work;
    cudaMalloc((float **)&d_work, sizeof(float) * max_work_size);

    sketch_r1(
        d_A,  d_B, d_C, M, N, &gen, &cublasH, d_work, c_work
    );
    


    cudaMemcpy( int4b_arrayB.data(), d_B, sizeof(float) * M, cudaMemcpyDeviceToHost);
    cudaMemcpy( int32b_arrayC.data(), d_C, sizeof(float) * N, cudaMemcpyDeviceToHost);
    printf("\n");

    for (int i = 0; i < M; ++i) {
        printf("%f,",(int4b_arrayB[i]));
    }
    printf("\n");
    for (int i = 0; i < N; ++i) {
        printf("%f,",(int32b_arrayC[i]));
    }
    printf("\n");
    printf("\n");

    cublas_gemm_rowmajor(
        &cublasH, d_B, d_C, d_A,  M,  1,
        1,  N, 1.0,  0.0);    
    cudaMemcpy( int4b_arrayA.data(), d_A, sizeof(float) * N* M, cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; ++i) {
        for(int j=0;j<N;j++){
            printf("%d,",static_cast<int>(int4b_arrayA[i*N+j]));
        }
        printf("\n");
    }


}




void skxigemm_acc(){
    int max = 1024;
    float *matrixA = (float *)malloc(sizeof(float) * max*max);
    float *matrixB = (float *)malloc(sizeof(float) * max*max);
    float *matrixC = (float *)malloc(sizeof(float) * max*max);
    float *matrixCQ = (float *)malloc(sizeof(float) * max*max);
    float *matrixR = (float *)malloc(sizeof(float) * max*max);

    int M=max , N=max, K = max;

    generate_matrix<float>(matrixA,M,K,'u');
    generate_matrix<float>(matrixB,K,N,'u');    

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

    skxigemm<float,8>(A_d,B_d,C_d,M,K,K,N,10, &cusolverH, &cublasH);
    cudaMemcpy( matrixCQ,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 

    printf("%.7f\n",R3);


    return ;
}

void precision_test(){
    int test_para[2048][5] = {

        // {2048,2048,2048,50},
        // {2048,2048,2048,40},
        // {2048,2048,2048,30},
        // {2048,2048,2048,20},
        // {2048,2048,2048,10},

        {4096,4096,4096,10,'n'},
        {4096,4096,4096,10,'u'},
        {4096,4096,4096,10,'s'},
        {4096,4096,4096,10,'e'},
        {4096,4096,4096,10,'k'},
        {4096,4096,4096,10,'p'},

        // {128,128,128,10,'k'},
        // {256,256,256,10,'k'},
        // {384,384,384,10,'k'},
        // {512,512,512,10,'k'},
        // {640,640,640,10,'k'},
        // {768,768,768,10,'k'},
        // {896,896,896,10,'k'},
        // {2048,2048,2048,10,'k'},
        // {1152,1152,1152,10,'k'},
        // {1280,1280,1280,10,'k'},
        // {1408,1408,1408,10,'k'},
        // {1536,1536,1536,10,'k'},
        // {1664,1664,1664,10,'k'},
        // {1792,1792,1792,10,'k'},
        // {1920,1920,1920,10,'k'},
        // {2048,2048,2048,10,'k'},
        // {2176,2176,2176,10,'k'},
        // {2304,2304,2304,10,'k'},
        // {2432,2432,2432,10,'k'},
        // {2560,2560,2560,10,'k'},
        // {2688,2688,2688,10,'k'},
        // {2816,2816,2816,10,'k'},
        // {2944,2944,2944,10,'k'},
        // {3072,3072,3072,10,'k'},
        // {3200,3200,3200,10,'k'},
        // {3328,3328,3328,10,'k'},
    }; 
  
    cublasHandle_t cublasH = NULL;
    cusolverDnHandle_t cusolverH = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    int max = 8192;
    float *matrixA = (float *)malloc(sizeof(float) * max*max);
    float *matrixB = (float *)malloc(sizeof(float) * max*max);
    float *matrixC = (float *)malloc(sizeof(float) * max*max);
    float *matrixCQ = (float *)malloc(sizeof(float) * max*max);
    float *matrixR = (float *)malloc(sizeof(float) * max*max);

    std::cout<<"M\tN\tK\ttype\trank\torigin\t\tLrxigemm\tsketch\n";
    const int digit = 8;
    
    float *A_d, *B_d, *C_d;
    for(int i=0;i<5;i++){
        

        int N=test_para[i][0],M=test_para[i][1],K=test_para[i][2];
        int rank =test_para[i][3];
        char type =test_para[i][4];

        float alpha = 1.0, beta = 0.0;

        if(i!=0) {
            if(N!=test_para[i-1][0]||M!=test_para[i-1][1]||K!=test_para[i-1][2]||type!=test_para[i-1][4]){
                generate_matrix<float>(matrixA,M,K,type);
                generate_matrix<float>(matrixB,K,N,type);
            }
        } else {
            generate_matrix<float>(matrixA,M,K,type);
            generate_matrix<float>(matrixB,K,N,type);            
        }
        cudaMalloc((float **)&A_d, sizeof(float) * M*K);
        cudaMalloc((float **)&B_d, sizeof(float) * K*N);
        cudaMalloc((float **)&C_d, sizeof(float) * M*N);
        cudaMemcpy(A_d, matrixA, sizeof(float) * M*K, cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, matrixB, sizeof(float) * K*N, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        std::cout<<M<<"\t"<<N<<"\t"<<K<<"\t"<<type<<"\t"<<rank<<"\t";

        //计算float和int矩阵乘法得到结果矩阵

        // xgemm(matrixA,matrixB,matrixC,M,K,K,N);
        cublas_gemm_rowmajor(
            &cublasH, A_d, B_d, C_d, M, K,
            K, N, alpha, beta);
        cudaMemcpy( matrixC,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();




        {
            xigemm<float,8>(A_d,B_d,C_d,M,K,K,N);
            cudaMemcpy( matrixCQ,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            float R2 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
            printf("%.7f\t",R2);
            // for(int k=0;k<40;k++){
            //     printf("%f, ",matrixCQ[k]);
            // }
        }
        {
            
            lrxigemm<float,8>(A_d,B_d,C_d,M,K,K,N,10, &cusolverH, &cublasH);
            cudaMemcpy( matrixCQ,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 

            printf("%.7f\t",R3);
        }
        {
            skxigemm<float,8>(A_d,B_d,C_d,M,K,K,N,1, &cusolverH, &cublasH);
            cudaMemcpy( matrixCQ,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
            printf("%.7f\n",R3);
        }
    }
    return;        
}


void performance_test(){
    int test_para[2048][5] = {

        // {2048,2048,2048,50},
        // {2048,2048,2048,40},
        // {2048,2048,2048,30},
        // {2048,2048,2048,20},
        // {2048,2048,2048,10},
        // {2048,2048,2048,1},
        // {16384,16384,16384,1},
        // {2048,2048,2048,1},
        // {4096,4096,4096,1},
        {4096*4,4096*4,4096*4,1},
        // {16384,16384,16384,1},


        // {128,128,128,10,'k'},
        // {256,256,256,10,'k'},
        // {384,384,384,10,'k'},
        // {512,512,512,10,'k'},
        // {640,640,640,10,'k'},
        // {768,768,768,10,'k'},
        // {896,896,896,10,'k'},
        // {2048,2048,2048,10,'k'},
        // {1152,1152,1152,10,'k'},
        // {1280,1280,1280,10,'k'},
        // {1408,1408,1408,10,'k'},
        // {1536,1536,1536,10,'k'},
        // {1664,1664,1664,10,'k'},
        // {1792,1792,1792,10,'k'},
        // {1920,1920,1920,10,'k'},
        // {2048,2048,2048,10,'k'},
        // {2176,2176,2176,10,'k'},
        // {2304,2304,2304,10,'k'},
        // {2432,2432,2432,10,'k'},
        // {2560,2560,2560,10,'k'},
        // {2688,2688,2688,10,'k'},
        // {2816,2816,2816,10,'k'},
        // {2944,2944,2944,10,'k'},
        // {3072,3072,3072,10,'k'},
        // {3200,3200,3200,10,'k'},
        // {3328,3328,3328,10,'k'},
    }; 
  
    cublasHandle_t cublasH = NULL;
    cusolverDnHandle_t cusolverH = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    int max = 32678;
    float *matrixA = (float *)malloc(sizeof(float) * max*max);
    float *matrixB = (float *)malloc(sizeof(float) * max*max);
    float *matrixC = (float *)malloc(sizeof(float) * max*max);
    float *matrixCQ = (float *)malloc(sizeof(float) * max*max);
    float *matrixR = (float *)malloc(sizeof(float) * max*max);

    std::cout<<"M\tN\tK\trank\tSGEMM\t\torigin\t\tLrxigemm\tsketch\n";
    const int digit = 8;
    
    float *A_d, *B_d, *C_d;
    for(int i=0;i<5;i++){
        

        int N=test_para[i][0],M=test_para[i][1],K=test_para[i][2];
        int rank =test_para[i][3];

        float alpha = 1.0, beta = 0.0;
        char type = 'u';
        if(i!=0) {
            if(N!=test_para[i-1][0]||M!=test_para[i-1][1]||K!=test_para[i-1][2]||type!=test_para[i-1][4]){
                generate_matrix<float>(matrixA,M,K,type);
                generate_matrix<float>(matrixB,K,N,type);
            }
        } else {
            generate_matrix<float>(matrixA,M,K,type);
            generate_matrix<float>(matrixB,K,N,type);            
        } 
        if(M==0) return;
        cudaMalloc((float **)&A_d, sizeof(float) * M*K);
        cudaMalloc((float **)&B_d, sizeof(float) * K*N);
        cudaMalloc((float **)&C_d, sizeof(float) * M*N);
        cudaMemcpy(A_d, matrixA, sizeof(float) * M*K, cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, matrixB, sizeof(float) * K*N, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        std::cout<<M<<"\t"<<N<<"\t"<<K<<"\t"<<rank<<"\t";

        //计算float和int矩阵乘法得到结果矩阵

        // xgemm(matrixA,matrixB,matrixC,M,K,K,N);

        {
            auto start = std::chrono::high_resolution_clock::now();
            cublas_gemm_rowmajor(
                &cublasH, A_d, B_d, C_d, M, K,
                K, N, alpha, beta);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> diff = end - start;
            double time  = diff.count();
            printf("%.7lf\t",time);
            cudaDeviceSynchronize();
        }



        {
            auto start = std::chrono::high_resolution_clock::now();
            //xigemm<float,8>(A_d,B_d,C_d,M,K,K,N);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> diff = end - start;
            double time  = diff.count();
            printf("%.7lf\t",time);
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            //lrxigemm<float,8>(A_d,B_d,C_d,M,K,K,N,10, &cusolverH, &cublasH);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> diff = end - start;
            double time  = diff.count();
            printf("%.7lf\t",time);
        }
        {
            skxigemm<float,8>(A_d,B_d,C_d,128,128,128,128,1, &cusolverH, &cublasH);
            auto start = std::chrono::high_resolution_clock::now();
            skxigemm<float,8>(A_d,B_d,C_d,M,K,K,N,1, &cusolverH, &cublasH);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> diff = end - start;   
            double time  = diff.count();
            printf("%.7lf\n",time);
        }
    }
    return;        
}

int main(){
    //skxigemm_acc();
    //curand_test();
    //sketch_acc_test();
    performance_test();
    //precision_test();
}