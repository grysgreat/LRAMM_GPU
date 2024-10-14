#include "../lrxigemm.cuh"
#include <chrono>
#include "../gen_matrix.cuh"


void print_MatrixE(float matrix[], int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            printf("%.2e ", matrix[index]);
        }
        std::cout << std::endl;
    }
}
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

    sketch_r1_re(
        d_A,  d_B, d_C, M, N, &gen, &cublasH
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
        {1024,1024,1024,10,'n'},
        {1024,1024,1024,10,'n'},
        {1024,1024,1024,10,'u'},
        {1024,1024,1024,10,'s'},
        {1024,1024,1024,10,'e'},
        {1024,1024,1024,10,'k'},
        {1024,1024,1024,10,'p'},

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

    std::cout<<"M\tN\tK\ttype\trank\torigin\t\tLrxigemm\tsketch\t\tsketch_fu\thgemm\n";
    const int digit = 8;
    char * work;
    cudaMalloc((char **)&work, sizeof(float) * (max*max*8+max*5));

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
            xigemm_mem<float,8>(A_d,B_d,C_d,work,M,K,K,N);
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
            //skxigemm<float,8>(A_d,B_d,C_d,M,K,K,N,1, &cusolverH, &cublasH);
            skxigemm_mem<float,8>(A_d,B_d,C_d,work,M,K,K,N,1, &cusolverH, &cublasH);
            cudaMemcpy( matrixCQ,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
            printf("%.7f\t",R3);
        }
        {
            //skxigemm<float,8>(A_d,B_d,C_d,M,K,K,N,1, &cusolverH, &cublasH);
            skxigemm_mem_fusion<float,8>(A_d,B_d,C_d,work,M,K,K,N,1, &cublasH);
            cudaMemcpy( matrixCQ,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
            printf("%.7f\t",R3);
        }
        {
            shgemm(A_d,B_d,C_d,M,K,K,N,&cublasH);
            cudaMemcpy( matrixCQ,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 

            printf("%.7f\n",R3);

        }
    }
    return;        
}

static void gpu_helper(std::string info, bool print_info=false) {
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    if (cudaSuccess != cuda_status) {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = (total_db - free_db) / 1024.0 / 1024.0;
    if (print_info)
        std::cout << info << "   used GPU memory " << used_db << " MB   ,"<< "   total GPU memory " << total_byte/ 1024.0 / 1024.0 << " MB\n";
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
        {2048,2048,2048,1},
        {4096,4096,4096,1},
        {4096*2,4096*2,4096*2,1},
        {4096*3,4096*3,4096*3,1},
        {4096*4,4096*4,4096*4,1},
        {4096*5,4096*5,4096*5,1},
        {4096*6,4096*6,4096*6,1},
        // {4096*7,4096*7,4096*7,1},
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

    long long int max = 4096*6;
    float *matrixA = (float *)malloc(sizeof(float) * max*max);
    float *matrixB = (float *)malloc(sizeof(float) * max*max);
    float *matrixC = (float *)malloc(sizeof(float) * max*max);
    float *matrixCQ = (float *)malloc(sizeof(float) * max*max);
    float *matrixR = (float *)malloc(sizeof(float) * max*max);

    char * work;

    const int digit = 8;
    
    float *A_d, *B_d, *C_d;
    // generate_matrix<float>(matrixA,max,max,'u');
    // generate_matrix<float>(matrixB,max,max,'u');   

    cudaMalloc((float **)&A_d, sizeof(float) * max*max);
    cudaMalloc((float **)&B_d, sizeof(float) * max*max);
    cudaMalloc((float **)&C_d, sizeof(float) * max*max);

    // gpu_helper("GPU Memory Info", true);

    cudaError_t status_work = cudaMalloc((char **)&work, sizeof(float) * (max*max*6+max*5));
    
    double wantsize = (double)(max*max*6+max*5)*4/(1024.0*1024.0);
    std::cout<<"want size = "<<wantsize<<" MB\n"<<"max = "<<max*max<<"\n";
    if (status_work == cudaSuccess) {
        std::cout << "cudaMalloc succeeded!" << std::endl;
    } else {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(status_work) << std::endl;
        return ; // 返回错误码
    }

    cudaMemcpy(A_d, matrixA, sizeof(float) * max*max, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, matrixB, sizeof(float) * max*max, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();   

    gpu_helper("GPU Memory Info", true);

    std::cout<<"M\tN\tK\trank\tSGEMM\t\torigin\t\tLrxigemm\tsketch\t\tskecth-fusion\n";

    for(int i=0;i<10;i++){
        

        int N=test_para[i][0],M=test_para[i][1],K=test_para[i][2];
        int rank =test_para[i][3];

        float alpha = 1.0, beta = 0.0;
        if(M==0) return;

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
            xigemm_mem<float,8>(A_d,B_d,C_d,work,M,K,K,N);
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
            skxigemm_mem<float,8>(A_d,B_d,C_d,work,128,128,128,128,1, &cusolverH, &cublasH);
            //skxigemm<float,8>(A_d,B_d,C_d,128,128,128,128,1, &cusolverH, &cublasH);
            auto start = std::chrono::high_resolution_clock::now();
            //skxigemm<float,8>(A_d,B_d,C_d,M,K,K,N,1, &cusolverH, &cublasH);
            skxigemm_mem<float,8>(A_d,B_d,C_d,work,M,K,K,N,1, &cusolverH, &cublasH);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> diff = end - start;   
            double time  = diff.count();
            printf("%.7lf\t",time);
        }
        {
            skxigemm_mem_fusion<float,8>(A_d,B_d,C_d,work,128,128,128,128,1, &cublasH);
            //skxigemm<float,8>(A_d,B_d,C_d,128,128,128,128,1, &cusolverH, &cublasH);
            auto start = std::chrono::high_resolution_clock::now();
            //skxigemm<float,8>(A_d,B_d,C_d,M,K,K,N,1, &cusolverH, &cublasH);
            skxigemm_mem_fusion<float,8>(A_d,B_d,C_d,work,M,K,K,N,1, &cublasH);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> diff = end - start;   
            double time  = diff.count();
            printf("%.7lf\n",time);
        }
        //gpu_helper("GPU Memory Info2", true);
    }
    return;        
}



void nsys_perf_test(){
    int test_para[2048][5] = {
        {4096*4,4096*4,4096*4,1},
    }; 
  
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));

    int max = 4096*4;
    float *matrixA = (float *)malloc(sizeof(float) * max*max);
    float *matrixB = (float *)malloc(sizeof(float) * max*max);
    float *matrixC = (float *)malloc(sizeof(float) * max*max);
    float *matrixCQ = (float *)malloc(sizeof(float) * max*max);
    float *matrixR = (float *)malloc(sizeof(float) * max*max);

    char * work;
    cudaMalloc((char **)&work, sizeof(float) * (max*max*6+max*5));

    std::cout<<"M\tN\tK\trank\tSGEMM\t\torigin\t\tLrxigemm\tsketch\n";
    const int digit = 8;
    
    float *A_d, *B_d, *C_d;
    // generate_matrix<float>(matrixA,max,max,'u');
    // generate_matrix<float>(matrixB,max,max,'u');       
    for(int i=0;i<1;i++){
        

        int N=test_para[i][0],M=test_para[i][1],K=test_para[i][2];
        int rank =test_para[i][3];

        float alpha = 1.0, beta = 0.0;
        if(M==0) return;
        cudaMalloc((float **)&A_d, sizeof(float) * M*K);
        cudaMalloc((float **)&B_d, sizeof(float) * K*N);
        cudaMalloc((float **)&C_d, sizeof(float) * M*N);
        // cudaMemcpy(A_d, matrixA, sizeof(float) * M*K, cudaMemcpyHostToDevice);
        // cudaMemcpy(B_d, matrixB, sizeof(float) * K*N, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        //计算float和int矩阵乘法得到结果矩阵
        auto start = std::chrono::high_resolution_clock::now();
        skxigemm_mem_fusion<float,8>(A_d,B_d,C_d,work,M,K,K,N,1, &cublasH);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;   
        double time  = diff.count();
        printf("%.7lf\n",time);
    }
    return;        
}

void compare_print_test(){
    int test_para[2048][5] = {

        // {2048,2048,2048,50},
        // {2048,2048,2048,40},
        // {2048,2048,2048,30},
        // {2048,2048,2048,20},
        // {2048,2048,2048,10},
        {16,16,16,10,'k'},
    }; 
  
    cublasHandle_t cublasH = NULL;
    cusolverDnHandle_t cusolverH = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    int max = 8192;
    float *matrixA = (float *)malloc(sizeof(float) * max*max);
    float *matrixAR = (float *)malloc(sizeof(float) * max*max);
    float *matrixAR2 = (float *)malloc(sizeof(float) * max*max);
    std::cout<<"M\tN\tK\ttype\trank\torigin\t\tLrxigemm\tsketch\n";
    const int digit = 8;
    char * work;
    cudaMalloc((char **)&work, sizeof(float) * (max*max*8+max*5));

    float *A_d, *AL_d, *AR_d, *A2_d, *RA_d, *RA2_d, *PA_d;
    int8_t *AI_d;
    for(int i=0;i<1;i++){
        

        int N=test_para[i][0],M=test_para[i][1],K=test_para[i][2];
        int rank =test_para[i][3];
        char type =test_para[i][4];

        float alpha = 1.0, beta = 0.0;

        if(i!=0) {
            if(N!=test_para[i-1][0]||M!=test_para[i-1][1]||K!=test_para[i-1][2]||type!=test_para[i-1][4]){
                generate_matrix<float>(matrixA,M,K,type);
            }
        } else {
            generate_matrix<float>(matrixA,M,K,type);
        }
        cudaMalloc((float **)&A_d, sizeof(float) * M*N);
        cudaMalloc((float **)&A2_d, sizeof(float) * M*N);
        cudaMalloc((float **)&RA_d, sizeof(float) * M*N);
        cudaMalloc((float **)&RA2_d, sizeof(float) * M*N);
        cudaMalloc((float **)&PA_d, sizeof(float) * M*N);
        cudaMalloc((int8_t **)&AI_d, sizeof(int8_t) * M*N);

        cudaMalloc((float **)&AL_d, sizeof(float) * M);
        cudaMalloc((float **)&AR_d, sizeof(float) * N);
        cudaMemcpy(A_d, matrixA, sizeof(float) * M*N, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();




        const int max_int = (1<<(8-1)) - 1;
        float max_mA = cublas_absmax(&cublasH, A_d, M*N);
        float lambdaA = (float)max_int/max_mA;

        
        quantitize_getR_int8(A_d, AI_d, PA_d, RA_d, M, N, lambdaA);

        cudaMemcpy(matrixAR , RA_d, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        printf("AR = \n");
        print_MatrixE(matrixAR,M,N);


        //计算float和int矩阵乘法得到结果矩阵
        curandGenerator_t gen;
        sketch_r1_re( RA_d, AL_d, AR_d,M, N, &gen, &cublasH);

        cublas_gemm_rowmajor(
            &cublasH, AL_d, AR_d, RA2_d,  M,  1,
            1,  N, alpha,  beta);
        cudaDeviceSynchronize();
        cudaMemcpy(matrixAR2 , RA2_d, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        printf("\n\n\nSKETCH-AR = \n");
        print_MatrixE(matrixAR2,M,N);
        alpha = -1.0;
        cublas_saxpy(RA_d, RA2_d ,alpha, M*N, cublasH);
        cudaDeviceSynchronize();
        cudaMemcpy(matrixAR2 , RA2_d, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        printf("\n\n\nSKETCH - RAR = \n");
        print_MatrixE(matrixAR2,M,N);

        float sum =0,sumabs = 0;
        for(int j=0;j<M*N;j++){
            sum += matrixAR2[j];
            sumabs  +=abs(matrixAR2[j]);
        }
        printf("\n\nsum=%.4f, sumabs=%.4f, avg = %.4f, avg_abs=%.4f\n",sum,sumabs,sum/(float(M*N)),sumabs/(float(M*N)));
        alpha = 1.0;
        //计算float和int矩阵乘法得到结果矩阵
        cusolver_rsvd_LR(M, N, RA_d, AL_d, AR_d, 1, &cusolverH);

        cublas_gemm_rowmajor(
            &cublasH, AL_d, AR_d, RA2_d,  M,  1,
            1,  N, alpha,  beta);
        cudaDeviceSynchronize();
        cudaMemcpy(matrixAR2 , RA2_d, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        printf("\n\n\nCUSOLVER - AR = \n");
        print_MatrixE(matrixAR2,M,N);

        alpha = -1.0;
        cublas_saxpy(RA_d, RA2_d ,alpha, M*N, cublasH);
        cudaDeviceSynchronize();
        cudaMemcpy(matrixAR2 , RA2_d, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        printf("\n\n\nCUSOLVER - RAR = \n");
        print_MatrixE(matrixAR2,M,N);

    }
    return;        
}

void xhgemm_acc(){
    int max = 1024;
    float *matrixA = (float *)malloc(sizeof(float) * max*max);
    float *matrixB = (float *)malloc(sizeof(float) * max*max);
    float *matrixC = (float *)malloc(sizeof(float) * max*max);
    float *matrixCQ = (float *)malloc(sizeof(float) * max*max);
    float *matrixR = (float *)malloc(sizeof(float) * max*max);

    int M=max , N=max, K = max;

    generate_matrix<float>(matrixA,M,K,'n');
    generate_matrix<float>(matrixB,K,N,'n');    

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
    CUBLAS_CHECK(cublasCreate(&cublasH));

    shgemm(A_d,B_d,C_d,M,K,K,N,&cublasH);
    cudaMemcpy( matrixCQ,C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 

    printf("%.7f\n",R3);


    return ;
}


void mslag2h_withR(float *in, half *out, float *res, float *pin,int size){
    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0; i<size; i++){
        out[i] = (half)in[i];
        res[i] = in[i] - (float)out[i];
        pin[i] = (float)out[i];
    }
}


void sketchhgemm_acc_test(){
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    // 定义数组的大小
    int M=128,N=128,K=128;
    // 创建一个使用float类型的数组
    std::vector<float> arrayA(M*K);
    std::vector<float> arrayB(K*N);
    std::vector<float> arrayC(M*N);
    std::vector<float> arrayhfC(M*N);


    generate_matrix<float>(arrayA.data(),M,K,'u');
    generate_matrix<float>(arrayB.data(),K,N,'u');    


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
    std::vector<float> arrayhRA(M*K);
    std::vector<float> arrayhRB(K*N);
    std::vector<float> arrayhPA(M*K);
    std::vector<float> arrayhPB(K*N);
    std::vector<float> arrayhC(M*N);


    mslag2h_withR(arrayA.data(),arrayhA.data(), arrayhRA.data(),arrayhPA.data(),M*K);
    mslag2h_withR(arrayB.data(),arrayhB.data(), arrayhRB.data(),arrayhPB.data(),N*K);
    // mslag2d(arrayC.data(),arrayhC.data(),M*N);

    for(int i=0;i<10;i++){
        printf("%.8f,%.8f,%.8f,%.8f,%.12f\n",arrayA[i],(float)arrayhA[i],arrayhPA[i],arrayhRA[i],(arrayhPA[i]-(float)arrayhA[i]));
    }

    half* d_hA;
    half* d_hB;
    float* d_hC;
    float* d_hRA;
    float* d_hPA;
    float* d_hRB;
    float* d_hPB;


    cudaMalloc((void**)&d_hA, sizeof(half) * M*K);
    cudaMalloc((void**)&d_hB, sizeof(half) * K*N);
    cudaMalloc((void**)&d_hRA, sizeof(float) * M*K);
    cudaMalloc((void**)&d_hPA, sizeof(float) * M*K);
    cudaMalloc((void**)&d_hRB, sizeof(float) * K*N);
    cudaMalloc((void**)&d_hPB, sizeof(float) * K*N);
    cudaMalloc((void**)&d_hC, sizeof(float) * M*N);
    cudaMemcpy(d_hA, arrayhA.data(), sizeof(half) * M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hB, arrayhB.data(), sizeof(half) * K*N, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_hRA, arrayhRA.data(), sizeof(float) * M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hRB, arrayhRB.data(), sizeof(float) * K*N, cudaMemcpyHostToDevice);

    cudaMemcpy(d_hPA, arrayhPA.data(), sizeof(float) * M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hPB, arrayhPB.data(), sizeof(float) * K*N, cudaMemcpyHostToDevice);
  
    {
        float  beta2 = 0.0, alpha2 = 1.0;    
        skxhgemm(
        d_hA, d_hB, d_hC, d_hRA, d_hRB,  d_hPA, d_hPB, 
        M, K, K, N, 1, &cublasH);

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
    {
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
    {
        float  beta2 = 0.0, alpha2 = 1.0;    
        cublas_gemm_rowmajor(
        &cublasH, d_hPA, d_hPB, d_hC, M, K,
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
}


int main(){
    //skxigemm_acc();
    //curand_test();
    //sketch_acc_test();
    //  performance_test();
    //precision_test();

    // nsys_perf_test();
    //xhgemm_acc();
    //compare_print_test();

    sketchhgemm_acc_test();
}