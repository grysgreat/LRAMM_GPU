#include "../operator_matrix.cuh"
#include "cutlass_gemm_op.cuh"



int main() {
    // 定义数组的大小
    int M=1024,N=1024,K=512;
    // 创建一个使用input_t类型的数组
    std::vector<input_t> int4b_arrayA(M*K);
    std::vector<input_t> int4b_arrayB(K*N);
    std::vector<int32_t> int32b_arrayC(M*N);


    //初始化数组
    for (int i = 0; i < M; ++i) {

        for(int j=0;j<K;j++){
            // 将每个元素初始化为它的索引值，注意这里只是示例，实际值可能需要根据量化规则来确定
            // if(i==0||i==1)int4b_arrayA[i*K+j] = static_cast<input_t>(i+1);
            // if(j==0||j==1)int4b_arrayB[i+j*K] = static_cast<input_t>(j+1);
            int4b_arrayA[i*K+j] = static_cast<input_t>(1);
        }

    }


    for (int i = 0; i < K; ++i) {
        for(int j=0;j< N;j++){
            int4b_arrayB[i*N+j] = static_cast<input_t>(2);
        }
    }
    for (int i = 0; i < M; ++i) {
        for(int j=0;j< N;j++){
            int32b_arrayC[i*N+j] = 0;
        }
    }
    // for (int i = 0; i < M; ++i) {
    //     for(int j=0;j<K;j++){
    //         printf("%d,",static_cast<int>(int4b_arrayA[i*M+j]));
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // for (int i = 0; i < K; ++i) {
    //     for(int j=0;j< N;j++){
    //         printf("%d,",static_cast<int>(int4b_arrayB[i*M+j]));
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    input_t* d_A;
    input_t* d_B;
    input_t* d_B_TP;
    int32_t* d_C;
    cudaMalloc((void**)&d_A, sizeof(input_t) * M*N);
    cudaMalloc((void**)&d_B, sizeof(input_t) * M*N);
    cudaMalloc((void**)&d_C, sizeof(int32_t) * M*N);
    cudaMalloc((void**)&d_B_TP, sizeof(int32_t) * M*N);
    cudaMemcpy(d_A, int4b_arrayA.data(), sizeof(input_t) * M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, int4b_arrayB.data(), sizeof(input_t) * M*N, cudaMemcpyHostToDevice);

    I8trans(d_B_TP,d_B,K,N);
    cut_gemm(d_A, d_B_TP, d_C,M,K, K,N);

    
    cudaMemcpy( int32b_arrayC.data(),d_C, sizeof(int32_t) * M*N, cudaMemcpyDeviceToHost);
  
    for (int i = 0; i < 32; ++i) {

        for(int j=0;j< 32;j++){
            printf("%d,",static_cast<int>(int32b_arrayC[i*M+j]));
        }
        printf("\n");
    }



    return 0;
}