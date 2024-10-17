#include "../operator_matrix.cuh"
#include "cutlass_gemm_op.cuh"
#include <chrono>


void i8_perf_test(){

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

    // 定义数组的大小

    int max = 4096*8;
    // 创建一个使用float类型的数组
    std::vector<int8_t> int4b_arrayA(max*max);
    std::vector<int8_t> int4b_arrayB(max*max);
    std::vector<int32_t> int32b_arrayC(max*max);

    int8_t* d_A;
    int8_t* d_B;
    int32_t* d_C;
    cudaMalloc((void**)&d_A, sizeof(int8_t) * max*max);
    cudaMalloc((void**)&d_B, sizeof(int8_t) * max*max);
    cudaMalloc((void**)&d_C, sizeof(int32_t) * max*max);
    cudaMemcpy(d_A, int4b_arrayA.data(), sizeof(int8_t) * max*max, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, int4b_arrayB.data(), sizeof(int8_t) * max*max, cudaMemcpyHostToDevice);

    float beta = 0.0, alpha = 1.0;

    for(int i=0;i<32;i++){
        

        int N=test_para[i][0],M=test_para[i][1],K=test_para[i][2];
        float alpha = 1.0, beta = 0.0;
        if(M==0) return;

        std::cout<<M<<"\t"<<N<<"\t"<<K<<"\t";

        //计算float和int矩阵乘法得到结果矩阵
        cut_gemm(d_A, d_B, d_C,M,K, K,N);
        cudaDeviceSynchronize();
        {
            auto start = std::chrono::high_resolution_clock::now();
            cut_gemm(d_A, d_B, d_C,M,K, K,N);
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

void i8_acc_test() {
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
    cudaDeviceSynchronize();
    
    cudaMemcpy( int32b_arrayC.data(),d_C, sizeof(int32_t) * M*N, cudaMemcpyDeviceToHost);
  
    for (int i = 0; i < 32; ++i) {
        for(int j=0;j< 32;j++){
            printf("%d,",static_cast<int>(int32b_arrayC[i*M+j]));
        }
        printf("\n");
    }
    return ;
}

void i4_acc_test() {
    // 定义数组的大小
    int M=1024,N=1024,K=1024;
    // 创建一个使用input_t类型的数组
    std::vector<cutlass::int4b_t> int4b_arrayA(M*K);
    std::vector<cutlass::int4b_t> int4b_arrayB(K*N);
    std::vector<int32_t> int32b_arrayC(M*N);


    //初始化数组
    for (int i = 0; i < M; ++i) {

        for(int j=0;j<K;j++){
            // 将每个元素初始化为它的索引值，注意这里只是示例，实际值可能需要根据量化规则来确定
            // if(i==0||i==1)int4b_arrayA[i*K+j] = static_cast<input_t>(i+1);
            // if(j==0||j==1)int4b_arrayB[i+j*K] = static_cast<input_t>(j+1);
            if(i*K+j%2==0) int4b_arrayA[i*K+j] = static_cast<cutlass::int4b_t>(1);
            else int4b_arrayA[i*K+j] = static_cast<cutlass::int4b_t>(2);
        }

    }


    for (int i = 0; i < K; ++i) {
        for(int j=0;j< N;j++){
            if(i*N+j%2==0)int4b_arrayB[i*N+j] = static_cast<cutlass::int4b_t>(1);
            else int4b_arrayB[i*N+j] = static_cast<cutlass::int4b_t>(2);
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

    cutlass::int4b_t* d_A;
    cutlass::int4b_t* d_B;
    cutlass::int4b_t* d_B_TP;
    int32_t* d_C;
    cudaMalloc((void**)&d_A, sizeof(cutlass::int4b_t) * M*N);
    cudaMalloc((void**)&d_B, sizeof(cutlass::int4b_t) * M*N);
    cudaMalloc((void**)&d_C, sizeof(int32_t) * M*N);
    cudaMalloc((void**)&d_B_TP, sizeof(cutlass::int4b_t) * M*N);
    cudaMemcpy(d_A, int4b_arrayA.data(), sizeof(cutlass::int4b_t) * M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, int4b_arrayB.data(), sizeof(cutlass::int4b_t) * M*N, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cut_gemm4(d_A, d_B, d_C,M,K, K,N);
    cudaDeviceSynchronize();
    
    cudaMemcpy( int32b_arrayC.data(),d_C, sizeof(int32_t) * M*N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < 32; ++i) {
        for(int j=0;j< 32;j++){
            printf("%d,",static_cast<int>(int4b_arrayB[i*32+j]));
        }
        printf("\n");
    }
    cudaMemcpy( int4b_arrayB.data(),d_B, sizeof(cutlass::int4b_t) * K*N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < 32; ++i) {
        for(int j=0;j< 32;j++){
            printf("%d,",static_cast<int>(int32b_arrayC[i*32+j]));
        }
        printf("\n");
    }


    return ;
}


int main(){
    i8_acc_test();
    printf("\n");
    printf("\n");
    printf("\n");
    i4_acc_test();

    //i8_perf_test();
}


// #include <cutlass/cutlass.h>
// #include <cutlass/gemm/device/gemm.h>
// #include <cutlass/util/host_tensor.h>
// #include <cutlass/util/reference/host/tensor_fill.h>
// #include <cutlass/util/reference/host/tensor_compare.h>
// #include <cutlass/util/reference/host/gemm.h>
// #include <cutlass/util/tensor_view_io.h>

// int main() {
//     using ElementA = cutlass::int4b_t;
//     using ElementB = cutlass::int4b_t;
//     using ElementC = int32_t;
//     using LayoutA = cutlass::layout::RowMajor;
//     using LayoutB = cutlass::layout::ColumnMajor;
//     using LayoutC = cutlass::layout::RowMajor;
//     using ElementOutput = int32_t;
//     using ElementAccumulator = int32_t;
//     using ElementCompute = int32_t;
//     using Gemm = cutlass::gemm::device::Gemm<
//     cutlass::int4b_t,
//     cutlass::layout::RowMajor,
//     cutlass::int4b_t,
//     cutlass::layout::ColumnMajor,
//     ElementOutput,
//     cutlass::layout::RowMajor,
//     ElementAccumulator,
//     cutlass::arch::OpClassTensorOp,
//     cutlass::arch::Sm80,
//     cutlass::gemm::GemmShape<128, 256, 128>,
//     cutlass::gemm::GemmShape<64, 64, 128>,
//     cutlass::gemm::GemmShape<8, 8, 32>,
//     cutlass::epilogue::thread::LinearCombinationClamp<
//         ElementOutput,
//         128 / cutlass::sizeof_bits<ElementOutput>::value,
//         ElementAccumulator,
//         ElementCompute
//     >,
//     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
//     2
//     >;


//     int M = 1024;
//     int N = 1024;
//     int K = 1024;

//     cutlass::HostTensor<ElementA, LayoutA> tensor_A({M, K});
//     cutlass::HostTensor<ElementB, LayoutB> tensor_B({K, N});
//     cutlass::HostTensor<ElementC, LayoutC> tensor_C({M, N});
//     cutlass::HostTensor<ElementC, LayoutC> tensor_D({M, N});

//     cutlass::int4b_t oob_value = cutlass::int4b_t(1);
//     int32_t oob_value2 = int32_t(0);
//     int32_t oob_value3 = int32_t(1);
//     cutlass::reference::host::TensorFill(tensor_A.host_view(), oob_value);
//     cutlass::reference::host::TensorFill(tensor_B.host_view(), oob_value);
//     cutlass::reference::host::TensorFill(tensor_C.host_view(), oob_value2);

//     tensor_A.sync_device();
//     tensor_B.sync_device();
//     tensor_C.sync_device();

//     typename Gemm::Arguments arguments{
//         {M, N, K},
//         tensor_A.device_ref(),
//         tensor_B.device_ref(),
//         tensor_C.device_ref(),
//         tensor_D.device_ref(),
//         {1, 0}
//     };

//     Gemm gemm_op;
//     size_t workspace_size = Gemm::get_workspace_size(arguments);
//     cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

//     cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
//     if (status != cutlass::Status::kSuccess) {
//         std::cerr << "Failed to initialize GEMM operation" << std::endl;
//         return -1;
//     }

//     status = gemm_op();
//     if (status != cutlass::Status::kSuccess) {
//         std::cerr << "Failed to execute GEMM operation" << std::endl;
//         return -1;
//     }

//     tensor_D.sync_host();
//     cutlass::reference::host::Gemm<
//     cutlass::int4b_t, cutlass::layout::RowMajor, 
//     cutlass::int4b_t, cutlass::layout::ColumnMajor,
//     int32_t, cutlass::layout::RowMajor, 
//     int32_t, int32_t> reference_gemm;

//     reference_gemm(
//         {M, N, K},
//         oob_value3,
//         tensor_A.host_ref(),
//         tensor_B.host_ref(),
//         oob_value2,
//         tensor_C.host_ref(),
//         int32_t()
//     );

//     bool correct = cutlass::reference::host::TensorEquals(
//         tensor_D.host_view(),
//         tensor_C.host_view()
//     );

//     if (correct) {
//         std::cout << "GEMM result is correct." << std::endl;
//     } else {
//         std::cout << "GEMM result is incorrect." << std::endl;
//     }

//     return 0;
// }