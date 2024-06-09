#include "../operator_matrix.cuh"
#include "../gen_matrix.cuh"
#include "../print_matrix.cuh"

#include "help_func.cu"
#include <iomanip>
#include "stdio.h"
#include <chrono>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;

void scopy_strans_acc_test(){
    int N=128,M=128,K=128;


    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);
    float *matrixCQ = (float *)malloc(sizeof(float) * M*N);
    float *matrixTmp = (float *)malloc(sizeof(float) * M*N);
    float *matrixA_dev;
    float *matrixB_dev;
    float *matrixTmp_dev;

    generate_matrix<float>(matrixA,M,K,'k');

    cudaMalloc((void**)&matrixA_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixB_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixTmp_dev, sizeof(float) * M*N);
    cudaMemcpy(matrixA_dev, matrixA, sizeof(float) * M*N, cudaMemcpyHostToDevice);
 

    dim3 block(32);
    //二维线程网格，128×128
    dim3 grid((M)/block.x);

    scopy<<<grid,block>>>(matrixA_dev,matrixB_dev,M,N);

    cudaMemcpy(matrixB, matrixB_dev, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
    


    strans(matrixTmp_dev,matrixB_dev,M,N);

    cudaMemcpy(matrixTmp, matrixTmp_dev, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
    

    int flag = 0;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            if(matrixTmp[j*N+i]!=matrixA[i*N+j]){
                flag++;
                //printf("error!!");
            }
        }
    }
    if(flag==0){
        printf("correct!!\n");
    } else {
        printf("wrong error num = %d!!\n",flag);
    }

}


void scopy_strans_perf_test(){
    int num = 8192;
    int N=num,M=num,K=num;


    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);
    float *matrixCQ = (float *)malloc(sizeof(float) * M*N);
    float *matrixTmp = (float *)malloc(sizeof(float) * M*N);
    float *matrixA_dev;
    float *matrixB_dev;
    float *matrixTmp_dev;

    generate_matrix<float>(matrixA,M,K,'k');

    cudaMalloc((void**)&matrixA_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixB_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixTmp_dev, sizeof(float) * M*N);
    cudaMemcpy(matrixA_dev, matrixA, sizeof(float) * M*N, cudaMemcpyHostToDevice);
 

    dim3 block(32);
    //二维线程网格，128×128
    dim3 grid((M)/block.x);


    // 开始计时，使用chrono计时，不支持其它计时方式

    auto start = std::chrono::high_resolution_clock::now();

    scopy<<<grid,block>>>(matrixA_dev,matrixB_dev,M,N);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    int time  = diff.count()*1000*1000;
    std::cout<<"size="<<M<<"//scopy - gpu time:" << std::fixed << std::setprecision(6) << time << std::endl;



    cudaMemcpy(matrixB, matrixB_dev, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    dim3 threadsPerBlock(32, 32);
    // dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x, 
    //                    (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 blocksPerGrid(M / threadsPerBlock.x, 
                    N  /threadsPerBlock.y);
    start = std::chrono::high_resolution_clock::now();

    //strans<<<blocksPerGrid,threadsPerBlock>>>(matrixB_dev,matrixTmp_dev,M,N);


    strans(matrixTmp_dev,matrixB_dev,M,N);

    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    time  = diff.count()*1000*1000;
    std::cout<<"size="<<M<<"//strans - gpu time:" << std::fixed << std::setprecision(6) << time << std::endl;


    cudaMemcpy(matrixTmp, matrixTmp_dev, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
    
    // int flag = 0;
    // for(int i=0;i<M;i++){
    //     for(int j=0;j<N;j++){
    //         if(matrixTmp[j*N+i]!=matrixA[i*N+j]){
    //             flag++;
    //             //printf("error!!");
    //         }
    //     }
    // }
    // if(flag==0){
    //     printf("correct!!\n");
    // } else {
    //     printf("wrong error num = %d!!\n",flag);
    // }


}

template <typename T>
T get_max(T* matrix,int rows,int cols){
    T maxM=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            maxM = max(std::abs(maxM),std::abs(matrix[i*cols+j]));
        }
    }
    return maxM;
}


void quant_perf_test(){
    int num = 1024*8;
    int N=num,M=num,K=num;

    float *matrixA = (float *)malloc(sizeof(float) * M*N);
    int8_t *matrixA8 = (int8_t *)malloc(sizeof(int8_t) * M*N);
    float *matrixB = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,N,'u');


    float max_mAR =get_max<float>(matrixA,M,N);
    const int max_int = 1<<(8-1) - 1;
    float lambdaAnew = (float)max_int/max_mAR;

    // for(int i=0;i<=10;i++){
    //     int a = matrixA8[i];
    //     std::cout<<a<<" ";
    // }
    // std::cout<<std::endl;
    float *matrixA_dev;
    float *matrixB_dev;
    float *matrixC_dev;
    int8_t* matrixA8_dev;
    int8_t* matrixA8_dev_ans= (int8_t *)malloc(sizeof(int8_t) * M*N);
    cudaMalloc((void**)&matrixA_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixB_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixC_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixA8_dev, sizeof(int8_t) * M*N);


    // start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(matrixA_dev, matrixA, sizeof(float) * M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_dev, matrixB, sizeof(float) * M*N, cudaMemcpyHostToDevice);
    // end = std::chrono::high_resolution_clock::now();
    // diff = end - start;
    // time  = diff.count()*1000*1000;
    // std::cout <<"cuda copt time:"<< std::fixed << std::setprecision(6) << time << std::endl;






    auto start = std::chrono::high_resolution_clock::now();

    quantitize_int8(matrixA_dev,matrixA8_dev,M,N,lambdaAnew);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    int time  = diff.count();
    std::cout<<"size="<<M<<"//quant - gpu time:" << std::fixed << std::setprecision(6) << time << std::endl;


}

void quant_acc_test(){
    int num = 32;
    int N=num,M=num,K=num;

    float *matrixA = (float *)malloc(sizeof(float) * M*N);
    int8_t *matrixA8 = (int8_t *)malloc(sizeof(int8_t) * M*N);
    float *matrixB = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,N,'u');


    float max_mAR =get_max<float>(matrixA,M,N);
    const int max_int = (1<<(8-1)) - 1;
    float lambdaAnew = (float)max_int/max_mAR;

    printMatrix(matrixA,M,N);
    std::cout<<std::endl;
    float *matrixA_dev;
    float *matrixB_dev;
    float *matrixC_dev;
    int8_t* matrixA8_dev;
    int8_t* matrixA8_dev_ans= (int8_t *)malloc(sizeof(int8_t) * M*N);
    cudaMalloc((void**)&matrixA_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixB_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixC_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixA8_dev, sizeof(int8_t) * M*N);

    cudaMemcpy(matrixA_dev, matrixA, sizeof(float) * M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_dev, matrixB, sizeof(float) * M*N, cudaMemcpyHostToDevice);
  

    quantitize_int8(matrixA_dev,matrixA8_dev,M,N,lambdaAnew);

    printf("%f,%f\n",lambdaAnew,max_mAR);

    cudaMemcpy( matrixA8_dev_ans,matrixA8_dev, sizeof(int8_t) * M*N, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < M; ++i) { // 遍历行
        for (size_t j = 0; j < N; ++j) { // 遍历列
            // 将int8_t转换为int以打印
            std::cout << static_cast<int>(matrixA8_dev_ans[i*N+j]) << " ";
        }
        std::cout << std::endl; // 每行后换行
    }

    dequantitize_int8(matrixA8_dev,matrixA_dev,M,N,lambdaAnew);
    cudaMemcpy( matrixB,matrixA_dev, sizeof(float) * M*N, cudaMemcpyDeviceToHost);

    printMatrix(matrixB,M,N);
}

// this function calls the CUDA kernel
void max_min_abs_test() {
    int number= 8192;
    int size = number*number;
    float gloden_avg=0;
    float *array = (float *)malloc(sizeof(float) * size);
    // fill the array with random numbers (could be any set of numbers)
    for(int i=0; i<size; i++) {
        array[i] = (rand() % 99) +1; // simply assigning a random number between 0 and 99
    }
    
    array[1]=-100.3;
    array[3]=-0.003;

    double tmp_sum=0;
    for(int i=0; i<size; i++) {
        tmp_sum+=(std::abs(array[i]));
    }
    gloden_avg=tmp_sum/((float)size);

    float max_value;
    float *d_in, *d_out;

    // allocate device memory
    cudaMalloc((void **)&d_in, size * sizeof(float));
    cudaMalloc((void **)&d_out, sizeof(float) * ((size/1024)+1));

    // transfer the array to the GPU
    cudaMemcpy(d_in, array, size * sizeof(float), cudaMemcpyHostToDevice);

    // calculating number of blocks based on array size
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    float partial_max[blocksPerGrid];



    max_value = max_abs(d_in,d_out,partial_max,size);
    // launch kernel using blocks and threads
    // IMPORTANT: we are passing the size of the shared memory explicitly to kernel
    auto start = std::chrono::high_resolution_clock::now();

    max_value = max_abs(d_in,d_out,partial_max,size);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    int time  = diff.count()*1000*1000;
    std::cout<<"size="<<size<<"//find_max - gpu time:" << std::fixed << std::setprecision(6) << time << std::endl;



    // Finish the reduction on CPU
    

    if(std::abs(max_value-100.3)>1e-4){
        printf("error! maximum = %f\n",max_value);
    } else {
        printf("find max pass!,max = %f\n",max_value);
    }


    float min_value;
    start = std::chrono::high_resolution_clock::now();

    min_value = min_abs(d_in,d_out,partial_max,size);

    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    time  = diff.count()*1000*1000;
    std::cout<<"size="<<size<<"//find_min - gpu time:" << std::fixed << std::setprecision(6) << time << std::endl;
    
    if(min_value-0.003>1e-6*min_value){
        printf("error! maximum = %f\n",min_value);
    } else {
        printf("find min pass!,min = %f\n",min_value);
    }


    float sum_value=0;


    start = std::chrono::high_resolution_clock::now();

    sum_value = avg_abs(d_in,d_out,partial_max,size);


    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    time  = diff.count()*1000*1000;
    std::cout<<"size="<<size<<"//find_min - gpu time:" << std::fixed << std::setprecision(6) << time << std::endl;
  


    if(std::abs(sum_value-gloden_avg)>1e-4){
        printf("error! sum_value = %f,gloden_avg=%f\n",sum_value,gloden_avg);
    } else {
        printf("find min pass!,avg = %f\n",sum_value);
    }




    // free the memory allocated on the GPU
    cudaFree(d_in);
    cudaFree(d_out);


}

int max_vec_perf_test(){
    int num = 16384;
    int N=num,M=num,K=num;

    float *matrixA = (float *)malloc(sizeof(float) * M*N);
    int8_t *matrixA8 = (int8_t *)malloc(sizeof(int8_t) * M*N);
    float *matrixB = (float *)malloc(sizeof(float) * M*N);

    float *vec_row = (float *)malloc(sizeof(float) * M*N);
    float *vec_col = (float *)malloc(sizeof(float) * M*N);
    float *work = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,N,'u');



    // for(int i=0;i<=10;i++){
    //     int a = matrixA8[i];
    //     std::cout<<a<<" ";
    // }
    // std::cout<<std::endl;
    float *matrixA_dev;
    float *vec_row_dev;
    float *vec_col_dev;
    float *work_dev;
    cudaMalloc((void**)&matrixA_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&vec_row_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&vec_col_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&work_dev, sizeof(float) * M*N);
    // start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(matrixA_dev, matrixA, sizeof(float) * M*N, cudaMemcpyHostToDevice);


    dim3 block(32, 32);
    //二维线程网格，128×128
    dim3 grid((M)/block.x, (N)/block.y);

    auto start = std::chrono::high_resolution_clock::now();


    max_abs_vec(matrixA_dev,work_dev,work,vec_row,M,N,'r');

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    int time  = diff.count()*1000*1000;
    std::cout<<"size="<<M<<"//max_vec - gpu time:" << std::fixed << std::setprecision(6) << time << std::endl;



}

int max_vec_acc_test(){
    int num = 8192;
    int N=num,M=num,K=num;

    float *matrixA = (float *)malloc(sizeof(float) * M*N);
    int8_t *matrixA8 = (int8_t *)malloc(sizeof(int8_t) * M*N);
    float *matrixB = (float *)malloc(sizeof(float) * M*N);

    float *vec_row = (float *)malloc(sizeof(float) * M*N);
    float *vec_col = (float *)malloc(sizeof(float) * M*N);

    float *vec_row_gold = (float *)malloc(sizeof(float) * M*N);
    float *vec_col_gold = (float *)malloc(sizeof(float) * M*N);
    
    float *work = (float *)malloc(sizeof(float) * M*N);

    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            matrixA[i*N+j] = (float)i+j;
        }
    }

    get_avg_vec<float>(matrixA,vec_row_gold, M,N,'r');
    get_avg_vec<float>(matrixA,vec_col_gold, M,N,'c');
    // for(int i=0;i<=10;i++){
    //     int a = matrixA8[i];
    //     std::cout<<a<<" ";
    // }
    // std::cout<<std::endl;
    float *matrixA_dev;
    float *vec_row_dev;
    float *vec_col_dev;
    float *work_dev;
    cudaMalloc((void**)&matrixA_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&vec_row_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&vec_col_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&work_dev, sizeof(float) * M*N);
    // start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(matrixA_dev, matrixA, sizeof(float) * M*N, cudaMemcpyHostToDevice);



    avg_abs_vec(matrixA_dev,work_dev,work,vec_row,M,N,'r');
    avg_abs_vec(matrixA_dev,work_dev,work,vec_col,M,N,'c');


    int flag = 0;
    for(int i=0;i<M;i++){
        if(vec_row[i]!=vec_row_gold[i]){
            printf("row error, gold = %f, cuda = %f\n",vec_row_gold[i],vec_row[i]);
            // return 0;
        } 
    }
    for(int i=0;i<N;i++){
        if(vec_col[i]!=vec_col_gold[i]){
            printf("col error, gold = %f, cuda = %f\n",vec_col_gold[i],vec_row[i]);
            return 0;
        } 
    }

    if(flag == 0) printf("max vec correct!\n");

}

int main(){


    //max_min_abs_test();


    // scopy_strans_acc_test();

    //scopy_strans_perf_test();

    //quant_perf_test();
    
     quant_acc_test();

    return 0;
    // max_vec_perf_test();
    // max_vec_acc_test();
}