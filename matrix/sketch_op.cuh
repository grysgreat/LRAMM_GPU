#include "operator_matrix.cuh"
#include "cutlass_gemm_op.cuh"
#include "./cublas/cublas_lib.cuh"
#include <curand.h>
#include <curand_kernel.h>




// void sketch_r1(
//     float* A,  float* A_L,float* A_R, int rowsA, int colsA, 
//     curandGenerator_t *gen, cublasHandle_t *cublashandler, 
//     float* d_work,float* c_work
// ) {
//     float *Sketch;
//     float *B;
//     float *A_Sketch;
//     float *Q;
//     float *US;
//     float *AT;
//     float *ATSketch_tmp;
//     const int iter = 1;

//     cudaMalloc((void**)&Sketch, sizeof(float) * colsA);
//     cudaMalloc((void**)&B, sizeof(float) * colsA);
//     cudaMalloc((void**)&A_Sketch, sizeof(float) * rowsA);
//     cudaMalloc((void**)&Q, sizeof(float) * rowsA);
//     cudaMalloc((void**)&US, sizeof(float) * rowsA);
//     cudaMalloc((void**)&AT, sizeof(float) * rowsA * colsA);
//     cudaMalloc((void**)&ATSketch_tmp, sizeof(float) * colsA);

//     cublasHandle_t cublasH = *cublashandler;
//     strans(AT, A, rowsA, colsA);

//     curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT);
//     curandSetPseudoRandomGeneratorSeed(*gen,9872349ULL);

//     curandGenerateNormal(*gen, Sketch, colsA, 0.0f, 1.0f);


//     //Compute Y = AS
//     float  beta = 0.0, alpha = 1.0;
//     cublasSgemv(cublasH, CUBLAS_OP_N, rowsA, colsA, &alpha, AT, rowsA, Sketch, 1, &beta, A_Sketch, 1);  
//     cudaDeviceSynchronize();


//     for(int i=0;i<iter;i++){
//         cublasSgemv(cublasH, CUBLAS_OP_N, colsA, rowsA, &alpha, A, colsA, A_Sketch, 1, &beta, ATSketch_tmp, 1);  
//         cudaDeviceSynchronize();
//         cublasSgemv(cublasH, CUBLAS_OP_N, rowsA, colsA, &alpha, AT, rowsA, ATSketch_tmp, 1, &beta, A_Sketch, 1);  
//         cudaDeviceSynchronize();
//     }

//     // //Construct a matrix Q 
//     float Sum_AS = get_Sum_sq2(A_Sketch, d_work, c_work, rowsA);
//     float Sum_AS_d1 = ((float)1.0)/Sum_AS;

//     s_axnoy(A_Sketch, Q, rowsA, Sum_AS_d1);


//     // B = QT A
//     cublasSgemv(cublasH, CUBLAS_OP_N, colsA, rowsA, &alpha, A, colsA, Q, 1, &beta, B, 1);  
//     cudaDeviceSynchronize();
//     // B = U S V, Sum_B = US(1*1); V��1*colA��
//     float Sum_B = get_Sum_sq2(B, d_work, c_work, colsA);
//     float Sum_B_d1 = ((float)1.0)/(float)Sum_B;
//     s_axnoy(B, A_R, colsA, Sum_B_d1);
//     s_axnoy(Q, A_L, rowsA, Sum_B);


// }


void sketch_r1(
    float* A,  float* A_L,float* A_R, int rowsA, int colsA, 
    curandGenerator_t *gen, cublasHandle_t *cublashandler
) {
    float *Sketch;
    float *B;
    float *A_Sketch;
    float *Q;
    float *US;
    float *ATSketch_tmp;
    const int iter = 1;

    cudaMalloc((void**)&Sketch, sizeof(float) * colsA);
    cudaMalloc((void**)&B, sizeof(float) * colsA);
    cudaMalloc((void**)&A_Sketch, sizeof(float) * rowsA);
    cudaMalloc((void**)&Q, sizeof(float) * rowsA);
    cudaMalloc((void**)&US, sizeof(float) * rowsA);
    cudaMalloc((void**)&ATSketch_tmp, sizeof(float) * colsA);

    cublasHandle_t cublasH = *cublashandler;

    curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(*gen,9872349ULL);

    curandGenerateNormal(*gen, Sketch, colsA, 0.0f, 1.0f);


    //Compute Y = AS
    float  beta = 0.0, alpha = 1.0;

    cublas_gemv_rowmajor( &cublasH,A, Sketch, A_Sketch, rowsA, colsA, alpha, beta);
    // cublasSgemv(cublasH, CUBLAS_OP_N, rowsA, colsA, &alpha, AT, rowsA, Sketch, 1, &beta, A_Sketch, 1);  
    // cudaDeviceSynchronize();


    for(int i=0;i<iter;i++){
        cublasSgemv(cublasH, CUBLAS_OP_N, colsA, rowsA, &alpha, A, colsA, A_Sketch, 1, &beta, ATSketch_tmp, 1);  
        cudaDeviceSynchronize();
        cublas_gemv_rowmajor( &cublasH, A, ATSketch_tmp, A_Sketch, rowsA, colsA, alpha, beta);
        cudaDeviceSynchronize();
    }

    // //Construct a matrix Q 
   
    float Sum_AS =  cublas_norm2(&cublasH, A_Sketch, rowsA);//get_Sum_sq2(A_Sketch, d_work, c_work, rowsA);
    float Sum_AS_d1 = ((float)1.0)/Sum_AS;

    s_axnoy(A_Sketch, Q, rowsA, Sum_AS_d1);

    // B = QT A
    cublasSgemv(cublasH, CUBLAS_OP_N, colsA, rowsA, &alpha, A, colsA, Q, 1, &beta, B, 1);  
    cudaDeviceSynchronize();
    // B = U S V, Sum_B = US(1*1); V��1*colA��
    float Sum_B = cublas_norm2(&cublasH, B, colsA);//get_Sum_sq2(B, d_work, c_work, colsA);
    float Sum_B_d1 = ((float)1.0)/(float)Sum_B;
    s_axnoy(B, A_R, colsA, Sum_B_d1);
    s_axnoy(Q, A_L, rowsA, Sum_B);


}

void sketch_r1_re(
    float* A,  float* A_L,float* A_R, int rowsA, int colsA, 
    curandGenerator_t *gen, cublasHandle_t *cublashandler
) {
    float *P;
    float *Q;
    float *tmp;
    float *tmp2;
    float *Sketch;
    float  beta = 0.0, alpha = 1.0;
    cudaMalloc((void**)&Sketch, sizeof(float) * colsA);
    cudaMalloc((void**)&P, sizeof(float) * rowsA);
    cudaMalloc((void**)&Q, sizeof(float) * colsA);
    cudaMalloc((void**)&tmp, sizeof(float) * rowsA);
    cudaMalloc((void**)&tmp2, sizeof(float) * colsA);
    cublasHandle_t cublasH = *cublashandler;


    curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(*gen,9872349ULL);
    curandGenerateNormal(*gen, Sketch, colsA, 0.0f, 1.0f);    

    cublas_gemv_rowmajor( &cublasH,A, Sketch, tmp, rowsA, colsA, alpha, beta);
    cublas_gemv_rowmajor_trans( &cublasH,A, tmp, tmp2, rowsA, colsA, alpha, beta);
    cublas_gemv_rowmajor( &cublasH,A, tmp2, A_L, rowsA, colsA, alpha, beta);
    cublas_gemv_rowmajor_trans( &cublasH,A, A_L, A_R, rowsA, colsA, alpha, beta);

    float normP =  cublas_norm2(&cublasH, A_L, rowsA);
    float normQ =  cublas_norm2(&cublasH, A_R, colsA);


    float Var_AL = normQ/(normP*normP);
    float Var_AR = 1.0/normQ;

    cublas_sscal(&cublasH, A_L, rowsA, Var_AL);
    cublas_sscal(&cublasH, A_R, colsA, Var_AR);
}


void sketch_r1_stream(
    float* A,  float* A_L,float* A_R, float * work, int rowsA, int colsA, 
    curandGenerator_t *gen, cublasHandle_t *cublashandler, cudaStream_t *pstream

) {
    float *P;
    float *Q;
    float *tmp;
    float *tmp2;
    float *Sketch;
    float  beta = 0.0, alpha = 1.0;

    Sketch = (float *)(&work[0]);
    P = (float *)(&Sketch[colsA]);
    Q = (float *)(&P[rowsA]);
    tmp = (float *)(&Q[colsA]);
    tmp2 = (float *)(&tmp[rowsA]);
    // cudaMalloc((void**)&Sketch, sizeof(float) * colsA);
    // cudaMalloc((void**)&P, sizeof(float) * rowsA);
    // cudaMalloc((void**)&Q, sizeof(float) * colsA);
    // cudaMalloc((void**)&tmp, sizeof(float) * rowsA);
    // cudaMalloc((void**)&tmp2, sizeof(float) * colsA);
    cublasHandle_t cublasH = *cublashandler;
    cudaStream_t stream = *pstream;

    curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(*gen,9872349ULL);
    curandGenerateNormal(*gen, Sketch, colsA, 0.0f, 1.0f);    

    cublas_gemv_rowmajor_i( &cublasH,A, Sketch, tmp, rowsA, colsA, alpha, beta, pstream);

    cublas_gemv_rowmajor_trans_i( &cublasH,A, tmp, tmp2, rowsA, colsA, alpha, beta, pstream);

    cublas_gemv_rowmajor_i( &cublasH,A, tmp2, A_L, rowsA, colsA, alpha, beta, pstream);

    cublas_gemv_rowmajor_trans_i( &cublasH,A, A_L, A_R, rowsA, colsA, alpha, beta, pstream);

    float normP =  cublas_norm2_i(&cublasH, A_L, rowsA);
    //cudaStreamSynchronize (stream);
    float normQ =  cublas_norm2_i(&cublasH, A_R, colsA);
    //cudaStreamSynchronize (stream);

    float Var_AL = normQ/(normP*normP);
    float Var_AR = 1.0/normQ;

    cublas_sscal_i(&cublasH, A_L, rowsA, Var_AL);
    //cudaStreamSynchronize (stream);
    cublas_sscal_i(&cublasH, A_R, colsA, Var_AR);
    //cudaStreamSynchronize (stream);
}