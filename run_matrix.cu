#include "matrix/gen_matrix.cuh"



int main(){
    int N=400,M=400,K=400;


    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);
    float *matrixCQ = (float *)malloc(sizeof(float) * M*N);
    float *matrixR = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,K,'k');
    generate_matrix<float>(matrixB,K,N,'k');
    generate_ZDmatrix(matrixB,K,N);
}