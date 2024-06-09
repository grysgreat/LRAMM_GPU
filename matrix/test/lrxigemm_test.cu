#include <random>
#include <chrono>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "cuda_runtime.h"
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

int main(){
    int max = 32;
    float *matrixA = (float *)malloc(sizeof(float) * max*max);
    float *matrixB = (float *)malloc(sizeof(float) * max*max);
    float *matrixC = (float *)malloc(sizeof(float) * max*max);
    float *matrixCQ = (float *)malloc(sizeof(float) * max*max);
    float *matrixR = (float *)malloc(sizeof(float) * max*max);

    int M=max , N=max, K = max;
    generate_matrix<float>(matrixA,M,K,'u');
    generate_matrix<float>(matrixB,K,N,'u');    

    xgemm(matrixA,matrixB,matrixC,M,K,K,N);

    xigemm<float,8>(matrixA,matrixB,matrixCQ,M,K,K,N);


    printMatrix(matrixC,M,N);
    printf("\n\n");
    printMatrix(matrixCQ,M,N);
    printf("\n\n");
    float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 

    printf("%.7f\n",R3);


    return ;
}