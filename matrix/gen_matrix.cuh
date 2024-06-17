#include <random>
#include <omp.h>

template <typename T>
void generate_matrix(T* matrix,int rows,int cols,char type ){
    // 创建一个随机数引擎
    std::random_device rd;
    std::mt19937 gen(rd());
    // 创建一个均匀分布，范围是[0, 1)
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    //std::normal_real_distribution<float> dis(0.0, 1.0);
    int max_omp_thread = omp_get_max_threads();
    if(type == 'u'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                matrix[i*cols+j] = dis(gen);
                if(i==j&&i!=rows-1) matrix[i*cols+j] = (matrix[i*cols+j]);
                else  matrix[i*cols+j]=(matrix[i*cols+j]);
            }
        }        
    }
    else if(type == 'n'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                std::normal_distribution<double> dis(10.0, 3.0); // 定义随机数分布器对象dis，期望为0.0，标准差为1.0的正态分布
                matrix[i*cols+j] = dis(gen);
            }
        }        
    }else if(type == 'e'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                std::exponential_distribution<double> dis(1.0/(1.0/4.0)); // 定义随机数分布器对象dis，期望为0.0，标准差为1.0的正态分布
                matrix[i*cols+j] = dis(gen);
            }
        }        
    }else if(type == 'k'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                std::chi_squared_distribution<> dis(2);
                matrix[i*cols+j] = dis(gen);
            }
        }        
    }
};

void generate_ZDmatrix(float* matrix,int rows,int cols );