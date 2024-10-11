// get performance of gemm(f16,i8,i4), gemv(f32), quant
#include "get_opsPerf.cuh"
/**
 * default config direction is ../config.dat
 */

namespace fuseConfig{
    #include "cublas/cublas_lib.cuh"
    std::string getDeviceName(){
        cudaDeviceProp deviceProp;
        cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, 0);

        return deviceProp.name;
    }

    void initConfig(std::string deviceName, std::ofstream *pfile){
        //std::ofstream file = *pfile;
        *pfile << deviceName <<"\n";
        std::vector<float> runtime(16*16*16);

        cublasHandle_t cublasH = NULL;
        cublasCreate(&cublasH);

        long long int max_size = 4096*8;
        half* d_A;
        half* d_B;
        half* d_C;
        cudaMalloc((void**)&d_A, sizeof(half) * max_size*max_size);
        cudaMalloc((void**)&d_B, sizeof(half) * max_size*max_size);
        cudaMalloc((void**)&d_C, sizeof(half) * max_size*max_size);

        half alpha = 1.0, beta = 0.0;
        for (int m = 2048; m <= 8*2048; m+=2048) {
            for (int n = 2048; n <= 8*2048; n+=2048) {
                for (int k = 2048; k <= 8*2048; k+=2048) {
                    cublas_gemm_rowmajor(
                        &cublasH, d_A, d_B, d_C, m, k,
                        k, n, alpha, beta);
                    auto start = std::chrono::high_resolution_clock::now();
                    cublas_gemm_rowmajor(
                        &cublasH, d_A, d_B, d_C, m, k,
                        k, n, alpha, beta);
                    auto end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> diff = end - start;
                    double time  = diff.count();
                    *pfile << time << " ";
                }
                *pfile << "\n";
            }
        }

        // gemm_test fp16 2048~32678
    }

    Config::Config(){
        bool rebuild=false;
        std::string dir = "../../../config.dat";
        
        deviceName = getDeviceName();

        std::ifstream rconfigFile(dir);
        

        if (!rconfigFile.is_open()) {
            printf("no config file, will build.\n");
            rebuild = true;
        } else {
            std::string configDeviceName;
            std::getline(rconfigFile, configDeviceName);
            if(configDeviceName.compare(deviceName)) {

                printf("123\n %s \n%s \n",deviceName.c_str(),configDeviceName.c_str());
                printf("device name not match, will rebuild.\n");
                rebuild = true;
            }
        }
        rconfigFile.close();

        if(rebuild) {
            std::ofstream wconfigFile(dir);
            initConfig(deviceName, &wconfigFile);
            wconfigFile.close();
        }
        
    }

}