// get performance of gemm(f16,i8,i4), gemv(f32), quant
#include "get_opsPerf.cuh"
#include "cutlass_gemm_op.cuh"
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

    template <typename TAB, typename TC>
    void test_gemm(long long int max_size, int stride, std::ofstream *pfile, char type){

        cublasHandle_t cublasH = NULL;
        cublasCreate(&cublasH);

        TAB* d_A;
        TAB* d_B;
        TC* d_C;
        cudaMalloc((void**)&d_A, sizeof(TAB) * max_size*max_size);
        cudaMalloc((void**)&d_B, sizeof(TAB) * max_size*max_size);
        cudaMalloc((void**)&d_C, sizeof(TC) * max_size*max_size);
        TC alphah = 1.0, betah = 0.0;
        for (int m = 2048; m <= max_size; m+=stride) {
            for (int n = 2048; n <= max_size; n+=stride) {
                for (int k = 2048; k <= max_size; k+=stride) {
                    std::chrono::duration<double, std::milli> diff;
                    if( type == 'I'){
                        cut_gemm((int8_t *)d_A, (int8_t *)d_B, (int32_t *)d_C,m, k, k, n);
                        cudaDeviceSynchronize();
                        auto start = std::chrono::high_resolution_clock::now();
                        cut_gemm((int8_t *)d_A, (int8_t *)d_B, (int32_t *)d_C,m, k, k, n);
                        cudaDeviceSynchronize();
                        auto end = std::chrono::high_resolution_clock::now();
                        diff = end - start;
                    } else if(type == 'H'){
                        cublas_gemm_rowmajor(
                            &cublasH, (half *)d_A, (half *)d_B, (half *)d_C, m, k,
                            k, n, alphah, betah);
                        cudaDeviceSynchronize();
                        auto start = std::chrono::high_resolution_clock::now();
                        cublas_gemm_rowmajor(
                            &cublasH, (half *)d_A, (half *)d_B, (half *)d_C, m, k,
                            k, n, alphah, betah);
                        cudaDeviceSynchronize();
                        auto end = std::chrono::high_resolution_clock::now();
                        diff = end - start;
                    }
                    double time  = diff.count();
                    *pfile << time << " ";
                }
                *pfile << "\n";
            }
        }
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    
    void initConfig(std::string deviceName, std::ofstream *pfile){
        //std::ofstream file = *pfile;
        *pfile << deviceName <<"\n";
        std::vector<float> runtime(16*16*16);
        long long int max_size = 4096*4;
        int stride = 2048;
        test_gemm<half, half>(max_size, stride, pfile, 'H');
        *pfile << "\n";
        test_gemm<int8_t, int32_t>(max_size, stride, pfile, 'I');


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