// get performance of gemm(f16,i8,i4), gemv(f32), quant
#include "get_opsPerf.cuh"
#include "cutlass_gemm_op.cuh"
#include "operator_matrix.cuh"
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
    void test_write_level3(long long int max_size, int stride, std::ofstream *pfile, calculation_type type){

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
                    if( type == calculation_type::I8GEMM){
                        cut_gemm((int8_t *)d_A, (int8_t *)d_B, (int32_t *)d_C,m, k, k, n);
                        cudaDeviceSynchronize();
                        auto start = std::chrono::high_resolution_clock::now();
                        cut_gemm((int8_t *)d_A, (int8_t *)d_B, (int32_t *)d_C,m, k, k, n);
                        cudaDeviceSynchronize();
                        auto end = std::chrono::high_resolution_clock::now();
                        diff = end - start;
                    } else if(type == calculation_type::I4GEMM){
                        cut_gemm4((cutlass::int4b_t *)d_A, (cutlass::int4b_t *)d_B, (int32_t *)d_C,m, k, k, n);
                        cudaDeviceSynchronize();
                        auto start = std::chrono::high_resolution_clock::now();
                        cut_gemm4((cutlass::int4b_t *)d_A, (cutlass::int4b_t *)d_B, (int32_t *)d_C,m, k, k, n);
                        cudaDeviceSynchronize();
                        auto end = std::chrono::high_resolution_clock::now();
                        diff = end - start;
                    } else if(type == calculation_type::HGEMM){
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

    void test_read_level3(int size , std::ifstream *pfile, std::vector<float> *data){
        std::string line;
        int cnt=0;
        for(int i=0;i<size*size;i++){
            
            std::getline(*pfile, line);
            std::istringstream iss(line);
            float value;

            while (iss >> value) {
                (*data)[cnt++] = value;
            }
        }
    }
    void test_read_level2(int size , std::ifstream *pfile, std::vector<float> *data){
        std::string line;
        int cnt=0;
        for(int i=0;i<size;i++){
            
            std::getline(*pfile, line);
            std::istringstream iss(line);
            float value;

            while (iss >> value) {
                (*data)[cnt++] = value;
            }
        }
    }

    template <typename TA, typename TB>
    void test_write_level2(long long int max_size, int stride, std::ofstream *pfile, calculation_type type){

        cublasHandle_t cublasH = NULL;
        cublasCreate(&cublasH);

        TA* d_A;
        TB* d_B;
        TB* d_C;
        cudaMalloc((void**)&d_A, sizeof(TA) * max_size*max_size);
        cudaMalloc((void**)&d_B, sizeof(TB) * max_size*max_size);
        cudaMalloc((void**)&d_C, sizeof(TB) * max_size*max_size);
        float alpha = 1.0, beta = 0.0;
        for (int m = 2048; m <= max_size; m+=stride) {
            for (int n = 2048; n <= max_size; n+=stride) {

                std::chrono::duration<double, std::milli> diff;
                if( type == calculation_type::QUANT){
                    quantitize_int8((float *)d_A, (int8_t *)d_B, m, n, alpha);
                    cudaDeviceSynchronize();
                    auto start = std::chrono::high_resolution_clock::now();
                    quantitize_int8((float *)d_A, (int8_t *)d_B, m, n, alpha);
                    cudaDeviceSynchronize();
                    auto end = std::chrono::high_resolution_clock::now();
                    diff = end - start;
                } else if(type == calculation_type::SGEMV){
                    cublas_gemv_rowmajor( &cublasH,(float *)d_A, (float *)d_B, (float *)d_C, m, n, alpha, beta);
                    cudaDeviceSynchronize();
                    auto start = std::chrono::high_resolution_clock::now();
                    cublas_gemv_rowmajor( &cublasH,(float *)d_A, (float *)d_B, (float *)d_C, m, n, alpha, beta);
                    cudaDeviceSynchronize();
                    auto end = std::chrono::high_resolution_clock::now();
                    diff = end - start;
                } else if(type == calculation_type::SGEMV_TRANS){
                    cublas_gemv_rowmajor_trans( &cublasH,(float *)d_A, (float *)d_B, (float *)d_C, m, n, alpha, beta);
                    cudaDeviceSynchronize();
                    auto start = std::chrono::high_resolution_clock::now();
                    cublas_gemv_rowmajor_trans( &cublasH,(float *)d_A, (float *)d_B, (float *)d_C, m, n, alpha, beta);
                    cudaDeviceSynchronize();
                    auto end = std::chrono::high_resolution_clock::now();
                    diff = end - start;
                }
                double time  = diff.count();
                *pfile << time << " ";
            }
            *pfile << "\n";
        }
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    void Config::initConfig(std::string deviceName, std::ofstream *pfile){
        //std::ofstream file = *pfile;
        *pfile << deviceName <<"\n";
        std::vector<float> runtime(16*16*16);
        test_write_level3<half, half>(max_size, stride, pfile, calculation_type::HGEMM);
        *pfile << "\n";
        test_write_level3<int8_t, int32_t>(max_size, stride, pfile, calculation_type::I8GEMM);
        *pfile << "\n";
        test_write_level3<cutlass::int4b_t, int32_t>(max_size, stride, pfile, calculation_type::I4GEMM);
        *pfile << "\n";
        test_write_level2<float, float>(max_size, stride, pfile, calculation_type::SGEMV);
        *pfile << "\n";
        // test_write_level2<float, float>(max_size, stride, pfile, calculation_type::SGEMV_TRANS);
        // *pfile << "\n";
        test_write_level2<float, int8_t>(max_size, stride, pfile, calculation_type::QUANT);
        // gemm_test fp16 2048~32678
    }
    void Config::readConfig(std::ifstream *pfile){
        std::getline(*pfile, deviceName);
        std::string tmp;
        int size = max_size/stride;
        test_read_level3(size , pfile, &PerfHGemm);
        std::getline(*pfile, tmp);
        test_read_level3(size , pfile, &PerfI8Gemm);
        std::getline(*pfile, tmp);
        test_read_level3(size , pfile, &PerfI4Gemm);
        std::getline(*pfile, tmp);
        test_read_level2(size , pfile, &PerfSGemv);
        std::getline(*pfile, tmp);
        test_read_level2(size , pfile, &PerfQuant);
    }

    Config::Config(){
        bool rebuild=false;
        std::string dir = "./config.dat";
        
        int size = max_size/stride;
        PerfHGemm.resize(size*size*size);
        PerfI8Gemm.resize(size*size*size);
        PerfI4Gemm.resize(size*size*size);
        PerfQuant.resize(size*size);
        PerfSGemv.resize(size*size);
                
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

        std::ifstream rconfigFile2(dir);
        readConfig(&rconfigFile2);
        rconfigFile2.close();
    }

}