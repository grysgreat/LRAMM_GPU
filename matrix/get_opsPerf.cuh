#ifndef SRC_PW_STRESS_MULTI_DEVICE_H
#define SRC_PW_STRESS_MULTI_DEVICE_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include "./cublas/cublas_utils.h"
enum class calculation_type
{
    SGEMM,
    HGEMM,
    I8GEMM,
    I4GEMM,
    SGEMV,
    SGEMV_TRANS,
    QUANT
};

namespace fuseConfig{

    class Config{
        public:
            Config();
            Config(std::string condigDir);

            void initConfig(std::string deviceName, std::ofstream *pfile);
            void readConfig(std::ifstream *pfile);

            long long int max_size = 4096*4;
            int stride = 2048;
            std::string deviceName;
            int memSize;
            std::vector<float> PerfHGemm;
            std::vector<float> PerfI8Gemm;
            std::vector<float> PerfI4Gemm;
            std::vector<float> PerfQuant;
            std::vector<float> PerfSGemv;
    
    };



}

#endif //SRC_PW_STRESS_MULTI_DEVICE_H