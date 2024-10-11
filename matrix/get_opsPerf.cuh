#ifndef SRC_PW_STRESS_MULTI_DEVICE_H
#define SRC_PW_STRESS_MULTI_DEVICE_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include "./cublas/cublas_utils.h"
enum calculation_type
{
    Te
};

namespace fuseConfig{

    class Config{
        public:
            Config();
            Config(std::string condigDir);

            std::string deviceName;
            int memSize;
            float *PerfHGemm;
            float *PerfI8Gemm;
            float *PerfQuant;
            float *PerfSGemv;
    
    };



}

#endif //SRC_PW_STRESS_MULTI_DEVICE_H