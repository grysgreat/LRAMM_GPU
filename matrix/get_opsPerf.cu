// get performance of gemm(f16,i8,i4), gemv(f32), quant

#include "get_opsPerf.cuh"
/**
 * default config direction is ../config.dat
 */
namespace fuseConfig{

    std::string getDeviceName(){
        cudaDeviceProp deviceProp;
        cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, 0);

        return deviceProp.name;
    }

    Config::Config(){
        bool rebuild=false;
        std::string dir = "../../../config.dat";

        deviceName = getDeviceName();

        std::ifstream configFile(dir);
        if (!configFile.is_open()) {
            printf("no config file, will build.\n");
            rebuild = true;
        } else {
            std::string configDeviceName;
            std::getline(configFile, configDeviceName);
            if(configDeviceName.compare(deviceName)) {
                printf("device name not match, will rebuild.\n");
                rebuild = true;
            }
        }
        std::cout<<rebuild;
    }

}