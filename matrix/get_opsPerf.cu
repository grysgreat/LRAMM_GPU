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

    void initConfig(std::string deviceName, std::ofstream *pfile){
        //std::ofstream file = *pfile;
        *pfile << deviceName <<"\n";

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