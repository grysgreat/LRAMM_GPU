#include "../get_opsPerf.cuh"

int main(){
    fuseConfig::Config config;

    printf("%s\n",config.deviceName.c_str());
    int len = 8;
    for (int m = 0; m < len; m++) {
        for (int n = 0; n < len; n++) {
            for (int k = 0; k < len; k++) {
                printf("%f ",config.PerfHGemm[m*(len*len)+n*len+k]);
                
            }
            printf("\n");
        }
    }    
}