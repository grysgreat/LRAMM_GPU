#include "../get_opsPerf.cuh"

int main(){
    fuseConfig::Config config;

    printf("%s\n",config.deviceName.c_str());
    int len = 8;
    int cnt = 0;
    for (int m = 0; m < len; m++) {
        for (int n = 0; n < len; n++) {
            for (int k = 0; k < len; k++) {
                printf("%f ",config.PerfHGemm[cnt++]);
                
            }
            printf("\n");
        }
    }    
    printf("\n");
    cnt = 0;
    for (int m = 0; m < len; m++) {
        for (int n = 0; n < len; n++) {
            for (int k = 0; k < len; k++) {
                printf("%f ",config.PerfI8Gemm[cnt++]);
                
            }
            printf("\n");
        }
    }  
    printf("\n");
    cnt = 0;
    for (int m = 0; m < len; m++) {
        for (int n = 0; n < len; n++) {
            printf("%f ",config.PerfSGemv[cnt++]);
            
        }
        printf("\n");
    }  
    printf("\n");
    cnt = 0;
    for (int m = 0; m < len; m++) {
        for (int n = 0; n < len; n++) {
            printf("%f ",config.PerfQuant[cnt++]);
            
        }
        printf("\n");
    }  
}