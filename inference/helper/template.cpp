#include "inference/helper/reader.h"
#include "inference/helper/writer.h"
#include "../layers/convolution_layer/convolution.h"
#include "../layers/pooling_layer/pooling.h"
#include "../layers/fully_connected_layer/fully_connected.h"
#include "../layers/relu_layer/relu.h"

int main() 
{
    float gemm_time_total = 0.0;
    float run_time_total = 0.0;
    
    for(int i=0; i < im_num; i++)
    {   
        clock_t run_time_start = clock();    
        
        
        clock_t run_time_end = clock();
        double run_time = (double) (run_time_end-run_time_start) / CLOCKS_PER_SEC;
    }

    cout << "-----------------------------" << endl;

    float avg_run_time = 0.0;            
    avg_run_time = run_time_total / im_num;
    cout << "average online run time: " << avg_run_time << endl;

    float avg_gemm_time = 0.0;
    avg_gemm_time = gemm_time_total / im_num;
    cout << "average total time for GEMM: " << avg_gemm_time << endl;

    return 0; 
}
