#include "helper/reader.h"
#include "helper/writer.h"
#include "../layers/convolution_layer/convolution.h"
#include "../layers/pooling_layer/pooling.h"
#include "../layers/fully_connected_layer/fully_connected.h"
#include "../layers/relu_layer/relu.h"

using Eigen::Matrix;
using Eigen::RowMajor;

int main() 
{
    float gemm_time_total = 0.0;
    float run_time_total = 0.0;
    
    for(int i=0; i < im_num; i++)
    {   
        clock_t run_time_start = clock();    
        
        MatrixXd img;
        if ( train.rows() != 1 ) {
            img = train.block<1,im_size_1*im_depth_1>(i,0);
        }
        else {          
            img = train;
        }
        
        MatrixXd image = Map<Matrix<double, im_depth_1, im_size_1, RowMajor>>(img.data());
        
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
