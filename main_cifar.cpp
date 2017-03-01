#include "reader.h"
#include "convolution_layer/convolution.h"
#include "pooling_layer/pooling.h"
#include "fully_connected_layer/fully_connected.h"
#include "relu_layer/relu.h"

using Eigen::IOFormat;
using Eigen::StreamPrecision;
using Eigen::DontAlignCols;
using Eigen::Matrix;
using Eigen::RowMajor;

void write_to_csv(string name, MatrixXd matrix) {
    const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
    
    ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
}

int main() 
{
    // CONVOLUTION 1 //
    // Define Input.
    const int im_height = 32; // OPTION
    const int im_width = 32; // OPTION
    const int im_depth = 3; // OPTION
    const int im_size = im_height*im_width;
    // Define Kernels.
    const int k_num = 32; // OPTION
    const int k_size = 25; // OPTION
    const int stride = 1; // OPTION
    const int k_depth = im_depth;
    // Define Padding.
    //const int p1 = ((stride * im_height) - stride - im_height + sqrt(k_size)) / 2;
    //const int p2 = ((stride * im_width) - stride - im_width + sqrt(k_size)) / 2;
    const int p1 = 2; // OPTION
    const int p2 = 2; // OPTION
    // Define Output.
    const int output_height = (((im_height+(2*p1)) - sqrt(k_size))/stride) + 1;
    const int output_width = (((im_width+(2*p2)) - sqrt(k_size))/stride) + 1;
    const int output_size = output_height * output_width;
    // Define Weights.
    MatrixXd conv1_weights = read_cifar10_conv1_weights();
    Map<MatrixXd> w(conv1_weights.data(), k_num, k_size * k_depth);
    // Define Biases.
    MatrixXd conv1_biases = read_cifar10_conv1_biases();
    VectorXd conv1_b(Map<VectorXd>(conv1_biases.data(), conv1_biases.cols()*conv1_biases.rows()));  

    // POOLING 1 //
    // Define Pooling behaviour.   
    const int f = 2; //OPTION
    const int s = 2; //OPTION

    // ReLU 1 //
    // Nothing to define...

    // CONVOLUTION 2 //
    // Define Input.
    const int im_height_2 = ((output_height - f) / s) + 1;
    const int im_width_2 = ((output_width - f) / s) + 1;
    const int im_depth_2 = k_num;
    const int im_size_2 = im_height_2 * im_width_2;
    // Define Kernels.
    const int k_num_2 = 32; // OPTION
    const int k_size_2 = 25; // OPTION
    const int stride_2 = 1; // OPTION
    const int k_depth_2 = im_depth_2;
    // Define Padding.
    //const int p1_2 = ((stride_2 * im_height_2) - stride_2 - im_height_2 + sqrt(k_size_2)) / 2;
    //const int p2_2 = ((stride_2 * im_width_2) - stride_2 - im_width_2 + sqrt(k_size_2)) / 2;
    const int p1_2 = 2;
    const int p2_2 = 2; 
    // Define Output.       
    const int output_height_2 = (((im_height_2+(2*p1_2)) - sqrt(k_size_2))/stride_2) + 1;
    const int output_width_2 = (((im_width_2+(2*p2_2)) - sqrt(k_size_2))/stride_2) + 1;
    const int output_size_2 = output_height_2 * output_width_2;
    // Define Weights.
    MatrixXd conv2_weights = read_cifar10_conv2_weights();
    MatrixXd w_2 = conv2_weights;
    // Define Biases.
    MatrixXd conv2_biases = read_cifar10_conv2_biases();
    VectorXd conv2_b(Map<VectorXd>(conv2_biases.data(), conv2_biases.cols()*conv2_biases.rows()));
    
    // ReLU 2 //
    // Nothing to define...

    // POOLING 2 //
    // Define Pooling behaviour.   
    const int f_2 = 2; //OPTION
    const int s_2 = 2; //OPTION
    
    // CONVOLUTION 3 //
    // Define Input.
    const int im_height_3 = ((output_height_2 - f_2) / s_2) + 1;
    const int im_width_3 = ((output_width_2 - f_2) / s_2) + 1;
    const int im_depth_3 = k_num_2;
    const int im_size_3 = im_height_3 * im_width_3;
    // Define Kernels.
    const int k_num_3 = 64; // OPTION
    const int k_size_3 = 25; // OPTION
    const int stride_3 = 1; // OPTION
    const int k_depth_3 = im_depth_3;
    // Define Padding.
    //const int p1_3 = ((stride_3 * im_height_3) - stride_3 - im_height_3 + sqrt(k_size_3)) / 2;
    //const int p2_3 = ((stride_3 * im_width_3) - stride_3 - im_width_3 + sqrt(k_size_3)) / 2;
    const int p1_3 = 2;
    const int p2_3 = 2;
    // Define Output.        
    const int output_height_3 = (((im_height_3+(2*p1_3)) - sqrt(k_size_3))/stride_3) + 1;
    const int output_width_3 = (((im_width_3+(2*p2_3)) - sqrt(k_size_3))/stride_3) + 1;
    const int output_size_3 = output_height_3 * output_width_3;
    // Define Weights.        
    MatrixXd conv3_weights = read_cifar10_conv3_weights();
    MatrixXd w_3 = conv3_weights;
    // Define Biases.
    MatrixXd conv3_biases = read_cifar10_conv3_biases();
    VectorXd conv3_b(Map<VectorXd>(conv3_biases.data(), conv3_biases.cols()*conv3_biases.rows()));    

    // ReLU 3 //
    // Nothing to define...

    // POOLING 3 //
    // Define Pooling behaviour.   
    const int f_3 = 2; //OPTION
    const int s_3 = 2; //OPTION  

    // FULLY CONNECTED 1 //
    // Define Weights.
    MatrixXd fc1_weights = read_cifar10_fc1_weights();
    // Define Biases.
    MatrixXd fc1_biases = read_cifar10_fc1_biases();
    VectorXd fc1_b(Map<VectorXd>(fc1_biases.data(), fc1_biases.cols()*fc1_biases.rows()));      

    // FULLY CONNECTED 2 //
    // Define Weights.
    MatrixXd fc2_weights = read_cifar10_fc2_weights();
    // Define Biases.
    MatrixXd fc2_biases = read_cifar10_fc2_biases();
    VectorXd fc2_b(Map<VectorXd>(fc2_biases.data(), fc2_biases.cols()*fc2_biases.rows()));
    
    // Define Times.
    float gemm_time_total = 0.0;
    float run_time_total = 0.0;
    // Read in images.
    const int im_num = 100; // OPTION
    MatrixXd train = read_cifar10();
    for(int i=0; i < im_num; i++)  // For every row (image)...
    {  
        clock_t run_time_start = clock();  

        // Parse image with full depth (no labels this time).        
        MatrixXd img = train.block<1,im_size*im_depth>(i,0);
        MatrixXd image = Map<Matrix<double, im_depth, im_size, RowMajor>>(img.data());
     
        // Convolve 1.   
        MatrixXd convolved_1;
        double gemm_time_1;
        double offline_time_1;
        std::tie(convolved_1, gemm_time_1, offline_time_1) = convolve(image, im_size, im_height, im_width, im_depth, k_size, stride, conv1_b, p1, p2, w, output_size);               

        // Pool 1.
        MatrixXd pooled_1 = pool(convolved_1, f, s, output_width, output_height);
        
        // ReLU 1.
        MatrixXd relud_1 = relu(pooled_1);

        // Convolve 2.
        MatrixXd convolved_2;
        double gemm_time_2;
        double offline_time_2;
        std::tie(convolved_2, gemm_time_2, offline_time_2) = convolve(relud_1, im_size_2, im_height_2, im_width_2, im_depth_2, k_size_2, stride_2, conv2_b, p1_2, p2_2, w_2, output_size_2);

        // ReLU 2.
        MatrixXd relud_2 = relu(convolved_2);

        // Pool 2.
        MatrixXd pooled_2 = pool(relud_2, f_2, s_2, output_width_2, output_height_2);

        // Convolve 3.      
        MatrixXd convolved_3;
        double gemm_time_3;
        double offline_time_3;
        std::tie(convolved_3, gemm_time_3, offline_time_3) = convolve(pooled_2, im_size_3, im_height_3, im_width_3, im_depth_3, k_size_3, stride_3, conv3_b, p1_3, p2_3, w_3, output_size_3);  

        // ReLU 3.
        MatrixXd relud_3 = relu(convolved_3);

        // Pool 3.
        MatrixXd pooled_3 = pool(relud_3, f_3, s_3, output_width_3, output_height_3);

        // Fully Connect 1.
        MatrixXd fc1 = fully_connect(pooled_3, pooled_3.rows(), fc1_weights, fc1_b);

        // Fully Connect 2.
        MatrixXd fc2 = fully_connect(fc1, fc1.rows(), fc2_weights, fc2_b);

        clock_t run_time_end = clock();
        double run_time = (double) (run_time_end-run_time_start) / CLOCKS_PER_SEC;

        //Aggregate run_time_total.
        run_time_total += (run_time - offline_time_1 - offline_time_2 - offline_time_3);
    
        //Aggregate gemm_time_total.
        gemm_time_total += gemm_time_1 + gemm_time_2 + gemm_time_3;

        // Write features to file.
        std::string name = "data/cifar10/features/conv1_" + std::to_string(i) + ".csv";
        write_to_csv(name, convolved_1);
        std::string name2 = "data/cifar10/features/pool1_" + std::to_string(i) + ".csv";
        write_to_csv(name2, pooled_1);
        std::string name3 = "data/cifar10/features/relu1_" + std::to_string(i) + ".csv";
        write_to_csv(name3, relud_1);
        std::string name_4 = "data/cifar10/features/conv2_" + std::to_string(i) + ".csv";
        write_to_csv(name_4, convolved_2);  
        std::string name5 = "data/cifar10/features/relu2_" + std::to_string(i) + ".csv";
        write_to_csv(name5, relud_2);
        std::string name6 = "data/cifar10/features/pool2_" + std::to_string(i) + ".csv";
        write_to_csv(name6, pooled_2);
        std::string name_7 = "data/cifar10/features/conv3_" + std::to_string(i) + ".csv";
        write_to_csv(name_7, convolved_3);
        std::string name8 = "data/cifar10/features/relu3_" + std::to_string(i) + ".csv";
        write_to_csv(name8, relud_3);
        std::string name9 = "data/cifar10/features/pool3_" + std::to_string(i) + ".csv";
        write_to_csv(name9, pooled_3);
        std::string name10 = "data/cifar10/features/fc1_" + std::to_string(i) + ".csv";
        write_to_csv(name10, fc1);
        std::string name11 = "data/cifar10/features/fc2_" + std::to_string(i) + ".csv";
        write_to_csv(name11, fc2);
    }

    // Print average timings.
    cout << "-----------------------------" << endl;

    float avg_run_time = 0.0;            
    avg_run_time = run_time_total / im_num;
    cout << "average online run time: " << avg_run_time << endl;

    float avg_gemm_time = 0.0;
    avg_gemm_time = gemm_time_total / im_num;
    cout << "average total time for GEMM: " << avg_gemm_time << endl;

    return 0;
}
