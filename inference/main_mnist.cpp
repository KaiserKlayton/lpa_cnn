#include "helper/reader.h"
#include "helper/writer.h"
#include "../layers/convolution_layer/convolution.h"
#include "../layers/pooling_layer/pooling.h"
#include "../layers/fully_connected_layer/fully_connected.h"
#include "../layers/relu_layer/relu.h"
#include "../layers/eltwise_layer/eltwise.h"

using Eigen::Matrix;
using Eigen::RowMajor;

int main() 
{
	const int im_height_1 = 28;
	const int im_width_1 = 28;
	const int im_depth_1 = 1;
	const int im_size_1 = im_height_1*im_width_1;
	
	const int k_num_1 = 20;
	const int k_size_1 = 25;
	const int stride_1 = 1;
	const int k_depth_1 = im_depth_1;
	
	const int p1_1 = 0;
	const int p2_1 = 0;
	
	const int output_height_1 = (((im_height_1+(2*p1_1)) - sqrt(k_size_1))/stride_1) + 1;
	const int output_width_1 = (((im_width_1+(2*p2_1)) - sqrt(k_size_1))/stride_1) + 1;
	const int output_size_1 = output_height_1 * output_width_1;
	
	MatrixXd conv1_weights = load_csv_arma<MatrixXd>("../weights/mnist/conv1_weights.csv");
	Map<MatrixXd> conv1_w(conv1_weights.data(), k_num_1, k_size_1 * k_depth_1);
	
	MatrixXd conv1_biases = load_csv_arma<MatrixXd>("../weights/mnist/conv1_biases.csv");
	VectorXd conv1_b(Map<VectorXd>(conv1_biases.data(), conv1_biases.cols()*conv1_biases.rows()));
	
	const int f_1 = 2;
	const int s_1 = 2;
	
	const int pp1_1 = 0;
	const int pp2_1 = 0;
	
	const int im_height_2 = ((output_height_1 - f_1 + 2 * pp1_1) / s_1) + 1;
	const int im_width_2 = ((output_width_1 - f_1 + 2 * pp2_1) / s_1) + 1;
	const int im_depth_2 = k_num_1;
	const int im_size_2 = im_height_2 * im_width_2;
	
	const int k_num_2 = 50;
	const int k_size_2 = 25;
	const int stride_2 = 1;
	const int k_depth_2 = im_depth_2;
	
	const int p1_2 = 0;
	const int p2_2 = 0;
	
	const int output_height_2 = (((im_height_2+(2*p1_2)) - sqrt(k_size_2))/stride_2) + 1;
	const int output_width_2 = (((im_width_2+(2*p2_2)) - sqrt(k_size_2))/stride_2) + 1;
	const int output_size_2 = output_height_2 * output_width_2;
	
	MatrixXd conv2_weights = load_csv_arma<MatrixXd>("../weights/mnist/conv2_weights.csv");
	MatrixXd conv2_w = conv2_weights;
	
	MatrixXd conv2_biases = load_csv_arma<MatrixXd>("../weights/mnist/conv2_biases.csv");
	VectorXd conv2_b(Map<VectorXd>(conv2_biases.data(), conv2_biases.cols()*conv2_biases.rows()));
	
	const int f_2 = 2;
	const int s_2 = 2;
	
	const int pp1_2 = 0;
	const int pp2_2 = 0;
	
	MatrixXd ip1_weights = load_csv_arma<MatrixXd>("../weights/mnist/ip1_weights.csv");
	
	MatrixXd ip1_biases = load_csv_arma<MatrixXd>("../weights/mnist/ip1_biases.csv");
	VectorXd ip1_b(Map<VectorXd>(ip1_biases.data(), ip1_biases.cols()*ip1_biases.rows()));
	
	MatrixXd ip2_weights = load_csv_arma<MatrixXd>("../weights/mnist/ip2_weights.csv");
	
	MatrixXd ip2_biases = load_csv_arma<MatrixXd>("../weights/mnist/ip2_biases.csv");
	VectorXd ip2_b(Map<VectorXd>(ip2_biases.data(), ip2_biases.cols()*ip2_biases.rows()));
	
	const int im_num = 1000;
	MatrixXd train = load_csv_arma<MatrixXd>("../inputs/mnist/production/mnist.1000.csv");
	
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
        
		MatrixXd conv1;
		double gemm_time_1;
		double offline_time_1;
		std::tie(conv1, gemm_time_1, offline_time_1) = convolve(image, im_size_1, im_height_1, im_width_1, im_depth_1, k_size_1, stride_1, conv1_b, p1_1, p2_1, conv1_w, output_size_1);
		
		MatrixXd pool1 = pool(conv1, f_1, s_1, output_width_1, output_height_1, pp1_1, pp2_1);
		
		MatrixXd conv2;
		double gemm_time_2;
		double offline_time_2;
		std::tie(conv2, gemm_time_2, offline_time_2) = convolve(pool1, im_size_2, im_height_2, im_width_2, im_depth_2, k_size_2, stride_2, conv2_b, p1_2, p2_2, conv2_w, output_size_2);
		
		MatrixXd pool2 = pool(conv2, f_2, s_2, output_width_2, output_height_2, pp1_2, pp2_2);
		
		MatrixXd ip1 = fully_connect(pool2, pool2.rows(), ip1_weights, ip1_b);
		
		MatrixXd relu1 = relu(ip1);
		
		MatrixXd ip2 = fully_connect(relu1, relu1.rows(), ip2_weights, ip2_b);
		
        clock_t run_time_end = clock();
        
        double run_time = (double) (run_time_end-run_time_start) / CLOCKS_PER_SEC;   
		run_time_total += (run_time - offline_time_1 - offline_time_2);
		gemm_time_total += 0.0 + gemm_time_1 + gemm_time_2;
		
		std::string name_1 = "../features/mnist/ip2_" + std::to_string(i) + ".csv";
		write_to_csv(name_1, ip2);
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
