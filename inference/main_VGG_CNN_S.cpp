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
	const int im_height_1 = 224;
	const int im_width_1 = 224;
	const int im_depth_1 = 3;
	const int im_size_1 = im_height_1*im_width_1;
	
	const int k_num_1 = 96;
	const int k_size_1 = 49;
	const int stride_1 = 2;
	const int k_depth_1 = im_depth_1;
	
	const int p1_1 = 0;
	const int p2_1 = 0;
	
	const int output_height_1 = (((im_height_1+(2*p1_1)) - sqrt(k_size_1))/stride_1) + 1;
	const int output_width_1 = (((im_width_1+(2*p2_1)) - sqrt(k_size_1))/stride_1) + 1;
	const int output_size_1 = output_height_1 * output_width_1;
	
	MatrixXd conv1_weights = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/conv1_weights.csv");
	Map<MatrixXd> conv1_w(conv1_weights.data(), k_num_1, k_size_1 * k_depth_1);
	
	MatrixXd conv1_biases = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/conv1_biases.csv");
	VectorXd conv1_b(Map<VectorXd>(conv1_biases.data(), conv1_biases.cols()*conv1_biases.rows()));
	
	const int f_1 = 2;
	const int s_1 = 3;
	
	const int pp1_1 = 1;
	const int pp2_1 = 1;
	
	const int im_height_2 = ((output_height_1 - f_1 + 2 * pp1_1) / s_1) + 1;
	const int im_width_2 = ((output_width_1 - f_1 + 2 * pp2_1) / s_1) + 1;
	const int im_depth_2 = k_num_1;
	const int im_size_2 = im_height_2 * im_width_2;
	
	const int k_num_2 = 256;
	const int k_size_2 = 25;
	const int stride_2 = 1;
	const int k_depth_2 = im_depth_2;
	
	const int p1_2 = 0;
	const int p2_2 = 0;
	
	const int output_height_2 = (((im_height_2+(2*p1_2)) - sqrt(k_size_2))/stride_2) + 1;
	const int output_width_2 = (((im_width_2+(2*p2_2)) - sqrt(k_size_2))/stride_2) + 1;
	const int output_size_2 = output_height_2 * output_width_2;
	
	MatrixXd conv2_weights = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/conv2_weights.csv");
	MatrixXd conv2_w = conv2_weights;
	
	MatrixXd conv2_biases = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/conv2_biases.csv");
	VectorXd conv2_b(Map<VectorXd>(conv2_biases.data(), conv2_biases.cols()*conv2_biases.rows()));
	
	const int f_2 = 2;
	const int s_2 = 2;
	
	const int pp1_2 = 0;
	const int pp2_2 = 0;
	
	const int im_height_3 = ((output_height_2 - f_2 + 2 * pp1_2) / s_2) + 1;
	const int im_width_3 = ((output_width_2 - f_2 + 2 * pp2_2) / s_2) + 1;
	const int im_depth_3 = k_num_2;
	const int im_size_3 = im_height_3 * im_width_3;
	
	const int k_num_3 = 512;
	const int k_size_3 = 9;
	const int stride_3 = 1;
	const int k_depth_3 = im_depth_3;
	
	const int p1_3 = 1;
	const int p2_3 = 1;
	
	const int output_height_3 = (((im_height_3+(2*p1_3)) - sqrt(k_size_3))/stride_3) + 1;
	const int output_width_3 = (((im_width_3+(2*p2_3)) - sqrt(k_size_3))/stride_3) + 1;
	const int output_size_3 = output_height_3 * output_width_3;
	
	MatrixXd conv3_weights = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/conv3_weights.csv");
	MatrixXd conv3_w = conv3_weights;
	
	MatrixXd conv3_biases = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/conv3_biases.csv");
	VectorXd conv3_b(Map<VectorXd>(conv3_biases.data(), conv3_biases.cols()*conv3_biases.rows()));
	
	const int im_height_4 = (output_height_3);
	const int im_width_4 = (output_width_3);
	const int im_depth_4 = k_num_3;
	const int im_size_4 = im_height_4 * im_width_4;
	
	const int k_num_4 = 512;
	const int k_size_4 = 9;
	const int stride_4 = 1;
	const int k_depth_4 = im_depth_4;
	
	const int p1_4 = 1;
	const int p2_4 = 1;
	
	const int output_height_4 = (((im_height_4+(2*p1_4)) - sqrt(k_size_4))/stride_4) + 1;
	const int output_width_4 = (((im_width_4+(2*p2_4)) - sqrt(k_size_4))/stride_4) + 1;
	const int output_size_4 = output_height_4 * output_width_4;
	
	MatrixXd conv4_weights = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/conv4_weights.csv");
	MatrixXd conv4_w = conv4_weights;
	
	MatrixXd conv4_biases = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/conv4_biases.csv");
	VectorXd conv4_b(Map<VectorXd>(conv4_biases.data(), conv4_biases.cols()*conv4_biases.rows()));
	
	const int im_height_5 = (output_height_4);
	const int im_width_5 = (output_width_4);
	const int im_depth_5 = k_num_4;
	const int im_size_5 = im_height_5 * im_width_5;
	
	const int k_num_5 = 512;
	const int k_size_5 = 9;
	const int stride_5 = 1;
	const int k_depth_5 = im_depth_5;
	
	const int p1_5 = 1;
	const int p2_5 = 1;
	
	const int output_height_5 = (((im_height_5+(2*p1_5)) - sqrt(k_size_5))/stride_5) + 1;
	const int output_width_5 = (((im_width_5+(2*p2_5)) - sqrt(k_size_5))/stride_5) + 1;
	const int output_size_5 = output_height_5 * output_width_5;
	
	MatrixXd conv5_weights = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/conv5_weights.csv");
	MatrixXd conv5_w = conv5_weights;
	
	MatrixXd conv5_biases = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/conv5_biases.csv");
	VectorXd conv5_b(Map<VectorXd>(conv5_biases.data(), conv5_biases.cols()*conv5_biases.rows()));
	
	const int f_3 = 2;
	const int s_3 = 3;
	
	const int pp1_3 = 1;
	const int pp2_3 = 1;
	
	MatrixXd fc6_weights = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/fc6_weights.csv");
	
	MatrixXd fc6_biases = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/fc6_biases.csv");
	VectorXd fc6_b(Map<VectorXd>(fc6_biases.data(), fc6_biases.cols()*fc6_biases.rows()));
	
	MatrixXd fc7_weights = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/fc7_weights.csv");
	
	MatrixXd fc7_biases = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/fc7_biases.csv");
	VectorXd fc7_b(Map<VectorXd>(fc7_biases.data(), fc7_biases.cols()*fc7_biases.rows()));
	
	MatrixXd fc8_weights = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/fc8_weights.csv");
	
	MatrixXd fc8_biases = load_csv_arma<MatrixXd>("../weights/VGG_CNN_S/fc8_biases.csv");
	VectorXd fc8_b(Map<VectorXd>(fc8_biases.data(), fc8_biases.cols()*fc8_biases.rows()));
	
	const int im_num = 1;
	MatrixXd train = load_csv_arma<MatrixXd>("../inputs/VGG_CNN_S/production/VGG_CNN_S.csv");
	
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
		
		MatrixXd relu1 = relu(conv1);
		
		MatrixXd pool1 = pool(relu1, f_1, s_1, output_width_1, output_height_1, pp1_1, pp2_1);
		
		MatrixXd conv2;
		double gemm_time_2;
		double offline_time_2;
		std::tie(conv2, gemm_time_2, offline_time_2) = convolve(pool1, im_size_2, im_height_2, im_width_2, im_depth_2, k_size_2, stride_2, conv2_b, p1_2, p2_2, conv2_w, output_size_2);
		
		MatrixXd relu2 = relu(conv2);
		
		MatrixXd pool2 = pool(relu2, f_2, s_2, output_width_2, output_height_2, pp1_2, pp2_2);
		
		MatrixXd conv3;
		double gemm_time_3;
		double offline_time_3;
		std::tie(conv3, gemm_time_3, offline_time_3) = convolve(pool2, im_size_3, im_height_3, im_width_3, im_depth_3, k_size_3, stride_3, conv3_b, p1_3, p2_3, conv3_w, output_size_3);
		
		MatrixXd relu3 = relu(conv3);
		
		MatrixXd conv4;
		double gemm_time_4;
		double offline_time_4;
		std::tie(conv4, gemm_time_4, offline_time_4) = convolve(relu3, im_size_4, im_height_4, im_width_4, im_depth_4, k_size_4, stride_4, conv4_b, p1_4, p2_4, conv4_w, output_size_4);
		
		MatrixXd relu4 = relu(conv4);
		
		MatrixXd conv5;
		double gemm_time_5;
		double offline_time_5;
		std::tie(conv5, gemm_time_5, offline_time_5) = convolve(relu4, im_size_5, im_height_5, im_width_5, im_depth_5, k_size_5, stride_5, conv5_b, p1_5, p2_5, conv5_w, output_size_5);
		
		MatrixXd relu5 = relu(conv5);
		
		MatrixXd pool5 = pool(relu5, f_3, s_3, output_width_5, output_height_5, pp1_3, pp2_3);
		
		MatrixXd fc6 = fully_connect(pool5, pool5.rows(), fc6_weights, fc6_b);
		
		MatrixXd relu6 = relu(fc6);
		
		MatrixXd fc7 = fully_connect(relu6, relu6.rows(), fc7_weights, fc7_b);
		
		MatrixXd relu7 = relu(fc7);
		
		MatrixXd fc8 = fully_connect(relu7, relu7.rows(), fc8_weights, fc8_b);
		
        clock_t run_time_end = clock();
        
        double run_time = (double) (run_time_end-run_time_start) / CLOCKS_PER_SEC;   
		run_time_total += (run_time - offline_time_1 - offline_time_2 - offline_time_3 - offline_time_4 - offline_time_5);
		gemm_time_total += 0.0 + gemm_time_1 + gemm_time_2 + gemm_time_3 + gemm_time_4 + gemm_time_5;
		
		std::string name_1 = "../features/VGG_CNN_S/conv1_" + std::to_string(i) + ".csv";
		write_to_csv(name_1, conv1);
		std::string name_2 = "../features/VGG_CNN_S/relu1_" + std::to_string(i) + ".csv";
		write_to_csv(name_2, relu1);
		std::string name_3 = "../features/VGG_CNN_S/pool1_" + std::to_string(i) + ".csv";
		write_to_csv(name_3, pool1);
		std::string name_4 = "../features/VGG_CNN_S/conv2_" + std::to_string(i) + ".csv";
		write_to_csv(name_4, conv2);
		std::string name_5 = "../features/VGG_CNN_S/relu2_" + std::to_string(i) + ".csv";
		write_to_csv(name_5, relu2);
		std::string name_6 = "../features/VGG_CNN_S/pool2_" + std::to_string(i) + ".csv";
		write_to_csv(name_6, pool2);
		std::string name_7 = "../features/VGG_CNN_S/conv3_" + std::to_string(i) + ".csv";
		write_to_csv(name_7, conv3);
		std::string name_8 = "../features/VGG_CNN_S/relu3_" + std::to_string(i) + ".csv";
		write_to_csv(name_8, relu3);
		std::string name_9 = "../features/VGG_CNN_S/conv4_" + std::to_string(i) + ".csv";
		write_to_csv(name_9, conv4);
		std::string name_10 = "../features/VGG_CNN_S/relu4_" + std::to_string(i) + ".csv";
		write_to_csv(name_10, relu4);
		std::string name_11 = "../features/VGG_CNN_S/conv5_" + std::to_string(i) + ".csv";
		write_to_csv(name_11, conv5);
		std::string name_12 = "../features/VGG_CNN_S/relu5_" + std::to_string(i) + ".csv";
		write_to_csv(name_12, relu5);
		std::string name_13 = "../features/VGG_CNN_S/pool5_" + std::to_string(i) + ".csv";
		write_to_csv(name_13, pool5);
		std::string name_14 = "../features/VGG_CNN_S/fc6_" + std::to_string(i) + ".csv";
		write_to_csv(name_14, fc6);
		std::string name_15 = "../features/VGG_CNN_S/relu6_" + std::to_string(i) + ".csv";
		write_to_csv(name_15, relu6);
		std::string name_16 = "../features/VGG_CNN_S/fc7_" + std::to_string(i) + ".csv";
		write_to_csv(name_16, fc7);
		std::string name_17 = "../features/VGG_CNN_S/relu7_" + std::to_string(i) + ".csv";
		write_to_csv(name_17, relu7);
		std::string name_18 = "../features/VGG_CNN_S/fc8_" + std::to_string(i) + ".csv";
		write_to_csv(name_18, fc8);
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
