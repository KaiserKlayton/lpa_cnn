#include "helper/reader.h"
#include "helper/input_parser.h"
#include "helper/writer.h"
#include "../layers/convolution_layer/convolution.h"
#include "../layers/pooling_layer/pooling.h"
#include "../layers/fully_connected_layer/fully_connected.h"
#include "../layers/relu_layer/relu.h"
#include "../layers/eltwise_layer/eltwise.h"

#include <string.h>

using Eigen::Matrix;
using Eigen::RowMajor;

int main(int argc, char *argv[])
{
    if (argc == 2) {
        if (!strcmp(argv[1], "eigen") || !strcmp(argv[1], "gemmlowp")) {
            ;
        } else {
            cout << "ERROR --> usage: ./lpa_cnn.out <eigen> | <gemmlowp>" << endl;
            exit(1);
        }
    } else {
        cout << "ERROR --> usage: ./lpa_cnn.out <eigen> | <gemmlowp>" << endl;
        exit(1);
    }

    float gemm_time_total = 0.0;
    float run_time_total = 0.0;

    std::string mode = argv[1];

	const int im_height_1 = 224;
	const int im_width_1 = 224;
	const int im_depth_1 = 3;
	const int im_size_1 = im_height_1*im_width_1;
	
	const int k_num_1 = 64;
	const int k_size_1 = 9;
	const int stride_1 = 1;
	const int k_depth_1 = im_depth_1;
	
	const int p1_1 = 1;
	const int p2_1 = 1;
	
	const int output_height_1 = (((im_height_1+(2*p1_1)) - sqrt(k_size_1))/stride_1) + 1;
	const int output_width_1 = (((im_width_1+(2*p2_1)) - sqrt(k_size_1))/stride_1) + 1;
	const int output_size_1 = output_height_1 * output_width_1;
	
	MatrixXd conv1_1_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/conv1_1_weights.csv");
	Map<MatrixXd> conv1_1_w(conv1_1_weights.data(), k_num_1, k_size_1 * k_depth_1);
	
	MatrixXd conv1_1_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/conv1_1_biases.csv");
	VectorXd conv1_1_b(Map<VectorXd>(conv1_1_biases.data(), conv1_1_biases.cols()*conv1_1_biases.rows()));
	
	const int im_height_2 = output_height_1;
	const int im_width_2 = output_width_1;
	const int im_depth_2 = k_num_1;
	const int im_size_2 = im_height_2 * im_width_2;
	
	const int k_num_2 = 64;
	const int k_size_2 = 9;
	const int stride_2 = 1;
	const int k_depth_2 = im_depth_2;
	
	const int p1_2 = 1;
	const int p2_2 = 1;
	
	const int output_height_2 = (((im_height_2+(2*p1_2)) - sqrt(k_size_2))/stride_2) + 1;
	const int output_width_2 = (((im_width_2+(2*p2_2)) - sqrt(k_size_2))/stride_2) + 1;
	const int output_size_2 = output_height_2 * output_width_2;
	
	MatrixXd conv1_2_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/conv1_2_weights.csv");
	MatrixXd conv1_2_w = conv1_2_weights;
	
	MatrixXd conv1_2_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/conv1_2_biases.csv");
	VectorXd conv1_2_b(Map<VectorXd>(conv1_2_biases.data(), conv1_2_biases.cols()*conv1_2_biases.rows()));
	
	const int f_1 = 2;
	const int s_1 = 2;
	
	const int pp1_1 = 0;
	const int pp2_1 = 0;
	
	const int im_height_3 = ((output_height_2 - f_1 + 2 * pp1_1) / s_1) + 1;
	const int im_width_3 = ((output_width_2 - f_1 + 2 * pp2_1) / s_1) + 1;
	const int im_depth_3 = k_num_2;
	const int im_size_3 = im_height_3 * im_width_3;
	
	const int k_num_3 = 128;
	const int k_size_3 = 9;
	const int stride_3 = 1;
	const int k_depth_3 = im_depth_3;
	
	const int p1_3 = 1;
	const int p2_3 = 1;
	
	const int output_height_3 = (((im_height_3+(2*p1_3)) - sqrt(k_size_3))/stride_3) + 1;
	const int output_width_3 = (((im_width_3+(2*p2_3)) - sqrt(k_size_3))/stride_3) + 1;
	const int output_size_3 = output_height_3 * output_width_3;
	
	MatrixXd conv2_1_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/conv2_1_weights.csv");
	MatrixXd conv2_1_w = conv2_1_weights;
	
	MatrixXd conv2_1_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/conv2_1_biases.csv");
	VectorXd conv2_1_b(Map<VectorXd>(conv2_1_biases.data(), conv2_1_biases.cols()*conv2_1_biases.rows()));
	
	const int im_height_4 = output_height_3;
	const int im_width_4 = output_width_3;
	const int im_depth_4 = k_num_3;
	const int im_size_4 = im_height_4 * im_width_4;
	
	const int k_num_4 = 128;
	const int k_size_4 = 9;
	const int stride_4 = 1;
	const int k_depth_4 = im_depth_4;
	
	const int p1_4 = 1;
	const int p2_4 = 1;
	
	const int output_height_4 = (((im_height_4+(2*p1_4)) - sqrt(k_size_4))/stride_4) + 1;
	const int output_width_4 = (((im_width_4+(2*p2_4)) - sqrt(k_size_4))/stride_4) + 1;
	const int output_size_4 = output_height_4 * output_width_4;
	
	MatrixXd conv2_2_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/conv2_2_weights.csv");
	MatrixXd conv2_2_w = conv2_2_weights;
	
	MatrixXd conv2_2_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/conv2_2_biases.csv");
	VectorXd conv2_2_b(Map<VectorXd>(conv2_2_biases.data(), conv2_2_biases.cols()*conv2_2_biases.rows()));
	
	const int f_2 = 2;
	const int s_2 = 2;
	
	const int pp1_2 = 0;
	const int pp2_2 = 0;
	
	const int im_height_5 = ((output_height_4 - f_2 + 2 * pp1_2) / s_2) + 1;
	const int im_width_5 = ((output_width_4 - f_2 + 2 * pp2_2) / s_2) + 1;
	const int im_depth_5 = k_num_4;
	const int im_size_5 = im_height_5 * im_width_5;
	
	const int k_num_5 = 256;
	const int k_size_5 = 9;
	const int stride_5 = 1;
	const int k_depth_5 = im_depth_5;
	
	const int p1_5 = 1;
	const int p2_5 = 1;
	
	const int output_height_5 = (((im_height_5+(2*p1_5)) - sqrt(k_size_5))/stride_5) + 1;
	const int output_width_5 = (((im_width_5+(2*p2_5)) - sqrt(k_size_5))/stride_5) + 1;
	const int output_size_5 = output_height_5 * output_width_5;
	
	MatrixXd conv3_1_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/conv3_1_weights.csv");
	MatrixXd conv3_1_w = conv3_1_weights;
	
	MatrixXd conv3_1_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/conv3_1_biases.csv");
	VectorXd conv3_1_b(Map<VectorXd>(conv3_1_biases.data(), conv3_1_biases.cols()*conv3_1_biases.rows()));
	
	const int im_height_6 = output_height_5;
	const int im_width_6 = output_width_5;
	const int im_depth_6 = k_num_5;
	const int im_size_6 = im_height_6 * im_width_6;
	
	const int k_num_6 = 256;
	const int k_size_6 = 9;
	const int stride_6 = 1;
	const int k_depth_6 = im_depth_6;
	
	const int p1_6 = 1;
	const int p2_6 = 1;
	
	const int output_height_6 = (((im_height_6+(2*p1_6)) - sqrt(k_size_6))/stride_6) + 1;
	const int output_width_6 = (((im_width_6+(2*p2_6)) - sqrt(k_size_6))/stride_6) + 1;
	const int output_size_6 = output_height_6 * output_width_6;
	
	MatrixXd conv3_2_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/conv3_2_weights.csv");
	MatrixXd conv3_2_w = conv3_2_weights;
	
	MatrixXd conv3_2_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/conv3_2_biases.csv");
	VectorXd conv3_2_b(Map<VectorXd>(conv3_2_biases.data(), conv3_2_biases.cols()*conv3_2_biases.rows()));
	
	const int im_height_7 = output_height_6;
	const int im_width_7 = output_width_6;
	const int im_depth_7 = k_num_6;
	const int im_size_7 = im_height_7 * im_width_7;
	
	const int k_num_7 = 256;
	const int k_size_7 = 9;
	const int stride_7 = 1;
	const int k_depth_7 = im_depth_7;
	
	const int p1_7 = 1;
	const int p2_7 = 1;
	
	const int output_height_7 = (((im_height_7+(2*p1_7)) - sqrt(k_size_7))/stride_7) + 1;
	const int output_width_7 = (((im_width_7+(2*p2_7)) - sqrt(k_size_7))/stride_7) + 1;
	const int output_size_7 = output_height_7 * output_width_7;
	
	MatrixXd conv3_3_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/conv3_3_weights.csv");
	MatrixXd conv3_3_w = conv3_3_weights;
	
	MatrixXd conv3_3_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/conv3_3_biases.csv");
	VectorXd conv3_3_b(Map<VectorXd>(conv3_3_biases.data(), conv3_3_biases.cols()*conv3_3_biases.rows()));
	
	const int f_3 = 2;
	const int s_3 = 2;
	
	const int pp1_3 = 0;
	const int pp2_3 = 0;
	
	const int im_height_8 = ((output_height_7 - f_3 + 2 * pp1_3) / s_3) + 1;
	const int im_width_8 = ((output_width_7 - f_3 + 2 * pp2_3) / s_3) + 1;
	const int im_depth_8 = k_num_7;
	const int im_size_8 = im_height_8 * im_width_8;
	
	const int k_num_8 = 512;
	const int k_size_8 = 9;
	const int stride_8 = 1;
	const int k_depth_8 = im_depth_8;
	
	const int p1_8 = 1;
	const int p2_8 = 1;
	
	const int output_height_8 = (((im_height_8+(2*p1_8)) - sqrt(k_size_8))/stride_8) + 1;
	const int output_width_8 = (((im_width_8+(2*p2_8)) - sqrt(k_size_8))/stride_8) + 1;
	const int output_size_8 = output_height_8 * output_width_8;
	
	MatrixXd conv4_1_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/conv4_1_weights.csv");
	MatrixXd conv4_1_w = conv4_1_weights;
	
	MatrixXd conv4_1_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/conv4_1_biases.csv");
	VectorXd conv4_1_b(Map<VectorXd>(conv4_1_biases.data(), conv4_1_biases.cols()*conv4_1_biases.rows()));
	
	const int im_height_9 = output_height_8;
	const int im_width_9 = output_width_8;
	const int im_depth_9 = k_num_8;
	const int im_size_9 = im_height_9 * im_width_9;
	
	const int k_num_9 = 512;
	const int k_size_9 = 9;
	const int stride_9 = 1;
	const int k_depth_9 = im_depth_9;
	
	const int p1_9 = 1;
	const int p2_9 = 1;
	
	const int output_height_9 = (((im_height_9+(2*p1_9)) - sqrt(k_size_9))/stride_9) + 1;
	const int output_width_9 = (((im_width_9+(2*p2_9)) - sqrt(k_size_9))/stride_9) + 1;
	const int output_size_9 = output_height_9 * output_width_9;
	
	MatrixXd conv4_2_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/conv4_2_weights.csv");
	MatrixXd conv4_2_w = conv4_2_weights;
	
	MatrixXd conv4_2_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/conv4_2_biases.csv");
	VectorXd conv4_2_b(Map<VectorXd>(conv4_2_biases.data(), conv4_2_biases.cols()*conv4_2_biases.rows()));
	
	const int im_height_10 = output_height_9;
	const int im_width_10 = output_width_9;
	const int im_depth_10 = k_num_9;
	const int im_size_10 = im_height_10 * im_width_10;
	
	const int k_num_10 = 512;
	const int k_size_10 = 9;
	const int stride_10 = 1;
	const int k_depth_10 = im_depth_10;
	
	const int p1_10 = 1;
	const int p2_10 = 1;
	
	const int output_height_10 = (((im_height_10+(2*p1_10)) - sqrt(k_size_10))/stride_10) + 1;
	const int output_width_10 = (((im_width_10+(2*p2_10)) - sqrt(k_size_10))/stride_10) + 1;
	const int output_size_10 = output_height_10 * output_width_10;
	
	MatrixXd conv4_3_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/conv4_3_weights.csv");
	MatrixXd conv4_3_w = conv4_3_weights;
	
	MatrixXd conv4_3_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/conv4_3_biases.csv");
	VectorXd conv4_3_b(Map<VectorXd>(conv4_3_biases.data(), conv4_3_biases.cols()*conv4_3_biases.rows()));
	
	const int f_4 = 2;
	const int s_4 = 2;
	
	const int pp1_4 = 0;
	const int pp2_4 = 0;
	
	const int im_height_11 = ((output_height_10 - f_4 + 2 * pp1_4) / s_4) + 1;
	const int im_width_11 = ((output_width_10 - f_4 + 2 * pp2_4) / s_4) + 1;
	const int im_depth_11 = k_num_10;
	const int im_size_11 = im_height_11 * im_width_11;
	
	const int k_num_11 = 512;
	const int k_size_11 = 9;
	const int stride_11 = 1;
	const int k_depth_11 = im_depth_11;
	
	const int p1_11 = 1;
	const int p2_11 = 1;
	
	const int output_height_11 = (((im_height_11+(2*p1_11)) - sqrt(k_size_11))/stride_11) + 1;
	const int output_width_11 = (((im_width_11+(2*p2_11)) - sqrt(k_size_11))/stride_11) + 1;
	const int output_size_11 = output_height_11 * output_width_11;
	
	MatrixXd conv5_1_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/conv5_1_weights.csv");
	MatrixXd conv5_1_w = conv5_1_weights;
	
	MatrixXd conv5_1_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/conv5_1_biases.csv");
	VectorXd conv5_1_b(Map<VectorXd>(conv5_1_biases.data(), conv5_1_biases.cols()*conv5_1_biases.rows()));
	
	const int im_height_12 = output_height_11;
	const int im_width_12 = output_width_11;
	const int im_depth_12 = k_num_11;
	const int im_size_12 = im_height_12 * im_width_12;
	
	const int k_num_12 = 512;
	const int k_size_12 = 9;
	const int stride_12 = 1;
	const int k_depth_12 = im_depth_12;
	
	const int p1_12 = 1;
	const int p2_12 = 1;
	
	const int output_height_12 = (((im_height_12+(2*p1_12)) - sqrt(k_size_12))/stride_12) + 1;
	const int output_width_12 = (((im_width_12+(2*p2_12)) - sqrt(k_size_12))/stride_12) + 1;
	const int output_size_12 = output_height_12 * output_width_12;
	
	MatrixXd conv5_2_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/conv5_2_weights.csv");
	MatrixXd conv5_2_w = conv5_2_weights;
	
	MatrixXd conv5_2_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/conv5_2_biases.csv");
	VectorXd conv5_2_b(Map<VectorXd>(conv5_2_biases.data(), conv5_2_biases.cols()*conv5_2_biases.rows()));
	
	const int im_height_13 = output_height_12;
	const int im_width_13 = output_width_12;
	const int im_depth_13 = k_num_12;
	const int im_size_13 = im_height_13 * im_width_13;
	
	const int k_num_13 = 512;
	const int k_size_13 = 9;
	const int stride_13 = 1;
	const int k_depth_13 = im_depth_13;
	
	const int p1_13 = 1;
	const int p2_13 = 1;
	
	const int output_height_13 = (((im_height_13+(2*p1_13)) - sqrt(k_size_13))/stride_13) + 1;
	const int output_width_13 = (((im_width_13+(2*p2_13)) - sqrt(k_size_13))/stride_13) + 1;
	const int output_size_13 = output_height_13 * output_width_13;
	
	MatrixXd conv5_3_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/conv5_3_weights.csv");
	MatrixXd conv5_3_w = conv5_3_weights;
	
	MatrixXd conv5_3_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/conv5_3_biases.csv");
	VectorXd conv5_3_b(Map<VectorXd>(conv5_3_biases.data(), conv5_3_biases.cols()*conv5_3_biases.rows()));
	
	const int f_5 = 2;
	const int s_5 = 2;
	
	const int pp1_5 = 0;
	const int pp2_5 = 0;
	
	MatrixXd fc6_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/fc6_weights.csv");
	
	MatrixXd fc6_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/fc6_biases.csv");
	VectorXd fc6_b(Map<VectorXd>(fc6_biases.data(), fc6_biases.cols()*fc6_biases.rows()));
	
	MatrixXd fc7_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/fc7_weights.csv");
	
	MatrixXd fc7_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/fc7_biases.csv");
	VectorXd fc7_b(Map<VectorXd>(fc7_biases.data(), fc7_biases.cols()*fc7_biases.rows()));
	
	MatrixXd fc8_weights = load_csv_arma<MatrixXd>("../weights/VGG_16/fc8_weights.csv");
	
	MatrixXd fc8_biases = load_csv_arma<MatrixXd>("../weights/VGG_16/fc8_biases.csv");
	VectorXd fc8_b(Map<VectorXd>(fc8_biases.data(), fc8_biases.cols()*fc8_biases.rows()));
	
	const int im_num = 1000;
	
	ifstream infile;
	infile.open("../inputs/VGG_16/production/imagenet_pxl_norm_1000.csv");
	
    for(int i=0; i < im_num; ++i)
    {   
        cout << i << endl;
    
		MatrixXd line = load_csv<MatrixXd>(infile);
		
        MatrixXd img;
        if ( line.rows() != 1 ) {
            img = line.block<1,im_size_1*im_depth_1>(i,0);
        }
        else {          
            img = line.block<1,im_size_1*im_depth_1>(0,0);
        }
        
        MatrixXd image = Map<Matrix<double, im_depth_1, im_size_1, RowMajor>>(img.data());

        clock_t run_time_start = clock();
        
		MatrixXd conv1_1;
		double gemm_time_1;
		double offline_time_1;
		std::tie(conv1_1, gemm_time_1, offline_time_1) = convolve(image, im_size_1, im_height_1, im_width_1, im_depth_1, k_size_1, stride_1, conv1_1_b, p1_1, p2_1, conv1_1_w, output_size_1, mode);
		
		MatrixXd relu1_1 = relu(conv1_1);
		
		MatrixXd conv1_2;
		double gemm_time_2;
		double offline_time_2;
		std::tie(conv1_2, gemm_time_2, offline_time_2) = convolve(relu1_1, im_size_2, im_height_2, im_width_2, im_depth_2, k_size_2, stride_2, conv1_2_b, p1_2, p2_2, conv1_2_w, output_size_2, mode);
		
		MatrixXd relu1_2 = relu(conv1_2);
		
		MatrixXd pool1 = pool(relu1_2, f_1, s_1, output_width_2, output_height_2, pp1_1, pp2_1);
		
		MatrixXd conv2_1;
		double gemm_time_3;
		double offline_time_3;
		std::tie(conv2_1, gemm_time_3, offline_time_3) = convolve(pool1, im_size_3, im_height_3, im_width_3, im_depth_3, k_size_3, stride_3, conv2_1_b, p1_3, p2_3, conv2_1_w, output_size_3, mode);
		
		MatrixXd relu2_1 = relu(conv2_1);
		
		MatrixXd conv2_2;
		double gemm_time_4;
		double offline_time_4;
		std::tie(conv2_2, gemm_time_4, offline_time_4) = convolve(relu2_1, im_size_4, im_height_4, im_width_4, im_depth_4, k_size_4, stride_4, conv2_2_b, p1_4, p2_4, conv2_2_w, output_size_4, mode);
		
		MatrixXd relu2_2 = relu(conv2_2);
		
		MatrixXd pool2 = pool(relu2_2, f_2, s_2, output_width_4, output_height_4, pp1_2, pp2_2);
		
		MatrixXd conv3_1;
		double gemm_time_5;
		double offline_time_5;
		std::tie(conv3_1, gemm_time_5, offline_time_5) = convolve(pool2, im_size_5, im_height_5, im_width_5, im_depth_5, k_size_5, stride_5, conv3_1_b, p1_5, p2_5, conv3_1_w, output_size_5, mode);
		
		MatrixXd relu3_1 = relu(conv3_1);
		
		MatrixXd conv3_2;
		double gemm_time_6;
		double offline_time_6;
		std::tie(conv3_2, gemm_time_6, offline_time_6) = convolve(relu3_1, im_size_6, im_height_6, im_width_6, im_depth_6, k_size_6, stride_6, conv3_2_b, p1_6, p2_6, conv3_2_w, output_size_6, mode);
		
		MatrixXd relu3_2 = relu(conv3_2);
		
		MatrixXd conv3_3;
		double gemm_time_7;
		double offline_time_7;
		std::tie(conv3_3, gemm_time_7, offline_time_7) = convolve(relu3_2, im_size_7, im_height_7, im_width_7, im_depth_7, k_size_7, stride_7, conv3_3_b, p1_7, p2_7, conv3_3_w, output_size_7, mode);
		
		MatrixXd relu3_3 = relu(conv3_3);
		
		MatrixXd pool3 = pool(relu3_3, f_3, s_3, output_width_7, output_height_7, pp1_3, pp2_3);
		
		MatrixXd conv4_1;
		double gemm_time_8;
		double offline_time_8;
		std::tie(conv4_1, gemm_time_8, offline_time_8) = convolve(pool3, im_size_8, im_height_8, im_width_8, im_depth_8, k_size_8, stride_8, conv4_1_b, p1_8, p2_8, conv4_1_w, output_size_8, mode);
		
		MatrixXd relu4_1 = relu(conv4_1);
		
		MatrixXd conv4_2;
		double gemm_time_9;
		double offline_time_9;
		std::tie(conv4_2, gemm_time_9, offline_time_9) = convolve(relu4_1, im_size_9, im_height_9, im_width_9, im_depth_9, k_size_9, stride_9, conv4_2_b, p1_9, p2_9, conv4_2_w, output_size_9, mode);
		
		MatrixXd relu4_2 = relu(conv4_2);
		
		MatrixXd conv4_3;
		double gemm_time_10;
		double offline_time_10;
		std::tie(conv4_3, gemm_time_10, offline_time_10) = convolve(relu4_2, im_size_10, im_height_10, im_width_10, im_depth_10, k_size_10, stride_10, conv4_3_b, p1_10, p2_10, conv4_3_w, output_size_10, mode);
		
		MatrixXd relu4_3 = relu(conv4_3);
		
		MatrixXd pool4 = pool(relu4_3, f_4, s_4, output_width_10, output_height_10, pp1_4, pp2_4);
		
		MatrixXd conv5_1;
		double gemm_time_11;
		double offline_time_11;
		std::tie(conv5_1, gemm_time_11, offline_time_11) = convolve(pool4, im_size_11, im_height_11, im_width_11, im_depth_11, k_size_11, stride_11, conv5_1_b, p1_11, p2_11, conv5_1_w, output_size_11, mode);
		
		MatrixXd relu5_1 = relu(conv5_1);
		
		MatrixXd conv5_2;
		double gemm_time_12;
		double offline_time_12;
		std::tie(conv5_2, gemm_time_12, offline_time_12) = convolve(relu5_1, im_size_12, im_height_12, im_width_12, im_depth_12, k_size_12, stride_12, conv5_2_b, p1_12, p2_12, conv5_2_w, output_size_12, mode);
		
		MatrixXd relu5_2 = relu(conv5_2);
		
		MatrixXd conv5_3;
		double gemm_time_13;
		double offline_time_13;
		std::tie(conv5_3, gemm_time_13, offline_time_13) = convolve(relu5_2, im_size_13, im_height_13, im_width_13, im_depth_13, k_size_13, stride_13, conv5_3_b, p1_13, p2_13, conv5_3_w, output_size_13, mode);
		
		MatrixXd relu5_3 = relu(conv5_3);
		
		MatrixXd pool5 = pool(relu5_3, f_5, s_5, output_width_13, output_height_13, pp1_5, pp2_5);
		
		MatrixXd fc6 = fully_connect(pool5, pool5.rows(), fc6_weights, fc6_b);
		
		MatrixXd relu6 = relu(fc6);
		
		MatrixXd fc7 = fully_connect(relu6, relu6.rows(), fc7_weights, fc7_b);
		
		MatrixXd relu7 = relu(fc7);
		
		MatrixXd fc8 = fully_connect(relu7, relu7.rows(), fc8_weights, fc8_b);
		
        clock_t run_time_end = clock();
        
        double run_time = (double) (run_time_end-run_time_start) / CLOCKS_PER_SEC;   
		run_time_total += (run_time - offline_time_1 - offline_time_2 - offline_time_3 - offline_time_4 - offline_time_5 - offline_time_6 - offline_time_7 - offline_time_8 - offline_time_9 - offline_time_10 - offline_time_11 - offline_time_12 - offline_time_13);
		gemm_time_total += 0.0 + gemm_time_1 + gemm_time_2 + gemm_time_3 + gemm_time_4 + gemm_time_5 + gemm_time_6 + gemm_time_7 + gemm_time_8 + gemm_time_9 + gemm_time_10 + gemm_time_11 + gemm_time_12 + gemm_time_13;
		
		std::string name_1 = "../features/VGG_16/fc8_" + std::to_string(i) + ".csv";
		write_to_csv(name_1, fc8);
    }

    infile.close();

    cout << "-----------------------------" << endl;

    float avg_run_time = 0.0;            
    avg_run_time = run_time_total / im_num;
    cout << "average online run time: " << avg_run_time << endl;

    float avg_gemm_time = 0.0;
    avg_gemm_time = gemm_time_total / im_num;
    cout << "average total time for GEMM: " << avg_gemm_time << endl;

    return 0; 
}
