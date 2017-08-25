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
	const int k_size_1 = 49;
	const int stride_1 = 2;
	const int k_depth_1 = im_depth_1;
	
	const int p1_1 = 3;
	const int p2_1 = 3;
	
	const int output_height_1 = (((im_height_1+(2*p1_1)) - sqrt(k_size_1))/stride_1) + 1;
	const int output_width_1 = (((im_width_1+(2*p2_1)) - sqrt(k_size_1))/stride_1) + 1;
	const int output_size_1 = output_height_1 * output_width_1;
	
	MatrixXd conv1_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/conv1_weights.csv");
	Map<MatrixXd> conv1_w(conv1_weights.data(), k_num_1, k_size_1 * k_depth_1);
	
	MatrixXd conv1_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/conv1_biases.csv");
	VectorXd conv1_b(Map<VectorXd>(conv1_biases.data(), conv1_biases.cols()*conv1_biases.rows()));
	
	const int f_1 = 2;
	const int s_1 = 2;
	std::string mode_1 = "max";
	
	const int pp1_1 = 0;
	const int pp2_1 = 0;
	
	const int im_height_2 = ((output_height_1 - f_1 + 2 * pp1_1) / s_1) + 1;
	const int im_width_2 = ((output_width_1 - f_1 + 2 * pp2_1) / s_1) + 1;
	const int im_depth_2 = k_num_1;
	const int im_size_2 = im_height_2 * im_width_2;
	
	const int k_num_2 = 256;
	const int k_size_2 = 1;
	const int stride_2 = 1;
	const int k_depth_2 = im_depth_2;
	
	const int p1_2 = 0;
	const int p2_2 = 0;
	
	const int output_height_2 = (((im_height_2+(2*p1_2)) - sqrt(k_size_2))/stride_2) + 1;
	const int output_width_2 = (((im_width_2+(2*p2_2)) - sqrt(k_size_2))/stride_2) + 1;
	const int output_size_2 = output_height_2 * output_width_2;
	
	MatrixXd res2a_branch1_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res2a_branch1_weights.csv");
	MatrixXd res2a_branch1_w = res2a_branch1_weights;
	
	MatrixXd res2a_branch1_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res2a_branch1_biases.csv");
	VectorXd res2a_branch1_b(Map<VectorXd>(res2a_branch1_biases.data(), res2a_branch1_biases.cols()*res2a_branch1_biases.rows()));
	
	const int im_height_3 = im_height_2;
	const int im_width_3 = im_width_2;
	const int im_depth_3 = im_depth_2;
	const int im_size_3 = im_size_2;
	
	const int k_num_3 = 64;
	const int k_size_3 = 1;
	const int stride_3 = 1;
	const int k_depth_3 = im_depth_3;
	
	const int p1_3 = 0;
	const int p2_3 = 0;
	
	const int output_height_3 = (((im_height_3+(2*p1_3)) - sqrt(k_size_3))/stride_3) + 1;
	const int output_width_3 = (((im_width_3+(2*p2_3)) - sqrt(k_size_3))/stride_3) + 1;
	const int output_size_3 = output_height_3 * output_width_3;
	
	MatrixXd res2a_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res2a_branch2a_weights.csv");
	MatrixXd res2a_branch2a_w = res2a_branch2a_weights;
	
	MatrixXd res2a_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res2a_branch2a_biases.csv");
	VectorXd res2a_branch2a_b(Map<VectorXd>(res2a_branch2a_biases.data(), res2a_branch2a_biases.cols()*res2a_branch2a_biases.rows()));
	
	const int im_height_4 = output_height_3;
	const int im_width_4 = output_width_3;
	const int im_depth_4 = k_num_3;
	const int im_size_4 = im_height_4 * im_width_4;
	
	const int k_num_4 = 64;
	const int k_size_4 = 9;
	const int stride_4 = 1;
	const int k_depth_4 = im_depth_4;
	
	const int p1_4 = 1;
	const int p2_4 = 1;
	
	const int output_height_4 = (((im_height_4+(2*p1_4)) - sqrt(k_size_4))/stride_4) + 1;
	const int output_width_4 = (((im_width_4+(2*p2_4)) - sqrt(k_size_4))/stride_4) + 1;
	const int output_size_4 = output_height_4 * output_width_4;
	
	MatrixXd res2a_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res2a_branch2b_weights.csv");
	MatrixXd res2a_branch2b_w = res2a_branch2b_weights;
	
	MatrixXd res2a_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res2a_branch2b_biases.csv");
	VectorXd res2a_branch2b_b(Map<VectorXd>(res2a_branch2b_biases.data(), res2a_branch2b_biases.cols()*res2a_branch2b_biases.rows()));
	
	const int im_height_5 = output_height_4;
	const int im_width_5 = output_width_4;
	const int im_depth_5 = k_num_4;
	const int im_size_5 = im_height_5 * im_width_5;
	
	const int k_num_5 = 256;
	const int k_size_5 = 1;
	const int stride_5 = 1;
	const int k_depth_5 = im_depth_5;
	
	const int p1_5 = 0;
	const int p2_5 = 0;
	
	const int output_height_5 = (((im_height_5+(2*p1_5)) - sqrt(k_size_5))/stride_5) + 1;
	const int output_width_5 = (((im_width_5+(2*p2_5)) - sqrt(k_size_5))/stride_5) + 1;
	const int output_size_5 = output_height_5 * output_width_5;
	
	MatrixXd res2a_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res2a_branch2c_weights.csv");
	MatrixXd res2a_branch2c_w = res2a_branch2c_weights;
	
	MatrixXd res2a_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res2a_branch2c_biases.csv");
	VectorXd res2a_branch2c_b(Map<VectorXd>(res2a_branch2c_biases.data(), res2a_branch2c_biases.cols()*res2a_branch2c_biases.rows()));
	
	const int im_height_6 = output_height_5;
	const int im_width_6 = output_width_5;
	const int im_depth_6 = k_num_5;
	const int im_size_6 = im_height_6 * im_width_6;
	
	const int k_num_6 = 64;
	const int k_size_6 = 1;
	const int stride_6 = 1;
	const int k_depth_6 = im_depth_6;
	
	const int p1_6 = 0;
	const int p2_6 = 0;
	
	const int output_height_6 = (((im_height_6+(2*p1_6)) - sqrt(k_size_6))/stride_6) + 1;
	const int output_width_6 = (((im_width_6+(2*p2_6)) - sqrt(k_size_6))/stride_6) + 1;
	const int output_size_6 = output_height_6 * output_width_6;
	
	MatrixXd res2b_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res2b_branch2a_weights.csv");
	MatrixXd res2b_branch2a_w = res2b_branch2a_weights;
	
	MatrixXd res2b_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res2b_branch2a_biases.csv");
	VectorXd res2b_branch2a_b(Map<VectorXd>(res2b_branch2a_biases.data(), res2b_branch2a_biases.cols()*res2b_branch2a_biases.rows()));
	
	const int im_height_7 = output_height_6;
	const int im_width_7 = output_width_6;
	const int im_depth_7 = k_num_6;
	const int im_size_7 = im_height_7 * im_width_7;
	
	const int k_num_7 = 64;
	const int k_size_7 = 9;
	const int stride_7 = 1;
	const int k_depth_7 = im_depth_7;
	
	const int p1_7 = 1;
	const int p2_7 = 1;
	
	const int output_height_7 = (((im_height_7+(2*p1_7)) - sqrt(k_size_7))/stride_7) + 1;
	const int output_width_7 = (((im_width_7+(2*p2_7)) - sqrt(k_size_7))/stride_7) + 1;
	const int output_size_7 = output_height_7 * output_width_7;
	
	MatrixXd res2b_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res2b_branch2b_weights.csv");
	MatrixXd res2b_branch2b_w = res2b_branch2b_weights;
	
	MatrixXd res2b_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res2b_branch2b_biases.csv");
	VectorXd res2b_branch2b_b(Map<VectorXd>(res2b_branch2b_biases.data(), res2b_branch2b_biases.cols()*res2b_branch2b_biases.rows()));
	
	const int im_height_8 = output_height_7;
	const int im_width_8 = output_width_7;
	const int im_depth_8 = k_num_7;
	const int im_size_8 = im_height_8 * im_width_8;
	
	const int k_num_8 = 256;
	const int k_size_8 = 1;
	const int stride_8 = 1;
	const int k_depth_8 = im_depth_8;
	
	const int p1_8 = 0;
	const int p2_8 = 0;
	
	const int output_height_8 = (((im_height_8+(2*p1_8)) - sqrt(k_size_8))/stride_8) + 1;
	const int output_width_8 = (((im_width_8+(2*p2_8)) - sqrt(k_size_8))/stride_8) + 1;
	const int output_size_8 = output_height_8 * output_width_8;
	
	MatrixXd res2b_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res2b_branch2c_weights.csv");
	MatrixXd res2b_branch2c_w = res2b_branch2c_weights;
	
	MatrixXd res2b_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res2b_branch2c_biases.csv");
	VectorXd res2b_branch2c_b(Map<VectorXd>(res2b_branch2c_biases.data(), res2b_branch2c_biases.cols()*res2b_branch2c_biases.rows()));
	
	const int im_height_9 = output_height_8;
	const int im_width_9 = output_width_8;
	const int im_depth_9 = k_num_8;
	const int im_size_9 = im_height_9 * im_width_9;
	
	const int k_num_9 = 64;
	const int k_size_9 = 1;
	const int stride_9 = 1;
	const int k_depth_9 = im_depth_9;
	
	const int p1_9 = 0;
	const int p2_9 = 0;
	
	const int output_height_9 = (((im_height_9+(2*p1_9)) - sqrt(k_size_9))/stride_9) + 1;
	const int output_width_9 = (((im_width_9+(2*p2_9)) - sqrt(k_size_9))/stride_9) + 1;
	const int output_size_9 = output_height_9 * output_width_9;
	
	MatrixXd res2c_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res2c_branch2a_weights.csv");
	MatrixXd res2c_branch2a_w = res2c_branch2a_weights;
	
	MatrixXd res2c_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res2c_branch2a_biases.csv");
	VectorXd res2c_branch2a_b(Map<VectorXd>(res2c_branch2a_biases.data(), res2c_branch2a_biases.cols()*res2c_branch2a_biases.rows()));
	
	const int im_height_10 = output_height_9;
	const int im_width_10 = output_width_9;
	const int im_depth_10 = k_num_9;
	const int im_size_10 = im_height_10 * im_width_10;
	
	const int k_num_10 = 64;
	const int k_size_10 = 9;
	const int stride_10 = 1;
	const int k_depth_10 = im_depth_10;
	
	const int p1_10 = 1;
	const int p2_10 = 1;
	
	const int output_height_10 = (((im_height_10+(2*p1_10)) - sqrt(k_size_10))/stride_10) + 1;
	const int output_width_10 = (((im_width_10+(2*p2_10)) - sqrt(k_size_10))/stride_10) + 1;
	const int output_size_10 = output_height_10 * output_width_10;
	
	MatrixXd res2c_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res2c_branch2b_weights.csv");
	MatrixXd res2c_branch2b_w = res2c_branch2b_weights;
	
	MatrixXd res2c_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res2c_branch2b_biases.csv");
	VectorXd res2c_branch2b_b(Map<VectorXd>(res2c_branch2b_biases.data(), res2c_branch2b_biases.cols()*res2c_branch2b_biases.rows()));
	
	const int im_height_11 = output_height_10;
	const int im_width_11 = output_width_10;
	const int im_depth_11 = k_num_10;
	const int im_size_11 = im_height_11 * im_width_11;
	
	const int k_num_11 = 256;
	const int k_size_11 = 1;
	const int stride_11 = 1;
	const int k_depth_11 = im_depth_11;
	
	const int p1_11 = 0;
	const int p2_11 = 0;
	
	const int output_height_11 = (((im_height_11+(2*p1_11)) - sqrt(k_size_11))/stride_11) + 1;
	const int output_width_11 = (((im_width_11+(2*p2_11)) - sqrt(k_size_11))/stride_11) + 1;
	const int output_size_11 = output_height_11 * output_width_11;
	
	MatrixXd res2c_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res2c_branch2c_weights.csv");
	MatrixXd res2c_branch2c_w = res2c_branch2c_weights;
	
	MatrixXd res2c_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res2c_branch2c_biases.csv");
	VectorXd res2c_branch2c_b(Map<VectorXd>(res2c_branch2c_biases.data(), res2c_branch2c_biases.cols()*res2c_branch2c_biases.rows()));
	
	const int im_height_12 = output_height_11;
	const int im_width_12 = output_width_11;
	const int im_depth_12 = k_num_11;
	const int im_size_12 = im_height_12 * im_width_12;
	
	const int k_num_12 = 512;
	const int k_size_12 = 1;
	const int stride_12 = 2;
	const int k_depth_12 = im_depth_12;
	
	const int p1_12 = 0;
	const int p2_12 = 0;
	
	const int output_height_12 = (((im_height_12+(2*p1_12)) - sqrt(k_size_12))/stride_12) + 1;
	const int output_width_12 = (((im_width_12+(2*p2_12)) - sqrt(k_size_12))/stride_12) + 1;
	const int output_size_12 = output_height_12 * output_width_12;
	
	MatrixXd res3a_branch1_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res3a_branch1_weights.csv");
	MatrixXd res3a_branch1_w = res3a_branch1_weights;
	
	MatrixXd res3a_branch1_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res3a_branch1_biases.csv");
	VectorXd res3a_branch1_b(Map<VectorXd>(res3a_branch1_biases.data(), res3a_branch1_biases.cols()*res3a_branch1_biases.rows()));
	
	const int im_height_13 = im_height_12;
	const int im_width_13 = im_width_12;
	const int im_depth_13 = im_depth_12;
	const int im_size_13 = im_size_12;
	
	const int k_num_13 = 128;
	const int k_size_13 = 1;
	const int stride_13 = 2;
	const int k_depth_13 = im_depth_13;
	
	const int p1_13 = 0;
	const int p2_13 = 0;
	
	const int output_height_13 = (((im_height_13+(2*p1_13)) - sqrt(k_size_13))/stride_13) + 1;
	const int output_width_13 = (((im_width_13+(2*p2_13)) - sqrt(k_size_13))/stride_13) + 1;
	const int output_size_13 = output_height_13 * output_width_13;
	
	MatrixXd res3a_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res3a_branch2a_weights.csv");
	MatrixXd res3a_branch2a_w = res3a_branch2a_weights;
	
	MatrixXd res3a_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res3a_branch2a_biases.csv");
	VectorXd res3a_branch2a_b(Map<VectorXd>(res3a_branch2a_biases.data(), res3a_branch2a_biases.cols()*res3a_branch2a_biases.rows()));
	
	const int im_height_14 = output_height_13;
	const int im_width_14 = output_width_13;
	const int im_depth_14 = k_num_13;
	const int im_size_14 = im_height_14 * im_width_14;
	
	const int k_num_14 = 128;
	const int k_size_14 = 9;
	const int stride_14 = 1;
	const int k_depth_14 = im_depth_14;
	
	const int p1_14 = 1;
	const int p2_14 = 1;
	
	const int output_height_14 = (((im_height_14+(2*p1_14)) - sqrt(k_size_14))/stride_14) + 1;
	const int output_width_14 = (((im_width_14+(2*p2_14)) - sqrt(k_size_14))/stride_14) + 1;
	const int output_size_14 = output_height_14 * output_width_14;
	
	MatrixXd res3a_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res3a_branch2b_weights.csv");
	MatrixXd res3a_branch2b_w = res3a_branch2b_weights;
	
	MatrixXd res3a_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res3a_branch2b_biases.csv");
	VectorXd res3a_branch2b_b(Map<VectorXd>(res3a_branch2b_biases.data(), res3a_branch2b_biases.cols()*res3a_branch2b_biases.rows()));
	
	const int im_height_15 = output_height_14;
	const int im_width_15 = output_width_14;
	const int im_depth_15 = k_num_14;
	const int im_size_15 = im_height_15 * im_width_15;
	
	const int k_num_15 = 512;
	const int k_size_15 = 1;
	const int stride_15 = 1;
	const int k_depth_15 = im_depth_15;
	
	const int p1_15 = 0;
	const int p2_15 = 0;
	
	const int output_height_15 = (((im_height_15+(2*p1_15)) - sqrt(k_size_15))/stride_15) + 1;
	const int output_width_15 = (((im_width_15+(2*p2_15)) - sqrt(k_size_15))/stride_15) + 1;
	const int output_size_15 = output_height_15 * output_width_15;
	
	MatrixXd res3a_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res3a_branch2c_weights.csv");
	MatrixXd res3a_branch2c_w = res3a_branch2c_weights;
	
	MatrixXd res3a_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res3a_branch2c_biases.csv");
	VectorXd res3a_branch2c_b(Map<VectorXd>(res3a_branch2c_biases.data(), res3a_branch2c_biases.cols()*res3a_branch2c_biases.rows()));
	
	const int im_height_16 = output_height_15;
	const int im_width_16 = output_width_15;
	const int im_depth_16 = k_num_15;
	const int im_size_16 = im_height_16 * im_width_16;
	
	const int k_num_16 = 128;
	const int k_size_16 = 1;
	const int stride_16 = 1;
	const int k_depth_16 = im_depth_16;
	
	const int p1_16 = 0;
	const int p2_16 = 0;
	
	const int output_height_16 = (((im_height_16+(2*p1_16)) - sqrt(k_size_16))/stride_16) + 1;
	const int output_width_16 = (((im_width_16+(2*p2_16)) - sqrt(k_size_16))/stride_16) + 1;
	const int output_size_16 = output_height_16 * output_width_16;
	
	MatrixXd res3b1_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b1_branch2a_weights.csv");
	MatrixXd res3b1_branch2a_w = res3b1_branch2a_weights;
	
	MatrixXd res3b1_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b1_branch2a_biases.csv");
	VectorXd res3b1_branch2a_b(Map<VectorXd>(res3b1_branch2a_biases.data(), res3b1_branch2a_biases.cols()*res3b1_branch2a_biases.rows()));
	
	const int im_height_17 = output_height_16;
	const int im_width_17 = output_width_16;
	const int im_depth_17 = k_num_16;
	const int im_size_17 = im_height_17 * im_width_17;
	
	const int k_num_17 = 128;
	const int k_size_17 = 9;
	const int stride_17 = 1;
	const int k_depth_17 = im_depth_17;
	
	const int p1_17 = 1;
	const int p2_17 = 1;
	
	const int output_height_17 = (((im_height_17+(2*p1_17)) - sqrt(k_size_17))/stride_17) + 1;
	const int output_width_17 = (((im_width_17+(2*p2_17)) - sqrt(k_size_17))/stride_17) + 1;
	const int output_size_17 = output_height_17 * output_width_17;
	
	MatrixXd res3b1_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b1_branch2b_weights.csv");
	MatrixXd res3b1_branch2b_w = res3b1_branch2b_weights;
	
	MatrixXd res3b1_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b1_branch2b_biases.csv");
	VectorXd res3b1_branch2b_b(Map<VectorXd>(res3b1_branch2b_biases.data(), res3b1_branch2b_biases.cols()*res3b1_branch2b_biases.rows()));
	
	const int im_height_18 = output_height_17;
	const int im_width_18 = output_width_17;
	const int im_depth_18 = k_num_17;
	const int im_size_18 = im_height_18 * im_width_18;
	
	const int k_num_18 = 512;
	const int k_size_18 = 1;
	const int stride_18 = 1;
	const int k_depth_18 = im_depth_18;
	
	const int p1_18 = 0;
	const int p2_18 = 0;
	
	const int output_height_18 = (((im_height_18+(2*p1_18)) - sqrt(k_size_18))/stride_18) + 1;
	const int output_width_18 = (((im_width_18+(2*p2_18)) - sqrt(k_size_18))/stride_18) + 1;
	const int output_size_18 = output_height_18 * output_width_18;
	
	MatrixXd res3b1_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b1_branch2c_weights.csv");
	MatrixXd res3b1_branch2c_w = res3b1_branch2c_weights;
	
	MatrixXd res3b1_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b1_branch2c_biases.csv");
	VectorXd res3b1_branch2c_b(Map<VectorXd>(res3b1_branch2c_biases.data(), res3b1_branch2c_biases.cols()*res3b1_branch2c_biases.rows()));
	
	const int im_height_19 = output_height_18;
	const int im_width_19 = output_width_18;
	const int im_depth_19 = k_num_18;
	const int im_size_19 = im_height_19 * im_width_19;
	
	const int k_num_19 = 128;
	const int k_size_19 = 1;
	const int stride_19 = 1;
	const int k_depth_19 = im_depth_19;
	
	const int p1_19 = 0;
	const int p2_19 = 0;
	
	const int output_height_19 = (((im_height_19+(2*p1_19)) - sqrt(k_size_19))/stride_19) + 1;
	const int output_width_19 = (((im_width_19+(2*p2_19)) - sqrt(k_size_19))/stride_19) + 1;
	const int output_size_19 = output_height_19 * output_width_19;
	
	MatrixXd res3b2_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b2_branch2a_weights.csv");
	MatrixXd res3b2_branch2a_w = res3b2_branch2a_weights;
	
	MatrixXd res3b2_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b2_branch2a_biases.csv");
	VectorXd res3b2_branch2a_b(Map<VectorXd>(res3b2_branch2a_biases.data(), res3b2_branch2a_biases.cols()*res3b2_branch2a_biases.rows()));
	
	const int im_height_20 = output_height_19;
	const int im_width_20 = output_width_19;
	const int im_depth_20 = k_num_19;
	const int im_size_20 = im_height_20 * im_width_20;
	
	const int k_num_20 = 128;
	const int k_size_20 = 9;
	const int stride_20 = 1;
	const int k_depth_20 = im_depth_20;
	
	const int p1_20 = 1;
	const int p2_20 = 1;
	
	const int output_height_20 = (((im_height_20+(2*p1_20)) - sqrt(k_size_20))/stride_20) + 1;
	const int output_width_20 = (((im_width_20+(2*p2_20)) - sqrt(k_size_20))/stride_20) + 1;
	const int output_size_20 = output_height_20 * output_width_20;
	
	MatrixXd res3b2_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b2_branch2b_weights.csv");
	MatrixXd res3b2_branch2b_w = res3b2_branch2b_weights;
	
	MatrixXd res3b2_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b2_branch2b_biases.csv");
	VectorXd res3b2_branch2b_b(Map<VectorXd>(res3b2_branch2b_biases.data(), res3b2_branch2b_biases.cols()*res3b2_branch2b_biases.rows()));
	
	const int im_height_21 = output_height_20;
	const int im_width_21 = output_width_20;
	const int im_depth_21 = k_num_20;
	const int im_size_21 = im_height_21 * im_width_21;
	
	const int k_num_21 = 512;
	const int k_size_21 = 1;
	const int stride_21 = 1;
	const int k_depth_21 = im_depth_21;
	
	const int p1_21 = 0;
	const int p2_21 = 0;
	
	const int output_height_21 = (((im_height_21+(2*p1_21)) - sqrt(k_size_21))/stride_21) + 1;
	const int output_width_21 = (((im_width_21+(2*p2_21)) - sqrt(k_size_21))/stride_21) + 1;
	const int output_size_21 = output_height_21 * output_width_21;
	
	MatrixXd res3b2_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b2_branch2c_weights.csv");
	MatrixXd res3b2_branch2c_w = res3b2_branch2c_weights;
	
	MatrixXd res3b2_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b2_branch2c_biases.csv");
	VectorXd res3b2_branch2c_b(Map<VectorXd>(res3b2_branch2c_biases.data(), res3b2_branch2c_biases.cols()*res3b2_branch2c_biases.rows()));
	
	const int im_height_22 = output_height_21;
	const int im_width_22 = output_width_21;
	const int im_depth_22 = k_num_21;
	const int im_size_22 = im_height_22 * im_width_22;
	
	const int k_num_22 = 128;
	const int k_size_22 = 1;
	const int stride_22 = 1;
	const int k_depth_22 = im_depth_22;
	
	const int p1_22 = 0;
	const int p2_22 = 0;
	
	const int output_height_22 = (((im_height_22+(2*p1_22)) - sqrt(k_size_22))/stride_22) + 1;
	const int output_width_22 = (((im_width_22+(2*p2_22)) - sqrt(k_size_22))/stride_22) + 1;
	const int output_size_22 = output_height_22 * output_width_22;
	
	MatrixXd res3b3_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b3_branch2a_weights.csv");
	MatrixXd res3b3_branch2a_w = res3b3_branch2a_weights;
	
	MatrixXd res3b3_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b3_branch2a_biases.csv");
	VectorXd res3b3_branch2a_b(Map<VectorXd>(res3b3_branch2a_biases.data(), res3b3_branch2a_biases.cols()*res3b3_branch2a_biases.rows()));
	
	const int im_height_23 = output_height_22;
	const int im_width_23 = output_width_22;
	const int im_depth_23 = k_num_22;
	const int im_size_23 = im_height_23 * im_width_23;
	
	const int k_num_23 = 128;
	const int k_size_23 = 9;
	const int stride_23 = 1;
	const int k_depth_23 = im_depth_23;
	
	const int p1_23 = 1;
	const int p2_23 = 1;
	
	const int output_height_23 = (((im_height_23+(2*p1_23)) - sqrt(k_size_23))/stride_23) + 1;
	const int output_width_23 = (((im_width_23+(2*p2_23)) - sqrt(k_size_23))/stride_23) + 1;
	const int output_size_23 = output_height_23 * output_width_23;
	
	MatrixXd res3b3_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b3_branch2b_weights.csv");
	MatrixXd res3b3_branch2b_w = res3b3_branch2b_weights;
	
	MatrixXd res3b3_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b3_branch2b_biases.csv");
	VectorXd res3b3_branch2b_b(Map<VectorXd>(res3b3_branch2b_biases.data(), res3b3_branch2b_biases.cols()*res3b3_branch2b_biases.rows()));
	
	const int im_height_24 = output_height_23;
	const int im_width_24 = output_width_23;
	const int im_depth_24 = k_num_23;
	const int im_size_24 = im_height_24 * im_width_24;
	
	const int k_num_24 = 512;
	const int k_size_24 = 1;
	const int stride_24 = 1;
	const int k_depth_24 = im_depth_24;
	
	const int p1_24 = 0;
	const int p2_24 = 0;
	
	const int output_height_24 = (((im_height_24+(2*p1_24)) - sqrt(k_size_24))/stride_24) + 1;
	const int output_width_24 = (((im_width_24+(2*p2_24)) - sqrt(k_size_24))/stride_24) + 1;
	const int output_size_24 = output_height_24 * output_width_24;
	
	MatrixXd res3b3_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b3_branch2c_weights.csv");
	MatrixXd res3b3_branch2c_w = res3b3_branch2c_weights;
	
	MatrixXd res3b3_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res3b3_branch2c_biases.csv");
	VectorXd res3b3_branch2c_b(Map<VectorXd>(res3b3_branch2c_biases.data(), res3b3_branch2c_biases.cols()*res3b3_branch2c_biases.rows()));
	
	const int im_height_25 = output_height_24;
	const int im_width_25 = output_width_24;
	const int im_depth_25 = k_num_24;
	const int im_size_25 = im_height_25 * im_width_25;
	
	const int k_num_25 = 1024;
	const int k_size_25 = 1;
	const int stride_25 = 2;
	const int k_depth_25 = im_depth_25;
	
	const int p1_25 = 0;
	const int p2_25 = 0;
	
	const int output_height_25 = (((im_height_25+(2*p1_25)) - sqrt(k_size_25))/stride_25) + 1;
	const int output_width_25 = (((im_width_25+(2*p2_25)) - sqrt(k_size_25))/stride_25) + 1;
	const int output_size_25 = output_height_25 * output_width_25;
	
	MatrixXd res4a_branch1_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4a_branch1_weights.csv");
	MatrixXd res4a_branch1_w = res4a_branch1_weights;
	
	MatrixXd res4a_branch1_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4a_branch1_biases.csv");
	VectorXd res4a_branch1_b(Map<VectorXd>(res4a_branch1_biases.data(), res4a_branch1_biases.cols()*res4a_branch1_biases.rows()));
	
	const int im_height_26 = im_height_25;
	const int im_width_26 = im_width_25;
	const int im_depth_26 = im_depth_25;
	const int im_size_26 = im_size_25;
	
	const int k_num_26 = 256;
	const int k_size_26 = 1;
	const int stride_26 = 2;
	const int k_depth_26 = im_depth_26;
	
	const int p1_26 = 0;
	const int p2_26 = 0;
	
	const int output_height_26 = (((im_height_26+(2*p1_26)) - sqrt(k_size_26))/stride_26) + 1;
	const int output_width_26 = (((im_width_26+(2*p2_26)) - sqrt(k_size_26))/stride_26) + 1;
	const int output_size_26 = output_height_26 * output_width_26;
	
	MatrixXd res4a_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4a_branch2a_weights.csv");
	MatrixXd res4a_branch2a_w = res4a_branch2a_weights;
	
	MatrixXd res4a_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4a_branch2a_biases.csv");
	VectorXd res4a_branch2a_b(Map<VectorXd>(res4a_branch2a_biases.data(), res4a_branch2a_biases.cols()*res4a_branch2a_biases.rows()));
	
	const int im_height_27 = output_height_26;
	const int im_width_27 = output_width_26;
	const int im_depth_27 = k_num_26;
	const int im_size_27 = im_height_27 * im_width_27;
	
	const int k_num_27 = 256;
	const int k_size_27 = 9;
	const int stride_27 = 1;
	const int k_depth_27 = im_depth_27;
	
	const int p1_27 = 1;
	const int p2_27 = 1;
	
	const int output_height_27 = (((im_height_27+(2*p1_27)) - sqrt(k_size_27))/stride_27) + 1;
	const int output_width_27 = (((im_width_27+(2*p2_27)) - sqrt(k_size_27))/stride_27) + 1;
	const int output_size_27 = output_height_27 * output_width_27;
	
	MatrixXd res4a_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4a_branch2b_weights.csv");
	MatrixXd res4a_branch2b_w = res4a_branch2b_weights;
	
	MatrixXd res4a_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4a_branch2b_biases.csv");
	VectorXd res4a_branch2b_b(Map<VectorXd>(res4a_branch2b_biases.data(), res4a_branch2b_biases.cols()*res4a_branch2b_biases.rows()));
	
	const int im_height_28 = output_height_27;
	const int im_width_28 = output_width_27;
	const int im_depth_28 = k_num_27;
	const int im_size_28 = im_height_28 * im_width_28;
	
	const int k_num_28 = 1024;
	const int k_size_28 = 1;
	const int stride_28 = 1;
	const int k_depth_28 = im_depth_28;
	
	const int p1_28 = 0;
	const int p2_28 = 0;
	
	const int output_height_28 = (((im_height_28+(2*p1_28)) - sqrt(k_size_28))/stride_28) + 1;
	const int output_width_28 = (((im_width_28+(2*p2_28)) - sqrt(k_size_28))/stride_28) + 1;
	const int output_size_28 = output_height_28 * output_width_28;
	
	MatrixXd res4a_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4a_branch2c_weights.csv");
	MatrixXd res4a_branch2c_w = res4a_branch2c_weights;
	
	MatrixXd res4a_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4a_branch2c_biases.csv");
	VectorXd res4a_branch2c_b(Map<VectorXd>(res4a_branch2c_biases.data(), res4a_branch2c_biases.cols()*res4a_branch2c_biases.rows()));
	
	const int im_height_29 = output_height_28;
	const int im_width_29 = output_width_28;
	const int im_depth_29 = k_num_28;
	const int im_size_29 = im_height_29 * im_width_29;
	
	const int k_num_29 = 256;
	const int k_size_29 = 1;
	const int stride_29 = 1;
	const int k_depth_29 = im_depth_29;
	
	const int p1_29 = 0;
	const int p2_29 = 0;
	
	const int output_height_29 = (((im_height_29+(2*p1_29)) - sqrt(k_size_29))/stride_29) + 1;
	const int output_width_29 = (((im_width_29+(2*p2_29)) - sqrt(k_size_29))/stride_29) + 1;
	const int output_size_29 = output_height_29 * output_width_29;
	
	MatrixXd res4b1_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b1_branch2a_weights.csv");
	MatrixXd res4b1_branch2a_w = res4b1_branch2a_weights;
	
	MatrixXd res4b1_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b1_branch2a_biases.csv");
	VectorXd res4b1_branch2a_b(Map<VectorXd>(res4b1_branch2a_biases.data(), res4b1_branch2a_biases.cols()*res4b1_branch2a_biases.rows()));
	
	const int im_height_30 = output_height_29;
	const int im_width_30 = output_width_29;
	const int im_depth_30 = k_num_29;
	const int im_size_30 = im_height_30 * im_width_30;
	
	const int k_num_30 = 256;
	const int k_size_30 = 9;
	const int stride_30 = 1;
	const int k_depth_30 = im_depth_30;
	
	const int p1_30 = 1;
	const int p2_30 = 1;
	
	const int output_height_30 = (((im_height_30+(2*p1_30)) - sqrt(k_size_30))/stride_30) + 1;
	const int output_width_30 = (((im_width_30+(2*p2_30)) - sqrt(k_size_30))/stride_30) + 1;
	const int output_size_30 = output_height_30 * output_width_30;
	
	MatrixXd res4b1_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b1_branch2b_weights.csv");
	MatrixXd res4b1_branch2b_w = res4b1_branch2b_weights;
	
	MatrixXd res4b1_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b1_branch2b_biases.csv");
	VectorXd res4b1_branch2b_b(Map<VectorXd>(res4b1_branch2b_biases.data(), res4b1_branch2b_biases.cols()*res4b1_branch2b_biases.rows()));
	
	const int im_height_31 = output_height_30;
	const int im_width_31 = output_width_30;
	const int im_depth_31 = k_num_30;
	const int im_size_31 = im_height_31 * im_width_31;
	
	const int k_num_31 = 1024;
	const int k_size_31 = 1;
	const int stride_31 = 1;
	const int k_depth_31 = im_depth_31;
	
	const int p1_31 = 0;
	const int p2_31 = 0;
	
	const int output_height_31 = (((im_height_31+(2*p1_31)) - sqrt(k_size_31))/stride_31) + 1;
	const int output_width_31 = (((im_width_31+(2*p2_31)) - sqrt(k_size_31))/stride_31) + 1;
	const int output_size_31 = output_height_31 * output_width_31;
	
	MatrixXd res4b1_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b1_branch2c_weights.csv");
	MatrixXd res4b1_branch2c_w = res4b1_branch2c_weights;
	
	MatrixXd res4b1_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b1_branch2c_biases.csv");
	VectorXd res4b1_branch2c_b(Map<VectorXd>(res4b1_branch2c_biases.data(), res4b1_branch2c_biases.cols()*res4b1_branch2c_biases.rows()));
	
	const int im_height_32 = output_height_31;
	const int im_width_32 = output_width_31;
	const int im_depth_32 = k_num_31;
	const int im_size_32 = im_height_32 * im_width_32;
	
	const int k_num_32 = 256;
	const int k_size_32 = 1;
	const int stride_32 = 1;
	const int k_depth_32 = im_depth_32;
	
	const int p1_32 = 0;
	const int p2_32 = 0;
	
	const int output_height_32 = (((im_height_32+(2*p1_32)) - sqrt(k_size_32))/stride_32) + 1;
	const int output_width_32 = (((im_width_32+(2*p2_32)) - sqrt(k_size_32))/stride_32) + 1;
	const int output_size_32 = output_height_32 * output_width_32;
	
	MatrixXd res4b2_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b2_branch2a_weights.csv");
	MatrixXd res4b2_branch2a_w = res4b2_branch2a_weights;
	
	MatrixXd res4b2_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b2_branch2a_biases.csv");
	VectorXd res4b2_branch2a_b(Map<VectorXd>(res4b2_branch2a_biases.data(), res4b2_branch2a_biases.cols()*res4b2_branch2a_biases.rows()));
	
	const int im_height_33 = output_height_32;
	const int im_width_33 = output_width_32;
	const int im_depth_33 = k_num_32;
	const int im_size_33 = im_height_33 * im_width_33;
	
	const int k_num_33 = 256;
	const int k_size_33 = 9;
	const int stride_33 = 1;
	const int k_depth_33 = im_depth_33;
	
	const int p1_33 = 1;
	const int p2_33 = 1;
	
	const int output_height_33 = (((im_height_33+(2*p1_33)) - sqrt(k_size_33))/stride_33) + 1;
	const int output_width_33 = (((im_width_33+(2*p2_33)) - sqrt(k_size_33))/stride_33) + 1;
	const int output_size_33 = output_height_33 * output_width_33;
	
	MatrixXd res4b2_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b2_branch2b_weights.csv");
	MatrixXd res4b2_branch2b_w = res4b2_branch2b_weights;
	
	MatrixXd res4b2_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b2_branch2b_biases.csv");
	VectorXd res4b2_branch2b_b(Map<VectorXd>(res4b2_branch2b_biases.data(), res4b2_branch2b_biases.cols()*res4b2_branch2b_biases.rows()));
	
	const int im_height_34 = output_height_33;
	const int im_width_34 = output_width_33;
	const int im_depth_34 = k_num_33;
	const int im_size_34 = im_height_34 * im_width_34;
	
	const int k_num_34 = 1024;
	const int k_size_34 = 1;
	const int stride_34 = 1;
	const int k_depth_34 = im_depth_34;
	
	const int p1_34 = 0;
	const int p2_34 = 0;
	
	const int output_height_34 = (((im_height_34+(2*p1_34)) - sqrt(k_size_34))/stride_34) + 1;
	const int output_width_34 = (((im_width_34+(2*p2_34)) - sqrt(k_size_34))/stride_34) + 1;
	const int output_size_34 = output_height_34 * output_width_34;
	
	MatrixXd res4b2_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b2_branch2c_weights.csv");
	MatrixXd res4b2_branch2c_w = res4b2_branch2c_weights;
	
	MatrixXd res4b2_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b2_branch2c_biases.csv");
	VectorXd res4b2_branch2c_b(Map<VectorXd>(res4b2_branch2c_biases.data(), res4b2_branch2c_biases.cols()*res4b2_branch2c_biases.rows()));
	
	const int im_height_35 = output_height_34;
	const int im_width_35 = output_width_34;
	const int im_depth_35 = k_num_34;
	const int im_size_35 = im_height_35 * im_width_35;
	
	const int k_num_35 = 256;
	const int k_size_35 = 1;
	const int stride_35 = 1;
	const int k_depth_35 = im_depth_35;
	
	const int p1_35 = 0;
	const int p2_35 = 0;
	
	const int output_height_35 = (((im_height_35+(2*p1_35)) - sqrt(k_size_35))/stride_35) + 1;
	const int output_width_35 = (((im_width_35+(2*p2_35)) - sqrt(k_size_35))/stride_35) + 1;
	const int output_size_35 = output_height_35 * output_width_35;
	
	MatrixXd res4b3_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b3_branch2a_weights.csv");
	MatrixXd res4b3_branch2a_w = res4b3_branch2a_weights;
	
	MatrixXd res4b3_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b3_branch2a_biases.csv");
	VectorXd res4b3_branch2a_b(Map<VectorXd>(res4b3_branch2a_biases.data(), res4b3_branch2a_biases.cols()*res4b3_branch2a_biases.rows()));
	
	const int im_height_36 = output_height_35;
	const int im_width_36 = output_width_35;
	const int im_depth_36 = k_num_35;
	const int im_size_36 = im_height_36 * im_width_36;
	
	const int k_num_36 = 256;
	const int k_size_36 = 9;
	const int stride_36 = 1;
	const int k_depth_36 = im_depth_36;
	
	const int p1_36 = 1;
	const int p2_36 = 1;
	
	const int output_height_36 = (((im_height_36+(2*p1_36)) - sqrt(k_size_36))/stride_36) + 1;
	const int output_width_36 = (((im_width_36+(2*p2_36)) - sqrt(k_size_36))/stride_36) + 1;
	const int output_size_36 = output_height_36 * output_width_36;
	
	MatrixXd res4b3_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b3_branch2b_weights.csv");
	MatrixXd res4b3_branch2b_w = res4b3_branch2b_weights;
	
	MatrixXd res4b3_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b3_branch2b_biases.csv");
	VectorXd res4b3_branch2b_b(Map<VectorXd>(res4b3_branch2b_biases.data(), res4b3_branch2b_biases.cols()*res4b3_branch2b_biases.rows()));
	
	const int im_height_37 = output_height_36;
	const int im_width_37 = output_width_36;
	const int im_depth_37 = k_num_36;
	const int im_size_37 = im_height_37 * im_width_37;
	
	const int k_num_37 = 1024;
	const int k_size_37 = 1;
	const int stride_37 = 1;
	const int k_depth_37 = im_depth_37;
	
	const int p1_37 = 0;
	const int p2_37 = 0;
	
	const int output_height_37 = (((im_height_37+(2*p1_37)) - sqrt(k_size_37))/stride_37) + 1;
	const int output_width_37 = (((im_width_37+(2*p2_37)) - sqrt(k_size_37))/stride_37) + 1;
	const int output_size_37 = output_height_37 * output_width_37;
	
	MatrixXd res4b3_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b3_branch2c_weights.csv");
	MatrixXd res4b3_branch2c_w = res4b3_branch2c_weights;
	
	MatrixXd res4b3_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b3_branch2c_biases.csv");
	VectorXd res4b3_branch2c_b(Map<VectorXd>(res4b3_branch2c_biases.data(), res4b3_branch2c_biases.cols()*res4b3_branch2c_biases.rows()));
	
	const int im_height_38 = output_height_37;
	const int im_width_38 = output_width_37;
	const int im_depth_38 = k_num_37;
	const int im_size_38 = im_height_38 * im_width_38;
	
	const int k_num_38 = 256;
	const int k_size_38 = 1;
	const int stride_38 = 1;
	const int k_depth_38 = im_depth_38;
	
	const int p1_38 = 0;
	const int p2_38 = 0;
	
	const int output_height_38 = (((im_height_38+(2*p1_38)) - sqrt(k_size_38))/stride_38) + 1;
	const int output_width_38 = (((im_width_38+(2*p2_38)) - sqrt(k_size_38))/stride_38) + 1;
	const int output_size_38 = output_height_38 * output_width_38;
	
	MatrixXd res4b4_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b4_branch2a_weights.csv");
	MatrixXd res4b4_branch2a_w = res4b4_branch2a_weights;
	
	MatrixXd res4b4_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b4_branch2a_biases.csv");
	VectorXd res4b4_branch2a_b(Map<VectorXd>(res4b4_branch2a_biases.data(), res4b4_branch2a_biases.cols()*res4b4_branch2a_biases.rows()));
	
	const int im_height_39 = output_height_38;
	const int im_width_39 = output_width_38;
	const int im_depth_39 = k_num_38;
	const int im_size_39 = im_height_39 * im_width_39;
	
	const int k_num_39 = 256;
	const int k_size_39 = 9;
	const int stride_39 = 1;
	const int k_depth_39 = im_depth_39;
	
	const int p1_39 = 1;
	const int p2_39 = 1;
	
	const int output_height_39 = (((im_height_39+(2*p1_39)) - sqrt(k_size_39))/stride_39) + 1;
	const int output_width_39 = (((im_width_39+(2*p2_39)) - sqrt(k_size_39))/stride_39) + 1;
	const int output_size_39 = output_height_39 * output_width_39;
	
	MatrixXd res4b4_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b4_branch2b_weights.csv");
	MatrixXd res4b4_branch2b_w = res4b4_branch2b_weights;
	
	MatrixXd res4b4_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b4_branch2b_biases.csv");
	VectorXd res4b4_branch2b_b(Map<VectorXd>(res4b4_branch2b_biases.data(), res4b4_branch2b_biases.cols()*res4b4_branch2b_biases.rows()));
	
	const int im_height_40 = output_height_39;
	const int im_width_40 = output_width_39;
	const int im_depth_40 = k_num_39;
	const int im_size_40 = im_height_40 * im_width_40;
	
	const int k_num_40 = 1024;
	const int k_size_40 = 1;
	const int stride_40 = 1;
	const int k_depth_40 = im_depth_40;
	
	const int p1_40 = 0;
	const int p2_40 = 0;
	
	const int output_height_40 = (((im_height_40+(2*p1_40)) - sqrt(k_size_40))/stride_40) + 1;
	const int output_width_40 = (((im_width_40+(2*p2_40)) - sqrt(k_size_40))/stride_40) + 1;
	const int output_size_40 = output_height_40 * output_width_40;
	
	MatrixXd res4b4_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b4_branch2c_weights.csv");
	MatrixXd res4b4_branch2c_w = res4b4_branch2c_weights;
	
	MatrixXd res4b4_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b4_branch2c_biases.csv");
	VectorXd res4b4_branch2c_b(Map<VectorXd>(res4b4_branch2c_biases.data(), res4b4_branch2c_biases.cols()*res4b4_branch2c_biases.rows()));
	
	const int im_height_41 = output_height_40;
	const int im_width_41 = output_width_40;
	const int im_depth_41 = k_num_40;
	const int im_size_41 = im_height_41 * im_width_41;
	
	const int k_num_41 = 256;
	const int k_size_41 = 1;
	const int stride_41 = 1;
	const int k_depth_41 = im_depth_41;
	
	const int p1_41 = 0;
	const int p2_41 = 0;
	
	const int output_height_41 = (((im_height_41+(2*p1_41)) - sqrt(k_size_41))/stride_41) + 1;
	const int output_width_41 = (((im_width_41+(2*p2_41)) - sqrt(k_size_41))/stride_41) + 1;
	const int output_size_41 = output_height_41 * output_width_41;
	
	MatrixXd res4b5_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b5_branch2a_weights.csv");
	MatrixXd res4b5_branch2a_w = res4b5_branch2a_weights;
	
	MatrixXd res4b5_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b5_branch2a_biases.csv");
	VectorXd res4b5_branch2a_b(Map<VectorXd>(res4b5_branch2a_biases.data(), res4b5_branch2a_biases.cols()*res4b5_branch2a_biases.rows()));
	
	const int im_height_42 = output_height_41;
	const int im_width_42 = output_width_41;
	const int im_depth_42 = k_num_41;
	const int im_size_42 = im_height_42 * im_width_42;
	
	const int k_num_42 = 256;
	const int k_size_42 = 9;
	const int stride_42 = 1;
	const int k_depth_42 = im_depth_42;
	
	const int p1_42 = 1;
	const int p2_42 = 1;
	
	const int output_height_42 = (((im_height_42+(2*p1_42)) - sqrt(k_size_42))/stride_42) + 1;
	const int output_width_42 = (((im_width_42+(2*p2_42)) - sqrt(k_size_42))/stride_42) + 1;
	const int output_size_42 = output_height_42 * output_width_42;
	
	MatrixXd res4b5_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b5_branch2b_weights.csv");
	MatrixXd res4b5_branch2b_w = res4b5_branch2b_weights;
	
	MatrixXd res4b5_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b5_branch2b_biases.csv");
	VectorXd res4b5_branch2b_b(Map<VectorXd>(res4b5_branch2b_biases.data(), res4b5_branch2b_biases.cols()*res4b5_branch2b_biases.rows()));
	
	const int im_height_43 = output_height_42;
	const int im_width_43 = output_width_42;
	const int im_depth_43 = k_num_42;
	const int im_size_43 = im_height_43 * im_width_43;
	
	const int k_num_43 = 1024;
	const int k_size_43 = 1;
	const int stride_43 = 1;
	const int k_depth_43 = im_depth_43;
	
	const int p1_43 = 0;
	const int p2_43 = 0;
	
	const int output_height_43 = (((im_height_43+(2*p1_43)) - sqrt(k_size_43))/stride_43) + 1;
	const int output_width_43 = (((im_width_43+(2*p2_43)) - sqrt(k_size_43))/stride_43) + 1;
	const int output_size_43 = output_height_43 * output_width_43;
	
	MatrixXd res4b5_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b5_branch2c_weights.csv");
	MatrixXd res4b5_branch2c_w = res4b5_branch2c_weights;
	
	MatrixXd res4b5_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b5_branch2c_biases.csv");
	VectorXd res4b5_branch2c_b(Map<VectorXd>(res4b5_branch2c_biases.data(), res4b5_branch2c_biases.cols()*res4b5_branch2c_biases.rows()));
	
	const int im_height_44 = output_height_43;
	const int im_width_44 = output_width_43;
	const int im_depth_44 = k_num_43;
	const int im_size_44 = im_height_44 * im_width_44;
	
	const int k_num_44 = 256;
	const int k_size_44 = 1;
	const int stride_44 = 1;
	const int k_depth_44 = im_depth_44;
	
	const int p1_44 = 0;
	const int p2_44 = 0;
	
	const int output_height_44 = (((im_height_44+(2*p1_44)) - sqrt(k_size_44))/stride_44) + 1;
	const int output_width_44 = (((im_width_44+(2*p2_44)) - sqrt(k_size_44))/stride_44) + 1;
	const int output_size_44 = output_height_44 * output_width_44;
	
	MatrixXd res4b6_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b6_branch2a_weights.csv");
	MatrixXd res4b6_branch2a_w = res4b6_branch2a_weights;
	
	MatrixXd res4b6_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b6_branch2a_biases.csv");
	VectorXd res4b6_branch2a_b(Map<VectorXd>(res4b6_branch2a_biases.data(), res4b6_branch2a_biases.cols()*res4b6_branch2a_biases.rows()));
	
	const int im_height_45 = output_height_44;
	const int im_width_45 = output_width_44;
	const int im_depth_45 = k_num_44;
	const int im_size_45 = im_height_45 * im_width_45;
	
	const int k_num_45 = 256;
	const int k_size_45 = 9;
	const int stride_45 = 1;
	const int k_depth_45 = im_depth_45;
	
	const int p1_45 = 1;
	const int p2_45 = 1;
	
	const int output_height_45 = (((im_height_45+(2*p1_45)) - sqrt(k_size_45))/stride_45) + 1;
	const int output_width_45 = (((im_width_45+(2*p2_45)) - sqrt(k_size_45))/stride_45) + 1;
	const int output_size_45 = output_height_45 * output_width_45;
	
	MatrixXd res4b6_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b6_branch2b_weights.csv");
	MatrixXd res4b6_branch2b_w = res4b6_branch2b_weights;
	
	MatrixXd res4b6_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b6_branch2b_biases.csv");
	VectorXd res4b6_branch2b_b(Map<VectorXd>(res4b6_branch2b_biases.data(), res4b6_branch2b_biases.cols()*res4b6_branch2b_biases.rows()));
	
	const int im_height_46 = output_height_45;
	const int im_width_46 = output_width_45;
	const int im_depth_46 = k_num_45;
	const int im_size_46 = im_height_46 * im_width_46;
	
	const int k_num_46 = 1024;
	const int k_size_46 = 1;
	const int stride_46 = 1;
	const int k_depth_46 = im_depth_46;
	
	const int p1_46 = 0;
	const int p2_46 = 0;
	
	const int output_height_46 = (((im_height_46+(2*p1_46)) - sqrt(k_size_46))/stride_46) + 1;
	const int output_width_46 = (((im_width_46+(2*p2_46)) - sqrt(k_size_46))/stride_46) + 1;
	const int output_size_46 = output_height_46 * output_width_46;
	
	MatrixXd res4b6_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b6_branch2c_weights.csv");
	MatrixXd res4b6_branch2c_w = res4b6_branch2c_weights;
	
	MatrixXd res4b6_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b6_branch2c_biases.csv");
	VectorXd res4b6_branch2c_b(Map<VectorXd>(res4b6_branch2c_biases.data(), res4b6_branch2c_biases.cols()*res4b6_branch2c_biases.rows()));
	
	const int im_height_47 = output_height_46;
	const int im_width_47 = output_width_46;
	const int im_depth_47 = k_num_46;
	const int im_size_47 = im_height_47 * im_width_47;
	
	const int k_num_47 = 256;
	const int k_size_47 = 1;
	const int stride_47 = 1;
	const int k_depth_47 = im_depth_47;
	
	const int p1_47 = 0;
	const int p2_47 = 0;
	
	const int output_height_47 = (((im_height_47+(2*p1_47)) - sqrt(k_size_47))/stride_47) + 1;
	const int output_width_47 = (((im_width_47+(2*p2_47)) - sqrt(k_size_47))/stride_47) + 1;
	const int output_size_47 = output_height_47 * output_width_47;
	
	MatrixXd res4b7_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b7_branch2a_weights.csv");
	MatrixXd res4b7_branch2a_w = res4b7_branch2a_weights;
	
	MatrixXd res4b7_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b7_branch2a_biases.csv");
	VectorXd res4b7_branch2a_b(Map<VectorXd>(res4b7_branch2a_biases.data(), res4b7_branch2a_biases.cols()*res4b7_branch2a_biases.rows()));
	
	const int im_height_48 = output_height_47;
	const int im_width_48 = output_width_47;
	const int im_depth_48 = k_num_47;
	const int im_size_48 = im_height_48 * im_width_48;
	
	const int k_num_48 = 256;
	const int k_size_48 = 9;
	const int stride_48 = 1;
	const int k_depth_48 = im_depth_48;
	
	const int p1_48 = 1;
	const int p2_48 = 1;
	
	const int output_height_48 = (((im_height_48+(2*p1_48)) - sqrt(k_size_48))/stride_48) + 1;
	const int output_width_48 = (((im_width_48+(2*p2_48)) - sqrt(k_size_48))/stride_48) + 1;
	const int output_size_48 = output_height_48 * output_width_48;
	
	MatrixXd res4b7_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b7_branch2b_weights.csv");
	MatrixXd res4b7_branch2b_w = res4b7_branch2b_weights;
	
	MatrixXd res4b7_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b7_branch2b_biases.csv");
	VectorXd res4b7_branch2b_b(Map<VectorXd>(res4b7_branch2b_biases.data(), res4b7_branch2b_biases.cols()*res4b7_branch2b_biases.rows()));
	
	const int im_height_49 = output_height_48;
	const int im_width_49 = output_width_48;
	const int im_depth_49 = k_num_48;
	const int im_size_49 = im_height_49 * im_width_49;
	
	const int k_num_49 = 1024;
	const int k_size_49 = 1;
	const int stride_49 = 1;
	const int k_depth_49 = im_depth_49;
	
	const int p1_49 = 0;
	const int p2_49 = 0;
	
	const int output_height_49 = (((im_height_49+(2*p1_49)) - sqrt(k_size_49))/stride_49) + 1;
	const int output_width_49 = (((im_width_49+(2*p2_49)) - sqrt(k_size_49))/stride_49) + 1;
	const int output_size_49 = output_height_49 * output_width_49;
	
	MatrixXd res4b7_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b7_branch2c_weights.csv");
	MatrixXd res4b7_branch2c_w = res4b7_branch2c_weights;
	
	MatrixXd res4b7_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b7_branch2c_biases.csv");
	VectorXd res4b7_branch2c_b(Map<VectorXd>(res4b7_branch2c_biases.data(), res4b7_branch2c_biases.cols()*res4b7_branch2c_biases.rows()));
	
	const int im_height_50 = output_height_49;
	const int im_width_50 = output_width_49;
	const int im_depth_50 = k_num_49;
	const int im_size_50 = im_height_50 * im_width_50;
	
	const int k_num_50 = 256;
	const int k_size_50 = 1;
	const int stride_50 = 1;
	const int k_depth_50 = im_depth_50;
	
	const int p1_50 = 0;
	const int p2_50 = 0;
	
	const int output_height_50 = (((im_height_50+(2*p1_50)) - sqrt(k_size_50))/stride_50) + 1;
	const int output_width_50 = (((im_width_50+(2*p2_50)) - sqrt(k_size_50))/stride_50) + 1;
	const int output_size_50 = output_height_50 * output_width_50;
	
	MatrixXd res4b8_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b8_branch2a_weights.csv");
	MatrixXd res4b8_branch2a_w = res4b8_branch2a_weights;
	
	MatrixXd res4b8_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b8_branch2a_biases.csv");
	VectorXd res4b8_branch2a_b(Map<VectorXd>(res4b8_branch2a_biases.data(), res4b8_branch2a_biases.cols()*res4b8_branch2a_biases.rows()));
	
	const int im_height_51 = output_height_50;
	const int im_width_51 = output_width_50;
	const int im_depth_51 = k_num_50;
	const int im_size_51 = im_height_51 * im_width_51;
	
	const int k_num_51 = 256;
	const int k_size_51 = 9;
	const int stride_51 = 1;
	const int k_depth_51 = im_depth_51;
	
	const int p1_51 = 1;
	const int p2_51 = 1;
	
	const int output_height_51 = (((im_height_51+(2*p1_51)) - sqrt(k_size_51))/stride_51) + 1;
	const int output_width_51 = (((im_width_51+(2*p2_51)) - sqrt(k_size_51))/stride_51) + 1;
	const int output_size_51 = output_height_51 * output_width_51;
	
	MatrixXd res4b8_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b8_branch2b_weights.csv");
	MatrixXd res4b8_branch2b_w = res4b8_branch2b_weights;
	
	MatrixXd res4b8_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b8_branch2b_biases.csv");
	VectorXd res4b8_branch2b_b(Map<VectorXd>(res4b8_branch2b_biases.data(), res4b8_branch2b_biases.cols()*res4b8_branch2b_biases.rows()));
	
	const int im_height_52 = output_height_51;
	const int im_width_52 = output_width_51;
	const int im_depth_52 = k_num_51;
	const int im_size_52 = im_height_52 * im_width_52;
	
	const int k_num_52 = 1024;
	const int k_size_52 = 1;
	const int stride_52 = 1;
	const int k_depth_52 = im_depth_52;
	
	const int p1_52 = 0;
	const int p2_52 = 0;
	
	const int output_height_52 = (((im_height_52+(2*p1_52)) - sqrt(k_size_52))/stride_52) + 1;
	const int output_width_52 = (((im_width_52+(2*p2_52)) - sqrt(k_size_52))/stride_52) + 1;
	const int output_size_52 = output_height_52 * output_width_52;
	
	MatrixXd res4b8_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b8_branch2c_weights.csv");
	MatrixXd res4b8_branch2c_w = res4b8_branch2c_weights;
	
	MatrixXd res4b8_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b8_branch2c_biases.csv");
	VectorXd res4b8_branch2c_b(Map<VectorXd>(res4b8_branch2c_biases.data(), res4b8_branch2c_biases.cols()*res4b8_branch2c_biases.rows()));
	
	const int im_height_53 = output_height_52;
	const int im_width_53 = output_width_52;
	const int im_depth_53 = k_num_52;
	const int im_size_53 = im_height_53 * im_width_53;
	
	const int k_num_53 = 256;
	const int k_size_53 = 1;
	const int stride_53 = 1;
	const int k_depth_53 = im_depth_53;
	
	const int p1_53 = 0;
	const int p2_53 = 0;
	
	const int output_height_53 = (((im_height_53+(2*p1_53)) - sqrt(k_size_53))/stride_53) + 1;
	const int output_width_53 = (((im_width_53+(2*p2_53)) - sqrt(k_size_53))/stride_53) + 1;
	const int output_size_53 = output_height_53 * output_width_53;
	
	MatrixXd res4b9_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b9_branch2a_weights.csv");
	MatrixXd res4b9_branch2a_w = res4b9_branch2a_weights;
	
	MatrixXd res4b9_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b9_branch2a_biases.csv");
	VectorXd res4b9_branch2a_b(Map<VectorXd>(res4b9_branch2a_biases.data(), res4b9_branch2a_biases.cols()*res4b9_branch2a_biases.rows()));
	
	const int im_height_54 = output_height_53;
	const int im_width_54 = output_width_53;
	const int im_depth_54 = k_num_53;
	const int im_size_54 = im_height_54 * im_width_54;
	
	const int k_num_54 = 256;
	const int k_size_54 = 9;
	const int stride_54 = 1;
	const int k_depth_54 = im_depth_54;
	
	const int p1_54 = 1;
	const int p2_54 = 1;
	
	const int output_height_54 = (((im_height_54+(2*p1_54)) - sqrt(k_size_54))/stride_54) + 1;
	const int output_width_54 = (((im_width_54+(2*p2_54)) - sqrt(k_size_54))/stride_54) + 1;
	const int output_size_54 = output_height_54 * output_width_54;
	
	MatrixXd res4b9_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b9_branch2b_weights.csv");
	MatrixXd res4b9_branch2b_w = res4b9_branch2b_weights;
	
	MatrixXd res4b9_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b9_branch2b_biases.csv");
	VectorXd res4b9_branch2b_b(Map<VectorXd>(res4b9_branch2b_biases.data(), res4b9_branch2b_biases.cols()*res4b9_branch2b_biases.rows()));
	
	const int im_height_55 = output_height_54;
	const int im_width_55 = output_width_54;
	const int im_depth_55 = k_num_54;
	const int im_size_55 = im_height_55 * im_width_55;
	
	const int k_num_55 = 1024;
	const int k_size_55 = 1;
	const int stride_55 = 1;
	const int k_depth_55 = im_depth_55;
	
	const int p1_55 = 0;
	const int p2_55 = 0;
	
	const int output_height_55 = (((im_height_55+(2*p1_55)) - sqrt(k_size_55))/stride_55) + 1;
	const int output_width_55 = (((im_width_55+(2*p2_55)) - sqrt(k_size_55))/stride_55) + 1;
	const int output_size_55 = output_height_55 * output_width_55;
	
	MatrixXd res4b9_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b9_branch2c_weights.csv");
	MatrixXd res4b9_branch2c_w = res4b9_branch2c_weights;
	
	MatrixXd res4b9_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b9_branch2c_biases.csv");
	VectorXd res4b9_branch2c_b(Map<VectorXd>(res4b9_branch2c_biases.data(), res4b9_branch2c_biases.cols()*res4b9_branch2c_biases.rows()));
	
	const int im_height_56 = output_height_55;
	const int im_width_56 = output_width_55;
	const int im_depth_56 = k_num_55;
	const int im_size_56 = im_height_56 * im_width_56;
	
	const int k_num_56 = 256;
	const int k_size_56 = 1;
	const int stride_56 = 1;
	const int k_depth_56 = im_depth_56;
	
	const int p1_56 = 0;
	const int p2_56 = 0;
	
	const int output_height_56 = (((im_height_56+(2*p1_56)) - sqrt(k_size_56))/stride_56) + 1;
	const int output_width_56 = (((im_width_56+(2*p2_56)) - sqrt(k_size_56))/stride_56) + 1;
	const int output_size_56 = output_height_56 * output_width_56;
	
	MatrixXd res4b10_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b10_branch2a_weights.csv");
	MatrixXd res4b10_branch2a_w = res4b10_branch2a_weights;
	
	MatrixXd res4b10_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b10_branch2a_biases.csv");
	VectorXd res4b10_branch2a_b(Map<VectorXd>(res4b10_branch2a_biases.data(), res4b10_branch2a_biases.cols()*res4b10_branch2a_biases.rows()));
	
	const int im_height_57 = output_height_56;
	const int im_width_57 = output_width_56;
	const int im_depth_57 = k_num_56;
	const int im_size_57 = im_height_57 * im_width_57;
	
	const int k_num_57 = 256;
	const int k_size_57 = 9;
	const int stride_57 = 1;
	const int k_depth_57 = im_depth_57;
	
	const int p1_57 = 1;
	const int p2_57 = 1;
	
	const int output_height_57 = (((im_height_57+(2*p1_57)) - sqrt(k_size_57))/stride_57) + 1;
	const int output_width_57 = (((im_width_57+(2*p2_57)) - sqrt(k_size_57))/stride_57) + 1;
	const int output_size_57 = output_height_57 * output_width_57;
	
	MatrixXd res4b10_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b10_branch2b_weights.csv");
	MatrixXd res4b10_branch2b_w = res4b10_branch2b_weights;
	
	MatrixXd res4b10_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b10_branch2b_biases.csv");
	VectorXd res4b10_branch2b_b(Map<VectorXd>(res4b10_branch2b_biases.data(), res4b10_branch2b_biases.cols()*res4b10_branch2b_biases.rows()));
	
	const int im_height_58 = output_height_57;
	const int im_width_58 = output_width_57;
	const int im_depth_58 = k_num_57;
	const int im_size_58 = im_height_58 * im_width_58;
	
	const int k_num_58 = 1024;
	const int k_size_58 = 1;
	const int stride_58 = 1;
	const int k_depth_58 = im_depth_58;
	
	const int p1_58 = 0;
	const int p2_58 = 0;
	
	const int output_height_58 = (((im_height_58+(2*p1_58)) - sqrt(k_size_58))/stride_58) + 1;
	const int output_width_58 = (((im_width_58+(2*p2_58)) - sqrt(k_size_58))/stride_58) + 1;
	const int output_size_58 = output_height_58 * output_width_58;
	
	MatrixXd res4b10_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b10_branch2c_weights.csv");
	MatrixXd res4b10_branch2c_w = res4b10_branch2c_weights;
	
	MatrixXd res4b10_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b10_branch2c_biases.csv");
	VectorXd res4b10_branch2c_b(Map<VectorXd>(res4b10_branch2c_biases.data(), res4b10_branch2c_biases.cols()*res4b10_branch2c_biases.rows()));
	
	const int im_height_59 = output_height_58;
	const int im_width_59 = output_width_58;
	const int im_depth_59 = k_num_58;
	const int im_size_59 = im_height_59 * im_width_59;
	
	const int k_num_59 = 256;
	const int k_size_59 = 1;
	const int stride_59 = 1;
	const int k_depth_59 = im_depth_59;
	
	const int p1_59 = 0;
	const int p2_59 = 0;
	
	const int output_height_59 = (((im_height_59+(2*p1_59)) - sqrt(k_size_59))/stride_59) + 1;
	const int output_width_59 = (((im_width_59+(2*p2_59)) - sqrt(k_size_59))/stride_59) + 1;
	const int output_size_59 = output_height_59 * output_width_59;
	
	MatrixXd res4b11_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b11_branch2a_weights.csv");
	MatrixXd res4b11_branch2a_w = res4b11_branch2a_weights;
	
	MatrixXd res4b11_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b11_branch2a_biases.csv");
	VectorXd res4b11_branch2a_b(Map<VectorXd>(res4b11_branch2a_biases.data(), res4b11_branch2a_biases.cols()*res4b11_branch2a_biases.rows()));
	
	const int im_height_60 = output_height_59;
	const int im_width_60 = output_width_59;
	const int im_depth_60 = k_num_59;
	const int im_size_60 = im_height_60 * im_width_60;
	
	const int k_num_60 = 256;
	const int k_size_60 = 9;
	const int stride_60 = 1;
	const int k_depth_60 = im_depth_60;
	
	const int p1_60 = 1;
	const int p2_60 = 1;
	
	const int output_height_60 = (((im_height_60+(2*p1_60)) - sqrt(k_size_60))/stride_60) + 1;
	const int output_width_60 = (((im_width_60+(2*p2_60)) - sqrt(k_size_60))/stride_60) + 1;
	const int output_size_60 = output_height_60 * output_width_60;
	
	MatrixXd res4b11_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b11_branch2b_weights.csv");
	MatrixXd res4b11_branch2b_w = res4b11_branch2b_weights;
	
	MatrixXd res4b11_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b11_branch2b_biases.csv");
	VectorXd res4b11_branch2b_b(Map<VectorXd>(res4b11_branch2b_biases.data(), res4b11_branch2b_biases.cols()*res4b11_branch2b_biases.rows()));
	
	const int im_height_61 = output_height_60;
	const int im_width_61 = output_width_60;
	const int im_depth_61 = k_num_60;
	const int im_size_61 = im_height_61 * im_width_61;
	
	const int k_num_61 = 1024;
	const int k_size_61 = 1;
	const int stride_61 = 1;
	const int k_depth_61 = im_depth_61;
	
	const int p1_61 = 0;
	const int p2_61 = 0;
	
	const int output_height_61 = (((im_height_61+(2*p1_61)) - sqrt(k_size_61))/stride_61) + 1;
	const int output_width_61 = (((im_width_61+(2*p2_61)) - sqrt(k_size_61))/stride_61) + 1;
	const int output_size_61 = output_height_61 * output_width_61;
	
	MatrixXd res4b11_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b11_branch2c_weights.csv");
	MatrixXd res4b11_branch2c_w = res4b11_branch2c_weights;
	
	MatrixXd res4b11_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b11_branch2c_biases.csv");
	VectorXd res4b11_branch2c_b(Map<VectorXd>(res4b11_branch2c_biases.data(), res4b11_branch2c_biases.cols()*res4b11_branch2c_biases.rows()));
	
	const int im_height_62 = output_height_61;
	const int im_width_62 = output_width_61;
	const int im_depth_62 = k_num_61;
	const int im_size_62 = im_height_62 * im_width_62;
	
	const int k_num_62 = 256;
	const int k_size_62 = 1;
	const int stride_62 = 1;
	const int k_depth_62 = im_depth_62;
	
	const int p1_62 = 0;
	const int p2_62 = 0;
	
	const int output_height_62 = (((im_height_62+(2*p1_62)) - sqrt(k_size_62))/stride_62) + 1;
	const int output_width_62 = (((im_width_62+(2*p2_62)) - sqrt(k_size_62))/stride_62) + 1;
	const int output_size_62 = output_height_62 * output_width_62;
	
	MatrixXd res4b12_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b12_branch2a_weights.csv");
	MatrixXd res4b12_branch2a_w = res4b12_branch2a_weights;
	
	MatrixXd res4b12_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b12_branch2a_biases.csv");
	VectorXd res4b12_branch2a_b(Map<VectorXd>(res4b12_branch2a_biases.data(), res4b12_branch2a_biases.cols()*res4b12_branch2a_biases.rows()));
	
	const int im_height_63 = output_height_62;
	const int im_width_63 = output_width_62;
	const int im_depth_63 = k_num_62;
	const int im_size_63 = im_height_63 * im_width_63;
	
	const int k_num_63 = 256;
	const int k_size_63 = 9;
	const int stride_63 = 1;
	const int k_depth_63 = im_depth_63;
	
	const int p1_63 = 1;
	const int p2_63 = 1;
	
	const int output_height_63 = (((im_height_63+(2*p1_63)) - sqrt(k_size_63))/stride_63) + 1;
	const int output_width_63 = (((im_width_63+(2*p2_63)) - sqrt(k_size_63))/stride_63) + 1;
	const int output_size_63 = output_height_63 * output_width_63;
	
	MatrixXd res4b12_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b12_branch2b_weights.csv");
	MatrixXd res4b12_branch2b_w = res4b12_branch2b_weights;
	
	MatrixXd res4b12_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b12_branch2b_biases.csv");
	VectorXd res4b12_branch2b_b(Map<VectorXd>(res4b12_branch2b_biases.data(), res4b12_branch2b_biases.cols()*res4b12_branch2b_biases.rows()));
	
	const int im_height_64 = output_height_63;
	const int im_width_64 = output_width_63;
	const int im_depth_64 = k_num_63;
	const int im_size_64 = im_height_64 * im_width_64;
	
	const int k_num_64 = 1024;
	const int k_size_64 = 1;
	const int stride_64 = 1;
	const int k_depth_64 = im_depth_64;
	
	const int p1_64 = 0;
	const int p2_64 = 0;
	
	const int output_height_64 = (((im_height_64+(2*p1_64)) - sqrt(k_size_64))/stride_64) + 1;
	const int output_width_64 = (((im_width_64+(2*p2_64)) - sqrt(k_size_64))/stride_64) + 1;
	const int output_size_64 = output_height_64 * output_width_64;
	
	MatrixXd res4b12_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b12_branch2c_weights.csv");
	MatrixXd res4b12_branch2c_w = res4b12_branch2c_weights;
	
	MatrixXd res4b12_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b12_branch2c_biases.csv");
	VectorXd res4b12_branch2c_b(Map<VectorXd>(res4b12_branch2c_biases.data(), res4b12_branch2c_biases.cols()*res4b12_branch2c_biases.rows()));
	
	const int im_height_65 = output_height_64;
	const int im_width_65 = output_width_64;
	const int im_depth_65 = k_num_64;
	const int im_size_65 = im_height_65 * im_width_65;
	
	const int k_num_65 = 256;
	const int k_size_65 = 1;
	const int stride_65 = 1;
	const int k_depth_65 = im_depth_65;
	
	const int p1_65 = 0;
	const int p2_65 = 0;
	
	const int output_height_65 = (((im_height_65+(2*p1_65)) - sqrt(k_size_65))/stride_65) + 1;
	const int output_width_65 = (((im_width_65+(2*p2_65)) - sqrt(k_size_65))/stride_65) + 1;
	const int output_size_65 = output_height_65 * output_width_65;
	
	MatrixXd res4b13_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b13_branch2a_weights.csv");
	MatrixXd res4b13_branch2a_w = res4b13_branch2a_weights;
	
	MatrixXd res4b13_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b13_branch2a_biases.csv");
	VectorXd res4b13_branch2a_b(Map<VectorXd>(res4b13_branch2a_biases.data(), res4b13_branch2a_biases.cols()*res4b13_branch2a_biases.rows()));
	
	const int im_height_66 = output_height_65;
	const int im_width_66 = output_width_65;
	const int im_depth_66 = k_num_65;
	const int im_size_66 = im_height_66 * im_width_66;
	
	const int k_num_66 = 256;
	const int k_size_66 = 9;
	const int stride_66 = 1;
	const int k_depth_66 = im_depth_66;
	
	const int p1_66 = 1;
	const int p2_66 = 1;
	
	const int output_height_66 = (((im_height_66+(2*p1_66)) - sqrt(k_size_66))/stride_66) + 1;
	const int output_width_66 = (((im_width_66+(2*p2_66)) - sqrt(k_size_66))/stride_66) + 1;
	const int output_size_66 = output_height_66 * output_width_66;
	
	MatrixXd res4b13_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b13_branch2b_weights.csv");
	MatrixXd res4b13_branch2b_w = res4b13_branch2b_weights;
	
	MatrixXd res4b13_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b13_branch2b_biases.csv");
	VectorXd res4b13_branch2b_b(Map<VectorXd>(res4b13_branch2b_biases.data(), res4b13_branch2b_biases.cols()*res4b13_branch2b_biases.rows()));
	
	const int im_height_67 = output_height_66;
	const int im_width_67 = output_width_66;
	const int im_depth_67 = k_num_66;
	const int im_size_67 = im_height_67 * im_width_67;
	
	const int k_num_67 = 1024;
	const int k_size_67 = 1;
	const int stride_67 = 1;
	const int k_depth_67 = im_depth_67;
	
	const int p1_67 = 0;
	const int p2_67 = 0;
	
	const int output_height_67 = (((im_height_67+(2*p1_67)) - sqrt(k_size_67))/stride_67) + 1;
	const int output_width_67 = (((im_width_67+(2*p2_67)) - sqrt(k_size_67))/stride_67) + 1;
	const int output_size_67 = output_height_67 * output_width_67;
	
	MatrixXd res4b13_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b13_branch2c_weights.csv");
	MatrixXd res4b13_branch2c_w = res4b13_branch2c_weights;
	
	MatrixXd res4b13_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b13_branch2c_biases.csv");
	VectorXd res4b13_branch2c_b(Map<VectorXd>(res4b13_branch2c_biases.data(), res4b13_branch2c_biases.cols()*res4b13_branch2c_biases.rows()));
	
	const int im_height_68 = output_height_67;
	const int im_width_68 = output_width_67;
	const int im_depth_68 = k_num_67;
	const int im_size_68 = im_height_68 * im_width_68;
	
	const int k_num_68 = 256;
	const int k_size_68 = 1;
	const int stride_68 = 1;
	const int k_depth_68 = im_depth_68;
	
	const int p1_68 = 0;
	const int p2_68 = 0;
	
	const int output_height_68 = (((im_height_68+(2*p1_68)) - sqrt(k_size_68))/stride_68) + 1;
	const int output_width_68 = (((im_width_68+(2*p2_68)) - sqrt(k_size_68))/stride_68) + 1;
	const int output_size_68 = output_height_68 * output_width_68;
	
	MatrixXd res4b14_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b14_branch2a_weights.csv");
	MatrixXd res4b14_branch2a_w = res4b14_branch2a_weights;
	
	MatrixXd res4b14_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b14_branch2a_biases.csv");
	VectorXd res4b14_branch2a_b(Map<VectorXd>(res4b14_branch2a_biases.data(), res4b14_branch2a_biases.cols()*res4b14_branch2a_biases.rows()));
	
	const int im_height_69 = output_height_68;
	const int im_width_69 = output_width_68;
	const int im_depth_69 = k_num_68;
	const int im_size_69 = im_height_69 * im_width_69;
	
	const int k_num_69 = 256;
	const int k_size_69 = 9;
	const int stride_69 = 1;
	const int k_depth_69 = im_depth_69;
	
	const int p1_69 = 1;
	const int p2_69 = 1;
	
	const int output_height_69 = (((im_height_69+(2*p1_69)) - sqrt(k_size_69))/stride_69) + 1;
	const int output_width_69 = (((im_width_69+(2*p2_69)) - sqrt(k_size_69))/stride_69) + 1;
	const int output_size_69 = output_height_69 * output_width_69;
	
	MatrixXd res4b14_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b14_branch2b_weights.csv");
	MatrixXd res4b14_branch2b_w = res4b14_branch2b_weights;
	
	MatrixXd res4b14_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b14_branch2b_biases.csv");
	VectorXd res4b14_branch2b_b(Map<VectorXd>(res4b14_branch2b_biases.data(), res4b14_branch2b_biases.cols()*res4b14_branch2b_biases.rows()));
	
	const int im_height_70 = output_height_69;
	const int im_width_70 = output_width_69;
	const int im_depth_70 = k_num_69;
	const int im_size_70 = im_height_70 * im_width_70;
	
	const int k_num_70 = 1024;
	const int k_size_70 = 1;
	const int stride_70 = 1;
	const int k_depth_70 = im_depth_70;
	
	const int p1_70 = 0;
	const int p2_70 = 0;
	
	const int output_height_70 = (((im_height_70+(2*p1_70)) - sqrt(k_size_70))/stride_70) + 1;
	const int output_width_70 = (((im_width_70+(2*p2_70)) - sqrt(k_size_70))/stride_70) + 1;
	const int output_size_70 = output_height_70 * output_width_70;
	
	MatrixXd res4b14_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b14_branch2c_weights.csv");
	MatrixXd res4b14_branch2c_w = res4b14_branch2c_weights;
	
	MatrixXd res4b14_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b14_branch2c_biases.csv");
	VectorXd res4b14_branch2c_b(Map<VectorXd>(res4b14_branch2c_biases.data(), res4b14_branch2c_biases.cols()*res4b14_branch2c_biases.rows()));
	
	const int im_height_71 = output_height_70;
	const int im_width_71 = output_width_70;
	const int im_depth_71 = k_num_70;
	const int im_size_71 = im_height_71 * im_width_71;
	
	const int k_num_71 = 256;
	const int k_size_71 = 1;
	const int stride_71 = 1;
	const int k_depth_71 = im_depth_71;
	
	const int p1_71 = 0;
	const int p2_71 = 0;
	
	const int output_height_71 = (((im_height_71+(2*p1_71)) - sqrt(k_size_71))/stride_71) + 1;
	const int output_width_71 = (((im_width_71+(2*p2_71)) - sqrt(k_size_71))/stride_71) + 1;
	const int output_size_71 = output_height_71 * output_width_71;
	
	MatrixXd res4b15_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b15_branch2a_weights.csv");
	MatrixXd res4b15_branch2a_w = res4b15_branch2a_weights;
	
	MatrixXd res4b15_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b15_branch2a_biases.csv");
	VectorXd res4b15_branch2a_b(Map<VectorXd>(res4b15_branch2a_biases.data(), res4b15_branch2a_biases.cols()*res4b15_branch2a_biases.rows()));
	
	const int im_height_72 = output_height_71;
	const int im_width_72 = output_width_71;
	const int im_depth_72 = k_num_71;
	const int im_size_72 = im_height_72 * im_width_72;
	
	const int k_num_72 = 256;
	const int k_size_72 = 9;
	const int stride_72 = 1;
	const int k_depth_72 = im_depth_72;
	
	const int p1_72 = 1;
	const int p2_72 = 1;
	
	const int output_height_72 = (((im_height_72+(2*p1_72)) - sqrt(k_size_72))/stride_72) + 1;
	const int output_width_72 = (((im_width_72+(2*p2_72)) - sqrt(k_size_72))/stride_72) + 1;
	const int output_size_72 = output_height_72 * output_width_72;
	
	MatrixXd res4b15_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b15_branch2b_weights.csv");
	MatrixXd res4b15_branch2b_w = res4b15_branch2b_weights;
	
	MatrixXd res4b15_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b15_branch2b_biases.csv");
	VectorXd res4b15_branch2b_b(Map<VectorXd>(res4b15_branch2b_biases.data(), res4b15_branch2b_biases.cols()*res4b15_branch2b_biases.rows()));
	
	const int im_height_73 = output_height_72;
	const int im_width_73 = output_width_72;
	const int im_depth_73 = k_num_72;
	const int im_size_73 = im_height_73 * im_width_73;
	
	const int k_num_73 = 1024;
	const int k_size_73 = 1;
	const int stride_73 = 1;
	const int k_depth_73 = im_depth_73;
	
	const int p1_73 = 0;
	const int p2_73 = 0;
	
	const int output_height_73 = (((im_height_73+(2*p1_73)) - sqrt(k_size_73))/stride_73) + 1;
	const int output_width_73 = (((im_width_73+(2*p2_73)) - sqrt(k_size_73))/stride_73) + 1;
	const int output_size_73 = output_height_73 * output_width_73;
	
	MatrixXd res4b15_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b15_branch2c_weights.csv");
	MatrixXd res4b15_branch2c_w = res4b15_branch2c_weights;
	
	MatrixXd res4b15_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b15_branch2c_biases.csv");
	VectorXd res4b15_branch2c_b(Map<VectorXd>(res4b15_branch2c_biases.data(), res4b15_branch2c_biases.cols()*res4b15_branch2c_biases.rows()));
	
	const int im_height_74 = output_height_73;
	const int im_width_74 = output_width_73;
	const int im_depth_74 = k_num_73;
	const int im_size_74 = im_height_74 * im_width_74;
	
	const int k_num_74 = 256;
	const int k_size_74 = 1;
	const int stride_74 = 1;
	const int k_depth_74 = im_depth_74;
	
	const int p1_74 = 0;
	const int p2_74 = 0;
	
	const int output_height_74 = (((im_height_74+(2*p1_74)) - sqrt(k_size_74))/stride_74) + 1;
	const int output_width_74 = (((im_width_74+(2*p2_74)) - sqrt(k_size_74))/stride_74) + 1;
	const int output_size_74 = output_height_74 * output_width_74;
	
	MatrixXd res4b16_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b16_branch2a_weights.csv");
	MatrixXd res4b16_branch2a_w = res4b16_branch2a_weights;
	
	MatrixXd res4b16_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b16_branch2a_biases.csv");
	VectorXd res4b16_branch2a_b(Map<VectorXd>(res4b16_branch2a_biases.data(), res4b16_branch2a_biases.cols()*res4b16_branch2a_biases.rows()));
	
	const int im_height_75 = output_height_74;
	const int im_width_75 = output_width_74;
	const int im_depth_75 = k_num_74;
	const int im_size_75 = im_height_75 * im_width_75;
	
	const int k_num_75 = 256;
	const int k_size_75 = 9;
	const int stride_75 = 1;
	const int k_depth_75 = im_depth_75;
	
	const int p1_75 = 1;
	const int p2_75 = 1;
	
	const int output_height_75 = (((im_height_75+(2*p1_75)) - sqrt(k_size_75))/stride_75) + 1;
	const int output_width_75 = (((im_width_75+(2*p2_75)) - sqrt(k_size_75))/stride_75) + 1;
	const int output_size_75 = output_height_75 * output_width_75;
	
	MatrixXd res4b16_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b16_branch2b_weights.csv");
	MatrixXd res4b16_branch2b_w = res4b16_branch2b_weights;
	
	MatrixXd res4b16_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b16_branch2b_biases.csv");
	VectorXd res4b16_branch2b_b(Map<VectorXd>(res4b16_branch2b_biases.data(), res4b16_branch2b_biases.cols()*res4b16_branch2b_biases.rows()));
	
	const int im_height_76 = output_height_75;
	const int im_width_76 = output_width_75;
	const int im_depth_76 = k_num_75;
	const int im_size_76 = im_height_76 * im_width_76;
	
	const int k_num_76 = 1024;
	const int k_size_76 = 1;
	const int stride_76 = 1;
	const int k_depth_76 = im_depth_76;
	
	const int p1_76 = 0;
	const int p2_76 = 0;
	
	const int output_height_76 = (((im_height_76+(2*p1_76)) - sqrt(k_size_76))/stride_76) + 1;
	const int output_width_76 = (((im_width_76+(2*p2_76)) - sqrt(k_size_76))/stride_76) + 1;
	const int output_size_76 = output_height_76 * output_width_76;
	
	MatrixXd res4b16_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b16_branch2c_weights.csv");
	MatrixXd res4b16_branch2c_w = res4b16_branch2c_weights;
	
	MatrixXd res4b16_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b16_branch2c_biases.csv");
	VectorXd res4b16_branch2c_b(Map<VectorXd>(res4b16_branch2c_biases.data(), res4b16_branch2c_biases.cols()*res4b16_branch2c_biases.rows()));
	
	const int im_height_77 = output_height_76;
	const int im_width_77 = output_width_76;
	const int im_depth_77 = k_num_76;
	const int im_size_77 = im_height_77 * im_width_77;
	
	const int k_num_77 = 256;
	const int k_size_77 = 1;
	const int stride_77 = 1;
	const int k_depth_77 = im_depth_77;
	
	const int p1_77 = 0;
	const int p2_77 = 0;
	
	const int output_height_77 = (((im_height_77+(2*p1_77)) - sqrt(k_size_77))/stride_77) + 1;
	const int output_width_77 = (((im_width_77+(2*p2_77)) - sqrt(k_size_77))/stride_77) + 1;
	const int output_size_77 = output_height_77 * output_width_77;
	
	MatrixXd res4b17_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b17_branch2a_weights.csv");
	MatrixXd res4b17_branch2a_w = res4b17_branch2a_weights;
	
	MatrixXd res4b17_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b17_branch2a_biases.csv");
	VectorXd res4b17_branch2a_b(Map<VectorXd>(res4b17_branch2a_biases.data(), res4b17_branch2a_biases.cols()*res4b17_branch2a_biases.rows()));
	
	const int im_height_78 = output_height_77;
	const int im_width_78 = output_width_77;
	const int im_depth_78 = k_num_77;
	const int im_size_78 = im_height_78 * im_width_78;
	
	const int k_num_78 = 256;
	const int k_size_78 = 9;
	const int stride_78 = 1;
	const int k_depth_78 = im_depth_78;
	
	const int p1_78 = 1;
	const int p2_78 = 1;
	
	const int output_height_78 = (((im_height_78+(2*p1_78)) - sqrt(k_size_78))/stride_78) + 1;
	const int output_width_78 = (((im_width_78+(2*p2_78)) - sqrt(k_size_78))/stride_78) + 1;
	const int output_size_78 = output_height_78 * output_width_78;
	
	MatrixXd res4b17_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b17_branch2b_weights.csv");
	MatrixXd res4b17_branch2b_w = res4b17_branch2b_weights;
	
	MatrixXd res4b17_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b17_branch2b_biases.csv");
	VectorXd res4b17_branch2b_b(Map<VectorXd>(res4b17_branch2b_biases.data(), res4b17_branch2b_biases.cols()*res4b17_branch2b_biases.rows()));
	
	const int im_height_79 = output_height_78;
	const int im_width_79 = output_width_78;
	const int im_depth_79 = k_num_78;
	const int im_size_79 = im_height_79 * im_width_79;
	
	const int k_num_79 = 1024;
	const int k_size_79 = 1;
	const int stride_79 = 1;
	const int k_depth_79 = im_depth_79;
	
	const int p1_79 = 0;
	const int p2_79 = 0;
	
	const int output_height_79 = (((im_height_79+(2*p1_79)) - sqrt(k_size_79))/stride_79) + 1;
	const int output_width_79 = (((im_width_79+(2*p2_79)) - sqrt(k_size_79))/stride_79) + 1;
	const int output_size_79 = output_height_79 * output_width_79;
	
	MatrixXd res4b17_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b17_branch2c_weights.csv");
	MatrixXd res4b17_branch2c_w = res4b17_branch2c_weights;
	
	MatrixXd res4b17_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b17_branch2c_biases.csv");
	VectorXd res4b17_branch2c_b(Map<VectorXd>(res4b17_branch2c_biases.data(), res4b17_branch2c_biases.cols()*res4b17_branch2c_biases.rows()));
	
	const int im_height_80 = output_height_79;
	const int im_width_80 = output_width_79;
	const int im_depth_80 = k_num_79;
	const int im_size_80 = im_height_80 * im_width_80;
	
	const int k_num_80 = 256;
	const int k_size_80 = 1;
	const int stride_80 = 1;
	const int k_depth_80 = im_depth_80;
	
	const int p1_80 = 0;
	const int p2_80 = 0;
	
	const int output_height_80 = (((im_height_80+(2*p1_80)) - sqrt(k_size_80))/stride_80) + 1;
	const int output_width_80 = (((im_width_80+(2*p2_80)) - sqrt(k_size_80))/stride_80) + 1;
	const int output_size_80 = output_height_80 * output_width_80;
	
	MatrixXd res4b18_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b18_branch2a_weights.csv");
	MatrixXd res4b18_branch2a_w = res4b18_branch2a_weights;
	
	MatrixXd res4b18_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b18_branch2a_biases.csv");
	VectorXd res4b18_branch2a_b(Map<VectorXd>(res4b18_branch2a_biases.data(), res4b18_branch2a_biases.cols()*res4b18_branch2a_biases.rows()));
	
	const int im_height_81 = output_height_80;
	const int im_width_81 = output_width_80;
	const int im_depth_81 = k_num_80;
	const int im_size_81 = im_height_81 * im_width_81;
	
	const int k_num_81 = 256;
	const int k_size_81 = 9;
	const int stride_81 = 1;
	const int k_depth_81 = im_depth_81;
	
	const int p1_81 = 1;
	const int p2_81 = 1;
	
	const int output_height_81 = (((im_height_81+(2*p1_81)) - sqrt(k_size_81))/stride_81) + 1;
	const int output_width_81 = (((im_width_81+(2*p2_81)) - sqrt(k_size_81))/stride_81) + 1;
	const int output_size_81 = output_height_81 * output_width_81;
	
	MatrixXd res4b18_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b18_branch2b_weights.csv");
	MatrixXd res4b18_branch2b_w = res4b18_branch2b_weights;
	
	MatrixXd res4b18_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b18_branch2b_biases.csv");
	VectorXd res4b18_branch2b_b(Map<VectorXd>(res4b18_branch2b_biases.data(), res4b18_branch2b_biases.cols()*res4b18_branch2b_biases.rows()));
	
	const int im_height_82 = output_height_81;
	const int im_width_82 = output_width_81;
	const int im_depth_82 = k_num_81;
	const int im_size_82 = im_height_82 * im_width_82;
	
	const int k_num_82 = 1024;
	const int k_size_82 = 1;
	const int stride_82 = 1;
	const int k_depth_82 = im_depth_82;
	
	const int p1_82 = 0;
	const int p2_82 = 0;
	
	const int output_height_82 = (((im_height_82+(2*p1_82)) - sqrt(k_size_82))/stride_82) + 1;
	const int output_width_82 = (((im_width_82+(2*p2_82)) - sqrt(k_size_82))/stride_82) + 1;
	const int output_size_82 = output_height_82 * output_width_82;
	
	MatrixXd res4b18_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b18_branch2c_weights.csv");
	MatrixXd res4b18_branch2c_w = res4b18_branch2c_weights;
	
	MatrixXd res4b18_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b18_branch2c_biases.csv");
	VectorXd res4b18_branch2c_b(Map<VectorXd>(res4b18_branch2c_biases.data(), res4b18_branch2c_biases.cols()*res4b18_branch2c_biases.rows()));
	
	const int im_height_83 = output_height_82;
	const int im_width_83 = output_width_82;
	const int im_depth_83 = k_num_82;
	const int im_size_83 = im_height_83 * im_width_83;
	
	const int k_num_83 = 256;
	const int k_size_83 = 1;
	const int stride_83 = 1;
	const int k_depth_83 = im_depth_83;
	
	const int p1_83 = 0;
	const int p2_83 = 0;
	
	const int output_height_83 = (((im_height_83+(2*p1_83)) - sqrt(k_size_83))/stride_83) + 1;
	const int output_width_83 = (((im_width_83+(2*p2_83)) - sqrt(k_size_83))/stride_83) + 1;
	const int output_size_83 = output_height_83 * output_width_83;
	
	MatrixXd res4b19_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b19_branch2a_weights.csv");
	MatrixXd res4b19_branch2a_w = res4b19_branch2a_weights;
	
	MatrixXd res4b19_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b19_branch2a_biases.csv");
	VectorXd res4b19_branch2a_b(Map<VectorXd>(res4b19_branch2a_biases.data(), res4b19_branch2a_biases.cols()*res4b19_branch2a_biases.rows()));
	
	const int im_height_84 = output_height_83;
	const int im_width_84 = output_width_83;
	const int im_depth_84 = k_num_83;
	const int im_size_84 = im_height_84 * im_width_84;
	
	const int k_num_84 = 256;
	const int k_size_84 = 9;
	const int stride_84 = 1;
	const int k_depth_84 = im_depth_84;
	
	const int p1_84 = 1;
	const int p2_84 = 1;
	
	const int output_height_84 = (((im_height_84+(2*p1_84)) - sqrt(k_size_84))/stride_84) + 1;
	const int output_width_84 = (((im_width_84+(2*p2_84)) - sqrt(k_size_84))/stride_84) + 1;
	const int output_size_84 = output_height_84 * output_width_84;
	
	MatrixXd res4b19_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b19_branch2b_weights.csv");
	MatrixXd res4b19_branch2b_w = res4b19_branch2b_weights;
	
	MatrixXd res4b19_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b19_branch2b_biases.csv");
	VectorXd res4b19_branch2b_b(Map<VectorXd>(res4b19_branch2b_biases.data(), res4b19_branch2b_biases.cols()*res4b19_branch2b_biases.rows()));
	
	const int im_height_85 = output_height_84;
	const int im_width_85 = output_width_84;
	const int im_depth_85 = k_num_84;
	const int im_size_85 = im_height_85 * im_width_85;
	
	const int k_num_85 = 1024;
	const int k_size_85 = 1;
	const int stride_85 = 1;
	const int k_depth_85 = im_depth_85;
	
	const int p1_85 = 0;
	const int p2_85 = 0;
	
	const int output_height_85 = (((im_height_85+(2*p1_85)) - sqrt(k_size_85))/stride_85) + 1;
	const int output_width_85 = (((im_width_85+(2*p2_85)) - sqrt(k_size_85))/stride_85) + 1;
	const int output_size_85 = output_height_85 * output_width_85;
	
	MatrixXd res4b19_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b19_branch2c_weights.csv");
	MatrixXd res4b19_branch2c_w = res4b19_branch2c_weights;
	
	MatrixXd res4b19_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b19_branch2c_biases.csv");
	VectorXd res4b19_branch2c_b(Map<VectorXd>(res4b19_branch2c_biases.data(), res4b19_branch2c_biases.cols()*res4b19_branch2c_biases.rows()));
	
	const int im_height_86 = output_height_85;
	const int im_width_86 = output_width_85;
	const int im_depth_86 = k_num_85;
	const int im_size_86 = im_height_86 * im_width_86;
	
	const int k_num_86 = 256;
	const int k_size_86 = 1;
	const int stride_86 = 1;
	const int k_depth_86 = im_depth_86;
	
	const int p1_86 = 0;
	const int p2_86 = 0;
	
	const int output_height_86 = (((im_height_86+(2*p1_86)) - sqrt(k_size_86))/stride_86) + 1;
	const int output_width_86 = (((im_width_86+(2*p2_86)) - sqrt(k_size_86))/stride_86) + 1;
	const int output_size_86 = output_height_86 * output_width_86;
	
	MatrixXd res4b20_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b20_branch2a_weights.csv");
	MatrixXd res4b20_branch2a_w = res4b20_branch2a_weights;
	
	MatrixXd res4b20_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b20_branch2a_biases.csv");
	VectorXd res4b20_branch2a_b(Map<VectorXd>(res4b20_branch2a_biases.data(), res4b20_branch2a_biases.cols()*res4b20_branch2a_biases.rows()));
	
	const int im_height_87 = output_height_86;
	const int im_width_87 = output_width_86;
	const int im_depth_87 = k_num_86;
	const int im_size_87 = im_height_87 * im_width_87;
	
	const int k_num_87 = 256;
	const int k_size_87 = 9;
	const int stride_87 = 1;
	const int k_depth_87 = im_depth_87;
	
	const int p1_87 = 1;
	const int p2_87 = 1;
	
	const int output_height_87 = (((im_height_87+(2*p1_87)) - sqrt(k_size_87))/stride_87) + 1;
	const int output_width_87 = (((im_width_87+(2*p2_87)) - sqrt(k_size_87))/stride_87) + 1;
	const int output_size_87 = output_height_87 * output_width_87;
	
	MatrixXd res4b20_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b20_branch2b_weights.csv");
	MatrixXd res4b20_branch2b_w = res4b20_branch2b_weights;
	
	MatrixXd res4b20_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b20_branch2b_biases.csv");
	VectorXd res4b20_branch2b_b(Map<VectorXd>(res4b20_branch2b_biases.data(), res4b20_branch2b_biases.cols()*res4b20_branch2b_biases.rows()));
	
	const int im_height_88 = output_height_87;
	const int im_width_88 = output_width_87;
	const int im_depth_88 = k_num_87;
	const int im_size_88 = im_height_88 * im_width_88;
	
	const int k_num_88 = 1024;
	const int k_size_88 = 1;
	const int stride_88 = 1;
	const int k_depth_88 = im_depth_88;
	
	const int p1_88 = 0;
	const int p2_88 = 0;
	
	const int output_height_88 = (((im_height_88+(2*p1_88)) - sqrt(k_size_88))/stride_88) + 1;
	const int output_width_88 = (((im_width_88+(2*p2_88)) - sqrt(k_size_88))/stride_88) + 1;
	const int output_size_88 = output_height_88 * output_width_88;
	
	MatrixXd res4b20_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b20_branch2c_weights.csv");
	MatrixXd res4b20_branch2c_w = res4b20_branch2c_weights;
	
	MatrixXd res4b20_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b20_branch2c_biases.csv");
	VectorXd res4b20_branch2c_b(Map<VectorXd>(res4b20_branch2c_biases.data(), res4b20_branch2c_biases.cols()*res4b20_branch2c_biases.rows()));
	
	const int im_height_89 = output_height_88;
	const int im_width_89 = output_width_88;
	const int im_depth_89 = k_num_88;
	const int im_size_89 = im_height_89 * im_width_89;
	
	const int k_num_89 = 256;
	const int k_size_89 = 1;
	const int stride_89 = 1;
	const int k_depth_89 = im_depth_89;
	
	const int p1_89 = 0;
	const int p2_89 = 0;
	
	const int output_height_89 = (((im_height_89+(2*p1_89)) - sqrt(k_size_89))/stride_89) + 1;
	const int output_width_89 = (((im_width_89+(2*p2_89)) - sqrt(k_size_89))/stride_89) + 1;
	const int output_size_89 = output_height_89 * output_width_89;
	
	MatrixXd res4b21_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b21_branch2a_weights.csv");
	MatrixXd res4b21_branch2a_w = res4b21_branch2a_weights;
	
	MatrixXd res4b21_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b21_branch2a_biases.csv");
	VectorXd res4b21_branch2a_b(Map<VectorXd>(res4b21_branch2a_biases.data(), res4b21_branch2a_biases.cols()*res4b21_branch2a_biases.rows()));
	
	const int im_height_90 = output_height_89;
	const int im_width_90 = output_width_89;
	const int im_depth_90 = k_num_89;
	const int im_size_90 = im_height_90 * im_width_90;
	
	const int k_num_90 = 256;
	const int k_size_90 = 9;
	const int stride_90 = 1;
	const int k_depth_90 = im_depth_90;
	
	const int p1_90 = 1;
	const int p2_90 = 1;
	
	const int output_height_90 = (((im_height_90+(2*p1_90)) - sqrt(k_size_90))/stride_90) + 1;
	const int output_width_90 = (((im_width_90+(2*p2_90)) - sqrt(k_size_90))/stride_90) + 1;
	const int output_size_90 = output_height_90 * output_width_90;
	
	MatrixXd res4b21_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b21_branch2b_weights.csv");
	MatrixXd res4b21_branch2b_w = res4b21_branch2b_weights;
	
	MatrixXd res4b21_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b21_branch2b_biases.csv");
	VectorXd res4b21_branch2b_b(Map<VectorXd>(res4b21_branch2b_biases.data(), res4b21_branch2b_biases.cols()*res4b21_branch2b_biases.rows()));
	
	const int im_height_91 = output_height_90;
	const int im_width_91 = output_width_90;
	const int im_depth_91 = k_num_90;
	const int im_size_91 = im_height_91 * im_width_91;
	
	const int k_num_91 = 1024;
	const int k_size_91 = 1;
	const int stride_91 = 1;
	const int k_depth_91 = im_depth_91;
	
	const int p1_91 = 0;
	const int p2_91 = 0;
	
	const int output_height_91 = (((im_height_91+(2*p1_91)) - sqrt(k_size_91))/stride_91) + 1;
	const int output_width_91 = (((im_width_91+(2*p2_91)) - sqrt(k_size_91))/stride_91) + 1;
	const int output_size_91 = output_height_91 * output_width_91;
	
	MatrixXd res4b21_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b21_branch2c_weights.csv");
	MatrixXd res4b21_branch2c_w = res4b21_branch2c_weights;
	
	MatrixXd res4b21_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b21_branch2c_biases.csv");
	VectorXd res4b21_branch2c_b(Map<VectorXd>(res4b21_branch2c_biases.data(), res4b21_branch2c_biases.cols()*res4b21_branch2c_biases.rows()));
	
	const int im_height_92 = output_height_91;
	const int im_width_92 = output_width_91;
	const int im_depth_92 = k_num_91;
	const int im_size_92 = im_height_92 * im_width_92;
	
	const int k_num_92 = 256;
	const int k_size_92 = 1;
	const int stride_92 = 1;
	const int k_depth_92 = im_depth_92;
	
	const int p1_92 = 0;
	const int p2_92 = 0;
	
	const int output_height_92 = (((im_height_92+(2*p1_92)) - sqrt(k_size_92))/stride_92) + 1;
	const int output_width_92 = (((im_width_92+(2*p2_92)) - sqrt(k_size_92))/stride_92) + 1;
	const int output_size_92 = output_height_92 * output_width_92;
	
	MatrixXd res4b22_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b22_branch2a_weights.csv");
	MatrixXd res4b22_branch2a_w = res4b22_branch2a_weights;
	
	MatrixXd res4b22_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b22_branch2a_biases.csv");
	VectorXd res4b22_branch2a_b(Map<VectorXd>(res4b22_branch2a_biases.data(), res4b22_branch2a_biases.cols()*res4b22_branch2a_biases.rows()));
	
	const int im_height_93 = output_height_92;
	const int im_width_93 = output_width_92;
	const int im_depth_93 = k_num_92;
	const int im_size_93 = im_height_93 * im_width_93;
	
	const int k_num_93 = 256;
	const int k_size_93 = 9;
	const int stride_93 = 1;
	const int k_depth_93 = im_depth_93;
	
	const int p1_93 = 1;
	const int p2_93 = 1;
	
	const int output_height_93 = (((im_height_93+(2*p1_93)) - sqrt(k_size_93))/stride_93) + 1;
	const int output_width_93 = (((im_width_93+(2*p2_93)) - sqrt(k_size_93))/stride_93) + 1;
	const int output_size_93 = output_height_93 * output_width_93;
	
	MatrixXd res4b22_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b22_branch2b_weights.csv");
	MatrixXd res4b22_branch2b_w = res4b22_branch2b_weights;
	
	MatrixXd res4b22_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b22_branch2b_biases.csv");
	VectorXd res4b22_branch2b_b(Map<VectorXd>(res4b22_branch2b_biases.data(), res4b22_branch2b_biases.cols()*res4b22_branch2b_biases.rows()));
	
	const int im_height_94 = output_height_93;
	const int im_width_94 = output_width_93;
	const int im_depth_94 = k_num_93;
	const int im_size_94 = im_height_94 * im_width_94;
	
	const int k_num_94 = 1024;
	const int k_size_94 = 1;
	const int stride_94 = 1;
	const int k_depth_94 = im_depth_94;
	
	const int p1_94 = 0;
	const int p2_94 = 0;
	
	const int output_height_94 = (((im_height_94+(2*p1_94)) - sqrt(k_size_94))/stride_94) + 1;
	const int output_width_94 = (((im_width_94+(2*p2_94)) - sqrt(k_size_94))/stride_94) + 1;
	const int output_size_94 = output_height_94 * output_width_94;
	
	MatrixXd res4b22_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b22_branch2c_weights.csv");
	MatrixXd res4b22_branch2c_w = res4b22_branch2c_weights;
	
	MatrixXd res4b22_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res4b22_branch2c_biases.csv");
	VectorXd res4b22_branch2c_b(Map<VectorXd>(res4b22_branch2c_biases.data(), res4b22_branch2c_biases.cols()*res4b22_branch2c_biases.rows()));
	
	const int im_height_95 = output_height_94;
	const int im_width_95 = output_width_94;
	const int im_depth_95 = k_num_94;
	const int im_size_95 = im_height_95 * im_width_95;
	
	const int k_num_95 = 2048;
	const int k_size_95 = 1;
	const int stride_95 = 2;
	const int k_depth_95 = im_depth_95;
	
	const int p1_95 = 0;
	const int p2_95 = 0;
	
	const int output_height_95 = (((im_height_95+(2*p1_95)) - sqrt(k_size_95))/stride_95) + 1;
	const int output_width_95 = (((im_width_95+(2*p2_95)) - sqrt(k_size_95))/stride_95) + 1;
	const int output_size_95 = output_height_95 * output_width_95;
	
	MatrixXd res5a_branch1_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res5a_branch1_weights.csv");
	MatrixXd res5a_branch1_w = res5a_branch1_weights;
	
	MatrixXd res5a_branch1_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res5a_branch1_biases.csv");
	VectorXd res5a_branch1_b(Map<VectorXd>(res5a_branch1_biases.data(), res5a_branch1_biases.cols()*res5a_branch1_biases.rows()));
	
	const int im_height_96 = im_height_95;
	const int im_width_96 = im_width_95;
	const int im_depth_96 = im_depth_95;
	const int im_size_96 = im_size_95;
	
	const int k_num_96 = 512;
	const int k_size_96 = 1;
	const int stride_96 = 2;
	const int k_depth_96 = im_depth_96;
	
	const int p1_96 = 0;
	const int p2_96 = 0;
	
	const int output_height_96 = (((im_height_96+(2*p1_96)) - sqrt(k_size_96))/stride_96) + 1;
	const int output_width_96 = (((im_width_96+(2*p2_96)) - sqrt(k_size_96))/stride_96) + 1;
	const int output_size_96 = output_height_96 * output_width_96;
	
	MatrixXd res5a_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res5a_branch2a_weights.csv");
	MatrixXd res5a_branch2a_w = res5a_branch2a_weights;
	
	MatrixXd res5a_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res5a_branch2a_biases.csv");
	VectorXd res5a_branch2a_b(Map<VectorXd>(res5a_branch2a_biases.data(), res5a_branch2a_biases.cols()*res5a_branch2a_biases.rows()));
	
	const int im_height_97 = output_height_96;
	const int im_width_97 = output_width_96;
	const int im_depth_97 = k_num_96;
	const int im_size_97 = im_height_97 * im_width_97;
	
	const int k_num_97 = 512;
	const int k_size_97 = 9;
	const int stride_97 = 1;
	const int k_depth_97 = im_depth_97;
	
	const int p1_97 = 1;
	const int p2_97 = 1;
	
	const int output_height_97 = (((im_height_97+(2*p1_97)) - sqrt(k_size_97))/stride_97) + 1;
	const int output_width_97 = (((im_width_97+(2*p2_97)) - sqrt(k_size_97))/stride_97) + 1;
	const int output_size_97 = output_height_97 * output_width_97;
	
	MatrixXd res5a_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res5a_branch2b_weights.csv");
	MatrixXd res5a_branch2b_w = res5a_branch2b_weights;
	
	MatrixXd res5a_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res5a_branch2b_biases.csv");
	VectorXd res5a_branch2b_b(Map<VectorXd>(res5a_branch2b_biases.data(), res5a_branch2b_biases.cols()*res5a_branch2b_biases.rows()));
	
	const int im_height_98 = output_height_97;
	const int im_width_98 = output_width_97;
	const int im_depth_98 = k_num_97;
	const int im_size_98 = im_height_98 * im_width_98;
	
	const int k_num_98 = 2048;
	const int k_size_98 = 1;
	const int stride_98 = 1;
	const int k_depth_98 = im_depth_98;
	
	const int p1_98 = 0;
	const int p2_98 = 0;
	
	const int output_height_98 = (((im_height_98+(2*p1_98)) - sqrt(k_size_98))/stride_98) + 1;
	const int output_width_98 = (((im_width_98+(2*p2_98)) - sqrt(k_size_98))/stride_98) + 1;
	const int output_size_98 = output_height_98 * output_width_98;
	
	MatrixXd res5a_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res5a_branch2c_weights.csv");
	MatrixXd res5a_branch2c_w = res5a_branch2c_weights;
	
	MatrixXd res5a_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res5a_branch2c_biases.csv");
	VectorXd res5a_branch2c_b(Map<VectorXd>(res5a_branch2c_biases.data(), res5a_branch2c_biases.cols()*res5a_branch2c_biases.rows()));
	
	const int im_height_99 = output_height_98;
	const int im_width_99 = output_width_98;
	const int im_depth_99 = k_num_98;
	const int im_size_99 = im_height_99 * im_width_99;
	
	const int k_num_99 = 512;
	const int k_size_99 = 1;
	const int stride_99 = 1;
	const int k_depth_99 = im_depth_99;
	
	const int p1_99 = 0;
	const int p2_99 = 0;
	
	const int output_height_99 = (((im_height_99+(2*p1_99)) - sqrt(k_size_99))/stride_99) + 1;
	const int output_width_99 = (((im_width_99+(2*p2_99)) - sqrt(k_size_99))/stride_99) + 1;
	const int output_size_99 = output_height_99 * output_width_99;
	
	MatrixXd res5b_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res5b_branch2a_weights.csv");
	MatrixXd res5b_branch2a_w = res5b_branch2a_weights;
	
	MatrixXd res5b_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res5b_branch2a_biases.csv");
	VectorXd res5b_branch2a_b(Map<VectorXd>(res5b_branch2a_biases.data(), res5b_branch2a_biases.cols()*res5b_branch2a_biases.rows()));
	
	const int im_height_100 = output_height_99;
	const int im_width_100 = output_width_99;
	const int im_depth_100 = k_num_99;
	const int im_size_100 = im_height_100 * im_width_100;
	
	const int k_num_100 = 512;
	const int k_size_100 = 9;
	const int stride_100 = 1;
	const int k_depth_100 = im_depth_100;
	
	const int p1_100 = 1;
	const int p2_100 = 1;
	
	const int output_height_100 = (((im_height_100+(2*p1_100)) - sqrt(k_size_100))/stride_100) + 1;
	const int output_width_100 = (((im_width_100+(2*p2_100)) - sqrt(k_size_100))/stride_100) + 1;
	const int output_size_100 = output_height_100 * output_width_100;
	
	MatrixXd res5b_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res5b_branch2b_weights.csv");
	MatrixXd res5b_branch2b_w = res5b_branch2b_weights;
	
	MatrixXd res5b_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res5b_branch2b_biases.csv");
	VectorXd res5b_branch2b_b(Map<VectorXd>(res5b_branch2b_biases.data(), res5b_branch2b_biases.cols()*res5b_branch2b_biases.rows()));
	
	const int im_height_101 = output_height_100;
	const int im_width_101 = output_width_100;
	const int im_depth_101 = k_num_100;
	const int im_size_101 = im_height_101 * im_width_101;
	
	const int k_num_101 = 2048;
	const int k_size_101 = 1;
	const int stride_101 = 1;
	const int k_depth_101 = im_depth_101;
	
	const int p1_101 = 0;
	const int p2_101 = 0;
	
	const int output_height_101 = (((im_height_101+(2*p1_101)) - sqrt(k_size_101))/stride_101) + 1;
	const int output_width_101 = (((im_width_101+(2*p2_101)) - sqrt(k_size_101))/stride_101) + 1;
	const int output_size_101 = output_height_101 * output_width_101;
	
	MatrixXd res5b_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res5b_branch2c_weights.csv");
	MatrixXd res5b_branch2c_w = res5b_branch2c_weights;
	
	MatrixXd res5b_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res5b_branch2c_biases.csv");
	VectorXd res5b_branch2c_b(Map<VectorXd>(res5b_branch2c_biases.data(), res5b_branch2c_biases.cols()*res5b_branch2c_biases.rows()));
	
	const int im_height_102 = output_height_101;
	const int im_width_102 = output_width_101;
	const int im_depth_102 = k_num_101;
	const int im_size_102 = im_height_102 * im_width_102;
	
	const int k_num_102 = 512;
	const int k_size_102 = 1;
	const int stride_102 = 1;
	const int k_depth_102 = im_depth_102;
	
	const int p1_102 = 0;
	const int p2_102 = 0;
	
	const int output_height_102 = (((im_height_102+(2*p1_102)) - sqrt(k_size_102))/stride_102) + 1;
	const int output_width_102 = (((im_width_102+(2*p2_102)) - sqrt(k_size_102))/stride_102) + 1;
	const int output_size_102 = output_height_102 * output_width_102;
	
	MatrixXd res5c_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res5c_branch2a_weights.csv");
	MatrixXd res5c_branch2a_w = res5c_branch2a_weights;
	
	MatrixXd res5c_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res5c_branch2a_biases.csv");
	VectorXd res5c_branch2a_b(Map<VectorXd>(res5c_branch2a_biases.data(), res5c_branch2a_biases.cols()*res5c_branch2a_biases.rows()));
	
	const int im_height_103 = output_height_102;
	const int im_width_103 = output_width_102;
	const int im_depth_103 = k_num_102;
	const int im_size_103 = im_height_103 * im_width_103;
	
	const int k_num_103 = 512;
	const int k_size_103 = 9;
	const int stride_103 = 1;
	const int k_depth_103 = im_depth_103;
	
	const int p1_103 = 1;
	const int p2_103 = 1;
	
	const int output_height_103 = (((im_height_103+(2*p1_103)) - sqrt(k_size_103))/stride_103) + 1;
	const int output_width_103 = (((im_width_103+(2*p2_103)) - sqrt(k_size_103))/stride_103) + 1;
	const int output_size_103 = output_height_103 * output_width_103;
	
	MatrixXd res5c_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res5c_branch2b_weights.csv");
	MatrixXd res5c_branch2b_w = res5c_branch2b_weights;
	
	MatrixXd res5c_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res5c_branch2b_biases.csv");
	VectorXd res5c_branch2b_b(Map<VectorXd>(res5c_branch2b_biases.data(), res5c_branch2b_biases.cols()*res5c_branch2b_biases.rows()));
	
	const int im_height_104 = output_height_103;
	const int im_width_104 = output_width_103;
	const int im_depth_104 = k_num_103;
	const int im_size_104 = im_height_104 * im_width_104;
	
	const int k_num_104 = 2048;
	const int k_size_104 = 1;
	const int stride_104 = 1;
	const int k_depth_104 = im_depth_104;
	
	const int p1_104 = 0;
	const int p2_104 = 0;
	
	const int output_height_104 = (((im_height_104+(2*p1_104)) - sqrt(k_size_104))/stride_104) + 1;
	const int output_width_104 = (((im_width_104+(2*p2_104)) - sqrt(k_size_104))/stride_104) + 1;
	const int output_size_104 = output_height_104 * output_width_104;
	
	MatrixXd res5c_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/res5c_branch2c_weights.csv");
	MatrixXd res5c_branch2c_w = res5c_branch2c_weights;
	
	MatrixXd res5c_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/res5c_branch2c_biases.csv");
	VectorXd res5c_branch2c_b(Map<VectorXd>(res5c_branch2c_biases.data(), res5c_branch2c_biases.cols()*res5c_branch2c_biases.rows()));
	
	const int f_2 = 7;
	const int s_2 = 1;
	std::string mode_2 = "ave";
	
	const int pp1_2 = 0;
	const int pp2_2 = 0;
	
	MatrixXd fc1000_weights = load_csv_arma<MatrixXd>("../weights/ResNet101/fc1000_weights.csv");
	
	MatrixXd fc1000_biases = load_csv_arma<MatrixXd>("../weights/ResNet101/fc1000_biases.csv");
	VectorXd fc1000_b(Map<VectorXd>(fc1000_biases.data(), fc1000_biases.cols()*fc1000_biases.rows()));
	
	const int im_num = 1000;
	
	ifstream infile;
	infile.open("../inputs/ResNet101/production/imagenet_img_norm_1000.csv");
	
    for(int i=0; i < im_num; ++i)
    {
        cout << i << endl;

		MatrixXd line = load_csv<MatrixXd>(infile);
		
        MatrixXd img;
        img = line.block<1,im_size_1*im_depth_1>(0,1);

        MatrixXd image = Map<Matrix<double, im_depth_1, im_size_1, RowMajor>>(img.data());

        clock_t run_time_start = clock();

		MatrixXd conv1;
		double gemm_time_1;
		double offline_time_1;
		std::tie(conv1, gemm_time_1, offline_time_1) = convolve(image, im_size_1, im_height_1, im_width_1, im_depth_1, k_size_1, stride_1, conv1_b, p1_1, p2_1, conv1_w, output_size_1, mode);
		
		MatrixXd conv1_relu = relu(conv1);
		
		MatrixXd pool1 = pool(conv1_relu, f_1, s_1, output_width_1, output_height_1, pp1_1, pp2_1, mode_1);
		
		MatrixXd res2a_branch1;
		double gemm_time_2;
		double offline_time_2;
		std::tie(res2a_branch1, gemm_time_2, offline_time_2) = convolve(pool1, im_size_2, im_height_2, im_width_2, im_depth_2, k_size_2, stride_2, res2a_branch1_b, p1_2, p2_2, res2a_branch1_w, output_size_2, mode);
		
		MatrixXd res2a_branch2a;
		double gemm_time_3;
		double offline_time_3;
		std::tie(res2a_branch2a, gemm_time_3, offline_time_3) = convolve(pool1, im_size_3, im_height_3, im_width_3, im_depth_3, k_size_3, stride_3, res2a_branch2a_b, p1_3, p2_3, res2a_branch2a_w, output_size_3, mode);
		
		MatrixXd res2a_branch2a_relu = relu(res2a_branch2a);
		
		MatrixXd res2a_branch2b;
		double gemm_time_4;
		double offline_time_4;
		std::tie(res2a_branch2b, gemm_time_4, offline_time_4) = convolve(res2a_branch2a_relu, im_size_4, im_height_4, im_width_4, im_depth_4, k_size_4, stride_4, res2a_branch2b_b, p1_4, p2_4, res2a_branch2b_w, output_size_4, mode);
		
		MatrixXd res2a_branch2b_relu = relu(res2a_branch2b);
		
		MatrixXd res2a_branch2c;
		double gemm_time_5;
		double offline_time_5;
		std::tie(res2a_branch2c, gemm_time_5, offline_time_5) = convolve(res2a_branch2b_relu, im_size_5, im_height_5, im_width_5, im_depth_5, k_size_5, stride_5, res2a_branch2c_b, p1_5, p2_5, res2a_branch2c_w, output_size_5, mode);
		
		MatrixXd res2a = eltwise(res2a_branch2c, res2a_branch1);
		
		MatrixXd res2a_relu = relu(res2a);
		
		MatrixXd res2b_branch2a;
		double gemm_time_6;
		double offline_time_6;
		std::tie(res2b_branch2a, gemm_time_6, offline_time_6) = convolve(res2a_relu, im_size_6, im_height_6, im_width_6, im_depth_6, k_size_6, stride_6, res2b_branch2a_b, p1_6, p2_6, res2b_branch2a_w, output_size_6, mode);
		
		MatrixXd res2b_branch2a_relu = relu(res2b_branch2a);
		
		MatrixXd res2b_branch2b;
		double gemm_time_7;
		double offline_time_7;
		std::tie(res2b_branch2b, gemm_time_7, offline_time_7) = convolve(res2b_branch2a_relu, im_size_7, im_height_7, im_width_7, im_depth_7, k_size_7, stride_7, res2b_branch2b_b, p1_7, p2_7, res2b_branch2b_w, output_size_7, mode);
		
		MatrixXd res2b_branch2b_relu = relu(res2b_branch2b);
		
		MatrixXd res2b_branch2c;
		double gemm_time_8;
		double offline_time_8;
		std::tie(res2b_branch2c, gemm_time_8, offline_time_8) = convolve(res2b_branch2b_relu, im_size_8, im_height_8, im_width_8, im_depth_8, k_size_8, stride_8, res2b_branch2c_b, p1_8, p2_8, res2b_branch2c_w, output_size_8, mode);
		
		MatrixXd res2b = eltwise(res2b_branch2c, res2a_relu);
		
		MatrixXd res2b_relu = relu(res2b);
		
		MatrixXd res2c_branch2a;
		double gemm_time_9;
		double offline_time_9;
		std::tie(res2c_branch2a, gemm_time_9, offline_time_9) = convolve(res2b_relu, im_size_9, im_height_9, im_width_9, im_depth_9, k_size_9, stride_9, res2c_branch2a_b, p1_9, p2_9, res2c_branch2a_w, output_size_9, mode);
		
		MatrixXd res2c_branch2a_relu = relu(res2c_branch2a);
		
		MatrixXd res2c_branch2b;
		double gemm_time_10;
		double offline_time_10;
		std::tie(res2c_branch2b, gemm_time_10, offline_time_10) = convolve(res2c_branch2a_relu, im_size_10, im_height_10, im_width_10, im_depth_10, k_size_10, stride_10, res2c_branch2b_b, p1_10, p2_10, res2c_branch2b_w, output_size_10, mode);
		
		MatrixXd res2c_branch2b_relu = relu(res2c_branch2b);
		
		MatrixXd res2c_branch2c;
		double gemm_time_11;
		double offline_time_11;
		std::tie(res2c_branch2c, gemm_time_11, offline_time_11) = convolve(res2c_branch2b_relu, im_size_11, im_height_11, im_width_11, im_depth_11, k_size_11, stride_11, res2c_branch2c_b, p1_11, p2_11, res2c_branch2c_w, output_size_11, mode);
		
		MatrixXd res2c = eltwise(res2c_branch2c, res2b_relu);
		
		MatrixXd res2c_relu = relu(res2c);
		
		MatrixXd res3a_branch1;
		double gemm_time_12;
		double offline_time_12;
		std::tie(res3a_branch1, gemm_time_12, offline_time_12) = convolve(res2c_relu, im_size_12, im_height_12, im_width_12, im_depth_12, k_size_12, stride_12, res3a_branch1_b, p1_12, p2_12, res3a_branch1_w, output_size_12, mode);
		
		MatrixXd res3a_branch2a;
		double gemm_time_13;
		double offline_time_13;
		std::tie(res3a_branch2a, gemm_time_13, offline_time_13) = convolve(res2c_relu, im_size_13, im_height_13, im_width_13, im_depth_13, k_size_13, stride_13, res3a_branch2a_b, p1_13, p2_13, res3a_branch2a_w, output_size_13, mode);
		
		MatrixXd res3a_branch2a_relu = relu(res3a_branch2a);
		
		MatrixXd res3a_branch2b;
		double gemm_time_14;
		double offline_time_14;
		std::tie(res3a_branch2b, gemm_time_14, offline_time_14) = convolve(res3a_branch2a_relu, im_size_14, im_height_14, im_width_14, im_depth_14, k_size_14, stride_14, res3a_branch2b_b, p1_14, p2_14, res3a_branch2b_w, output_size_14, mode);
		
		MatrixXd res3a_branch2b_relu = relu(res3a_branch2b);
		
		MatrixXd res3a_branch2c;
		double gemm_time_15;
		double offline_time_15;
		std::tie(res3a_branch2c, gemm_time_15, offline_time_15) = convolve(res3a_branch2b_relu, im_size_15, im_height_15, im_width_15, im_depth_15, k_size_15, stride_15, res3a_branch2c_b, p1_15, p2_15, res3a_branch2c_w, output_size_15, mode);
		
		MatrixXd res3a = eltwise(res3a_branch2c, res3a_branch1);
		
		MatrixXd res3a_relu = relu(res3a);
		
		MatrixXd res3b1_branch2a;
		double gemm_time_16;
		double offline_time_16;
		std::tie(res3b1_branch2a, gemm_time_16, offline_time_16) = convolve(res3a_relu, im_size_16, im_height_16, im_width_16, im_depth_16, k_size_16, stride_16, res3b1_branch2a_b, p1_16, p2_16, res3b1_branch2a_w, output_size_16, mode);
		
		MatrixXd res3b1_branch2a_relu = relu(res3b1_branch2a);
		
		MatrixXd res3b1_branch2b;
		double gemm_time_17;
		double offline_time_17;
		std::tie(res3b1_branch2b, gemm_time_17, offline_time_17) = convolve(res3b1_branch2a_relu, im_size_17, im_height_17, im_width_17, im_depth_17, k_size_17, stride_17, res3b1_branch2b_b, p1_17, p2_17, res3b1_branch2b_w, output_size_17, mode);
		
		MatrixXd res3b1_branch2b_relu = relu(res3b1_branch2b);
		
		MatrixXd res3b1_branch2c;
		double gemm_time_18;
		double offline_time_18;
		std::tie(res3b1_branch2c, gemm_time_18, offline_time_18) = convolve(res3b1_branch2b_relu, im_size_18, im_height_18, im_width_18, im_depth_18, k_size_18, stride_18, res3b1_branch2c_b, p1_18, p2_18, res3b1_branch2c_w, output_size_18, mode);
		
		MatrixXd res3b1 = eltwise(res3b1_branch2c, res3a_relu);
		
		MatrixXd res3b1_relu = relu(res3b1);
		
		MatrixXd res3b2_branch2a;
		double gemm_time_19;
		double offline_time_19;
		std::tie(res3b2_branch2a, gemm_time_19, offline_time_19) = convolve(res3b1_relu, im_size_19, im_height_19, im_width_19, im_depth_19, k_size_19, stride_19, res3b2_branch2a_b, p1_19, p2_19, res3b2_branch2a_w, output_size_19, mode);
		
		MatrixXd res3b2_branch2a_relu = relu(res3b2_branch2a);
		
		MatrixXd res3b2_branch2b;
		double gemm_time_20;
		double offline_time_20;
		std::tie(res3b2_branch2b, gemm_time_20, offline_time_20) = convolve(res3b2_branch2a_relu, im_size_20, im_height_20, im_width_20, im_depth_20, k_size_20, stride_20, res3b2_branch2b_b, p1_20, p2_20, res3b2_branch2b_w, output_size_20, mode);
		
		MatrixXd res3b2_branch2b_relu = relu(res3b2_branch2b);
		
		MatrixXd res3b2_branch2c;
		double gemm_time_21;
		double offline_time_21;
		std::tie(res3b2_branch2c, gemm_time_21, offline_time_21) = convolve(res3b2_branch2b_relu, im_size_21, im_height_21, im_width_21, im_depth_21, k_size_21, stride_21, res3b2_branch2c_b, p1_21, p2_21, res3b2_branch2c_w, output_size_21, mode);
		
		MatrixXd res3b2 = eltwise(res3b2_branch2c, res3b1_relu);
		
		MatrixXd res3b2_relu = relu(res3b2);
		
		MatrixXd res3b3_branch2a;
		double gemm_time_22;
		double offline_time_22;
		std::tie(res3b3_branch2a, gemm_time_22, offline_time_22) = convolve(res3b2_relu, im_size_22, im_height_22, im_width_22, im_depth_22, k_size_22, stride_22, res3b3_branch2a_b, p1_22, p2_22, res3b3_branch2a_w, output_size_22, mode);
		
		MatrixXd res3b3_branch2a_relu = relu(res3b3_branch2a);
		
		MatrixXd res3b3_branch2b;
		double gemm_time_23;
		double offline_time_23;
		std::tie(res3b3_branch2b, gemm_time_23, offline_time_23) = convolve(res3b3_branch2a_relu, im_size_23, im_height_23, im_width_23, im_depth_23, k_size_23, stride_23, res3b3_branch2b_b, p1_23, p2_23, res3b3_branch2b_w, output_size_23, mode);
		
		MatrixXd res3b3_branch2b_relu = relu(res3b3_branch2b);
		
		MatrixXd res3b3_branch2c;
		double gemm_time_24;
		double offline_time_24;
		std::tie(res3b3_branch2c, gemm_time_24, offline_time_24) = convolve(res3b3_branch2b_relu, im_size_24, im_height_24, im_width_24, im_depth_24, k_size_24, stride_24, res3b3_branch2c_b, p1_24, p2_24, res3b3_branch2c_w, output_size_24, mode);
		
		MatrixXd res3b3 = eltwise(res3b3_branch2c, res3b2_relu);
		
		MatrixXd res3b3_relu = relu(res3b3);
		
		MatrixXd res4a_branch1;
		double gemm_time_25;
		double offline_time_25;
		std::tie(res4a_branch1, gemm_time_25, offline_time_25) = convolve(res3b3_relu, im_size_25, im_height_25, im_width_25, im_depth_25, k_size_25, stride_25, res4a_branch1_b, p1_25, p2_25, res4a_branch1_w, output_size_25, mode);
		
		MatrixXd res4a_branch2a;
		double gemm_time_26;
		double offline_time_26;
		std::tie(res4a_branch2a, gemm_time_26, offline_time_26) = convolve(res3b3_relu, im_size_26, im_height_26, im_width_26, im_depth_26, k_size_26, stride_26, res4a_branch2a_b, p1_26, p2_26, res4a_branch2a_w, output_size_26, mode);
		
		MatrixXd res4a_branch2a_relu = relu(res4a_branch2a);
		
		MatrixXd res4a_branch2b;
		double gemm_time_27;
		double offline_time_27;
		std::tie(res4a_branch2b, gemm_time_27, offline_time_27) = convolve(res4a_branch2a_relu, im_size_27, im_height_27, im_width_27, im_depth_27, k_size_27, stride_27, res4a_branch2b_b, p1_27, p2_27, res4a_branch2b_w, output_size_27, mode);
		
		MatrixXd res4a_branch2b_relu = relu(res4a_branch2b);
		
		MatrixXd res4a_branch2c;
		double gemm_time_28;
		double offline_time_28;
		std::tie(res4a_branch2c, gemm_time_28, offline_time_28) = convolve(res4a_branch2b_relu, im_size_28, im_height_28, im_width_28, im_depth_28, k_size_28, stride_28, res4a_branch2c_b, p1_28, p2_28, res4a_branch2c_w, output_size_28, mode);
		
		MatrixXd res4a = eltwise(res4a_branch2c, res4a_branch1);
		
		MatrixXd res4a_relu = relu(res4a);
		
		MatrixXd res4b1_branch2a;
		double gemm_time_29;
		double offline_time_29;
		std::tie(res4b1_branch2a, gemm_time_29, offline_time_29) = convolve(res4a_relu, im_size_29, im_height_29, im_width_29, im_depth_29, k_size_29, stride_29, res4b1_branch2a_b, p1_29, p2_29, res4b1_branch2a_w, output_size_29, mode);
		
		MatrixXd res4b1_branch2a_relu = relu(res4b1_branch2a);
		
		MatrixXd res4b1_branch2b;
		double gemm_time_30;
		double offline_time_30;
		std::tie(res4b1_branch2b, gemm_time_30, offline_time_30) = convolve(res4b1_branch2a_relu, im_size_30, im_height_30, im_width_30, im_depth_30, k_size_30, stride_30, res4b1_branch2b_b, p1_30, p2_30, res4b1_branch2b_w, output_size_30, mode);
		
		MatrixXd res4b1_branch2b_relu = relu(res4b1_branch2b);
		
		MatrixXd res4b1_branch2c;
		double gemm_time_31;
		double offline_time_31;
		std::tie(res4b1_branch2c, gemm_time_31, offline_time_31) = convolve(res4b1_branch2b_relu, im_size_31, im_height_31, im_width_31, im_depth_31, k_size_31, stride_31, res4b1_branch2c_b, p1_31, p2_31, res4b1_branch2c_w, output_size_31, mode);
		
		MatrixXd res4b1 = eltwise(res4b1_branch2c, res4a_relu);
		
		MatrixXd res4b1_relu = relu(res4b1);
		
		MatrixXd res4b2_branch2a;
		double gemm_time_32;
		double offline_time_32;
		std::tie(res4b2_branch2a, gemm_time_32, offline_time_32) = convolve(res4b1_relu, im_size_32, im_height_32, im_width_32, im_depth_32, k_size_32, stride_32, res4b2_branch2a_b, p1_32, p2_32, res4b2_branch2a_w, output_size_32, mode);
		
		MatrixXd res4b2_branch2a_relu = relu(res4b2_branch2a);
		
		MatrixXd res4b2_branch2b;
		double gemm_time_33;
		double offline_time_33;
		std::tie(res4b2_branch2b, gemm_time_33, offline_time_33) = convolve(res4b2_branch2a_relu, im_size_33, im_height_33, im_width_33, im_depth_33, k_size_33, stride_33, res4b2_branch2b_b, p1_33, p2_33, res4b2_branch2b_w, output_size_33, mode);
		
		MatrixXd res4b2_branch2b_relu = relu(res4b2_branch2b);
		
		MatrixXd res4b2_branch2c;
		double gemm_time_34;
		double offline_time_34;
		std::tie(res4b2_branch2c, gemm_time_34, offline_time_34) = convolve(res4b2_branch2b_relu, im_size_34, im_height_34, im_width_34, im_depth_34, k_size_34, stride_34, res4b2_branch2c_b, p1_34, p2_34, res4b2_branch2c_w, output_size_34, mode);
		
		MatrixXd res4b2 = eltwise(res4b2_branch2c, res4b1_relu);
		
		MatrixXd res4b2_relu = relu(res4b2);
		
		MatrixXd res4b3_branch2a;
		double gemm_time_35;
		double offline_time_35;
		std::tie(res4b3_branch2a, gemm_time_35, offline_time_35) = convolve(res4b2_relu, im_size_35, im_height_35, im_width_35, im_depth_35, k_size_35, stride_35, res4b3_branch2a_b, p1_35, p2_35, res4b3_branch2a_w, output_size_35, mode);
		
		MatrixXd res4b3_branch2a_relu = relu(res4b3_branch2a);
		
		MatrixXd res4b3_branch2b;
		double gemm_time_36;
		double offline_time_36;
		std::tie(res4b3_branch2b, gemm_time_36, offline_time_36) = convolve(res4b3_branch2a_relu, im_size_36, im_height_36, im_width_36, im_depth_36, k_size_36, stride_36, res4b3_branch2b_b, p1_36, p2_36, res4b3_branch2b_w, output_size_36, mode);
		
		MatrixXd res4b3_branch2b_relu = relu(res4b3_branch2b);
		
		MatrixXd res4b3_branch2c;
		double gemm_time_37;
		double offline_time_37;
		std::tie(res4b3_branch2c, gemm_time_37, offline_time_37) = convolve(res4b3_branch2b_relu, im_size_37, im_height_37, im_width_37, im_depth_37, k_size_37, stride_37, res4b3_branch2c_b, p1_37, p2_37, res4b3_branch2c_w, output_size_37, mode);
		
		MatrixXd res4b3 = eltwise(res4b3_branch2c, res4b2_relu);
		
		MatrixXd res4b3_relu = relu(res4b3);
		
		MatrixXd res4b4_branch2a;
		double gemm_time_38;
		double offline_time_38;
		std::tie(res4b4_branch2a, gemm_time_38, offline_time_38) = convolve(res4b3_relu, im_size_38, im_height_38, im_width_38, im_depth_38, k_size_38, stride_38, res4b4_branch2a_b, p1_38, p2_38, res4b4_branch2a_w, output_size_38, mode);
		
		MatrixXd res4b4_branch2a_relu = relu(res4b4_branch2a);
		
		MatrixXd res4b4_branch2b;
		double gemm_time_39;
		double offline_time_39;
		std::tie(res4b4_branch2b, gemm_time_39, offline_time_39) = convolve(res4b4_branch2a_relu, im_size_39, im_height_39, im_width_39, im_depth_39, k_size_39, stride_39, res4b4_branch2b_b, p1_39, p2_39, res4b4_branch2b_w, output_size_39, mode);
		
		MatrixXd res4b4_branch2b_relu = relu(res4b4_branch2b);
		
		MatrixXd res4b4_branch2c;
		double gemm_time_40;
		double offline_time_40;
		std::tie(res4b4_branch2c, gemm_time_40, offline_time_40) = convolve(res4b4_branch2b_relu, im_size_40, im_height_40, im_width_40, im_depth_40, k_size_40, stride_40, res4b4_branch2c_b, p1_40, p2_40, res4b4_branch2c_w, output_size_40, mode);
		
		MatrixXd res4b4 = eltwise(res4b4_branch2c, res4b3_relu);
		
		MatrixXd res4b4_relu = relu(res4b4);
		
		MatrixXd res4b5_branch2a;
		double gemm_time_41;
		double offline_time_41;
		std::tie(res4b5_branch2a, gemm_time_41, offline_time_41) = convolve(res4b4_relu, im_size_41, im_height_41, im_width_41, im_depth_41, k_size_41, stride_41, res4b5_branch2a_b, p1_41, p2_41, res4b5_branch2a_w, output_size_41, mode);
		
		MatrixXd res4b5_branch2a_relu = relu(res4b5_branch2a);
		
		MatrixXd res4b5_branch2b;
		double gemm_time_42;
		double offline_time_42;
		std::tie(res4b5_branch2b, gemm_time_42, offline_time_42) = convolve(res4b5_branch2a_relu, im_size_42, im_height_42, im_width_42, im_depth_42, k_size_42, stride_42, res4b5_branch2b_b, p1_42, p2_42, res4b5_branch2b_w, output_size_42, mode);
		
		MatrixXd res4b5_branch2b_relu = relu(res4b5_branch2b);
		
		MatrixXd res4b5_branch2c;
		double gemm_time_43;
		double offline_time_43;
		std::tie(res4b5_branch2c, gemm_time_43, offline_time_43) = convolve(res4b5_branch2b_relu, im_size_43, im_height_43, im_width_43, im_depth_43, k_size_43, stride_43, res4b5_branch2c_b, p1_43, p2_43, res4b5_branch2c_w, output_size_43, mode);
		
		MatrixXd res4b5 = eltwise(res4b5_branch2c, res4b4_relu);
		
		MatrixXd res4b5_relu = relu(res4b5);
		
		MatrixXd res4b6_branch2a;
		double gemm_time_44;
		double offline_time_44;
		std::tie(res4b6_branch2a, gemm_time_44, offline_time_44) = convolve(res4b5_relu, im_size_44, im_height_44, im_width_44, im_depth_44, k_size_44, stride_44, res4b6_branch2a_b, p1_44, p2_44, res4b6_branch2a_w, output_size_44, mode);
		
		MatrixXd res4b6_branch2a_relu = relu(res4b6_branch2a);
		
		MatrixXd res4b6_branch2b;
		double gemm_time_45;
		double offline_time_45;
		std::tie(res4b6_branch2b, gemm_time_45, offline_time_45) = convolve(res4b6_branch2a_relu, im_size_45, im_height_45, im_width_45, im_depth_45, k_size_45, stride_45, res4b6_branch2b_b, p1_45, p2_45, res4b6_branch2b_w, output_size_45, mode);
		
		MatrixXd res4b6_branch2b_relu = relu(res4b6_branch2b);
		
		MatrixXd res4b6_branch2c;
		double gemm_time_46;
		double offline_time_46;
		std::tie(res4b6_branch2c, gemm_time_46, offline_time_46) = convolve(res4b6_branch2b_relu, im_size_46, im_height_46, im_width_46, im_depth_46, k_size_46, stride_46, res4b6_branch2c_b, p1_46, p2_46, res4b6_branch2c_w, output_size_46, mode);
		
		MatrixXd res4b6 = eltwise(res4b6_branch2c, res4b5_relu);
		
		MatrixXd res4b6_relu = relu(res4b6);
		
		MatrixXd res4b7_branch2a;
		double gemm_time_47;
		double offline_time_47;
		std::tie(res4b7_branch2a, gemm_time_47, offline_time_47) = convolve(res4b6_relu, im_size_47, im_height_47, im_width_47, im_depth_47, k_size_47, stride_47, res4b7_branch2a_b, p1_47, p2_47, res4b7_branch2a_w, output_size_47, mode);
		
		MatrixXd res4b7_branch2a_relu = relu(res4b7_branch2a);
		
		MatrixXd res4b7_branch2b;
		double gemm_time_48;
		double offline_time_48;
		std::tie(res4b7_branch2b, gemm_time_48, offline_time_48) = convolve(res4b7_branch2a_relu, im_size_48, im_height_48, im_width_48, im_depth_48, k_size_48, stride_48, res4b7_branch2b_b, p1_48, p2_48, res4b7_branch2b_w, output_size_48, mode);
		
		MatrixXd res4b7_branch2b_relu = relu(res4b7_branch2b);
		
		MatrixXd res4b7_branch2c;
		double gemm_time_49;
		double offline_time_49;
		std::tie(res4b7_branch2c, gemm_time_49, offline_time_49) = convolve(res4b7_branch2b_relu, im_size_49, im_height_49, im_width_49, im_depth_49, k_size_49, stride_49, res4b7_branch2c_b, p1_49, p2_49, res4b7_branch2c_w, output_size_49, mode);
		
		MatrixXd res4b7 = eltwise(res4b7_branch2c, res4b6_relu);
		
		MatrixXd res4b7_relu = relu(res4b7);
		
		MatrixXd res4b8_branch2a;
		double gemm_time_50;
		double offline_time_50;
		std::tie(res4b8_branch2a, gemm_time_50, offline_time_50) = convolve(res4b7_relu, im_size_50, im_height_50, im_width_50, im_depth_50, k_size_50, stride_50, res4b8_branch2a_b, p1_50, p2_50, res4b8_branch2a_w, output_size_50, mode);
		
		MatrixXd res4b8_branch2a_relu = relu(res4b8_branch2a);
		
		MatrixXd res4b8_branch2b;
		double gemm_time_51;
		double offline_time_51;
		std::tie(res4b8_branch2b, gemm_time_51, offline_time_51) = convolve(res4b8_branch2a_relu, im_size_51, im_height_51, im_width_51, im_depth_51, k_size_51, stride_51, res4b8_branch2b_b, p1_51, p2_51, res4b8_branch2b_w, output_size_51, mode);
		
		MatrixXd res4b8_branch2b_relu = relu(res4b8_branch2b);
		
		MatrixXd res4b8_branch2c;
		double gemm_time_52;
		double offline_time_52;
		std::tie(res4b8_branch2c, gemm_time_52, offline_time_52) = convolve(res4b8_branch2b_relu, im_size_52, im_height_52, im_width_52, im_depth_52, k_size_52, stride_52, res4b8_branch2c_b, p1_52, p2_52, res4b8_branch2c_w, output_size_52, mode);
		
		MatrixXd res4b8 = eltwise(res4b8_branch2c, res4b7_relu);
		
		MatrixXd res4b8_relu = relu(res4b8);
		
		MatrixXd res4b9_branch2a;
		double gemm_time_53;
		double offline_time_53;
		std::tie(res4b9_branch2a, gemm_time_53, offline_time_53) = convolve(res4b8_relu, im_size_53, im_height_53, im_width_53, im_depth_53, k_size_53, stride_53, res4b9_branch2a_b, p1_53, p2_53, res4b9_branch2a_w, output_size_53, mode);
		
		MatrixXd res4b9_branch2a_relu = relu(res4b9_branch2a);
		
		MatrixXd res4b9_branch2b;
		double gemm_time_54;
		double offline_time_54;
		std::tie(res4b9_branch2b, gemm_time_54, offline_time_54) = convolve(res4b9_branch2a_relu, im_size_54, im_height_54, im_width_54, im_depth_54, k_size_54, stride_54, res4b9_branch2b_b, p1_54, p2_54, res4b9_branch2b_w, output_size_54, mode);
		
		MatrixXd res4b9_branch2b_relu = relu(res4b9_branch2b);
		
		MatrixXd res4b9_branch2c;
		double gemm_time_55;
		double offline_time_55;
		std::tie(res4b9_branch2c, gemm_time_55, offline_time_55) = convolve(res4b9_branch2b_relu, im_size_55, im_height_55, im_width_55, im_depth_55, k_size_55, stride_55, res4b9_branch2c_b, p1_55, p2_55, res4b9_branch2c_w, output_size_55, mode);
		
		MatrixXd res4b9 = eltwise(res4b9_branch2c, res4b8_relu);
		
		MatrixXd res4b9_relu = relu(res4b9);
		
		MatrixXd res4b10_branch2a;
		double gemm_time_56;
		double offline_time_56;
		std::tie(res4b10_branch2a, gemm_time_56, offline_time_56) = convolve(res4b9_relu, im_size_56, im_height_56, im_width_56, im_depth_56, k_size_56, stride_56, res4b10_branch2a_b, p1_56, p2_56, res4b10_branch2a_w, output_size_56, mode);
		
		MatrixXd res4b10_branch2a_relu = relu(res4b10_branch2a);
		
		MatrixXd res4b10_branch2b;
		double gemm_time_57;
		double offline_time_57;
		std::tie(res4b10_branch2b, gemm_time_57, offline_time_57) = convolve(res4b10_branch2a_relu, im_size_57, im_height_57, im_width_57, im_depth_57, k_size_57, stride_57, res4b10_branch2b_b, p1_57, p2_57, res4b10_branch2b_w, output_size_57, mode);
		
		MatrixXd res4b10_branch2b_relu = relu(res4b10_branch2b);
		
		MatrixXd res4b10_branch2c;
		double gemm_time_58;
		double offline_time_58;
		std::tie(res4b10_branch2c, gemm_time_58, offline_time_58) = convolve(res4b10_branch2b_relu, im_size_58, im_height_58, im_width_58, im_depth_58, k_size_58, stride_58, res4b10_branch2c_b, p1_58, p2_58, res4b10_branch2c_w, output_size_58, mode);
		
		MatrixXd res4b10 = eltwise(res4b10_branch2c, res4b9_relu);
		
		MatrixXd res4b10_relu = relu(res4b10);
		
		MatrixXd res4b11_branch2a;
		double gemm_time_59;
		double offline_time_59;
		std::tie(res4b11_branch2a, gemm_time_59, offline_time_59) = convolve(res4b10_relu, im_size_59, im_height_59, im_width_59, im_depth_59, k_size_59, stride_59, res4b11_branch2a_b, p1_59, p2_59, res4b11_branch2a_w, output_size_59, mode);
		
		MatrixXd res4b11_branch2a_relu = relu(res4b11_branch2a);
		
		MatrixXd res4b11_branch2b;
		double gemm_time_60;
		double offline_time_60;
		std::tie(res4b11_branch2b, gemm_time_60, offline_time_60) = convolve(res4b11_branch2a_relu, im_size_60, im_height_60, im_width_60, im_depth_60, k_size_60, stride_60, res4b11_branch2b_b, p1_60, p2_60, res4b11_branch2b_w, output_size_60, mode);
		
		MatrixXd res4b11_branch2b_relu = relu(res4b11_branch2b);
		
		MatrixXd res4b11_branch2c;
		double gemm_time_61;
		double offline_time_61;
		std::tie(res4b11_branch2c, gemm_time_61, offline_time_61) = convolve(res4b11_branch2b_relu, im_size_61, im_height_61, im_width_61, im_depth_61, k_size_61, stride_61, res4b11_branch2c_b, p1_61, p2_61, res4b11_branch2c_w, output_size_61, mode);
		
		MatrixXd res4b11 = eltwise(res4b11_branch2c, res4b10_relu);
		
		MatrixXd res4b11_relu = relu(res4b11);
		
		MatrixXd res4b12_branch2a;
		double gemm_time_62;
		double offline_time_62;
		std::tie(res4b12_branch2a, gemm_time_62, offline_time_62) = convolve(res4b11_relu, im_size_62, im_height_62, im_width_62, im_depth_62, k_size_62, stride_62, res4b12_branch2a_b, p1_62, p2_62, res4b12_branch2a_w, output_size_62, mode);
		
		MatrixXd res4b12_branch2a_relu = relu(res4b12_branch2a);
		
		MatrixXd res4b12_branch2b;
		double gemm_time_63;
		double offline_time_63;
		std::tie(res4b12_branch2b, gemm_time_63, offline_time_63) = convolve(res4b12_branch2a_relu, im_size_63, im_height_63, im_width_63, im_depth_63, k_size_63, stride_63, res4b12_branch2b_b, p1_63, p2_63, res4b12_branch2b_w, output_size_63, mode);
		
		MatrixXd res4b12_branch2b_relu = relu(res4b12_branch2b);
		
		MatrixXd res4b12_branch2c;
		double gemm_time_64;
		double offline_time_64;
		std::tie(res4b12_branch2c, gemm_time_64, offline_time_64) = convolve(res4b12_branch2b_relu, im_size_64, im_height_64, im_width_64, im_depth_64, k_size_64, stride_64, res4b12_branch2c_b, p1_64, p2_64, res4b12_branch2c_w, output_size_64, mode);
		
		MatrixXd res4b12 = eltwise(res4b12_branch2c, res4b11_relu);
		
		MatrixXd res4b12_relu = relu(res4b12);
		
		MatrixXd res4b13_branch2a;
		double gemm_time_65;
		double offline_time_65;
		std::tie(res4b13_branch2a, gemm_time_65, offline_time_65) = convolve(res4b12_relu, im_size_65, im_height_65, im_width_65, im_depth_65, k_size_65, stride_65, res4b13_branch2a_b, p1_65, p2_65, res4b13_branch2a_w, output_size_65, mode);
		
		MatrixXd res4b13_branch2a_relu = relu(res4b13_branch2a);
		
		MatrixXd res4b13_branch2b;
		double gemm_time_66;
		double offline_time_66;
		std::tie(res4b13_branch2b, gemm_time_66, offline_time_66) = convolve(res4b13_branch2a_relu, im_size_66, im_height_66, im_width_66, im_depth_66, k_size_66, stride_66, res4b13_branch2b_b, p1_66, p2_66, res4b13_branch2b_w, output_size_66, mode);
		
		MatrixXd res4b13_branch2b_relu = relu(res4b13_branch2b);
		
		MatrixXd res4b13_branch2c;
		double gemm_time_67;
		double offline_time_67;
		std::tie(res4b13_branch2c, gemm_time_67, offline_time_67) = convolve(res4b13_branch2b_relu, im_size_67, im_height_67, im_width_67, im_depth_67, k_size_67, stride_67, res4b13_branch2c_b, p1_67, p2_67, res4b13_branch2c_w, output_size_67, mode);
		
		MatrixXd res4b13 = eltwise(res4b13_branch2c, res4b12_relu);
		
		MatrixXd res4b13_relu = relu(res4b13);
		
		MatrixXd res4b14_branch2a;
		double gemm_time_68;
		double offline_time_68;
		std::tie(res4b14_branch2a, gemm_time_68, offline_time_68) = convolve(res4b13_relu, im_size_68, im_height_68, im_width_68, im_depth_68, k_size_68, stride_68, res4b14_branch2a_b, p1_68, p2_68, res4b14_branch2a_w, output_size_68, mode);
		
		MatrixXd res4b14_branch2a_relu = relu(res4b14_branch2a);
		
		MatrixXd res4b14_branch2b;
		double gemm_time_69;
		double offline_time_69;
		std::tie(res4b14_branch2b, gemm_time_69, offline_time_69) = convolve(res4b14_branch2a_relu, im_size_69, im_height_69, im_width_69, im_depth_69, k_size_69, stride_69, res4b14_branch2b_b, p1_69, p2_69, res4b14_branch2b_w, output_size_69, mode);
		
		MatrixXd res4b14_branch2b_relu = relu(res4b14_branch2b);
		
		MatrixXd res4b14_branch2c;
		double gemm_time_70;
		double offline_time_70;
		std::tie(res4b14_branch2c, gemm_time_70, offline_time_70) = convolve(res4b14_branch2b_relu, im_size_70, im_height_70, im_width_70, im_depth_70, k_size_70, stride_70, res4b14_branch2c_b, p1_70, p2_70, res4b14_branch2c_w, output_size_70, mode);
		
		MatrixXd res4b14 = eltwise(res4b14_branch2c, res4b13_relu);
		
		MatrixXd res4b14_relu = relu(res4b14);
		
		MatrixXd res4b15_branch2a;
		double gemm_time_71;
		double offline_time_71;
		std::tie(res4b15_branch2a, gemm_time_71, offline_time_71) = convolve(res4b14_relu, im_size_71, im_height_71, im_width_71, im_depth_71, k_size_71, stride_71, res4b15_branch2a_b, p1_71, p2_71, res4b15_branch2a_w, output_size_71, mode);
		
		MatrixXd res4b15_branch2a_relu = relu(res4b15_branch2a);
		
		MatrixXd res4b15_branch2b;
		double gemm_time_72;
		double offline_time_72;
		std::tie(res4b15_branch2b, gemm_time_72, offline_time_72) = convolve(res4b15_branch2a_relu, im_size_72, im_height_72, im_width_72, im_depth_72, k_size_72, stride_72, res4b15_branch2b_b, p1_72, p2_72, res4b15_branch2b_w, output_size_72, mode);
		
		MatrixXd res4b15_branch2b_relu = relu(res4b15_branch2b);
		
		MatrixXd res4b15_branch2c;
		double gemm_time_73;
		double offline_time_73;
		std::tie(res4b15_branch2c, gemm_time_73, offline_time_73) = convolve(res4b15_branch2b_relu, im_size_73, im_height_73, im_width_73, im_depth_73, k_size_73, stride_73, res4b15_branch2c_b, p1_73, p2_73, res4b15_branch2c_w, output_size_73, mode);
		
		MatrixXd res4b15 = eltwise(res4b15_branch2c, res4b14_relu);
		
		MatrixXd res4b15_relu = relu(res4b15);
		
		MatrixXd res4b16_branch2a;
		double gemm_time_74;
		double offline_time_74;
		std::tie(res4b16_branch2a, gemm_time_74, offline_time_74) = convolve(res4b15_relu, im_size_74, im_height_74, im_width_74, im_depth_74, k_size_74, stride_74, res4b16_branch2a_b, p1_74, p2_74, res4b16_branch2a_w, output_size_74, mode);
		
		MatrixXd res4b16_branch2a_relu = relu(res4b16_branch2a);
		
		MatrixXd res4b16_branch2b;
		double gemm_time_75;
		double offline_time_75;
		std::tie(res4b16_branch2b, gemm_time_75, offline_time_75) = convolve(res4b16_branch2a_relu, im_size_75, im_height_75, im_width_75, im_depth_75, k_size_75, stride_75, res4b16_branch2b_b, p1_75, p2_75, res4b16_branch2b_w, output_size_75, mode);
		
		MatrixXd res4b16_branch2b_relu = relu(res4b16_branch2b);
		
		MatrixXd res4b16_branch2c;
		double gemm_time_76;
		double offline_time_76;
		std::tie(res4b16_branch2c, gemm_time_76, offline_time_76) = convolve(res4b16_branch2b_relu, im_size_76, im_height_76, im_width_76, im_depth_76, k_size_76, stride_76, res4b16_branch2c_b, p1_76, p2_76, res4b16_branch2c_w, output_size_76, mode);
		
		MatrixXd res4b16 = eltwise(res4b16_branch2c, res4b15_relu);
		
		MatrixXd res4b16_relu = relu(res4b16);
		
		MatrixXd res4b17_branch2a;
		double gemm_time_77;
		double offline_time_77;
		std::tie(res4b17_branch2a, gemm_time_77, offline_time_77) = convolve(res4b16_relu, im_size_77, im_height_77, im_width_77, im_depth_77, k_size_77, stride_77, res4b17_branch2a_b, p1_77, p2_77, res4b17_branch2a_w, output_size_77, mode);
		
		MatrixXd res4b17_branch2a_relu = relu(res4b17_branch2a);
		
		MatrixXd res4b17_branch2b;
		double gemm_time_78;
		double offline_time_78;
		std::tie(res4b17_branch2b, gemm_time_78, offline_time_78) = convolve(res4b17_branch2a_relu, im_size_78, im_height_78, im_width_78, im_depth_78, k_size_78, stride_78, res4b17_branch2b_b, p1_78, p2_78, res4b17_branch2b_w, output_size_78, mode);
		
		MatrixXd res4b17_branch2b_relu = relu(res4b17_branch2b);
		
		MatrixXd res4b17_branch2c;
		double gemm_time_79;
		double offline_time_79;
		std::tie(res4b17_branch2c, gemm_time_79, offline_time_79) = convolve(res4b17_branch2b_relu, im_size_79, im_height_79, im_width_79, im_depth_79, k_size_79, stride_79, res4b17_branch2c_b, p1_79, p2_79, res4b17_branch2c_w, output_size_79, mode);
		
		MatrixXd res4b17 = eltwise(res4b17_branch2c, res4b16_relu);
		
		MatrixXd res4b17_relu = relu(res4b17);
		
		MatrixXd res4b18_branch2a;
		double gemm_time_80;
		double offline_time_80;
		std::tie(res4b18_branch2a, gemm_time_80, offline_time_80) = convolve(res4b17_relu, im_size_80, im_height_80, im_width_80, im_depth_80, k_size_80, stride_80, res4b18_branch2a_b, p1_80, p2_80, res4b18_branch2a_w, output_size_80, mode);
		
		MatrixXd res4b18_branch2a_relu = relu(res4b18_branch2a);
		
		MatrixXd res4b18_branch2b;
		double gemm_time_81;
		double offline_time_81;
		std::tie(res4b18_branch2b, gemm_time_81, offline_time_81) = convolve(res4b18_branch2a_relu, im_size_81, im_height_81, im_width_81, im_depth_81, k_size_81, stride_81, res4b18_branch2b_b, p1_81, p2_81, res4b18_branch2b_w, output_size_81, mode);
		
		MatrixXd res4b18_branch2b_relu = relu(res4b18_branch2b);
		
		MatrixXd res4b18_branch2c;
		double gemm_time_82;
		double offline_time_82;
		std::tie(res4b18_branch2c, gemm_time_82, offline_time_82) = convolve(res4b18_branch2b_relu, im_size_82, im_height_82, im_width_82, im_depth_82, k_size_82, stride_82, res4b18_branch2c_b, p1_82, p2_82, res4b18_branch2c_w, output_size_82, mode);
		
		MatrixXd res4b18 = eltwise(res4b18_branch2c, res4b17_relu);
		
		MatrixXd res4b18_relu = relu(res4b18);
		
		MatrixXd res4b19_branch2a;
		double gemm_time_83;
		double offline_time_83;
		std::tie(res4b19_branch2a, gemm_time_83, offline_time_83) = convolve(res4b18_relu, im_size_83, im_height_83, im_width_83, im_depth_83, k_size_83, stride_83, res4b19_branch2a_b, p1_83, p2_83, res4b19_branch2a_w, output_size_83, mode);
		
		MatrixXd res4b19_branch2a_relu = relu(res4b19_branch2a);
		
		MatrixXd res4b19_branch2b;
		double gemm_time_84;
		double offline_time_84;
		std::tie(res4b19_branch2b, gemm_time_84, offline_time_84) = convolve(res4b19_branch2a_relu, im_size_84, im_height_84, im_width_84, im_depth_84, k_size_84, stride_84, res4b19_branch2b_b, p1_84, p2_84, res4b19_branch2b_w, output_size_84, mode);
		
		MatrixXd res4b19_branch2b_relu = relu(res4b19_branch2b);
		
		MatrixXd res4b19_branch2c;
		double gemm_time_85;
		double offline_time_85;
		std::tie(res4b19_branch2c, gemm_time_85, offline_time_85) = convolve(res4b19_branch2b_relu, im_size_85, im_height_85, im_width_85, im_depth_85, k_size_85, stride_85, res4b19_branch2c_b, p1_85, p2_85, res4b19_branch2c_w, output_size_85, mode);
		
		MatrixXd res4b19 = eltwise(res4b19_branch2c, res4b18_relu);
		
		MatrixXd res4b19_relu = relu(res4b19);
		
		MatrixXd res4b20_branch2a;
		double gemm_time_86;
		double offline_time_86;
		std::tie(res4b20_branch2a, gemm_time_86, offline_time_86) = convolve(res4b19_relu, im_size_86, im_height_86, im_width_86, im_depth_86, k_size_86, stride_86, res4b20_branch2a_b, p1_86, p2_86, res4b20_branch2a_w, output_size_86, mode);
		
		MatrixXd res4b20_branch2a_relu = relu(res4b20_branch2a);
		
		MatrixXd res4b20_branch2b;
		double gemm_time_87;
		double offline_time_87;
		std::tie(res4b20_branch2b, gemm_time_87, offline_time_87) = convolve(res4b20_branch2a_relu, im_size_87, im_height_87, im_width_87, im_depth_87, k_size_87, stride_87, res4b20_branch2b_b, p1_87, p2_87, res4b20_branch2b_w, output_size_87, mode);
		
		MatrixXd res4b20_branch2b_relu = relu(res4b20_branch2b);
		
		MatrixXd res4b20_branch2c;
		double gemm_time_88;
		double offline_time_88;
		std::tie(res4b20_branch2c, gemm_time_88, offline_time_88) = convolve(res4b20_branch2b_relu, im_size_88, im_height_88, im_width_88, im_depth_88, k_size_88, stride_88, res4b20_branch2c_b, p1_88, p2_88, res4b20_branch2c_w, output_size_88, mode);
		
		MatrixXd res4b20 = eltwise(res4b20_branch2c, res4b19_relu);
		
		MatrixXd res4b20_relu = relu(res4b20);
		
		MatrixXd res4b21_branch2a;
		double gemm_time_89;
		double offline_time_89;
		std::tie(res4b21_branch2a, gemm_time_89, offline_time_89) = convolve(res4b20_relu, im_size_89, im_height_89, im_width_89, im_depth_89, k_size_89, stride_89, res4b21_branch2a_b, p1_89, p2_89, res4b21_branch2a_w, output_size_89, mode);
		
		MatrixXd res4b21_branch2a_relu = relu(res4b21_branch2a);
		
		MatrixXd res4b21_branch2b;
		double gemm_time_90;
		double offline_time_90;
		std::tie(res4b21_branch2b, gemm_time_90, offline_time_90) = convolve(res4b21_branch2a_relu, im_size_90, im_height_90, im_width_90, im_depth_90, k_size_90, stride_90, res4b21_branch2b_b, p1_90, p2_90, res4b21_branch2b_w, output_size_90, mode);
		
		MatrixXd res4b21_branch2b_relu = relu(res4b21_branch2b);
		
		MatrixXd res4b21_branch2c;
		double gemm_time_91;
		double offline_time_91;
		std::tie(res4b21_branch2c, gemm_time_91, offline_time_91) = convolve(res4b21_branch2b_relu, im_size_91, im_height_91, im_width_91, im_depth_91, k_size_91, stride_91, res4b21_branch2c_b, p1_91, p2_91, res4b21_branch2c_w, output_size_91, mode);
		
		MatrixXd res4b21 = eltwise(res4b21_branch2c, res4b20_relu);
		
		MatrixXd res4b21_relu = relu(res4b21);
		
		MatrixXd res4b22_branch2a;
		double gemm_time_92;
		double offline_time_92;
		std::tie(res4b22_branch2a, gemm_time_92, offline_time_92) = convolve(res4b21_relu, im_size_92, im_height_92, im_width_92, im_depth_92, k_size_92, stride_92, res4b22_branch2a_b, p1_92, p2_92, res4b22_branch2a_w, output_size_92, mode);
		
		MatrixXd res4b22_branch2a_relu = relu(res4b22_branch2a);
		
		MatrixXd res4b22_branch2b;
		double gemm_time_93;
		double offline_time_93;
		std::tie(res4b22_branch2b, gemm_time_93, offline_time_93) = convolve(res4b22_branch2a_relu, im_size_93, im_height_93, im_width_93, im_depth_93, k_size_93, stride_93, res4b22_branch2b_b, p1_93, p2_93, res4b22_branch2b_w, output_size_93, mode);
		
		MatrixXd res4b22_branch2b_relu = relu(res4b22_branch2b);
		
		MatrixXd res4b22_branch2c;
		double gemm_time_94;
		double offline_time_94;
		std::tie(res4b22_branch2c, gemm_time_94, offline_time_94) = convolve(res4b22_branch2b_relu, im_size_94, im_height_94, im_width_94, im_depth_94, k_size_94, stride_94, res4b22_branch2c_b, p1_94, p2_94, res4b22_branch2c_w, output_size_94, mode);
		
		MatrixXd res4b22 = eltwise(res4b22_branch2c, res4b21_relu);
		
		MatrixXd res4b22_relu = relu(res4b22);
		
		MatrixXd res5a_branch1;
		double gemm_time_95;
		double offline_time_95;
		std::tie(res5a_branch1, gemm_time_95, offline_time_95) = convolve(res4b22_relu, im_size_95, im_height_95, im_width_95, im_depth_95, k_size_95, stride_95, res5a_branch1_b, p1_95, p2_95, res5a_branch1_w, output_size_95, mode);
		
		MatrixXd res5a_branch2a;
		double gemm_time_96;
		double offline_time_96;
		std::tie(res5a_branch2a, gemm_time_96, offline_time_96) = convolve(res4b22_relu, im_size_96, im_height_96, im_width_96, im_depth_96, k_size_96, stride_96, res5a_branch2a_b, p1_96, p2_96, res5a_branch2a_w, output_size_96, mode);
		
		MatrixXd res5a_branch2a_relu = relu(res5a_branch2a);
		
		MatrixXd res5a_branch2b;
		double gemm_time_97;
		double offline_time_97;
		std::tie(res5a_branch2b, gemm_time_97, offline_time_97) = convolve(res5a_branch2a_relu, im_size_97, im_height_97, im_width_97, im_depth_97, k_size_97, stride_97, res5a_branch2b_b, p1_97, p2_97, res5a_branch2b_w, output_size_97, mode);
		
		MatrixXd res5a_branch2b_relu = relu(res5a_branch2b);
		
		MatrixXd res5a_branch2c;
		double gemm_time_98;
		double offline_time_98;
		std::tie(res5a_branch2c, gemm_time_98, offline_time_98) = convolve(res5a_branch2b_relu, im_size_98, im_height_98, im_width_98, im_depth_98, k_size_98, stride_98, res5a_branch2c_b, p1_98, p2_98, res5a_branch2c_w, output_size_98, mode);
		
		MatrixXd res5a = eltwise(res5a_branch2c, res5a_branch1);
		
		MatrixXd res5a_relu = relu(res5a);
		
		MatrixXd res5b_branch2a;
		double gemm_time_99;
		double offline_time_99;
		std::tie(res5b_branch2a, gemm_time_99, offline_time_99) = convolve(res5a_relu, im_size_99, im_height_99, im_width_99, im_depth_99, k_size_99, stride_99, res5b_branch2a_b, p1_99, p2_99, res5b_branch2a_w, output_size_99, mode);
		
		MatrixXd res5b_branch2a_relu = relu(res5b_branch2a);
		
		MatrixXd res5b_branch2b;
		double gemm_time_100;
		double offline_time_100;
		std::tie(res5b_branch2b, gemm_time_100, offline_time_100) = convolve(res5b_branch2a_relu, im_size_100, im_height_100, im_width_100, im_depth_100, k_size_100, stride_100, res5b_branch2b_b, p1_100, p2_100, res5b_branch2b_w, output_size_100, mode);
		
		MatrixXd res5b_branch2b_relu = relu(res5b_branch2b);
		
		MatrixXd res5b_branch2c;
		double gemm_time_101;
		double offline_time_101;
		std::tie(res5b_branch2c, gemm_time_101, offline_time_101) = convolve(res5b_branch2b_relu, im_size_101, im_height_101, im_width_101, im_depth_101, k_size_101, stride_101, res5b_branch2c_b, p1_101, p2_101, res5b_branch2c_w, output_size_101, mode);
		
		MatrixXd res5b = eltwise(res5b_branch2c, res5a_relu);
		
		MatrixXd res5b_relu = relu(res5b);
		
		MatrixXd res5c_branch2a;
		double gemm_time_102;
		double offline_time_102;
		std::tie(res5c_branch2a, gemm_time_102, offline_time_102) = convolve(res5b_relu, im_size_102, im_height_102, im_width_102, im_depth_102, k_size_102, stride_102, res5c_branch2a_b, p1_102, p2_102, res5c_branch2a_w, output_size_102, mode);
		
		MatrixXd res5c_branch2a_relu = relu(res5c_branch2a);
		
		MatrixXd res5c_branch2b;
		double gemm_time_103;
		double offline_time_103;
		std::tie(res5c_branch2b, gemm_time_103, offline_time_103) = convolve(res5c_branch2a_relu, im_size_103, im_height_103, im_width_103, im_depth_103, k_size_103, stride_103, res5c_branch2b_b, p1_103, p2_103, res5c_branch2b_w, output_size_103, mode);
		
		MatrixXd res5c_branch2b_relu = relu(res5c_branch2b);
		
		MatrixXd res5c_branch2c;
		double gemm_time_104;
		double offline_time_104;
		std::tie(res5c_branch2c, gemm_time_104, offline_time_104) = convolve(res5c_branch2b_relu, im_size_104, im_height_104, im_width_104, im_depth_104, k_size_104, stride_104, res5c_branch2c_b, p1_104, p2_104, res5c_branch2c_w, output_size_104, mode);
		
		MatrixXd res5c = eltwise(res5c_branch2c, res5b_relu);
		
		MatrixXd res5c_relu = relu(res5c);
		
		MatrixXd pool5 = pool(res5c_relu, f_2, s_2, output_width_104, output_height_104, pp1_2, pp2_2, mode_2);
		
		MatrixXd fc1000 = fully_connect(pool5, pool5.rows(), fc1000_weights, fc1000_b);
		
        clock_t run_time_end = clock();

        double run_time = (double) (run_time_end-run_time_start) / CLOCKS_PER_SEC;
		run_time_total += (run_time - offline_time_1 - offline_time_2 - offline_time_3 - offline_time_4 - offline_time_5 - offline_time_6 - offline_time_7 - offline_time_8 - offline_time_9 - offline_time_10 - offline_time_11 - offline_time_12 - offline_time_13 - offline_time_14 - offline_time_15 - offline_time_16 - offline_time_17 - offline_time_18 - offline_time_19 - offline_time_20 - offline_time_21 - offline_time_22 - offline_time_23 - offline_time_24 - offline_time_25 - offline_time_26 - offline_time_27 - offline_time_28 - offline_time_29 - offline_time_30 - offline_time_31 - offline_time_32 - offline_time_33 - offline_time_34 - offline_time_35 - offline_time_36 - offline_time_37 - offline_time_38 - offline_time_39 - offline_time_40 - offline_time_41 - offline_time_42 - offline_time_43 - offline_time_44 - offline_time_45 - offline_time_46 - offline_time_47 - offline_time_48 - offline_time_49 - offline_time_50 - offline_time_51 - offline_time_52 - offline_time_53 - offline_time_54 - offline_time_55 - offline_time_56 - offline_time_57 - offline_time_58 - offline_time_59 - offline_time_60 - offline_time_61 - offline_time_62 - offline_time_63 - offline_time_64 - offline_time_65 - offline_time_66 - offline_time_67 - offline_time_68 - offline_time_69 - offline_time_70 - offline_time_71 - offline_time_72 - offline_time_73 - offline_time_74 - offline_time_75 - offline_time_76 - offline_time_77 - offline_time_78 - offline_time_79 - offline_time_80 - offline_time_81 - offline_time_82 - offline_time_83 - offline_time_84 - offline_time_85 - offline_time_86 - offline_time_87 - offline_time_88 - offline_time_89 - offline_time_90 - offline_time_91 - offline_time_92 - offline_time_93 - offline_time_94 - offline_time_95 - offline_time_96 - offline_time_97 - offline_time_98 - offline_time_99 - offline_time_100 - offline_time_101 - offline_time_102 - offline_time_103 - offline_time_104);
		gemm_time_total += 0.0 + gemm_time_1 + gemm_time_2 + gemm_time_3 + gemm_time_4 + gemm_time_5 + gemm_time_6 + gemm_time_7 + gemm_time_8 + gemm_time_9 + gemm_time_10 + gemm_time_11 + gemm_time_12 + gemm_time_13 + gemm_time_14 + gemm_time_15 + gemm_time_16 + gemm_time_17 + gemm_time_18 + gemm_time_19 + gemm_time_20 + gemm_time_21 + gemm_time_22 + gemm_time_23 + gemm_time_24 + gemm_time_25 + gemm_time_26 + gemm_time_27 + gemm_time_28 + gemm_time_29 + gemm_time_30 + gemm_time_31 + gemm_time_32 + gemm_time_33 + gemm_time_34 + gemm_time_35 + gemm_time_36 + gemm_time_37 + gemm_time_38 + gemm_time_39 + gemm_time_40 + gemm_time_41 + gemm_time_42 + gemm_time_43 + gemm_time_44 + gemm_time_45 + gemm_time_46 + gemm_time_47 + gemm_time_48 + gemm_time_49 + gemm_time_50 + gemm_time_51 + gemm_time_52 + gemm_time_53 + gemm_time_54 + gemm_time_55 + gemm_time_56 + gemm_time_57 + gemm_time_58 + gemm_time_59 + gemm_time_60 + gemm_time_61 + gemm_time_62 + gemm_time_63 + gemm_time_64 + gemm_time_65 + gemm_time_66 + gemm_time_67 + gemm_time_68 + gemm_time_69 + gemm_time_70 + gemm_time_71 + gemm_time_72 + gemm_time_73 + gemm_time_74 + gemm_time_75 + gemm_time_76 + gemm_time_77 + gemm_time_78 + gemm_time_79 + gemm_time_80 + gemm_time_81 + gemm_time_82 + gemm_time_83 + gemm_time_84 + gemm_time_85 + gemm_time_86 + gemm_time_87 + gemm_time_88 + gemm_time_89 + gemm_time_90 + gemm_time_91 + gemm_time_92 + gemm_time_93 + gemm_time_94 + gemm_time_95 + gemm_time_96 + gemm_time_97 + gemm_time_98 + gemm_time_99 + gemm_time_100 + gemm_time_101 + gemm_time_102 + gemm_time_103 + gemm_time_104;
		
		std::string name_1 = "../features/ResNet101/" + mode + "/fc1000_" + std::to_string(i) + ".csv";
		write_to_csv(name_1, fc1000);
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
