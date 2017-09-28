#include "helper/reader.h"
#include "helper/input_parser.h"
#include "helper/writer.h"
#include "../layers/convolution_layer/convolution.h"
#include "../layers/pooling_layer/pooling.h"
#include "../layers/fully_connected_layer/fully_connected.h"
#include "../layers/relu_layer/relu.h"
#include "../layers/eltwise_layer/eltwise.h"
#include "../layers/scale_layer/scale.h"
#include "../layers/batchnorm_layer/batchnorm.h"

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
	
	const int output_height_1 = (im_height_1 + 2 * p1_1 - sqrt(k_size_1)) / stride_1 + 1;
	const int output_width_1 = (im_width_1 + 2 * p2_1 - sqrt(k_size_1)) / stride_1 + 1;
	const int output_size_1 = output_height_1 * output_width_1;
	
	MatrixXd conv1_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/conv1_weights.csv");
	Map<MatrixXd> conv1_w(conv1_weights.data(), k_num_1, k_size_1 * k_depth_1);
	const float conv1_min = conv1_w.minCoeff();
	const float conv1_max = conv1_w.maxCoeff();
	
	MatrixXd conv1_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float conv1_result_min = conv1_result_params(0, 0);
	const float conv1_result_max = conv1_result_params(0, 1);
	
	MatrixXd conv1_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/conv1_biases.csv");
	VectorXd conv1_b(Map<VectorXd>(conv1_biases.data(), conv1_biases.cols()*conv1_biases.rows()));
	
	MatrixXd bn_conv1_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn_conv1_mean.csv");
	MatrixXd bn_conv1_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn_conv1_var.csv");
	
	MatrixXd scale_conv1_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale_conv1_weights.csv");
	
	MatrixXd scale_conv1_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale_conv1_biases.csv");
	VectorXd scale_conv1_b(Map<VectorXd>(scale_conv1_biases.data(), scale_conv1_biases.cols()*scale_conv1_biases.rows()));
	
	const int f_1 = 3;
	const int s_1 = 2;
	std::string mode_1 = "max";
	
	const int pp1_1 = 0;
	const int pp2_1 = 0;
	
	const int im_height_2 = static_cast<int>(ceil(static_cast<float>(output_height_1 + 2 * pp1_1 - f_1 ) / s_1)) + 1;
	const int im_width_2 = static_cast<int>(ceil(static_cast<float>(output_width_1 + 2 * pp2_1 - f_1) / s_1)) + 1;
	const int im_depth_2 = k_num_1;
	const int im_size_2 = im_height_2 * im_width_2;
	
	const int k_num_2 = 256;
	const int k_size_2 = 1;
	const int stride_2 = 1;
	const int k_depth_2 = im_depth_2;
	
	const int p1_2 = 0;
	const int p2_2 = 0;
	
	const int output_height_2 = (im_height_2 + 2 * p1_2 - sqrt(k_size_2)) / stride_2 + 1;
	const int output_width_2 = (im_width_2 + 2 * p2_2 - sqrt(k_size_2)) / stride_2 + 1;
	const int output_size_2 = output_height_2 * output_width_2;
	
	MatrixXd res2a_branch1_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res2a_branch1_weights.csv");
	MatrixXd res2a_branch1_w = res2a_branch1_weights;
	const float res2a_branch1_min = res2a_branch1_w.minCoeff();
	const float res2a_branch1_max = res2a_branch1_w.maxCoeff();
	
	MatrixXd res2a_branch1_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res2a_branch1_result_min = res2a_branch1_result_params(1, 0);
	const float res2a_branch1_result_max = res2a_branch1_result_params(1, 1);
	
	MatrixXd res2a_branch1_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res2a_branch1_biases.csv");
	VectorXd res2a_branch1_b(Map<VectorXd>(res2a_branch1_biases.data(), res2a_branch1_biases.cols()*res2a_branch1_biases.rows()));
	
	MatrixXd bn2a_branch1_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2a_branch1_mean.csv");
	MatrixXd bn2a_branch1_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2a_branch1_var.csv");
	
	MatrixXd scale2a_branch1_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2a_branch1_weights.csv");
	
	MatrixXd scale2a_branch1_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2a_branch1_biases.csv");
	VectorXd scale2a_branch1_b(Map<VectorXd>(scale2a_branch1_biases.data(), scale2a_branch1_biases.cols()*scale2a_branch1_biases.rows()));
	
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
	
	const int output_height_3 = (im_height_3 + 2 * p1_3 - sqrt(k_size_3)) / stride_3 + 1;
	const int output_width_3 = (im_width_3 + 2 * p2_3 - sqrt(k_size_3)) / stride_3 + 1;
	const int output_size_3 = output_height_3 * output_width_3;
	
	MatrixXd res2a_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res2a_branch2a_weights.csv");
	MatrixXd res2a_branch2a_w = res2a_branch2a_weights;
	const float res2a_branch2a_min = res2a_branch2a_w.minCoeff();
	const float res2a_branch2a_max = res2a_branch2a_w.maxCoeff();
	
	MatrixXd res2a_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res2a_branch2a_biases.csv");
	VectorXd res2a_branch2a_b(Map<VectorXd>(res2a_branch2a_biases.data(), res2a_branch2a_biases.cols()*res2a_branch2a_biases.rows()));
	
	MatrixXd res2a_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res2a_branch2a_result_min = res2a_branch2a_result_params(2, 0);
	const float res2a_branch2a_result_max = res2a_branch2a_result_params(2, 1);
	
	MatrixXd bn2a_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2a_branch2a_mean.csv");
	MatrixXd bn2a_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2a_branch2a_var.csv");
	
	MatrixXd scale2a_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2a_branch2a_weights.csv");
	
	MatrixXd scale2a_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2a_branch2a_biases.csv");
	VectorXd scale2a_branch2a_b(Map<VectorXd>(scale2a_branch2a_biases.data(), scale2a_branch2a_biases.cols()*scale2a_branch2a_biases.rows()));
	
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
	
	const int output_height_4 = (im_height_4 + 2 * p1_4 - sqrt(k_size_4)) / stride_4 + 1;
	const int output_width_4 = (im_width_4 + 2 * p2_4 - sqrt(k_size_4)) / stride_4 + 1;
	const int output_size_4 = output_height_4 * output_width_4;
	
	MatrixXd res2a_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res2a_branch2b_weights.csv");
	MatrixXd res2a_branch2b_w = res2a_branch2b_weights;
	const float res2a_branch2b_min = res2a_branch2b_w.minCoeff();
	const float res2a_branch2b_max = res2a_branch2b_w.maxCoeff();
	
	MatrixXd res2a_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res2a_branch2b_result_min = res2a_branch2b_result_params(3, 0);
	const float res2a_branch2b_result_max = res2a_branch2b_result_params(3, 1);
	
	MatrixXd res2a_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res2a_branch2b_biases.csv");
	VectorXd res2a_branch2b_b(Map<VectorXd>(res2a_branch2b_biases.data(), res2a_branch2b_biases.cols()*res2a_branch2b_biases.rows()));
	
	MatrixXd bn2a_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2a_branch2b_mean.csv");
	MatrixXd bn2a_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2a_branch2b_var.csv");
	
	MatrixXd scale2a_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2a_branch2b_weights.csv");
	
	MatrixXd scale2a_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2a_branch2b_biases.csv");
	VectorXd scale2a_branch2b_b(Map<VectorXd>(scale2a_branch2b_biases.data(), scale2a_branch2b_biases.cols()*scale2a_branch2b_biases.rows()));
	
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
	
	const int output_height_5 = (im_height_5 + 2 * p1_5 - sqrt(k_size_5)) / stride_5 + 1;
	const int output_width_5 = (im_width_5 + 2 * p2_5 - sqrt(k_size_5)) / stride_5 + 1;
	const int output_size_5 = output_height_5 * output_width_5;
	
	MatrixXd res2a_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res2a_branch2c_weights.csv");
	MatrixXd res2a_branch2c_w = res2a_branch2c_weights;
	const float res2a_branch2c_min = res2a_branch2c_w.minCoeff();
	const float res2a_branch2c_max = res2a_branch2c_w.maxCoeff();
	
	MatrixXd res2a_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res2a_branch2c_result_min = res2a_branch2c_result_params(4, 0);
	const float res2a_branch2c_result_max = res2a_branch2c_result_params(4, 1);
	
	MatrixXd res2a_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res2a_branch2c_biases.csv");
	VectorXd res2a_branch2c_b(Map<VectorXd>(res2a_branch2c_biases.data(), res2a_branch2c_biases.cols()*res2a_branch2c_biases.rows()));
	
	MatrixXd bn2a_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2a_branch2c_mean.csv");
	MatrixXd bn2a_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2a_branch2c_var.csv");
	
	MatrixXd scale2a_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2a_branch2c_weights.csv");
	
	MatrixXd scale2a_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2a_branch2c_biases.csv");
	VectorXd scale2a_branch2c_b(Map<VectorXd>(scale2a_branch2c_biases.data(), scale2a_branch2c_biases.cols()*scale2a_branch2c_biases.rows()));
	
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
	
	const int output_height_6 = (im_height_6 + 2 * p1_6 - sqrt(k_size_6)) / stride_6 + 1;
	const int output_width_6 = (im_width_6 + 2 * p2_6 - sqrt(k_size_6)) / stride_6 + 1;
	const int output_size_6 = output_height_6 * output_width_6;
	
	MatrixXd res2b_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res2b_branch2a_weights.csv");
	MatrixXd res2b_branch2a_w = res2b_branch2a_weights;
	const float res2b_branch2a_min = res2b_branch2a_w.minCoeff();
	const float res2b_branch2a_max = res2b_branch2a_w.maxCoeff();
	
	MatrixXd res2b_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res2b_branch2a_result_min = res2b_branch2a_result_params(5, 0);
	const float res2b_branch2a_result_max = res2b_branch2a_result_params(5, 1);
	
	MatrixXd res2b_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res2b_branch2a_biases.csv");
	VectorXd res2b_branch2a_b(Map<VectorXd>(res2b_branch2a_biases.data(), res2b_branch2a_biases.cols()*res2b_branch2a_biases.rows()));
	
	MatrixXd bn2b_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2b_branch2a_mean.csv");
	MatrixXd bn2b_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2b_branch2a_var.csv");
	
	MatrixXd scale2b_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2b_branch2a_weights.csv");
	
	MatrixXd scale2b_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2b_branch2a_biases.csv");
	VectorXd scale2b_branch2a_b(Map<VectorXd>(scale2b_branch2a_biases.data(), scale2b_branch2a_biases.cols()*scale2b_branch2a_biases.rows()));
	
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
	
	const int output_height_7 = (im_height_7 + 2 * p1_7 - sqrt(k_size_7)) / stride_7 + 1;
	const int output_width_7 = (im_width_7 + 2 * p2_7 - sqrt(k_size_7)) / stride_7 + 1;
	const int output_size_7 = output_height_7 * output_width_7;
	
	MatrixXd res2b_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res2b_branch2b_weights.csv");
	MatrixXd res2b_branch2b_w = res2b_branch2b_weights;
	const float res2b_branch2b_min = res2b_branch2b_w.minCoeff();
	const float res2b_branch2b_max = res2b_branch2b_w.maxCoeff();
	
	MatrixXd res2b_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res2b_branch2b_result_min = res2b_branch2b_result_params(6, 0);
	const float res2b_branch2b_result_max = res2b_branch2b_result_params(6, 1);
	
	MatrixXd res2b_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res2b_branch2b_biases.csv");
	VectorXd res2b_branch2b_b(Map<VectorXd>(res2b_branch2b_biases.data(), res2b_branch2b_biases.cols()*res2b_branch2b_biases.rows()));
	
	MatrixXd bn2b_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2b_branch2b_mean.csv");
	MatrixXd bn2b_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2b_branch2b_var.csv");
	
	MatrixXd scale2b_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2b_branch2b_weights.csv");
	
	MatrixXd scale2b_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2b_branch2b_biases.csv");
	VectorXd scale2b_branch2b_b(Map<VectorXd>(scale2b_branch2b_biases.data(), scale2b_branch2b_biases.cols()*scale2b_branch2b_biases.rows()));
	
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
	
	const int output_height_8 = (im_height_8 + 2 * p1_8 - sqrt(k_size_8)) / stride_8 + 1;
	const int output_width_8 = (im_width_8 + 2 * p2_8 - sqrt(k_size_8)) / stride_8 + 1;
	const int output_size_8 = output_height_8 * output_width_8;
	
	MatrixXd res2b_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res2b_branch2c_weights.csv");
	MatrixXd res2b_branch2c_w = res2b_branch2c_weights;
	const float res2b_branch2c_min = res2b_branch2c_w.minCoeff();
	const float res2b_branch2c_max = res2b_branch2c_w.maxCoeff();
	
	MatrixXd res2b_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res2b_branch2c_result_min = res2b_branch2c_result_params(7, 0);
	const float res2b_branch2c_result_max = res2b_branch2c_result_params(7, 1);
	
	MatrixXd res2b_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res2b_branch2c_biases.csv");
	VectorXd res2b_branch2c_b(Map<VectorXd>(res2b_branch2c_biases.data(), res2b_branch2c_biases.cols()*res2b_branch2c_biases.rows()));
	
	MatrixXd bn2b_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2b_branch2c_mean.csv");
	MatrixXd bn2b_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2b_branch2c_var.csv");
	
	MatrixXd scale2b_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2b_branch2c_weights.csv");
	
	MatrixXd scale2b_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2b_branch2c_biases.csv");
	VectorXd scale2b_branch2c_b(Map<VectorXd>(scale2b_branch2c_biases.data(), scale2b_branch2c_biases.cols()*scale2b_branch2c_biases.rows()));
	
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
	
	const int output_height_9 = (im_height_9 + 2 * p1_9 - sqrt(k_size_9)) / stride_9 + 1;
	const int output_width_9 = (im_width_9 + 2 * p2_9 - sqrt(k_size_9)) / stride_9 + 1;
	const int output_size_9 = output_height_9 * output_width_9;
	
	MatrixXd res2c_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res2c_branch2a_weights.csv");
	MatrixXd res2c_branch2a_w = res2c_branch2a_weights;
	const float res2c_branch2a_min = res2c_branch2a_w.minCoeff();
	const float res2c_branch2a_max = res2c_branch2a_w.maxCoeff();
	
	MatrixXd res2c_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res2c_branch2a_result_min = res2c_branch2a_result_params(8, 0);
	const float res2c_branch2a_result_max = res2c_branch2a_result_params(8, 1);
	
	MatrixXd res2c_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res2c_branch2a_biases.csv");
	VectorXd res2c_branch2a_b(Map<VectorXd>(res2c_branch2a_biases.data(), res2c_branch2a_biases.cols()*res2c_branch2a_biases.rows()));
	
	MatrixXd bn2c_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2c_branch2a_mean.csv");
	MatrixXd bn2c_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2c_branch2a_var.csv");
	
	MatrixXd scale2c_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2c_branch2a_weights.csv");
	
	MatrixXd scale2c_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2c_branch2a_biases.csv");
	VectorXd scale2c_branch2a_b(Map<VectorXd>(scale2c_branch2a_biases.data(), scale2c_branch2a_biases.cols()*scale2c_branch2a_biases.rows()));
	
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
	
	const int output_height_10 = (im_height_10 + 2 * p1_10 - sqrt(k_size_10)) / stride_10 + 1;
	const int output_width_10 = (im_width_10 + 2 * p2_10 - sqrt(k_size_10)) / stride_10 + 1;
	const int output_size_10 = output_height_10 * output_width_10;
	
	MatrixXd res2c_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res2c_branch2b_weights.csv");
	MatrixXd res2c_branch2b_w = res2c_branch2b_weights;
	const float res2c_branch2b_min = res2c_branch2b_w.minCoeff();
	const float res2c_branch2b_max = res2c_branch2b_w.maxCoeff();
	
	MatrixXd res2c_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res2c_branch2b_result_min = res2c_branch2b_result_params(9, 0);
	const float res2c_branch2b_result_max = res2c_branch2b_result_params(9, 1);
	
	MatrixXd res2c_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res2c_branch2b_biases.csv");
	VectorXd res2c_branch2b_b(Map<VectorXd>(res2c_branch2b_biases.data(), res2c_branch2b_biases.cols()*res2c_branch2b_biases.rows()));
	
	MatrixXd bn2c_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2c_branch2b_mean.csv");
	MatrixXd bn2c_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2c_branch2b_var.csv");
	
	MatrixXd scale2c_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2c_branch2b_weights.csv");
	
	MatrixXd scale2c_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2c_branch2b_biases.csv");
	VectorXd scale2c_branch2b_b(Map<VectorXd>(scale2c_branch2b_biases.data(), scale2c_branch2b_biases.cols()*scale2c_branch2b_biases.rows()));
	
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
	
	const int output_height_11 = (im_height_11 + 2 * p1_11 - sqrt(k_size_11)) / stride_11 + 1;
	const int output_width_11 = (im_width_11 + 2 * p2_11 - sqrt(k_size_11)) / stride_11 + 1;
	const int output_size_11 = output_height_11 * output_width_11;
	
	MatrixXd res2c_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res2c_branch2c_weights.csv");
	MatrixXd res2c_branch2c_w = res2c_branch2c_weights;
	const float res2c_branch2c_min = res2c_branch2c_w.minCoeff();
	const float res2c_branch2c_max = res2c_branch2c_w.maxCoeff();
	
	MatrixXd res2c_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res2c_branch2c_result_min = res2c_branch2c_result_params(10, 0);
	const float res2c_branch2c_result_max = res2c_branch2c_result_params(10, 1);
	
	MatrixXd res2c_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res2c_branch2c_biases.csv");
	VectorXd res2c_branch2c_b(Map<VectorXd>(res2c_branch2c_biases.data(), res2c_branch2c_biases.cols()*res2c_branch2c_biases.rows()));
	
	MatrixXd bn2c_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2c_branch2c_mean.csv");
	MatrixXd bn2c_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn2c_branch2c_var.csv");
	
	MatrixXd scale2c_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2c_branch2c_weights.csv");
	
	MatrixXd scale2c_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale2c_branch2c_biases.csv");
	VectorXd scale2c_branch2c_b(Map<VectorXd>(scale2c_branch2c_biases.data(), scale2c_branch2c_biases.cols()*scale2c_branch2c_biases.rows()));
	
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
	
	const int output_height_12 = (im_height_12 + 2 * p1_12 - sqrt(k_size_12)) / stride_12 + 1;
	const int output_width_12 = (im_width_12 + 2 * p2_12 - sqrt(k_size_12)) / stride_12 + 1;
	const int output_size_12 = output_height_12 * output_width_12;
	
	MatrixXd res3a_branch1_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res3a_branch1_weights.csv");
	MatrixXd res3a_branch1_w = res3a_branch1_weights;
	const float res3a_branch1_min = res3a_branch1_w.minCoeff();
	const float res3a_branch1_max = res3a_branch1_w.maxCoeff();
	
	MatrixXd res3a_branch1_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res3a_branch1_result_min = res3a_branch1_result_params(11, 0);
	const float res3a_branch1_result_max = res3a_branch1_result_params(11, 1);
	
	MatrixXd res3a_branch1_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res3a_branch1_biases.csv");
	VectorXd res3a_branch1_b(Map<VectorXd>(res3a_branch1_biases.data(), res3a_branch1_biases.cols()*res3a_branch1_biases.rows()));
	
	MatrixXd bn3a_branch1_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3a_branch1_mean.csv");
	MatrixXd bn3a_branch1_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3a_branch1_var.csv");
	
	MatrixXd scale3a_branch1_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3a_branch1_weights.csv");
	
	MatrixXd scale3a_branch1_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3a_branch1_biases.csv");
	VectorXd scale3a_branch1_b(Map<VectorXd>(scale3a_branch1_biases.data(), scale3a_branch1_biases.cols()*scale3a_branch1_biases.rows()));
	
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
	
	const int output_height_13 = (im_height_13 + 2 * p1_13 - sqrt(k_size_13)) / stride_13 + 1;
	const int output_width_13 = (im_width_13 + 2 * p2_13 - sqrt(k_size_13)) / stride_13 + 1;
	const int output_size_13 = output_height_13 * output_width_13;
	
	MatrixXd res3a_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res3a_branch2a_weights.csv");
	MatrixXd res3a_branch2a_w = res3a_branch2a_weights;
	const float res3a_branch2a_min = res3a_branch2a_w.minCoeff();
	const float res3a_branch2a_max = res3a_branch2a_w.maxCoeff();
	
	MatrixXd res3a_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res3a_branch2a_biases.csv");
	VectorXd res3a_branch2a_b(Map<VectorXd>(res3a_branch2a_biases.data(), res3a_branch2a_biases.cols()*res3a_branch2a_biases.rows()));
	
	MatrixXd res3a_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res3a_branch2a_result_min = res3a_branch2a_result_params(12, 0);
	const float res3a_branch2a_result_max = res3a_branch2a_result_params(12, 1);
	
	MatrixXd bn3a_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3a_branch2a_mean.csv");
	MatrixXd bn3a_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3a_branch2a_var.csv");
	
	MatrixXd scale3a_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3a_branch2a_weights.csv");
	
	MatrixXd scale3a_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3a_branch2a_biases.csv");
	VectorXd scale3a_branch2a_b(Map<VectorXd>(scale3a_branch2a_biases.data(), scale3a_branch2a_biases.cols()*scale3a_branch2a_biases.rows()));
	
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
	
	const int output_height_14 = (im_height_14 + 2 * p1_14 - sqrt(k_size_14)) / stride_14 + 1;
	const int output_width_14 = (im_width_14 + 2 * p2_14 - sqrt(k_size_14)) / stride_14 + 1;
	const int output_size_14 = output_height_14 * output_width_14;
	
	MatrixXd res3a_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res3a_branch2b_weights.csv");
	MatrixXd res3a_branch2b_w = res3a_branch2b_weights;
	const float res3a_branch2b_min = res3a_branch2b_w.minCoeff();
	const float res3a_branch2b_max = res3a_branch2b_w.maxCoeff();
	
	MatrixXd res3a_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res3a_branch2b_result_min = res3a_branch2b_result_params(13, 0);
	const float res3a_branch2b_result_max = res3a_branch2b_result_params(13, 1);
	
	MatrixXd res3a_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res3a_branch2b_biases.csv");
	VectorXd res3a_branch2b_b(Map<VectorXd>(res3a_branch2b_biases.data(), res3a_branch2b_biases.cols()*res3a_branch2b_biases.rows()));
	
	MatrixXd bn3a_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3a_branch2b_mean.csv");
	MatrixXd bn3a_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3a_branch2b_var.csv");
	
	MatrixXd scale3a_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3a_branch2b_weights.csv");
	
	MatrixXd scale3a_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3a_branch2b_biases.csv");
	VectorXd scale3a_branch2b_b(Map<VectorXd>(scale3a_branch2b_biases.data(), scale3a_branch2b_biases.cols()*scale3a_branch2b_biases.rows()));
	
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
	
	const int output_height_15 = (im_height_15 + 2 * p1_15 - sqrt(k_size_15)) / stride_15 + 1;
	const int output_width_15 = (im_width_15 + 2 * p2_15 - sqrt(k_size_15)) / stride_15 + 1;
	const int output_size_15 = output_height_15 * output_width_15;
	
	MatrixXd res3a_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res3a_branch2c_weights.csv");
	MatrixXd res3a_branch2c_w = res3a_branch2c_weights;
	const float res3a_branch2c_min = res3a_branch2c_w.minCoeff();
	const float res3a_branch2c_max = res3a_branch2c_w.maxCoeff();
	
	MatrixXd res3a_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res3a_branch2c_result_min = res3a_branch2c_result_params(14, 0);
	const float res3a_branch2c_result_max = res3a_branch2c_result_params(14, 1);
	
	MatrixXd res3a_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res3a_branch2c_biases.csv");
	VectorXd res3a_branch2c_b(Map<VectorXd>(res3a_branch2c_biases.data(), res3a_branch2c_biases.cols()*res3a_branch2c_biases.rows()));
	
	MatrixXd bn3a_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3a_branch2c_mean.csv");
	MatrixXd bn3a_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3a_branch2c_var.csv");
	
	MatrixXd scale3a_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3a_branch2c_weights.csv");
	
	MatrixXd scale3a_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3a_branch2c_biases.csv");
	VectorXd scale3a_branch2c_b(Map<VectorXd>(scale3a_branch2c_biases.data(), scale3a_branch2c_biases.cols()*scale3a_branch2c_biases.rows()));
	
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
	
	const int output_height_16 = (im_height_16 + 2 * p1_16 - sqrt(k_size_16)) / stride_16 + 1;
	const int output_width_16 = (im_width_16 + 2 * p2_16 - sqrt(k_size_16)) / stride_16 + 1;
	const int output_size_16 = output_height_16 * output_width_16;
	
	MatrixXd res3b_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res3b_branch2a_weights.csv");
	MatrixXd res3b_branch2a_w = res3b_branch2a_weights;
	const float res3b_branch2a_min = res3b_branch2a_w.minCoeff();
	const float res3b_branch2a_max = res3b_branch2a_w.maxCoeff();
	
	MatrixXd res3b_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res3b_branch2a_result_min = res3b_branch2a_result_params(15, 0);
	const float res3b_branch2a_result_max = res3b_branch2a_result_params(15, 1);
	
	MatrixXd res3b_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res3b_branch2a_biases.csv");
	VectorXd res3b_branch2a_b(Map<VectorXd>(res3b_branch2a_biases.data(), res3b_branch2a_biases.cols()*res3b_branch2a_biases.rows()));
	
	MatrixXd bn3b_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3b_branch2a_mean.csv");
	MatrixXd bn3b_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3b_branch2a_var.csv");
	
	MatrixXd scale3b_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3b_branch2a_weights.csv");
	
	MatrixXd scale3b_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3b_branch2a_biases.csv");
	VectorXd scale3b_branch2a_b(Map<VectorXd>(scale3b_branch2a_biases.data(), scale3b_branch2a_biases.cols()*scale3b_branch2a_biases.rows()));
	
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
	
	const int output_height_17 = (im_height_17 + 2 * p1_17 - sqrt(k_size_17)) / stride_17 + 1;
	const int output_width_17 = (im_width_17 + 2 * p2_17 - sqrt(k_size_17)) / stride_17 + 1;
	const int output_size_17 = output_height_17 * output_width_17;
	
	MatrixXd res3b_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res3b_branch2b_weights.csv");
	MatrixXd res3b_branch2b_w = res3b_branch2b_weights;
	const float res3b_branch2b_min = res3b_branch2b_w.minCoeff();
	const float res3b_branch2b_max = res3b_branch2b_w.maxCoeff();
	
	MatrixXd res3b_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res3b_branch2b_result_min = res3b_branch2b_result_params(16, 0);
	const float res3b_branch2b_result_max = res3b_branch2b_result_params(16, 1);
	
	MatrixXd res3b_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res3b_branch2b_biases.csv");
	VectorXd res3b_branch2b_b(Map<VectorXd>(res3b_branch2b_biases.data(), res3b_branch2b_biases.cols()*res3b_branch2b_biases.rows()));
	
	MatrixXd bn3b_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3b_branch2b_mean.csv");
	MatrixXd bn3b_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3b_branch2b_var.csv");
	
	MatrixXd scale3b_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3b_branch2b_weights.csv");
	
	MatrixXd scale3b_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3b_branch2b_biases.csv");
	VectorXd scale3b_branch2b_b(Map<VectorXd>(scale3b_branch2b_biases.data(), scale3b_branch2b_biases.cols()*scale3b_branch2b_biases.rows()));
	
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
	
	const int output_height_18 = (im_height_18 + 2 * p1_18 - sqrt(k_size_18)) / stride_18 + 1;
	const int output_width_18 = (im_width_18 + 2 * p2_18 - sqrt(k_size_18)) / stride_18 + 1;
	const int output_size_18 = output_height_18 * output_width_18;
	
	MatrixXd res3b_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res3b_branch2c_weights.csv");
	MatrixXd res3b_branch2c_w = res3b_branch2c_weights;
	const float res3b_branch2c_min = res3b_branch2c_w.minCoeff();
	const float res3b_branch2c_max = res3b_branch2c_w.maxCoeff();
	
	MatrixXd res3b_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res3b_branch2c_result_min = res3b_branch2c_result_params(17, 0);
	const float res3b_branch2c_result_max = res3b_branch2c_result_params(17, 1);
	
	MatrixXd res3b_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res3b_branch2c_biases.csv");
	VectorXd res3b_branch2c_b(Map<VectorXd>(res3b_branch2c_biases.data(), res3b_branch2c_biases.cols()*res3b_branch2c_biases.rows()));
	
	MatrixXd bn3b_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3b_branch2c_mean.csv");
	MatrixXd bn3b_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3b_branch2c_var.csv");
	
	MatrixXd scale3b_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3b_branch2c_weights.csv");
	
	MatrixXd scale3b_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3b_branch2c_biases.csv");
	VectorXd scale3b_branch2c_b(Map<VectorXd>(scale3b_branch2c_biases.data(), scale3b_branch2c_biases.cols()*scale3b_branch2c_biases.rows()));
	
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
	
	const int output_height_19 = (im_height_19 + 2 * p1_19 - sqrt(k_size_19)) / stride_19 + 1;
	const int output_width_19 = (im_width_19 + 2 * p2_19 - sqrt(k_size_19)) / stride_19 + 1;
	const int output_size_19 = output_height_19 * output_width_19;
	
	MatrixXd res3c_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res3c_branch2a_weights.csv");
	MatrixXd res3c_branch2a_w = res3c_branch2a_weights;
	const float res3c_branch2a_min = res3c_branch2a_w.minCoeff();
	const float res3c_branch2a_max = res3c_branch2a_w.maxCoeff();
	
	MatrixXd res3c_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res3c_branch2a_result_min = res3c_branch2a_result_params(18, 0);
	const float res3c_branch2a_result_max = res3c_branch2a_result_params(18, 1);
	
	MatrixXd res3c_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res3c_branch2a_biases.csv");
	VectorXd res3c_branch2a_b(Map<VectorXd>(res3c_branch2a_biases.data(), res3c_branch2a_biases.cols()*res3c_branch2a_biases.rows()));
	
	MatrixXd bn3c_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3c_branch2a_mean.csv");
	MatrixXd bn3c_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3c_branch2a_var.csv");
	
	MatrixXd scale3c_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3c_branch2a_weights.csv");
	
	MatrixXd scale3c_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3c_branch2a_biases.csv");
	VectorXd scale3c_branch2a_b(Map<VectorXd>(scale3c_branch2a_biases.data(), scale3c_branch2a_biases.cols()*scale3c_branch2a_biases.rows()));
	
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
	
	const int output_height_20 = (im_height_20 + 2 * p1_20 - sqrt(k_size_20)) / stride_20 + 1;
	const int output_width_20 = (im_width_20 + 2 * p2_20 - sqrt(k_size_20)) / stride_20 + 1;
	const int output_size_20 = output_height_20 * output_width_20;
	
	MatrixXd res3c_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res3c_branch2b_weights.csv");
	MatrixXd res3c_branch2b_w = res3c_branch2b_weights;
	const float res3c_branch2b_min = res3c_branch2b_w.minCoeff();
	const float res3c_branch2b_max = res3c_branch2b_w.maxCoeff();
	
	MatrixXd res3c_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res3c_branch2b_result_min = res3c_branch2b_result_params(19, 0);
	const float res3c_branch2b_result_max = res3c_branch2b_result_params(19, 1);
	
	MatrixXd res3c_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res3c_branch2b_biases.csv");
	VectorXd res3c_branch2b_b(Map<VectorXd>(res3c_branch2b_biases.data(), res3c_branch2b_biases.cols()*res3c_branch2b_biases.rows()));
	
	MatrixXd bn3c_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3c_branch2b_mean.csv");
	MatrixXd bn3c_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3c_branch2b_var.csv");
	
	MatrixXd scale3c_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3c_branch2b_weights.csv");
	
	MatrixXd scale3c_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3c_branch2b_biases.csv");
	VectorXd scale3c_branch2b_b(Map<VectorXd>(scale3c_branch2b_biases.data(), scale3c_branch2b_biases.cols()*scale3c_branch2b_biases.rows()));
	
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
	
	const int output_height_21 = (im_height_21 + 2 * p1_21 - sqrt(k_size_21)) / stride_21 + 1;
	const int output_width_21 = (im_width_21 + 2 * p2_21 - sqrt(k_size_21)) / stride_21 + 1;
	const int output_size_21 = output_height_21 * output_width_21;
	
	MatrixXd res3c_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res3c_branch2c_weights.csv");
	MatrixXd res3c_branch2c_w = res3c_branch2c_weights;
	const float res3c_branch2c_min = res3c_branch2c_w.minCoeff();
	const float res3c_branch2c_max = res3c_branch2c_w.maxCoeff();
	
	MatrixXd res3c_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res3c_branch2c_result_min = res3c_branch2c_result_params(20, 0);
	const float res3c_branch2c_result_max = res3c_branch2c_result_params(20, 1);
	
	MatrixXd res3c_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res3c_branch2c_biases.csv");
	VectorXd res3c_branch2c_b(Map<VectorXd>(res3c_branch2c_biases.data(), res3c_branch2c_biases.cols()*res3c_branch2c_biases.rows()));
	
	MatrixXd bn3c_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3c_branch2c_mean.csv");
	MatrixXd bn3c_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3c_branch2c_var.csv");
	
	MatrixXd scale3c_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3c_branch2c_weights.csv");
	
	MatrixXd scale3c_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3c_branch2c_biases.csv");
	VectorXd scale3c_branch2c_b(Map<VectorXd>(scale3c_branch2c_biases.data(), scale3c_branch2c_biases.cols()*scale3c_branch2c_biases.rows()));
	
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
	
	const int output_height_22 = (im_height_22 + 2 * p1_22 - sqrt(k_size_22)) / stride_22 + 1;
	const int output_width_22 = (im_width_22 + 2 * p2_22 - sqrt(k_size_22)) / stride_22 + 1;
	const int output_size_22 = output_height_22 * output_width_22;
	
	MatrixXd res3d_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res3d_branch2a_weights.csv");
	MatrixXd res3d_branch2a_w = res3d_branch2a_weights;
	const float res3d_branch2a_min = res3d_branch2a_w.minCoeff();
	const float res3d_branch2a_max = res3d_branch2a_w.maxCoeff();
	
	MatrixXd res3d_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res3d_branch2a_result_min = res3d_branch2a_result_params(21, 0);
	const float res3d_branch2a_result_max = res3d_branch2a_result_params(21, 1);
	
	MatrixXd res3d_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res3d_branch2a_biases.csv");
	VectorXd res3d_branch2a_b(Map<VectorXd>(res3d_branch2a_biases.data(), res3d_branch2a_biases.cols()*res3d_branch2a_biases.rows()));
	
	MatrixXd bn3d_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3d_branch2a_mean.csv");
	MatrixXd bn3d_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3d_branch2a_var.csv");
	
	MatrixXd scale3d_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3d_branch2a_weights.csv");
	
	MatrixXd scale3d_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3d_branch2a_biases.csv");
	VectorXd scale3d_branch2a_b(Map<VectorXd>(scale3d_branch2a_biases.data(), scale3d_branch2a_biases.cols()*scale3d_branch2a_biases.rows()));
	
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
	
	const int output_height_23 = (im_height_23 + 2 * p1_23 - sqrt(k_size_23)) / stride_23 + 1;
	const int output_width_23 = (im_width_23 + 2 * p2_23 - sqrt(k_size_23)) / stride_23 + 1;
	const int output_size_23 = output_height_23 * output_width_23;
	
	MatrixXd res3d_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res3d_branch2b_weights.csv");
	MatrixXd res3d_branch2b_w = res3d_branch2b_weights;
	const float res3d_branch2b_min = res3d_branch2b_w.minCoeff();
	const float res3d_branch2b_max = res3d_branch2b_w.maxCoeff();
	
	MatrixXd res3d_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res3d_branch2b_result_min = res3d_branch2b_result_params(22, 0);
	const float res3d_branch2b_result_max = res3d_branch2b_result_params(22, 1);
	
	MatrixXd res3d_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res3d_branch2b_biases.csv");
	VectorXd res3d_branch2b_b(Map<VectorXd>(res3d_branch2b_biases.data(), res3d_branch2b_biases.cols()*res3d_branch2b_biases.rows()));
	
	MatrixXd bn3d_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3d_branch2b_mean.csv");
	MatrixXd bn3d_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3d_branch2b_var.csv");
	
	MatrixXd scale3d_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3d_branch2b_weights.csv");
	
	MatrixXd scale3d_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3d_branch2b_biases.csv");
	VectorXd scale3d_branch2b_b(Map<VectorXd>(scale3d_branch2b_biases.data(), scale3d_branch2b_biases.cols()*scale3d_branch2b_biases.rows()));
	
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
	
	const int output_height_24 = (im_height_24 + 2 * p1_24 - sqrt(k_size_24)) / stride_24 + 1;
	const int output_width_24 = (im_width_24 + 2 * p2_24 - sqrt(k_size_24)) / stride_24 + 1;
	const int output_size_24 = output_height_24 * output_width_24;
	
	MatrixXd res3d_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res3d_branch2c_weights.csv");
	MatrixXd res3d_branch2c_w = res3d_branch2c_weights;
	const float res3d_branch2c_min = res3d_branch2c_w.minCoeff();
	const float res3d_branch2c_max = res3d_branch2c_w.maxCoeff();
	
	MatrixXd res3d_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res3d_branch2c_result_min = res3d_branch2c_result_params(23, 0);
	const float res3d_branch2c_result_max = res3d_branch2c_result_params(23, 1);
	
	MatrixXd res3d_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res3d_branch2c_biases.csv");
	VectorXd res3d_branch2c_b(Map<VectorXd>(res3d_branch2c_biases.data(), res3d_branch2c_biases.cols()*res3d_branch2c_biases.rows()));
	
	MatrixXd bn3d_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3d_branch2c_mean.csv");
	MatrixXd bn3d_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn3d_branch2c_var.csv");
	
	MatrixXd scale3d_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3d_branch2c_weights.csv");
	
	MatrixXd scale3d_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale3d_branch2c_biases.csv");
	VectorXd scale3d_branch2c_b(Map<VectorXd>(scale3d_branch2c_biases.data(), scale3d_branch2c_biases.cols()*scale3d_branch2c_biases.rows()));
	
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
	
	const int output_height_25 = (im_height_25 + 2 * p1_25 - sqrt(k_size_25)) / stride_25 + 1;
	const int output_width_25 = (im_width_25 + 2 * p2_25 - sqrt(k_size_25)) / stride_25 + 1;
	const int output_size_25 = output_height_25 * output_width_25;
	
	MatrixXd res4a_branch1_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4a_branch1_weights.csv");
	MatrixXd res4a_branch1_w = res4a_branch1_weights;
	const float res4a_branch1_min = res4a_branch1_w.minCoeff();
	const float res4a_branch1_max = res4a_branch1_w.maxCoeff();
	
	MatrixXd res4a_branch1_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4a_branch1_result_min = res4a_branch1_result_params(24, 0);
	const float res4a_branch1_result_max = res4a_branch1_result_params(24, 1);
	
	MatrixXd res4a_branch1_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4a_branch1_biases.csv");
	VectorXd res4a_branch1_b(Map<VectorXd>(res4a_branch1_biases.data(), res4a_branch1_biases.cols()*res4a_branch1_biases.rows()));
	
	MatrixXd bn4a_branch1_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4a_branch1_mean.csv");
	MatrixXd bn4a_branch1_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4a_branch1_var.csv");
	
	MatrixXd scale4a_branch1_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4a_branch1_weights.csv");
	
	MatrixXd scale4a_branch1_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4a_branch1_biases.csv");
	VectorXd scale4a_branch1_b(Map<VectorXd>(scale4a_branch1_biases.data(), scale4a_branch1_biases.cols()*scale4a_branch1_biases.rows()));
	
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
	
	const int output_height_26 = (im_height_26 + 2 * p1_26 - sqrt(k_size_26)) / stride_26 + 1;
	const int output_width_26 = (im_width_26 + 2 * p2_26 - sqrt(k_size_26)) / stride_26 + 1;
	const int output_size_26 = output_height_26 * output_width_26;
	
	MatrixXd res4a_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4a_branch2a_weights.csv");
	MatrixXd res4a_branch2a_w = res4a_branch2a_weights;
	const float res4a_branch2a_min = res4a_branch2a_w.minCoeff();
	const float res4a_branch2a_max = res4a_branch2a_w.maxCoeff();
	
	MatrixXd res4a_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4a_branch2a_biases.csv");
	VectorXd res4a_branch2a_b(Map<VectorXd>(res4a_branch2a_biases.data(), res4a_branch2a_biases.cols()*res4a_branch2a_biases.rows()));
	
	MatrixXd res4a_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4a_branch2a_result_min = res4a_branch2a_result_params(25, 0);
	const float res4a_branch2a_result_max = res4a_branch2a_result_params(25, 1);
	
	MatrixXd bn4a_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4a_branch2a_mean.csv");
	MatrixXd bn4a_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4a_branch2a_var.csv");
	
	MatrixXd scale4a_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4a_branch2a_weights.csv");
	
	MatrixXd scale4a_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4a_branch2a_biases.csv");
	VectorXd scale4a_branch2a_b(Map<VectorXd>(scale4a_branch2a_biases.data(), scale4a_branch2a_biases.cols()*scale4a_branch2a_biases.rows()));
	
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
	
	const int output_height_27 = (im_height_27 + 2 * p1_27 - sqrt(k_size_27)) / stride_27 + 1;
	const int output_width_27 = (im_width_27 + 2 * p2_27 - sqrt(k_size_27)) / stride_27 + 1;
	const int output_size_27 = output_height_27 * output_width_27;
	
	MatrixXd res4a_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4a_branch2b_weights.csv");
	MatrixXd res4a_branch2b_w = res4a_branch2b_weights;
	const float res4a_branch2b_min = res4a_branch2b_w.minCoeff();
	const float res4a_branch2b_max = res4a_branch2b_w.maxCoeff();
	
	MatrixXd res4a_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4a_branch2b_result_min = res4a_branch2b_result_params(26, 0);
	const float res4a_branch2b_result_max = res4a_branch2b_result_params(26, 1);
	
	MatrixXd res4a_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4a_branch2b_biases.csv");
	VectorXd res4a_branch2b_b(Map<VectorXd>(res4a_branch2b_biases.data(), res4a_branch2b_biases.cols()*res4a_branch2b_biases.rows()));
	
	MatrixXd bn4a_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4a_branch2b_mean.csv");
	MatrixXd bn4a_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4a_branch2b_var.csv");
	
	MatrixXd scale4a_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4a_branch2b_weights.csv");
	
	MatrixXd scale4a_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4a_branch2b_biases.csv");
	VectorXd scale4a_branch2b_b(Map<VectorXd>(scale4a_branch2b_biases.data(), scale4a_branch2b_biases.cols()*scale4a_branch2b_biases.rows()));
	
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
	
	const int output_height_28 = (im_height_28 + 2 * p1_28 - sqrt(k_size_28)) / stride_28 + 1;
	const int output_width_28 = (im_width_28 + 2 * p2_28 - sqrt(k_size_28)) / stride_28 + 1;
	const int output_size_28 = output_height_28 * output_width_28;
	
	MatrixXd res4a_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4a_branch2c_weights.csv");
	MatrixXd res4a_branch2c_w = res4a_branch2c_weights;
	const float res4a_branch2c_min = res4a_branch2c_w.minCoeff();
	const float res4a_branch2c_max = res4a_branch2c_w.maxCoeff();
	
	MatrixXd res4a_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4a_branch2c_result_min = res4a_branch2c_result_params(27, 0);
	const float res4a_branch2c_result_max = res4a_branch2c_result_params(27, 1);
	
	MatrixXd res4a_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4a_branch2c_biases.csv");
	VectorXd res4a_branch2c_b(Map<VectorXd>(res4a_branch2c_biases.data(), res4a_branch2c_biases.cols()*res4a_branch2c_biases.rows()));
	
	MatrixXd bn4a_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4a_branch2c_mean.csv");
	MatrixXd bn4a_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4a_branch2c_var.csv");
	
	MatrixXd scale4a_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4a_branch2c_weights.csv");
	
	MatrixXd scale4a_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4a_branch2c_biases.csv");
	VectorXd scale4a_branch2c_b(Map<VectorXd>(scale4a_branch2c_biases.data(), scale4a_branch2c_biases.cols()*scale4a_branch2c_biases.rows()));
	
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
	
	const int output_height_29 = (im_height_29 + 2 * p1_29 - sqrt(k_size_29)) / stride_29 + 1;
	const int output_width_29 = (im_width_29 + 2 * p2_29 - sqrt(k_size_29)) / stride_29 + 1;
	const int output_size_29 = output_height_29 * output_width_29;
	
	MatrixXd res4b_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4b_branch2a_weights.csv");
	MatrixXd res4b_branch2a_w = res4b_branch2a_weights;
	const float res4b_branch2a_min = res4b_branch2a_w.minCoeff();
	const float res4b_branch2a_max = res4b_branch2a_w.maxCoeff();
	
	MatrixXd res4b_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4b_branch2a_result_min = res4b_branch2a_result_params(28, 0);
	const float res4b_branch2a_result_max = res4b_branch2a_result_params(28, 1);
	
	MatrixXd res4b_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4b_branch2a_biases.csv");
	VectorXd res4b_branch2a_b(Map<VectorXd>(res4b_branch2a_biases.data(), res4b_branch2a_biases.cols()*res4b_branch2a_biases.rows()));
	
	MatrixXd bn4b_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4b_branch2a_mean.csv");
	MatrixXd bn4b_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4b_branch2a_var.csv");
	
	MatrixXd scale4b_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4b_branch2a_weights.csv");
	
	MatrixXd scale4b_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4b_branch2a_biases.csv");
	VectorXd scale4b_branch2a_b(Map<VectorXd>(scale4b_branch2a_biases.data(), scale4b_branch2a_biases.cols()*scale4b_branch2a_biases.rows()));
	
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
	
	const int output_height_30 = (im_height_30 + 2 * p1_30 - sqrt(k_size_30)) / stride_30 + 1;
	const int output_width_30 = (im_width_30 + 2 * p2_30 - sqrt(k_size_30)) / stride_30 + 1;
	const int output_size_30 = output_height_30 * output_width_30;
	
	MatrixXd res4b_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4b_branch2b_weights.csv");
	MatrixXd res4b_branch2b_w = res4b_branch2b_weights;
	const float res4b_branch2b_min = res4b_branch2b_w.minCoeff();
	const float res4b_branch2b_max = res4b_branch2b_w.maxCoeff();
	
	MatrixXd res4b_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4b_branch2b_result_min = res4b_branch2b_result_params(29, 0);
	const float res4b_branch2b_result_max = res4b_branch2b_result_params(29, 1);
	
	MatrixXd res4b_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4b_branch2b_biases.csv");
	VectorXd res4b_branch2b_b(Map<VectorXd>(res4b_branch2b_biases.data(), res4b_branch2b_biases.cols()*res4b_branch2b_biases.rows()));
	
	MatrixXd bn4b_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4b_branch2b_mean.csv");
	MatrixXd bn4b_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4b_branch2b_var.csv");
	
	MatrixXd scale4b_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4b_branch2b_weights.csv");
	
	MatrixXd scale4b_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4b_branch2b_biases.csv");
	VectorXd scale4b_branch2b_b(Map<VectorXd>(scale4b_branch2b_biases.data(), scale4b_branch2b_biases.cols()*scale4b_branch2b_biases.rows()));
	
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
	
	const int output_height_31 = (im_height_31 + 2 * p1_31 - sqrt(k_size_31)) / stride_31 + 1;
	const int output_width_31 = (im_width_31 + 2 * p2_31 - sqrt(k_size_31)) / stride_31 + 1;
	const int output_size_31 = output_height_31 * output_width_31;
	
	MatrixXd res4b_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4b_branch2c_weights.csv");
	MatrixXd res4b_branch2c_w = res4b_branch2c_weights;
	const float res4b_branch2c_min = res4b_branch2c_w.minCoeff();
	const float res4b_branch2c_max = res4b_branch2c_w.maxCoeff();
	
	MatrixXd res4b_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4b_branch2c_result_min = res4b_branch2c_result_params(30, 0);
	const float res4b_branch2c_result_max = res4b_branch2c_result_params(30, 1);
	
	MatrixXd res4b_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4b_branch2c_biases.csv");
	VectorXd res4b_branch2c_b(Map<VectorXd>(res4b_branch2c_biases.data(), res4b_branch2c_biases.cols()*res4b_branch2c_biases.rows()));
	
	MatrixXd bn4b_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4b_branch2c_mean.csv");
	MatrixXd bn4b_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4b_branch2c_var.csv");
	
	MatrixXd scale4b_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4b_branch2c_weights.csv");
	
	MatrixXd scale4b_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4b_branch2c_biases.csv");
	VectorXd scale4b_branch2c_b(Map<VectorXd>(scale4b_branch2c_biases.data(), scale4b_branch2c_biases.cols()*scale4b_branch2c_biases.rows()));
	
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
	
	const int output_height_32 = (im_height_32 + 2 * p1_32 - sqrt(k_size_32)) / stride_32 + 1;
	const int output_width_32 = (im_width_32 + 2 * p2_32 - sqrt(k_size_32)) / stride_32 + 1;
	const int output_size_32 = output_height_32 * output_width_32;
	
	MatrixXd res4c_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4c_branch2a_weights.csv");
	MatrixXd res4c_branch2a_w = res4c_branch2a_weights;
	const float res4c_branch2a_min = res4c_branch2a_w.minCoeff();
	const float res4c_branch2a_max = res4c_branch2a_w.maxCoeff();
	
	MatrixXd res4c_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4c_branch2a_result_min = res4c_branch2a_result_params(31, 0);
	const float res4c_branch2a_result_max = res4c_branch2a_result_params(31, 1);
	
	MatrixXd res4c_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4c_branch2a_biases.csv");
	VectorXd res4c_branch2a_b(Map<VectorXd>(res4c_branch2a_biases.data(), res4c_branch2a_biases.cols()*res4c_branch2a_biases.rows()));
	
	MatrixXd bn4c_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4c_branch2a_mean.csv");
	MatrixXd bn4c_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4c_branch2a_var.csv");
	
	MatrixXd scale4c_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4c_branch2a_weights.csv");
	
	MatrixXd scale4c_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4c_branch2a_biases.csv");
	VectorXd scale4c_branch2a_b(Map<VectorXd>(scale4c_branch2a_biases.data(), scale4c_branch2a_biases.cols()*scale4c_branch2a_biases.rows()));
	
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
	
	const int output_height_33 = (im_height_33 + 2 * p1_33 - sqrt(k_size_33)) / stride_33 + 1;
	const int output_width_33 = (im_width_33 + 2 * p2_33 - sqrt(k_size_33)) / stride_33 + 1;
	const int output_size_33 = output_height_33 * output_width_33;
	
	MatrixXd res4c_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4c_branch2b_weights.csv");
	MatrixXd res4c_branch2b_w = res4c_branch2b_weights;
	const float res4c_branch2b_min = res4c_branch2b_w.minCoeff();
	const float res4c_branch2b_max = res4c_branch2b_w.maxCoeff();
	
	MatrixXd res4c_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4c_branch2b_result_min = res4c_branch2b_result_params(32, 0);
	const float res4c_branch2b_result_max = res4c_branch2b_result_params(32, 1);
	
	MatrixXd res4c_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4c_branch2b_biases.csv");
	VectorXd res4c_branch2b_b(Map<VectorXd>(res4c_branch2b_biases.data(), res4c_branch2b_biases.cols()*res4c_branch2b_biases.rows()));
	
	MatrixXd bn4c_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4c_branch2b_mean.csv");
	MatrixXd bn4c_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4c_branch2b_var.csv");
	
	MatrixXd scale4c_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4c_branch2b_weights.csv");
	
	MatrixXd scale4c_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4c_branch2b_biases.csv");
	VectorXd scale4c_branch2b_b(Map<VectorXd>(scale4c_branch2b_biases.data(), scale4c_branch2b_biases.cols()*scale4c_branch2b_biases.rows()));
	
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
	
	const int output_height_34 = (im_height_34 + 2 * p1_34 - sqrt(k_size_34)) / stride_34 + 1;
	const int output_width_34 = (im_width_34 + 2 * p2_34 - sqrt(k_size_34)) / stride_34 + 1;
	const int output_size_34 = output_height_34 * output_width_34;
	
	MatrixXd res4c_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4c_branch2c_weights.csv");
	MatrixXd res4c_branch2c_w = res4c_branch2c_weights;
	const float res4c_branch2c_min = res4c_branch2c_w.minCoeff();
	const float res4c_branch2c_max = res4c_branch2c_w.maxCoeff();
	
	MatrixXd res4c_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4c_branch2c_result_min = res4c_branch2c_result_params(33, 0);
	const float res4c_branch2c_result_max = res4c_branch2c_result_params(33, 1);
	
	MatrixXd res4c_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4c_branch2c_biases.csv");
	VectorXd res4c_branch2c_b(Map<VectorXd>(res4c_branch2c_biases.data(), res4c_branch2c_biases.cols()*res4c_branch2c_biases.rows()));
	
	MatrixXd bn4c_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4c_branch2c_mean.csv");
	MatrixXd bn4c_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4c_branch2c_var.csv");
	
	MatrixXd scale4c_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4c_branch2c_weights.csv");
	
	MatrixXd scale4c_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4c_branch2c_biases.csv");
	VectorXd scale4c_branch2c_b(Map<VectorXd>(scale4c_branch2c_biases.data(), scale4c_branch2c_biases.cols()*scale4c_branch2c_biases.rows()));
	
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
	
	const int output_height_35 = (im_height_35 + 2 * p1_35 - sqrt(k_size_35)) / stride_35 + 1;
	const int output_width_35 = (im_width_35 + 2 * p2_35 - sqrt(k_size_35)) / stride_35 + 1;
	const int output_size_35 = output_height_35 * output_width_35;
	
	MatrixXd res4d_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4d_branch2a_weights.csv");
	MatrixXd res4d_branch2a_w = res4d_branch2a_weights;
	const float res4d_branch2a_min = res4d_branch2a_w.minCoeff();
	const float res4d_branch2a_max = res4d_branch2a_w.maxCoeff();
	
	MatrixXd res4d_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4d_branch2a_result_min = res4d_branch2a_result_params(34, 0);
	const float res4d_branch2a_result_max = res4d_branch2a_result_params(34, 1);
	
	MatrixXd res4d_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4d_branch2a_biases.csv");
	VectorXd res4d_branch2a_b(Map<VectorXd>(res4d_branch2a_biases.data(), res4d_branch2a_biases.cols()*res4d_branch2a_biases.rows()));
	
	MatrixXd bn4d_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4d_branch2a_mean.csv");
	MatrixXd bn4d_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4d_branch2a_var.csv");
	
	MatrixXd scale4d_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4d_branch2a_weights.csv");
	
	MatrixXd scale4d_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4d_branch2a_biases.csv");
	VectorXd scale4d_branch2a_b(Map<VectorXd>(scale4d_branch2a_biases.data(), scale4d_branch2a_biases.cols()*scale4d_branch2a_biases.rows()));
	
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
	
	const int output_height_36 = (im_height_36 + 2 * p1_36 - sqrt(k_size_36)) / stride_36 + 1;
	const int output_width_36 = (im_width_36 + 2 * p2_36 - sqrt(k_size_36)) / stride_36 + 1;
	const int output_size_36 = output_height_36 * output_width_36;
	
	MatrixXd res4d_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4d_branch2b_weights.csv");
	MatrixXd res4d_branch2b_w = res4d_branch2b_weights;
	const float res4d_branch2b_min = res4d_branch2b_w.minCoeff();
	const float res4d_branch2b_max = res4d_branch2b_w.maxCoeff();
	
	MatrixXd res4d_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4d_branch2b_result_min = res4d_branch2b_result_params(35, 0);
	const float res4d_branch2b_result_max = res4d_branch2b_result_params(35, 1);
	
	MatrixXd res4d_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4d_branch2b_biases.csv");
	VectorXd res4d_branch2b_b(Map<VectorXd>(res4d_branch2b_biases.data(), res4d_branch2b_biases.cols()*res4d_branch2b_biases.rows()));
	
	MatrixXd bn4d_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4d_branch2b_mean.csv");
	MatrixXd bn4d_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4d_branch2b_var.csv");
	
	MatrixXd scale4d_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4d_branch2b_weights.csv");
	
	MatrixXd scale4d_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4d_branch2b_biases.csv");
	VectorXd scale4d_branch2b_b(Map<VectorXd>(scale4d_branch2b_biases.data(), scale4d_branch2b_biases.cols()*scale4d_branch2b_biases.rows()));
	
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
	
	const int output_height_37 = (im_height_37 + 2 * p1_37 - sqrt(k_size_37)) / stride_37 + 1;
	const int output_width_37 = (im_width_37 + 2 * p2_37 - sqrt(k_size_37)) / stride_37 + 1;
	const int output_size_37 = output_height_37 * output_width_37;
	
	MatrixXd res4d_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4d_branch2c_weights.csv");
	MatrixXd res4d_branch2c_w = res4d_branch2c_weights;
	const float res4d_branch2c_min = res4d_branch2c_w.minCoeff();
	const float res4d_branch2c_max = res4d_branch2c_w.maxCoeff();
	
	MatrixXd res4d_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4d_branch2c_result_min = res4d_branch2c_result_params(36, 0);
	const float res4d_branch2c_result_max = res4d_branch2c_result_params(36, 1);
	
	MatrixXd res4d_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4d_branch2c_biases.csv");
	VectorXd res4d_branch2c_b(Map<VectorXd>(res4d_branch2c_biases.data(), res4d_branch2c_biases.cols()*res4d_branch2c_biases.rows()));
	
	MatrixXd bn4d_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4d_branch2c_mean.csv");
	MatrixXd bn4d_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4d_branch2c_var.csv");
	
	MatrixXd scale4d_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4d_branch2c_weights.csv");
	
	MatrixXd scale4d_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4d_branch2c_biases.csv");
	VectorXd scale4d_branch2c_b(Map<VectorXd>(scale4d_branch2c_biases.data(), scale4d_branch2c_biases.cols()*scale4d_branch2c_biases.rows()));
	
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
	
	const int output_height_38 = (im_height_38 + 2 * p1_38 - sqrt(k_size_38)) / stride_38 + 1;
	const int output_width_38 = (im_width_38 + 2 * p2_38 - sqrt(k_size_38)) / stride_38 + 1;
	const int output_size_38 = output_height_38 * output_width_38;
	
	MatrixXd res4e_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4e_branch2a_weights.csv");
	MatrixXd res4e_branch2a_w = res4e_branch2a_weights;
	const float res4e_branch2a_min = res4e_branch2a_w.minCoeff();
	const float res4e_branch2a_max = res4e_branch2a_w.maxCoeff();
	
	MatrixXd res4e_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4e_branch2a_result_min = res4e_branch2a_result_params(37, 0);
	const float res4e_branch2a_result_max = res4e_branch2a_result_params(37, 1);
	
	MatrixXd res4e_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4e_branch2a_biases.csv");
	VectorXd res4e_branch2a_b(Map<VectorXd>(res4e_branch2a_biases.data(), res4e_branch2a_biases.cols()*res4e_branch2a_biases.rows()));
	
	MatrixXd bn4e_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4e_branch2a_mean.csv");
	MatrixXd bn4e_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4e_branch2a_var.csv");
	
	MatrixXd scale4e_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4e_branch2a_weights.csv");
	
	MatrixXd scale4e_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4e_branch2a_biases.csv");
	VectorXd scale4e_branch2a_b(Map<VectorXd>(scale4e_branch2a_biases.data(), scale4e_branch2a_biases.cols()*scale4e_branch2a_biases.rows()));
	
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
	
	const int output_height_39 = (im_height_39 + 2 * p1_39 - sqrt(k_size_39)) / stride_39 + 1;
	const int output_width_39 = (im_width_39 + 2 * p2_39 - sqrt(k_size_39)) / stride_39 + 1;
	const int output_size_39 = output_height_39 * output_width_39;
	
	MatrixXd res4e_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4e_branch2b_weights.csv");
	MatrixXd res4e_branch2b_w = res4e_branch2b_weights;
	const float res4e_branch2b_min = res4e_branch2b_w.minCoeff();
	const float res4e_branch2b_max = res4e_branch2b_w.maxCoeff();
	
	MatrixXd res4e_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4e_branch2b_result_min = res4e_branch2b_result_params(38, 0);
	const float res4e_branch2b_result_max = res4e_branch2b_result_params(38, 1);
	
	MatrixXd res4e_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4e_branch2b_biases.csv");
	VectorXd res4e_branch2b_b(Map<VectorXd>(res4e_branch2b_biases.data(), res4e_branch2b_biases.cols()*res4e_branch2b_biases.rows()));
	
	MatrixXd bn4e_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4e_branch2b_mean.csv");
	MatrixXd bn4e_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4e_branch2b_var.csv");
	
	MatrixXd scale4e_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4e_branch2b_weights.csv");
	
	MatrixXd scale4e_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4e_branch2b_biases.csv");
	VectorXd scale4e_branch2b_b(Map<VectorXd>(scale4e_branch2b_biases.data(), scale4e_branch2b_biases.cols()*scale4e_branch2b_biases.rows()));
	
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
	
	const int output_height_40 = (im_height_40 + 2 * p1_40 - sqrt(k_size_40)) / stride_40 + 1;
	const int output_width_40 = (im_width_40 + 2 * p2_40 - sqrt(k_size_40)) / stride_40 + 1;
	const int output_size_40 = output_height_40 * output_width_40;
	
	MatrixXd res4e_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4e_branch2c_weights.csv");
	MatrixXd res4e_branch2c_w = res4e_branch2c_weights;
	const float res4e_branch2c_min = res4e_branch2c_w.minCoeff();
	const float res4e_branch2c_max = res4e_branch2c_w.maxCoeff();
	
	MatrixXd res4e_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4e_branch2c_result_min = res4e_branch2c_result_params(39, 0);
	const float res4e_branch2c_result_max = res4e_branch2c_result_params(39, 1);
	
	MatrixXd res4e_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4e_branch2c_biases.csv");
	VectorXd res4e_branch2c_b(Map<VectorXd>(res4e_branch2c_biases.data(), res4e_branch2c_biases.cols()*res4e_branch2c_biases.rows()));
	
	MatrixXd bn4e_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4e_branch2c_mean.csv");
	MatrixXd bn4e_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4e_branch2c_var.csv");
	
	MatrixXd scale4e_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4e_branch2c_weights.csv");
	
	MatrixXd scale4e_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4e_branch2c_biases.csv");
	VectorXd scale4e_branch2c_b(Map<VectorXd>(scale4e_branch2c_biases.data(), scale4e_branch2c_biases.cols()*scale4e_branch2c_biases.rows()));
	
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
	
	const int output_height_41 = (im_height_41 + 2 * p1_41 - sqrt(k_size_41)) / stride_41 + 1;
	const int output_width_41 = (im_width_41 + 2 * p2_41 - sqrt(k_size_41)) / stride_41 + 1;
	const int output_size_41 = output_height_41 * output_width_41;
	
	MatrixXd res4f_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4f_branch2a_weights.csv");
	MatrixXd res4f_branch2a_w = res4f_branch2a_weights;
	const float res4f_branch2a_min = res4f_branch2a_w.minCoeff();
	const float res4f_branch2a_max = res4f_branch2a_w.maxCoeff();
	
	MatrixXd res4f_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4f_branch2a_result_min = res4f_branch2a_result_params(40, 0);
	const float res4f_branch2a_result_max = res4f_branch2a_result_params(40, 1);
	
	MatrixXd res4f_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4f_branch2a_biases.csv");
	VectorXd res4f_branch2a_b(Map<VectorXd>(res4f_branch2a_biases.data(), res4f_branch2a_biases.cols()*res4f_branch2a_biases.rows()));
	
	MatrixXd bn4f_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4f_branch2a_mean.csv");
	MatrixXd bn4f_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4f_branch2a_var.csv");
	
	MatrixXd scale4f_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4f_branch2a_weights.csv");
	
	MatrixXd scale4f_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4f_branch2a_biases.csv");
	VectorXd scale4f_branch2a_b(Map<VectorXd>(scale4f_branch2a_biases.data(), scale4f_branch2a_biases.cols()*scale4f_branch2a_biases.rows()));
	
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
	
	const int output_height_42 = (im_height_42 + 2 * p1_42 - sqrt(k_size_42)) / stride_42 + 1;
	const int output_width_42 = (im_width_42 + 2 * p2_42 - sqrt(k_size_42)) / stride_42 + 1;
	const int output_size_42 = output_height_42 * output_width_42;
	
	MatrixXd res4f_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4f_branch2b_weights.csv");
	MatrixXd res4f_branch2b_w = res4f_branch2b_weights;
	const float res4f_branch2b_min = res4f_branch2b_w.minCoeff();
	const float res4f_branch2b_max = res4f_branch2b_w.maxCoeff();
	
	MatrixXd res4f_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4f_branch2b_result_min = res4f_branch2b_result_params(41, 0);
	const float res4f_branch2b_result_max = res4f_branch2b_result_params(41, 1);
	
	MatrixXd res4f_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4f_branch2b_biases.csv");
	VectorXd res4f_branch2b_b(Map<VectorXd>(res4f_branch2b_biases.data(), res4f_branch2b_biases.cols()*res4f_branch2b_biases.rows()));
	
	MatrixXd bn4f_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4f_branch2b_mean.csv");
	MatrixXd bn4f_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4f_branch2b_var.csv");
	
	MatrixXd scale4f_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4f_branch2b_weights.csv");
	
	MatrixXd scale4f_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4f_branch2b_biases.csv");
	VectorXd scale4f_branch2b_b(Map<VectorXd>(scale4f_branch2b_biases.data(), scale4f_branch2b_biases.cols()*scale4f_branch2b_biases.rows()));
	
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
	
	const int output_height_43 = (im_height_43 + 2 * p1_43 - sqrt(k_size_43)) / stride_43 + 1;
	const int output_width_43 = (im_width_43 + 2 * p2_43 - sqrt(k_size_43)) / stride_43 + 1;
	const int output_size_43 = output_height_43 * output_width_43;
	
	MatrixXd res4f_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res4f_branch2c_weights.csv");
	MatrixXd res4f_branch2c_w = res4f_branch2c_weights;
	const float res4f_branch2c_min = res4f_branch2c_w.minCoeff();
	const float res4f_branch2c_max = res4f_branch2c_w.maxCoeff();
	
	MatrixXd res4f_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res4f_branch2c_result_min = res4f_branch2c_result_params(42, 0);
	const float res4f_branch2c_result_max = res4f_branch2c_result_params(42, 1);
	
	MatrixXd res4f_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res4f_branch2c_biases.csv");
	VectorXd res4f_branch2c_b(Map<VectorXd>(res4f_branch2c_biases.data(), res4f_branch2c_biases.cols()*res4f_branch2c_biases.rows()));
	
	MatrixXd bn4f_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4f_branch2c_mean.csv");
	MatrixXd bn4f_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn4f_branch2c_var.csv");
	
	MatrixXd scale4f_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4f_branch2c_weights.csv");
	
	MatrixXd scale4f_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale4f_branch2c_biases.csv");
	VectorXd scale4f_branch2c_b(Map<VectorXd>(scale4f_branch2c_biases.data(), scale4f_branch2c_biases.cols()*scale4f_branch2c_biases.rows()));
	
	const int im_height_44 = output_height_43;
	const int im_width_44 = output_width_43;
	const int im_depth_44 = k_num_43;
	const int im_size_44 = im_height_44 * im_width_44;
	
	const int k_num_44 = 2048;
	const int k_size_44 = 1;
	const int stride_44 = 2;
	const int k_depth_44 = im_depth_44;
	
	const int p1_44 = 0;
	const int p2_44 = 0;
	
	const int output_height_44 = (im_height_44 + 2 * p1_44 - sqrt(k_size_44)) / stride_44 + 1;
	const int output_width_44 = (im_width_44 + 2 * p2_44 - sqrt(k_size_44)) / stride_44 + 1;
	const int output_size_44 = output_height_44 * output_width_44;
	
	MatrixXd res5a_branch1_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res5a_branch1_weights.csv");
	MatrixXd res5a_branch1_w = res5a_branch1_weights;
	const float res5a_branch1_min = res5a_branch1_w.minCoeff();
	const float res5a_branch1_max = res5a_branch1_w.maxCoeff();
	
	MatrixXd res5a_branch1_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res5a_branch1_result_min = res5a_branch1_result_params(43, 0);
	const float res5a_branch1_result_max = res5a_branch1_result_params(43, 1);
	
	MatrixXd res5a_branch1_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res5a_branch1_biases.csv");
	VectorXd res5a_branch1_b(Map<VectorXd>(res5a_branch1_biases.data(), res5a_branch1_biases.cols()*res5a_branch1_biases.rows()));
	
	MatrixXd bn5a_branch1_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5a_branch1_mean.csv");
	MatrixXd bn5a_branch1_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5a_branch1_var.csv");
	
	MatrixXd scale5a_branch1_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5a_branch1_weights.csv");
	
	MatrixXd scale5a_branch1_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5a_branch1_biases.csv");
	VectorXd scale5a_branch1_b(Map<VectorXd>(scale5a_branch1_biases.data(), scale5a_branch1_biases.cols()*scale5a_branch1_biases.rows()));
	
	const int im_height_45 = im_height_44;
	const int im_width_45 = im_width_44;
	const int im_depth_45 = im_depth_44;
	const int im_size_45 = im_size_44;
	
	const int k_num_45 = 512;
	const int k_size_45 = 1;
	const int stride_45 = 2;
	const int k_depth_45 = im_depth_45;
	
	const int p1_45 = 0;
	const int p2_45 = 0;
	
	const int output_height_45 = (im_height_45 + 2 * p1_45 - sqrt(k_size_45)) / stride_45 + 1;
	const int output_width_45 = (im_width_45 + 2 * p2_45 - sqrt(k_size_45)) / stride_45 + 1;
	const int output_size_45 = output_height_45 * output_width_45;
	
	MatrixXd res5a_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res5a_branch2a_weights.csv");
	MatrixXd res5a_branch2a_w = res5a_branch2a_weights;
	const float res5a_branch2a_min = res5a_branch2a_w.minCoeff();
	const float res5a_branch2a_max = res5a_branch2a_w.maxCoeff();
	
	MatrixXd res5a_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res5a_branch2a_biases.csv");
	VectorXd res5a_branch2a_b(Map<VectorXd>(res5a_branch2a_biases.data(), res5a_branch2a_biases.cols()*res5a_branch2a_biases.rows()));
	
	MatrixXd res5a_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res5a_branch2a_result_min = res5a_branch2a_result_params(44, 0);
	const float res5a_branch2a_result_max = res5a_branch2a_result_params(44, 1);
	
	MatrixXd bn5a_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5a_branch2a_mean.csv");
	MatrixXd bn5a_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5a_branch2a_var.csv");
	
	MatrixXd scale5a_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5a_branch2a_weights.csv");
	
	MatrixXd scale5a_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5a_branch2a_biases.csv");
	VectorXd scale5a_branch2a_b(Map<VectorXd>(scale5a_branch2a_biases.data(), scale5a_branch2a_biases.cols()*scale5a_branch2a_biases.rows()));
	
	const int im_height_46 = output_height_45;
	const int im_width_46 = output_width_45;
	const int im_depth_46 = k_num_45;
	const int im_size_46 = im_height_46 * im_width_46;
	
	const int k_num_46 = 512;
	const int k_size_46 = 9;
	const int stride_46 = 1;
	const int k_depth_46 = im_depth_46;
	
	const int p1_46 = 1;
	const int p2_46 = 1;
	
	const int output_height_46 = (im_height_46 + 2 * p1_46 - sqrt(k_size_46)) / stride_46 + 1;
	const int output_width_46 = (im_width_46 + 2 * p2_46 - sqrt(k_size_46)) / stride_46 + 1;
	const int output_size_46 = output_height_46 * output_width_46;
	
	MatrixXd res5a_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res5a_branch2b_weights.csv");
	MatrixXd res5a_branch2b_w = res5a_branch2b_weights;
	const float res5a_branch2b_min = res5a_branch2b_w.minCoeff();
	const float res5a_branch2b_max = res5a_branch2b_w.maxCoeff();
	
	MatrixXd res5a_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res5a_branch2b_result_min = res5a_branch2b_result_params(45, 0);
	const float res5a_branch2b_result_max = res5a_branch2b_result_params(45, 1);
	
	MatrixXd res5a_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res5a_branch2b_biases.csv");
	VectorXd res5a_branch2b_b(Map<VectorXd>(res5a_branch2b_biases.data(), res5a_branch2b_biases.cols()*res5a_branch2b_biases.rows()));
	
	MatrixXd bn5a_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5a_branch2b_mean.csv");
	MatrixXd bn5a_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5a_branch2b_var.csv");
	
	MatrixXd scale5a_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5a_branch2b_weights.csv");
	
	MatrixXd scale5a_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5a_branch2b_biases.csv");
	VectorXd scale5a_branch2b_b(Map<VectorXd>(scale5a_branch2b_biases.data(), scale5a_branch2b_biases.cols()*scale5a_branch2b_biases.rows()));
	
	const int im_height_47 = output_height_46;
	const int im_width_47 = output_width_46;
	const int im_depth_47 = k_num_46;
	const int im_size_47 = im_height_47 * im_width_47;
	
	const int k_num_47 = 2048;
	const int k_size_47 = 1;
	const int stride_47 = 1;
	const int k_depth_47 = im_depth_47;
	
	const int p1_47 = 0;
	const int p2_47 = 0;
	
	const int output_height_47 = (im_height_47 + 2 * p1_47 - sqrt(k_size_47)) / stride_47 + 1;
	const int output_width_47 = (im_width_47 + 2 * p2_47 - sqrt(k_size_47)) / stride_47 + 1;
	const int output_size_47 = output_height_47 * output_width_47;
	
	MatrixXd res5a_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res5a_branch2c_weights.csv");
	MatrixXd res5a_branch2c_w = res5a_branch2c_weights;
	const float res5a_branch2c_min = res5a_branch2c_w.minCoeff();
	const float res5a_branch2c_max = res5a_branch2c_w.maxCoeff();
	
	MatrixXd res5a_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res5a_branch2c_result_min = res5a_branch2c_result_params(46, 0);
	const float res5a_branch2c_result_max = res5a_branch2c_result_params(46, 1);
	
	MatrixXd res5a_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res5a_branch2c_biases.csv");
	VectorXd res5a_branch2c_b(Map<VectorXd>(res5a_branch2c_biases.data(), res5a_branch2c_biases.cols()*res5a_branch2c_biases.rows()));
	
	MatrixXd bn5a_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5a_branch2c_mean.csv");
	MatrixXd bn5a_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5a_branch2c_var.csv");
	
	MatrixXd scale5a_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5a_branch2c_weights.csv");
	
	MatrixXd scale5a_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5a_branch2c_biases.csv");
	VectorXd scale5a_branch2c_b(Map<VectorXd>(scale5a_branch2c_biases.data(), scale5a_branch2c_biases.cols()*scale5a_branch2c_biases.rows()));
	
	const int im_height_48 = output_height_47;
	const int im_width_48 = output_width_47;
	const int im_depth_48 = k_num_47;
	const int im_size_48 = im_height_48 * im_width_48;
	
	const int k_num_48 = 512;
	const int k_size_48 = 1;
	const int stride_48 = 1;
	const int k_depth_48 = im_depth_48;
	
	const int p1_48 = 0;
	const int p2_48 = 0;
	
	const int output_height_48 = (im_height_48 + 2 * p1_48 - sqrt(k_size_48)) / stride_48 + 1;
	const int output_width_48 = (im_width_48 + 2 * p2_48 - sqrt(k_size_48)) / stride_48 + 1;
	const int output_size_48 = output_height_48 * output_width_48;
	
	MatrixXd res5b_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res5b_branch2a_weights.csv");
	MatrixXd res5b_branch2a_w = res5b_branch2a_weights;
	const float res5b_branch2a_min = res5b_branch2a_w.minCoeff();
	const float res5b_branch2a_max = res5b_branch2a_w.maxCoeff();
	
	MatrixXd res5b_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res5b_branch2a_result_min = res5b_branch2a_result_params(47, 0);
	const float res5b_branch2a_result_max = res5b_branch2a_result_params(47, 1);
	
	MatrixXd res5b_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res5b_branch2a_biases.csv");
	VectorXd res5b_branch2a_b(Map<VectorXd>(res5b_branch2a_biases.data(), res5b_branch2a_biases.cols()*res5b_branch2a_biases.rows()));
	
	MatrixXd bn5b_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5b_branch2a_mean.csv");
	MatrixXd bn5b_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5b_branch2a_var.csv");
	
	MatrixXd scale5b_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5b_branch2a_weights.csv");
	
	MatrixXd scale5b_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5b_branch2a_biases.csv");
	VectorXd scale5b_branch2a_b(Map<VectorXd>(scale5b_branch2a_biases.data(), scale5b_branch2a_biases.cols()*scale5b_branch2a_biases.rows()));
	
	const int im_height_49 = output_height_48;
	const int im_width_49 = output_width_48;
	const int im_depth_49 = k_num_48;
	const int im_size_49 = im_height_49 * im_width_49;
	
	const int k_num_49 = 512;
	const int k_size_49 = 9;
	const int stride_49 = 1;
	const int k_depth_49 = im_depth_49;
	
	const int p1_49 = 1;
	const int p2_49 = 1;
	
	const int output_height_49 = (im_height_49 + 2 * p1_49 - sqrt(k_size_49)) / stride_49 + 1;
	const int output_width_49 = (im_width_49 + 2 * p2_49 - sqrt(k_size_49)) / stride_49 + 1;
	const int output_size_49 = output_height_49 * output_width_49;
	
	MatrixXd res5b_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res5b_branch2b_weights.csv");
	MatrixXd res5b_branch2b_w = res5b_branch2b_weights;
	const float res5b_branch2b_min = res5b_branch2b_w.minCoeff();
	const float res5b_branch2b_max = res5b_branch2b_w.maxCoeff();
	
	MatrixXd res5b_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res5b_branch2b_result_min = res5b_branch2b_result_params(48, 0);
	const float res5b_branch2b_result_max = res5b_branch2b_result_params(48, 1);
	
	MatrixXd res5b_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res5b_branch2b_biases.csv");
	VectorXd res5b_branch2b_b(Map<VectorXd>(res5b_branch2b_biases.data(), res5b_branch2b_biases.cols()*res5b_branch2b_biases.rows()));
	
	MatrixXd bn5b_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5b_branch2b_mean.csv");
	MatrixXd bn5b_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5b_branch2b_var.csv");
	
	MatrixXd scale5b_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5b_branch2b_weights.csv");
	
	MatrixXd scale5b_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5b_branch2b_biases.csv");
	VectorXd scale5b_branch2b_b(Map<VectorXd>(scale5b_branch2b_biases.data(), scale5b_branch2b_biases.cols()*scale5b_branch2b_biases.rows()));
	
	const int im_height_50 = output_height_49;
	const int im_width_50 = output_width_49;
	const int im_depth_50 = k_num_49;
	const int im_size_50 = im_height_50 * im_width_50;
	
	const int k_num_50 = 2048;
	const int k_size_50 = 1;
	const int stride_50 = 1;
	const int k_depth_50 = im_depth_50;
	
	const int p1_50 = 0;
	const int p2_50 = 0;
	
	const int output_height_50 = (im_height_50 + 2 * p1_50 - sqrt(k_size_50)) / stride_50 + 1;
	const int output_width_50 = (im_width_50 + 2 * p2_50 - sqrt(k_size_50)) / stride_50 + 1;
	const int output_size_50 = output_height_50 * output_width_50;
	
	MatrixXd res5b_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res5b_branch2c_weights.csv");
	MatrixXd res5b_branch2c_w = res5b_branch2c_weights;
	const float res5b_branch2c_min = res5b_branch2c_w.minCoeff();
	const float res5b_branch2c_max = res5b_branch2c_w.maxCoeff();
	
	MatrixXd res5b_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res5b_branch2c_result_min = res5b_branch2c_result_params(49, 0);
	const float res5b_branch2c_result_max = res5b_branch2c_result_params(49, 1);
	
	MatrixXd res5b_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res5b_branch2c_biases.csv");
	VectorXd res5b_branch2c_b(Map<VectorXd>(res5b_branch2c_biases.data(), res5b_branch2c_biases.cols()*res5b_branch2c_biases.rows()));
	
	MatrixXd bn5b_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5b_branch2c_mean.csv");
	MatrixXd bn5b_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5b_branch2c_var.csv");
	
	MatrixXd scale5b_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5b_branch2c_weights.csv");
	
	MatrixXd scale5b_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5b_branch2c_biases.csv");
	VectorXd scale5b_branch2c_b(Map<VectorXd>(scale5b_branch2c_biases.data(), scale5b_branch2c_biases.cols()*scale5b_branch2c_biases.rows()));
	
	const int im_height_51 = output_height_50;
	const int im_width_51 = output_width_50;
	const int im_depth_51 = k_num_50;
	const int im_size_51 = im_height_51 * im_width_51;
	
	const int k_num_51 = 512;
	const int k_size_51 = 1;
	const int stride_51 = 1;
	const int k_depth_51 = im_depth_51;
	
	const int p1_51 = 0;
	const int p2_51 = 0;
	
	const int output_height_51 = (im_height_51 + 2 * p1_51 - sqrt(k_size_51)) / stride_51 + 1;
	const int output_width_51 = (im_width_51 + 2 * p2_51 - sqrt(k_size_51)) / stride_51 + 1;
	const int output_size_51 = output_height_51 * output_width_51;
	
	MatrixXd res5c_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res5c_branch2a_weights.csv");
	MatrixXd res5c_branch2a_w = res5c_branch2a_weights;
	const float res5c_branch2a_min = res5c_branch2a_w.minCoeff();
	const float res5c_branch2a_max = res5c_branch2a_w.maxCoeff();
	
	MatrixXd res5c_branch2a_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res5c_branch2a_result_min = res5c_branch2a_result_params(50, 0);
	const float res5c_branch2a_result_max = res5c_branch2a_result_params(50, 1);
	
	MatrixXd res5c_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res5c_branch2a_biases.csv");
	VectorXd res5c_branch2a_b(Map<VectorXd>(res5c_branch2a_biases.data(), res5c_branch2a_biases.cols()*res5c_branch2a_biases.rows()));
	
	MatrixXd bn5c_branch2a_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5c_branch2a_mean.csv");
	MatrixXd bn5c_branch2a_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5c_branch2a_var.csv");
	
	MatrixXd scale5c_branch2a_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5c_branch2a_weights.csv");
	
	MatrixXd scale5c_branch2a_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5c_branch2a_biases.csv");
	VectorXd scale5c_branch2a_b(Map<VectorXd>(scale5c_branch2a_biases.data(), scale5c_branch2a_biases.cols()*scale5c_branch2a_biases.rows()));
	
	const int im_height_52 = output_height_51;
	const int im_width_52 = output_width_51;
	const int im_depth_52 = k_num_51;
	const int im_size_52 = im_height_52 * im_width_52;
	
	const int k_num_52 = 512;
	const int k_size_52 = 9;
	const int stride_52 = 1;
	const int k_depth_52 = im_depth_52;
	
	const int p1_52 = 1;
	const int p2_52 = 1;
	
	const int output_height_52 = (im_height_52 + 2 * p1_52 - sqrt(k_size_52)) / stride_52 + 1;
	const int output_width_52 = (im_width_52 + 2 * p2_52 - sqrt(k_size_52)) / stride_52 + 1;
	const int output_size_52 = output_height_52 * output_width_52;
	
	MatrixXd res5c_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res5c_branch2b_weights.csv");
	MatrixXd res5c_branch2b_w = res5c_branch2b_weights;
	const float res5c_branch2b_min = res5c_branch2b_w.minCoeff();
	const float res5c_branch2b_max = res5c_branch2b_w.maxCoeff();
	
	MatrixXd res5c_branch2b_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res5c_branch2b_result_min = res5c_branch2b_result_params(51, 0);
	const float res5c_branch2b_result_max = res5c_branch2b_result_params(51, 1);
	
	MatrixXd res5c_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res5c_branch2b_biases.csv");
	VectorXd res5c_branch2b_b(Map<VectorXd>(res5c_branch2b_biases.data(), res5c_branch2b_biases.cols()*res5c_branch2b_biases.rows()));
	
	MatrixXd bn5c_branch2b_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5c_branch2b_mean.csv");
	MatrixXd bn5c_branch2b_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5c_branch2b_var.csv");
	
	MatrixXd scale5c_branch2b_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5c_branch2b_weights.csv");
	
	MatrixXd scale5c_branch2b_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5c_branch2b_biases.csv");
	VectorXd scale5c_branch2b_b(Map<VectorXd>(scale5c_branch2b_biases.data(), scale5c_branch2b_biases.cols()*scale5c_branch2b_biases.rows()));
	
	const int im_height_53 = output_height_52;
	const int im_width_53 = output_width_52;
	const int im_depth_53 = k_num_52;
	const int im_size_53 = im_height_53 * im_width_53;
	
	const int k_num_53 = 2048;
	const int k_size_53 = 1;
	const int stride_53 = 1;
	const int k_depth_53 = im_depth_53;
	
	const int p1_53 = 0;
	const int p2_53 = 0;
	
	const int output_height_53 = (im_height_53 + 2 * p1_53 - sqrt(k_size_53)) / stride_53 + 1;
	const int output_width_53 = (im_width_53 + 2 * p2_53 - sqrt(k_size_53)) / stride_53 + 1;
	const int output_size_53 = output_height_53 * output_width_53;
	
	MatrixXd res5c_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/res5c_branch2c_weights.csv");
	MatrixXd res5c_branch2c_w = res5c_branch2c_weights;
	const float res5c_branch2c_min = res5c_branch2c_w.minCoeff();
	const float res5c_branch2c_max = res5c_branch2c_w.maxCoeff();
	
	MatrixXd res5c_branch2c_result_params = load_csv_arma<MatrixXd>("../features/ResNet50/caffe/result_params.csv");
	const float res5c_branch2c_result_min = res5c_branch2c_result_params(52, 0);
	const float res5c_branch2c_result_max = res5c_branch2c_result_params(52, 1);
	
	MatrixXd res5c_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/res5c_branch2c_biases.csv");
	VectorXd res5c_branch2c_b(Map<VectorXd>(res5c_branch2c_biases.data(), res5c_branch2c_biases.cols()*res5c_branch2c_biases.rows()));
	
	MatrixXd bn5c_branch2c_mean = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5c_branch2c_mean.csv");
	MatrixXd bn5c_branch2c_var = load_csv_arma<MatrixXd>("../weights/ResNet50/bn5c_branch2c_var.csv");
	
	MatrixXd scale5c_branch2c_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5c_branch2c_weights.csv");
	
	MatrixXd scale5c_branch2c_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/scale5c_branch2c_biases.csv");
	VectorXd scale5c_branch2c_b(Map<VectorXd>(scale5c_branch2c_biases.data(), scale5c_branch2c_biases.cols()*scale5c_branch2c_biases.rows()));
	
	const int f_2 = 7;
	const int s_2 = 1;
	std::string mode_2 = "ave";
	
	const int pp1_2 = 0;
	const int pp2_2 = 0;
	
	MatrixXd fc1000_weights = load_csv_arma<MatrixXd>("../weights/ResNet50/fc1000_weights.csv");
	
	MatrixXd fc1000_biases = load_csv_arma<MatrixXd>("../weights/ResNet50/fc1000_biases.csv");
	VectorXd fc1000_b(Map<VectorXd>(fc1000_biases.data(), fc1000_biases.cols()*fc1000_biases.rows()));
	
	const int im_num = 1000;
	
	ifstream infile;
	infile.open("../inputs/ResNet50/production/imagenet_img_norm_1000.csv");
	
    for(int i=0; i < im_num; ++i)
    {
        cout << "image: " << i << endl;

		MatrixXd line = load_csv<MatrixXd>(infile);
		
        MatrixXd img;
        img = line.block<1,im_size_1*im_depth_1>(0,1);

        MatrixXd image = Map<Matrix<double, im_depth_1, im_size_1, RowMajor>>(img.data());

        clock_t run_time_start = clock();

		MatrixXd conv1;
		float gemm_time_1;
		float offline_time_1;
		std::tie(conv1, gemm_time_1, offline_time_1) = convolve(image, im_size_1, im_height_1, im_width_1, im_depth_1, k_size_1, stride_1, conv1_b, p1_1, p2_1, conv1_w, output_size_1, mode, conv1_min, conv1_max, conv1_result_min, conv1_result_max);
		
		MatrixXd bn_conv1 = batchnorm(conv1, bn_conv1_mean, bn_conv1_var);
		
		MatrixXd scale_conv1 = scale(bn_conv1, scale_conv1_weights, scale_conv1_b);
		
		MatrixXd conv1_relu = relu(scale_conv1);
		
		MatrixXd pool1 = pool(conv1_relu, f_1, s_1, output_width_1, output_height_1, pp1_1, pp2_1, mode_1);
		
		MatrixXd res2a_branch1;
		float gemm_time_2;
		float offline_time_2;
		std::tie(res2a_branch1, gemm_time_2, offline_time_2) = convolve(pool1, im_size_2, im_height_2, im_width_2, im_depth_2, k_size_2, stride_2, res2a_branch1_b, p1_2, p2_2, res2a_branch1_w, output_size_2, mode, res2a_branch1_min, res2a_branch1_max, res2a_branch1_result_min, res2a_branch1_result_max);
		
		MatrixXd bn2a_branch1 = batchnorm(res2a_branch1, bn2a_branch1_mean, bn2a_branch1_var);
		
		MatrixXd scale2a_branch1 = scale(bn2a_branch1, scale2a_branch1_weights, scale2a_branch1_b);
		
		MatrixXd res2a_branch2a;
		float gemm_time_3;
		float offline_time_3;
		std::tie(res2a_branch2a, gemm_time_3, offline_time_3) = convolve(pool1, im_size_3, im_height_3, im_width_3, im_depth_3, k_size_3, stride_3, res2a_branch2a_b, p1_3, p2_3, res2a_branch2a_w, output_size_3, mode, res2a_branch2a_min, res2a_branch2a_max, res2a_branch2a_result_min, res2a_branch2a_result_max);
		
		MatrixXd bn2a_branch2a = batchnorm(res2a_branch2a, bn2a_branch2a_mean, bn2a_branch2a_var);
		
		MatrixXd scale2a_branch2a = scale(bn2a_branch2a, scale2a_branch2a_weights, scale2a_branch2a_b);
		
		MatrixXd res2a_branch2a_relu = relu(scale2a_branch2a);
		
		MatrixXd res2a_branch2b;
		float gemm_time_4;
		float offline_time_4;
		std::tie(res2a_branch2b, gemm_time_4, offline_time_4) = convolve(res2a_branch2a_relu, im_size_4, im_height_4, im_width_4, im_depth_4, k_size_4, stride_4, res2a_branch2b_b, p1_4, p2_4, res2a_branch2b_w, output_size_4, mode, res2a_branch2b_min, res2a_branch2b_max, res2a_branch2b_result_min, res2a_branch2b_result_max);
		
		MatrixXd bn2a_branch2b = batchnorm(res2a_branch2b, bn2a_branch2b_mean, bn2a_branch2b_var);
		
		MatrixXd scale2a_branch2b = scale(bn2a_branch2b, scale2a_branch2b_weights, scale2a_branch2b_b);
		
		MatrixXd res2a_branch2b_relu = relu(scale2a_branch2b);
		
		MatrixXd res2a_branch2c;
		float gemm_time_5;
		float offline_time_5;
		std::tie(res2a_branch2c, gemm_time_5, offline_time_5) = convolve(res2a_branch2b_relu, im_size_5, im_height_5, im_width_5, im_depth_5, k_size_5, stride_5, res2a_branch2c_b, p1_5, p2_5, res2a_branch2c_w, output_size_5, mode, res2a_branch2c_min, res2a_branch2c_max, res2a_branch2c_result_min, res2a_branch2c_result_max);
		
		MatrixXd bn2a_branch2c = batchnorm(res2a_branch2c, bn2a_branch2c_mean, bn2a_branch2c_var);
		
		MatrixXd scale2a_branch2c = scale(bn2a_branch2c, scale2a_branch2c_weights, scale2a_branch2c_b);
		
		MatrixXd res2a = eltwise(scale2a_branch2c, scale2a_branch1);
		
		MatrixXd res2a_relu = relu(res2a);
		
		MatrixXd res2b_branch2a;
		float gemm_time_6;
		float offline_time_6;
		std::tie(res2b_branch2a, gemm_time_6, offline_time_6) = convolve(res2a_relu, im_size_6, im_height_6, im_width_6, im_depth_6, k_size_6, stride_6, res2b_branch2a_b, p1_6, p2_6, res2b_branch2a_w, output_size_6, mode, res2b_branch2a_min, res2b_branch2a_max, res2b_branch2a_result_min, res2b_branch2a_result_max);
		
		MatrixXd bn2b_branch2a = batchnorm(res2b_branch2a, bn2b_branch2a_mean, bn2b_branch2a_var);
		
		MatrixXd scale2b_branch2a = scale(bn2b_branch2a, scale2b_branch2a_weights, scale2b_branch2a_b);
		
		MatrixXd res2b_branch2a_relu = relu(scale2b_branch2a);
		
		MatrixXd res2b_branch2b;
		float gemm_time_7;
		float offline_time_7;
		std::tie(res2b_branch2b, gemm_time_7, offline_time_7) = convolve(res2b_branch2a_relu, im_size_7, im_height_7, im_width_7, im_depth_7, k_size_7, stride_7, res2b_branch2b_b, p1_7, p2_7, res2b_branch2b_w, output_size_7, mode, res2b_branch2b_min, res2b_branch2b_max, res2b_branch2b_result_min, res2b_branch2b_result_max);
		
		MatrixXd bn2b_branch2b = batchnorm(res2b_branch2b, bn2b_branch2b_mean, bn2b_branch2b_var);
		
		MatrixXd scale2b_branch2b = scale(bn2b_branch2b, scale2b_branch2b_weights, scale2b_branch2b_b);
		
		MatrixXd res2b_branch2b_relu = relu(scale2b_branch2b);
		
		MatrixXd res2b_branch2c;
		float gemm_time_8;
		float offline_time_8;
		std::tie(res2b_branch2c, gemm_time_8, offline_time_8) = convolve(res2b_branch2b_relu, im_size_8, im_height_8, im_width_8, im_depth_8, k_size_8, stride_8, res2b_branch2c_b, p1_8, p2_8, res2b_branch2c_w, output_size_8, mode, res2b_branch2c_min, res2b_branch2c_max, res2b_branch2c_result_min, res2b_branch2c_result_max);
		
		MatrixXd bn2b_branch2c = batchnorm(res2b_branch2c, bn2b_branch2c_mean, bn2b_branch2c_var);
		
		MatrixXd scale2b_branch2c = scale(bn2b_branch2c, scale2b_branch2c_weights, scale2b_branch2c_b);
		
		MatrixXd res2b = eltwise(scale2b_branch2c, res2a_relu);
		
		MatrixXd res2b_relu = relu(res2b);
		
		MatrixXd res2c_branch2a;
		float gemm_time_9;
		float offline_time_9;
		std::tie(res2c_branch2a, gemm_time_9, offline_time_9) = convolve(res2b_relu, im_size_9, im_height_9, im_width_9, im_depth_9, k_size_9, stride_9, res2c_branch2a_b, p1_9, p2_9, res2c_branch2a_w, output_size_9, mode, res2c_branch2a_min, res2c_branch2a_max, res2c_branch2a_result_min, res2c_branch2a_result_max);
		
		MatrixXd bn2c_branch2a = batchnorm(res2c_branch2a, bn2c_branch2a_mean, bn2c_branch2a_var);
		
		MatrixXd scale2c_branch2a = scale(bn2c_branch2a, scale2c_branch2a_weights, scale2c_branch2a_b);
		
		MatrixXd res2c_branch2a_relu = relu(scale2c_branch2a);
		
		MatrixXd res2c_branch2b;
		float gemm_time_10;
		float offline_time_10;
		std::tie(res2c_branch2b, gemm_time_10, offline_time_10) = convolve(res2c_branch2a_relu, im_size_10, im_height_10, im_width_10, im_depth_10, k_size_10, stride_10, res2c_branch2b_b, p1_10, p2_10, res2c_branch2b_w, output_size_10, mode, res2c_branch2b_min, res2c_branch2b_max, res2c_branch2b_result_min, res2c_branch2b_result_max);
		
		MatrixXd bn2c_branch2b = batchnorm(res2c_branch2b, bn2c_branch2b_mean, bn2c_branch2b_var);
		
		MatrixXd scale2c_branch2b = scale(bn2c_branch2b, scale2c_branch2b_weights, scale2c_branch2b_b);
		
		MatrixXd res2c_branch2b_relu = relu(scale2c_branch2b);
		
		MatrixXd res2c_branch2c;
		float gemm_time_11;
		float offline_time_11;
		std::tie(res2c_branch2c, gemm_time_11, offline_time_11) = convolve(res2c_branch2b_relu, im_size_11, im_height_11, im_width_11, im_depth_11, k_size_11, stride_11, res2c_branch2c_b, p1_11, p2_11, res2c_branch2c_w, output_size_11, mode, res2c_branch2c_min, res2c_branch2c_max, res2c_branch2c_result_min, res2c_branch2c_result_max);
		
		MatrixXd bn2c_branch2c = batchnorm(res2c_branch2c, bn2c_branch2c_mean, bn2c_branch2c_var);
		
		MatrixXd scale2c_branch2c = scale(bn2c_branch2c, scale2c_branch2c_weights, scale2c_branch2c_b);
		
		MatrixXd res2c = eltwise(scale2c_branch2c, res2b_relu);
		
		MatrixXd res2c_relu = relu(res2c);
		
		MatrixXd res3a_branch1;
		float gemm_time_12;
		float offline_time_12;
		std::tie(res3a_branch1, gemm_time_12, offline_time_12) = convolve(res2c_relu, im_size_12, im_height_12, im_width_12, im_depth_12, k_size_12, stride_12, res3a_branch1_b, p1_12, p2_12, res3a_branch1_w, output_size_12, mode, res3a_branch1_min, res3a_branch1_max, res3a_branch1_result_min, res3a_branch1_result_max);
		
		MatrixXd bn3a_branch1 = batchnorm(res3a_branch1, bn3a_branch1_mean, bn3a_branch1_var);
		
		MatrixXd scale3a_branch1 = scale(bn3a_branch1, scale3a_branch1_weights, scale3a_branch1_b);
		
		MatrixXd res3a_branch2a;
		float gemm_time_13;
		float offline_time_13;
		std::tie(res3a_branch2a, gemm_time_13, offline_time_13) = convolve(res2c_relu, im_size_13, im_height_13, im_width_13, im_depth_13, k_size_13, stride_13, res3a_branch2a_b, p1_13, p2_13, res3a_branch2a_w, output_size_13, mode, res3a_branch2a_min, res3a_branch2a_max, res3a_branch2a_result_min, res3a_branch2a_result_max);
		
		MatrixXd bn3a_branch2a = batchnorm(res3a_branch2a, bn3a_branch2a_mean, bn3a_branch2a_var);
		
		MatrixXd scale3a_branch2a = scale(bn3a_branch2a, scale3a_branch2a_weights, scale3a_branch2a_b);
		
		MatrixXd res3a_branch2a_relu = relu(scale3a_branch2a);
		
		MatrixXd res3a_branch2b;
		float gemm_time_14;
		float offline_time_14;
		std::tie(res3a_branch2b, gemm_time_14, offline_time_14) = convolve(res3a_branch2a_relu, im_size_14, im_height_14, im_width_14, im_depth_14, k_size_14, stride_14, res3a_branch2b_b, p1_14, p2_14, res3a_branch2b_w, output_size_14, mode, res3a_branch2b_min, res3a_branch2b_max, res3a_branch2b_result_min, res3a_branch2b_result_max);
		
		MatrixXd bn3a_branch2b = batchnorm(res3a_branch2b, bn3a_branch2b_mean, bn3a_branch2b_var);
		
		MatrixXd scale3a_branch2b = scale(bn3a_branch2b, scale3a_branch2b_weights, scale3a_branch2b_b);
		
		MatrixXd res3a_branch2b_relu = relu(scale3a_branch2b);
		
		MatrixXd res3a_branch2c;
		float gemm_time_15;
		float offline_time_15;
		std::tie(res3a_branch2c, gemm_time_15, offline_time_15) = convolve(res3a_branch2b_relu, im_size_15, im_height_15, im_width_15, im_depth_15, k_size_15, stride_15, res3a_branch2c_b, p1_15, p2_15, res3a_branch2c_w, output_size_15, mode, res3a_branch2c_min, res3a_branch2c_max, res3a_branch2c_result_min, res3a_branch2c_result_max);
		
		MatrixXd bn3a_branch2c = batchnorm(res3a_branch2c, bn3a_branch2c_mean, bn3a_branch2c_var);
		
		MatrixXd scale3a_branch2c = scale(bn3a_branch2c, scale3a_branch2c_weights, scale3a_branch2c_b);
		
		MatrixXd res3a = eltwise(scale3a_branch2c, scale3a_branch1);
		
		MatrixXd res3a_relu = relu(res3a);
		
		MatrixXd res3b_branch2a;
		float gemm_time_16;
		float offline_time_16;
		std::tie(res3b_branch2a, gemm_time_16, offline_time_16) = convolve(res3a_relu, im_size_16, im_height_16, im_width_16, im_depth_16, k_size_16, stride_16, res3b_branch2a_b, p1_16, p2_16, res3b_branch2a_w, output_size_16, mode, res3b_branch2a_min, res3b_branch2a_max, res3b_branch2a_result_min, res3b_branch2a_result_max);
		
		MatrixXd bn3b_branch2a = batchnorm(res3b_branch2a, bn3b_branch2a_mean, bn3b_branch2a_var);
		
		MatrixXd scale3b_branch2a = scale(bn3b_branch2a, scale3b_branch2a_weights, scale3b_branch2a_b);
		
		MatrixXd res3b_branch2a_relu = relu(scale3b_branch2a);
		
		MatrixXd res3b_branch2b;
		float gemm_time_17;
		float offline_time_17;
		std::tie(res3b_branch2b, gemm_time_17, offline_time_17) = convolve(res3b_branch2a_relu, im_size_17, im_height_17, im_width_17, im_depth_17, k_size_17, stride_17, res3b_branch2b_b, p1_17, p2_17, res3b_branch2b_w, output_size_17, mode, res3b_branch2b_min, res3b_branch2b_max, res3b_branch2b_result_min, res3b_branch2b_result_max);
		
		MatrixXd bn3b_branch2b = batchnorm(res3b_branch2b, bn3b_branch2b_mean, bn3b_branch2b_var);
		
		MatrixXd scale3b_branch2b = scale(bn3b_branch2b, scale3b_branch2b_weights, scale3b_branch2b_b);
		
		MatrixXd res3b_branch2b_relu = relu(scale3b_branch2b);
		
		MatrixXd res3b_branch2c;
		float gemm_time_18;
		float offline_time_18;
		std::tie(res3b_branch2c, gemm_time_18, offline_time_18) = convolve(res3b_branch2b_relu, im_size_18, im_height_18, im_width_18, im_depth_18, k_size_18, stride_18, res3b_branch2c_b, p1_18, p2_18, res3b_branch2c_w, output_size_18, mode, res3b_branch2c_min, res3b_branch2c_max, res3b_branch2c_result_min, res3b_branch2c_result_max);
		
		MatrixXd bn3b_branch2c = batchnorm(res3b_branch2c, bn3b_branch2c_mean, bn3b_branch2c_var);
		
		MatrixXd scale3b_branch2c = scale(bn3b_branch2c, scale3b_branch2c_weights, scale3b_branch2c_b);
		
		MatrixXd res3b = eltwise(scale3b_branch2c, res3a_relu);
		
		MatrixXd res3b_relu = relu(res3b);
		
		MatrixXd res3c_branch2a;
		float gemm_time_19;
		float offline_time_19;
		std::tie(res3c_branch2a, gemm_time_19, offline_time_19) = convolve(res3b_relu, im_size_19, im_height_19, im_width_19, im_depth_19, k_size_19, stride_19, res3c_branch2a_b, p1_19, p2_19, res3c_branch2a_w, output_size_19, mode, res3c_branch2a_min, res3c_branch2a_max, res3c_branch2a_result_min, res3c_branch2a_result_max);
		
		MatrixXd bn3c_branch2a = batchnorm(res3c_branch2a, bn3c_branch2a_mean, bn3c_branch2a_var);
		
		MatrixXd scale3c_branch2a = scale(bn3c_branch2a, scale3c_branch2a_weights, scale3c_branch2a_b);
		
		MatrixXd res3c_branch2a_relu = relu(scale3c_branch2a);
		
		MatrixXd res3c_branch2b;
		float gemm_time_20;
		float offline_time_20;
		std::tie(res3c_branch2b, gemm_time_20, offline_time_20) = convolve(res3c_branch2a_relu, im_size_20, im_height_20, im_width_20, im_depth_20, k_size_20, stride_20, res3c_branch2b_b, p1_20, p2_20, res3c_branch2b_w, output_size_20, mode, res3c_branch2b_min, res3c_branch2b_max, res3c_branch2b_result_min, res3c_branch2b_result_max);
		
		MatrixXd bn3c_branch2b = batchnorm(res3c_branch2b, bn3c_branch2b_mean, bn3c_branch2b_var);
		
		MatrixXd scale3c_branch2b = scale(bn3c_branch2b, scale3c_branch2b_weights, scale3c_branch2b_b);
		
		MatrixXd res3c_branch2b_relu = relu(scale3c_branch2b);
		
		MatrixXd res3c_branch2c;
		float gemm_time_21;
		float offline_time_21;
		std::tie(res3c_branch2c, gemm_time_21, offline_time_21) = convolve(res3c_branch2b_relu, im_size_21, im_height_21, im_width_21, im_depth_21, k_size_21, stride_21, res3c_branch2c_b, p1_21, p2_21, res3c_branch2c_w, output_size_21, mode, res3c_branch2c_min, res3c_branch2c_max, res3c_branch2c_result_min, res3c_branch2c_result_max);
		
		MatrixXd bn3c_branch2c = batchnorm(res3c_branch2c, bn3c_branch2c_mean, bn3c_branch2c_var);
		
		MatrixXd scale3c_branch2c = scale(bn3c_branch2c, scale3c_branch2c_weights, scale3c_branch2c_b);
		
		MatrixXd res3c = eltwise(scale3c_branch2c, res3b_relu);
		
		MatrixXd res3c_relu = relu(res3c);
		
		MatrixXd res3d_branch2a;
		float gemm_time_22;
		float offline_time_22;
		std::tie(res3d_branch2a, gemm_time_22, offline_time_22) = convolve(res3c_relu, im_size_22, im_height_22, im_width_22, im_depth_22, k_size_22, stride_22, res3d_branch2a_b, p1_22, p2_22, res3d_branch2a_w, output_size_22, mode, res3d_branch2a_min, res3d_branch2a_max, res3d_branch2a_result_min, res3d_branch2a_result_max);
		
		MatrixXd bn3d_branch2a = batchnorm(res3d_branch2a, bn3d_branch2a_mean, bn3d_branch2a_var);
		
		MatrixXd scale3d_branch2a = scale(bn3d_branch2a, scale3d_branch2a_weights, scale3d_branch2a_b);
		
		MatrixXd res3d_branch2a_relu = relu(scale3d_branch2a);
		
		MatrixXd res3d_branch2b;
		float gemm_time_23;
		float offline_time_23;
		std::tie(res3d_branch2b, gemm_time_23, offline_time_23) = convolve(res3d_branch2a_relu, im_size_23, im_height_23, im_width_23, im_depth_23, k_size_23, stride_23, res3d_branch2b_b, p1_23, p2_23, res3d_branch2b_w, output_size_23, mode, res3d_branch2b_min, res3d_branch2b_max, res3d_branch2b_result_min, res3d_branch2b_result_max);
		
		MatrixXd bn3d_branch2b = batchnorm(res3d_branch2b, bn3d_branch2b_mean, bn3d_branch2b_var);
		
		MatrixXd scale3d_branch2b = scale(bn3d_branch2b, scale3d_branch2b_weights, scale3d_branch2b_b);
		
		MatrixXd res3d_branch2b_relu = relu(scale3d_branch2b);
		
		MatrixXd res3d_branch2c;
		float gemm_time_24;
		float offline_time_24;
		std::tie(res3d_branch2c, gemm_time_24, offline_time_24) = convolve(res3d_branch2b_relu, im_size_24, im_height_24, im_width_24, im_depth_24, k_size_24, stride_24, res3d_branch2c_b, p1_24, p2_24, res3d_branch2c_w, output_size_24, mode, res3d_branch2c_min, res3d_branch2c_max, res3d_branch2c_result_min, res3d_branch2c_result_max);
		
		MatrixXd bn3d_branch2c = batchnorm(res3d_branch2c, bn3d_branch2c_mean, bn3d_branch2c_var);
		
		MatrixXd scale3d_branch2c = scale(bn3d_branch2c, scale3d_branch2c_weights, scale3d_branch2c_b);
		
		MatrixXd res3d = eltwise(scale3d_branch2c, res3c_relu);
		
		MatrixXd res3d_relu = relu(res3d);
		
		MatrixXd res4a_branch1;
		float gemm_time_25;
		float offline_time_25;
		std::tie(res4a_branch1, gemm_time_25, offline_time_25) = convolve(res3d_relu, im_size_25, im_height_25, im_width_25, im_depth_25, k_size_25, stride_25, res4a_branch1_b, p1_25, p2_25, res4a_branch1_w, output_size_25, mode, res4a_branch1_min, res4a_branch1_max, res4a_branch1_result_min, res4a_branch1_result_max);
		
		MatrixXd bn4a_branch1 = batchnorm(res4a_branch1, bn4a_branch1_mean, bn4a_branch1_var);
		
		MatrixXd scale4a_branch1 = scale(bn4a_branch1, scale4a_branch1_weights, scale4a_branch1_b);
		
		MatrixXd res4a_branch2a;
		float gemm_time_26;
		float offline_time_26;
		std::tie(res4a_branch2a, gemm_time_26, offline_time_26) = convolve(res3d_relu, im_size_26, im_height_26, im_width_26, im_depth_26, k_size_26, stride_26, res4a_branch2a_b, p1_26, p2_26, res4a_branch2a_w, output_size_26, mode, res4a_branch2a_min, res4a_branch2a_max, res4a_branch2a_result_min, res4a_branch2a_result_max);
		
		MatrixXd bn4a_branch2a = batchnorm(res4a_branch2a, bn4a_branch2a_mean, bn4a_branch2a_var);
		
		MatrixXd scale4a_branch2a = scale(bn4a_branch2a, scale4a_branch2a_weights, scale4a_branch2a_b);
		
		MatrixXd res4a_branch2a_relu = relu(scale4a_branch2a);
		
		MatrixXd res4a_branch2b;
		float gemm_time_27;
		float offline_time_27;
		std::tie(res4a_branch2b, gemm_time_27, offline_time_27) = convolve(res4a_branch2a_relu, im_size_27, im_height_27, im_width_27, im_depth_27, k_size_27, stride_27, res4a_branch2b_b, p1_27, p2_27, res4a_branch2b_w, output_size_27, mode, res4a_branch2b_min, res4a_branch2b_max, res4a_branch2b_result_min, res4a_branch2b_result_max);
		
		MatrixXd bn4a_branch2b = batchnorm(res4a_branch2b, bn4a_branch2b_mean, bn4a_branch2b_var);
		
		MatrixXd scale4a_branch2b = scale(bn4a_branch2b, scale4a_branch2b_weights, scale4a_branch2b_b);
		
		MatrixXd res4a_branch2b_relu = relu(scale4a_branch2b);
		
		MatrixXd res4a_branch2c;
		float gemm_time_28;
		float offline_time_28;
		std::tie(res4a_branch2c, gemm_time_28, offline_time_28) = convolve(res4a_branch2b_relu, im_size_28, im_height_28, im_width_28, im_depth_28, k_size_28, stride_28, res4a_branch2c_b, p1_28, p2_28, res4a_branch2c_w, output_size_28, mode, res4a_branch2c_min, res4a_branch2c_max, res4a_branch2c_result_min, res4a_branch2c_result_max);
		
		MatrixXd bn4a_branch2c = batchnorm(res4a_branch2c, bn4a_branch2c_mean, bn4a_branch2c_var);
		
		MatrixXd scale4a_branch2c = scale(bn4a_branch2c, scale4a_branch2c_weights, scale4a_branch2c_b);
		
		MatrixXd res4a = eltwise(scale4a_branch2c, scale4a_branch1);
		
		MatrixXd res4a_relu = relu(res4a);
		
		MatrixXd res4b_branch2a;
		float gemm_time_29;
		float offline_time_29;
		std::tie(res4b_branch2a, gemm_time_29, offline_time_29) = convolve(res4a_relu, im_size_29, im_height_29, im_width_29, im_depth_29, k_size_29, stride_29, res4b_branch2a_b, p1_29, p2_29, res4b_branch2a_w, output_size_29, mode, res4b_branch2a_min, res4b_branch2a_max, res4b_branch2a_result_min, res4b_branch2a_result_max);
		
		MatrixXd bn4b_branch2a = batchnorm(res4b_branch2a, bn4b_branch2a_mean, bn4b_branch2a_var);
		
		MatrixXd scale4b_branch2a = scale(bn4b_branch2a, scale4b_branch2a_weights, scale4b_branch2a_b);
		
		MatrixXd res4b_branch2a_relu = relu(scale4b_branch2a);
		
		MatrixXd res4b_branch2b;
		float gemm_time_30;
		float offline_time_30;
		std::tie(res4b_branch2b, gemm_time_30, offline_time_30) = convolve(res4b_branch2a_relu, im_size_30, im_height_30, im_width_30, im_depth_30, k_size_30, stride_30, res4b_branch2b_b, p1_30, p2_30, res4b_branch2b_w, output_size_30, mode, res4b_branch2b_min, res4b_branch2b_max, res4b_branch2b_result_min, res4b_branch2b_result_max);
		
		MatrixXd bn4b_branch2b = batchnorm(res4b_branch2b, bn4b_branch2b_mean, bn4b_branch2b_var);
		
		MatrixXd scale4b_branch2b = scale(bn4b_branch2b, scale4b_branch2b_weights, scale4b_branch2b_b);
		
		MatrixXd res4b_branch2b_relu = relu(scale4b_branch2b);
		
		MatrixXd res4b_branch2c;
		float gemm_time_31;
		float offline_time_31;
		std::tie(res4b_branch2c, gemm_time_31, offline_time_31) = convolve(res4b_branch2b_relu, im_size_31, im_height_31, im_width_31, im_depth_31, k_size_31, stride_31, res4b_branch2c_b, p1_31, p2_31, res4b_branch2c_w, output_size_31, mode, res4b_branch2c_min, res4b_branch2c_max, res4b_branch2c_result_min, res4b_branch2c_result_max);
		
		MatrixXd bn4b_branch2c = batchnorm(res4b_branch2c, bn4b_branch2c_mean, bn4b_branch2c_var);
		
		MatrixXd scale4b_branch2c = scale(bn4b_branch2c, scale4b_branch2c_weights, scale4b_branch2c_b);
		
		MatrixXd res4b = eltwise(scale4b_branch2c, res4a_relu);
		
		MatrixXd res4b_relu = relu(res4b);
		
		MatrixXd res4c_branch2a;
		float gemm_time_32;
		float offline_time_32;
		std::tie(res4c_branch2a, gemm_time_32, offline_time_32) = convolve(res4b_relu, im_size_32, im_height_32, im_width_32, im_depth_32, k_size_32, stride_32, res4c_branch2a_b, p1_32, p2_32, res4c_branch2a_w, output_size_32, mode, res4c_branch2a_min, res4c_branch2a_max, res4c_branch2a_result_min, res4c_branch2a_result_max);
		
		MatrixXd bn4c_branch2a = batchnorm(res4c_branch2a, bn4c_branch2a_mean, bn4c_branch2a_var);
		
		MatrixXd scale4c_branch2a = scale(bn4c_branch2a, scale4c_branch2a_weights, scale4c_branch2a_b);
		
		MatrixXd res4c_branch2a_relu = relu(scale4c_branch2a);
		
		MatrixXd res4c_branch2b;
		float gemm_time_33;
		float offline_time_33;
		std::tie(res4c_branch2b, gemm_time_33, offline_time_33) = convolve(res4c_branch2a_relu, im_size_33, im_height_33, im_width_33, im_depth_33, k_size_33, stride_33, res4c_branch2b_b, p1_33, p2_33, res4c_branch2b_w, output_size_33, mode, res4c_branch2b_min, res4c_branch2b_max, res4c_branch2b_result_min, res4c_branch2b_result_max);
		
		MatrixXd bn4c_branch2b = batchnorm(res4c_branch2b, bn4c_branch2b_mean, bn4c_branch2b_var);
		
		MatrixXd scale4c_branch2b = scale(bn4c_branch2b, scale4c_branch2b_weights, scale4c_branch2b_b);
		
		MatrixXd res4c_branch2b_relu = relu(scale4c_branch2b);
		
		MatrixXd res4c_branch2c;
		float gemm_time_34;
		float offline_time_34;
		std::tie(res4c_branch2c, gemm_time_34, offline_time_34) = convolve(res4c_branch2b_relu, im_size_34, im_height_34, im_width_34, im_depth_34, k_size_34, stride_34, res4c_branch2c_b, p1_34, p2_34, res4c_branch2c_w, output_size_34, mode, res4c_branch2c_min, res4c_branch2c_max, res4c_branch2c_result_min, res4c_branch2c_result_max);
		
		MatrixXd bn4c_branch2c = batchnorm(res4c_branch2c, bn4c_branch2c_mean, bn4c_branch2c_var);
		
		MatrixXd scale4c_branch2c = scale(bn4c_branch2c, scale4c_branch2c_weights, scale4c_branch2c_b);
		
		MatrixXd res4c = eltwise(scale4c_branch2c, res4b_relu);
		
		MatrixXd res4c_relu = relu(res4c);
		
		MatrixXd res4d_branch2a;
		float gemm_time_35;
		float offline_time_35;
		std::tie(res4d_branch2a, gemm_time_35, offline_time_35) = convolve(res4c_relu, im_size_35, im_height_35, im_width_35, im_depth_35, k_size_35, stride_35, res4d_branch2a_b, p1_35, p2_35, res4d_branch2a_w, output_size_35, mode, res4d_branch2a_min, res4d_branch2a_max, res4d_branch2a_result_min, res4d_branch2a_result_max);
		
		MatrixXd bn4d_branch2a = batchnorm(res4d_branch2a, bn4d_branch2a_mean, bn4d_branch2a_var);
		
		MatrixXd scale4d_branch2a = scale(bn4d_branch2a, scale4d_branch2a_weights, scale4d_branch2a_b);
		
		MatrixXd res4d_branch2a_relu = relu(scale4d_branch2a);
		
		MatrixXd res4d_branch2b;
		float gemm_time_36;
		float offline_time_36;
		std::tie(res4d_branch2b, gemm_time_36, offline_time_36) = convolve(res4d_branch2a_relu, im_size_36, im_height_36, im_width_36, im_depth_36, k_size_36, stride_36, res4d_branch2b_b, p1_36, p2_36, res4d_branch2b_w, output_size_36, mode, res4d_branch2b_min, res4d_branch2b_max, res4d_branch2b_result_min, res4d_branch2b_result_max);
		
		MatrixXd bn4d_branch2b = batchnorm(res4d_branch2b, bn4d_branch2b_mean, bn4d_branch2b_var);
		
		MatrixXd scale4d_branch2b = scale(bn4d_branch2b, scale4d_branch2b_weights, scale4d_branch2b_b);
		
		MatrixXd res4d_branch2b_relu = relu(scale4d_branch2b);
		
		MatrixXd res4d_branch2c;
		float gemm_time_37;
		float offline_time_37;
		std::tie(res4d_branch2c, gemm_time_37, offline_time_37) = convolve(res4d_branch2b_relu, im_size_37, im_height_37, im_width_37, im_depth_37, k_size_37, stride_37, res4d_branch2c_b, p1_37, p2_37, res4d_branch2c_w, output_size_37, mode, res4d_branch2c_min, res4d_branch2c_max, res4d_branch2c_result_min, res4d_branch2c_result_max);
		
		MatrixXd bn4d_branch2c = batchnorm(res4d_branch2c, bn4d_branch2c_mean, bn4d_branch2c_var);
		
		MatrixXd scale4d_branch2c = scale(bn4d_branch2c, scale4d_branch2c_weights, scale4d_branch2c_b);
		
		MatrixXd res4d = eltwise(scale4d_branch2c, res4c_relu);
		
		MatrixXd res4d_relu = relu(res4d);
		
		MatrixXd res4e_branch2a;
		float gemm_time_38;
		float offline_time_38;
		std::tie(res4e_branch2a, gemm_time_38, offline_time_38) = convolve(res4d_relu, im_size_38, im_height_38, im_width_38, im_depth_38, k_size_38, stride_38, res4e_branch2a_b, p1_38, p2_38, res4e_branch2a_w, output_size_38, mode, res4e_branch2a_min, res4e_branch2a_max, res4e_branch2a_result_min, res4e_branch2a_result_max);
		
		MatrixXd bn4e_branch2a = batchnorm(res4e_branch2a, bn4e_branch2a_mean, bn4e_branch2a_var);
		
		MatrixXd scale4e_branch2a = scale(bn4e_branch2a, scale4e_branch2a_weights, scale4e_branch2a_b);
		
		MatrixXd res4e_branch2a_relu = relu(scale4e_branch2a);
		
		MatrixXd res4e_branch2b;
		float gemm_time_39;
		float offline_time_39;
		std::tie(res4e_branch2b, gemm_time_39, offline_time_39) = convolve(res4e_branch2a_relu, im_size_39, im_height_39, im_width_39, im_depth_39, k_size_39, stride_39, res4e_branch2b_b, p1_39, p2_39, res4e_branch2b_w, output_size_39, mode, res4e_branch2b_min, res4e_branch2b_max, res4e_branch2b_result_min, res4e_branch2b_result_max);
		
		MatrixXd bn4e_branch2b = batchnorm(res4e_branch2b, bn4e_branch2b_mean, bn4e_branch2b_var);
		
		MatrixXd scale4e_branch2b = scale(bn4e_branch2b, scale4e_branch2b_weights, scale4e_branch2b_b);
		
		MatrixXd res4e_branch2b_relu = relu(scale4e_branch2b);
		
		MatrixXd res4e_branch2c;
		float gemm_time_40;
		float offline_time_40;
		std::tie(res4e_branch2c, gemm_time_40, offline_time_40) = convolve(res4e_branch2b_relu, im_size_40, im_height_40, im_width_40, im_depth_40, k_size_40, stride_40, res4e_branch2c_b, p1_40, p2_40, res4e_branch2c_w, output_size_40, mode, res4e_branch2c_min, res4e_branch2c_max, res4e_branch2c_result_min, res4e_branch2c_result_max);
		
		MatrixXd bn4e_branch2c = batchnorm(res4e_branch2c, bn4e_branch2c_mean, bn4e_branch2c_var);
		
		MatrixXd scale4e_branch2c = scale(bn4e_branch2c, scale4e_branch2c_weights, scale4e_branch2c_b);
		
		MatrixXd res4e = eltwise(scale4e_branch2c, res4d_relu);
		
		MatrixXd res4e_relu = relu(res4e);
		
		MatrixXd res4f_branch2a;
		float gemm_time_41;
		float offline_time_41;
		std::tie(res4f_branch2a, gemm_time_41, offline_time_41) = convolve(res4e_relu, im_size_41, im_height_41, im_width_41, im_depth_41, k_size_41, stride_41, res4f_branch2a_b, p1_41, p2_41, res4f_branch2a_w, output_size_41, mode, res4f_branch2a_min, res4f_branch2a_max, res4f_branch2a_result_min, res4f_branch2a_result_max);
		
		MatrixXd bn4f_branch2a = batchnorm(res4f_branch2a, bn4f_branch2a_mean, bn4f_branch2a_var);
		
		MatrixXd scale4f_branch2a = scale(bn4f_branch2a, scale4f_branch2a_weights, scale4f_branch2a_b);
		
		MatrixXd res4f_branch2a_relu = relu(scale4f_branch2a);
		
		MatrixXd res4f_branch2b;
		float gemm_time_42;
		float offline_time_42;
		std::tie(res4f_branch2b, gemm_time_42, offline_time_42) = convolve(res4f_branch2a_relu, im_size_42, im_height_42, im_width_42, im_depth_42, k_size_42, stride_42, res4f_branch2b_b, p1_42, p2_42, res4f_branch2b_w, output_size_42, mode, res4f_branch2b_min, res4f_branch2b_max, res4f_branch2b_result_min, res4f_branch2b_result_max);
		
		MatrixXd bn4f_branch2b = batchnorm(res4f_branch2b, bn4f_branch2b_mean, bn4f_branch2b_var);
		
		MatrixXd scale4f_branch2b = scale(bn4f_branch2b, scale4f_branch2b_weights, scale4f_branch2b_b);
		
		MatrixXd res4f_branch2b_relu = relu(scale4f_branch2b);
		
		MatrixXd res4f_branch2c;
		float gemm_time_43;
		float offline_time_43;
		std::tie(res4f_branch2c, gemm_time_43, offline_time_43) = convolve(res4f_branch2b_relu, im_size_43, im_height_43, im_width_43, im_depth_43, k_size_43, stride_43, res4f_branch2c_b, p1_43, p2_43, res4f_branch2c_w, output_size_43, mode, res4f_branch2c_min, res4f_branch2c_max, res4f_branch2c_result_min, res4f_branch2c_result_max);
		
		MatrixXd bn4f_branch2c = batchnorm(res4f_branch2c, bn4f_branch2c_mean, bn4f_branch2c_var);
		
		MatrixXd scale4f_branch2c = scale(bn4f_branch2c, scale4f_branch2c_weights, scale4f_branch2c_b);
		
		MatrixXd res4f = eltwise(scale4f_branch2c, res4e_relu);
		
		MatrixXd res4f_relu = relu(res4f);
		
		MatrixXd res5a_branch1;
		float gemm_time_44;
		float offline_time_44;
		std::tie(res5a_branch1, gemm_time_44, offline_time_44) = convolve(res4f_relu, im_size_44, im_height_44, im_width_44, im_depth_44, k_size_44, stride_44, res5a_branch1_b, p1_44, p2_44, res5a_branch1_w, output_size_44, mode, res5a_branch1_min, res5a_branch1_max, res5a_branch1_result_min, res5a_branch1_result_max);
		
		MatrixXd bn5a_branch1 = batchnorm(res5a_branch1, bn5a_branch1_mean, bn5a_branch1_var);
		
		MatrixXd scale5a_branch1 = scale(bn5a_branch1, scale5a_branch1_weights, scale5a_branch1_b);
		
		MatrixXd res5a_branch2a;
		float gemm_time_45;
		float offline_time_45;
		std::tie(res5a_branch2a, gemm_time_45, offline_time_45) = convolve(res4f_relu, im_size_45, im_height_45, im_width_45, im_depth_45, k_size_45, stride_45, res5a_branch2a_b, p1_45, p2_45, res5a_branch2a_w, output_size_45, mode, res5a_branch2a_min, res5a_branch2a_max, res5a_branch2a_result_min, res5a_branch2a_result_max);
		
		MatrixXd bn5a_branch2a = batchnorm(res5a_branch2a, bn5a_branch2a_mean, bn5a_branch2a_var);
		
		MatrixXd scale5a_branch2a = scale(bn5a_branch2a, scale5a_branch2a_weights, scale5a_branch2a_b);
		
		MatrixXd res5a_branch2a_relu = relu(scale5a_branch2a);
		
		MatrixXd res5a_branch2b;
		float gemm_time_46;
		float offline_time_46;
		std::tie(res5a_branch2b, gemm_time_46, offline_time_46) = convolve(res5a_branch2a_relu, im_size_46, im_height_46, im_width_46, im_depth_46, k_size_46, stride_46, res5a_branch2b_b, p1_46, p2_46, res5a_branch2b_w, output_size_46, mode, res5a_branch2b_min, res5a_branch2b_max, res5a_branch2b_result_min, res5a_branch2b_result_max);
		
		MatrixXd bn5a_branch2b = batchnorm(res5a_branch2b, bn5a_branch2b_mean, bn5a_branch2b_var);
		
		MatrixXd scale5a_branch2b = scale(bn5a_branch2b, scale5a_branch2b_weights, scale5a_branch2b_b);
		
		MatrixXd res5a_branch2b_relu = relu(scale5a_branch2b);
		
		MatrixXd res5a_branch2c;
		float gemm_time_47;
		float offline_time_47;
		std::tie(res5a_branch2c, gemm_time_47, offline_time_47) = convolve(res5a_branch2b_relu, im_size_47, im_height_47, im_width_47, im_depth_47, k_size_47, stride_47, res5a_branch2c_b, p1_47, p2_47, res5a_branch2c_w, output_size_47, mode, res5a_branch2c_min, res5a_branch2c_max, res5a_branch2c_result_min, res5a_branch2c_result_max);
		
		MatrixXd bn5a_branch2c = batchnorm(res5a_branch2c, bn5a_branch2c_mean, bn5a_branch2c_var);
		
		MatrixXd scale5a_branch2c = scale(bn5a_branch2c, scale5a_branch2c_weights, scale5a_branch2c_b);
		
		MatrixXd res5a = eltwise(scale5a_branch2c, scale5a_branch1);
		
		MatrixXd res5a_relu = relu(res5a);
		
		MatrixXd res5b_branch2a;
		float gemm_time_48;
		float offline_time_48;
		std::tie(res5b_branch2a, gemm_time_48, offline_time_48) = convolve(res5a_relu, im_size_48, im_height_48, im_width_48, im_depth_48, k_size_48, stride_48, res5b_branch2a_b, p1_48, p2_48, res5b_branch2a_w, output_size_48, mode, res5b_branch2a_min, res5b_branch2a_max, res5b_branch2a_result_min, res5b_branch2a_result_max);
		
		MatrixXd bn5b_branch2a = batchnorm(res5b_branch2a, bn5b_branch2a_mean, bn5b_branch2a_var);
		
		MatrixXd scale5b_branch2a = scale(bn5b_branch2a, scale5b_branch2a_weights, scale5b_branch2a_b);
		
		MatrixXd res5b_branch2a_relu = relu(scale5b_branch2a);
		
		MatrixXd res5b_branch2b;
		float gemm_time_49;
		float offline_time_49;
		std::tie(res5b_branch2b, gemm_time_49, offline_time_49) = convolve(res5b_branch2a_relu, im_size_49, im_height_49, im_width_49, im_depth_49, k_size_49, stride_49, res5b_branch2b_b, p1_49, p2_49, res5b_branch2b_w, output_size_49, mode, res5b_branch2b_min, res5b_branch2b_max, res5b_branch2b_result_min, res5b_branch2b_result_max);
		
		MatrixXd bn5b_branch2b = batchnorm(res5b_branch2b, bn5b_branch2b_mean, bn5b_branch2b_var);
		
		MatrixXd scale5b_branch2b = scale(bn5b_branch2b, scale5b_branch2b_weights, scale5b_branch2b_b);
		
		MatrixXd res5b_branch2b_relu = relu(scale5b_branch2b);
		
		MatrixXd res5b_branch2c;
		float gemm_time_50;
		float offline_time_50;
		std::tie(res5b_branch2c, gemm_time_50, offline_time_50) = convolve(res5b_branch2b_relu, im_size_50, im_height_50, im_width_50, im_depth_50, k_size_50, stride_50, res5b_branch2c_b, p1_50, p2_50, res5b_branch2c_w, output_size_50, mode, res5b_branch2c_min, res5b_branch2c_max, res5b_branch2c_result_min, res5b_branch2c_result_max);
		
		MatrixXd bn5b_branch2c = batchnorm(res5b_branch2c, bn5b_branch2c_mean, bn5b_branch2c_var);
		
		MatrixXd scale5b_branch2c = scale(bn5b_branch2c, scale5b_branch2c_weights, scale5b_branch2c_b);
		
		MatrixXd res5b = eltwise(scale5b_branch2c, res5a_relu);
		
		MatrixXd res5b_relu = relu(res5b);
		
		MatrixXd res5c_branch2a;
		float gemm_time_51;
		float offline_time_51;
		std::tie(res5c_branch2a, gemm_time_51, offline_time_51) = convolve(res5b_relu, im_size_51, im_height_51, im_width_51, im_depth_51, k_size_51, stride_51, res5c_branch2a_b, p1_51, p2_51, res5c_branch2a_w, output_size_51, mode, res5c_branch2a_min, res5c_branch2a_max, res5c_branch2a_result_min, res5c_branch2a_result_max);
		
		MatrixXd bn5c_branch2a = batchnorm(res5c_branch2a, bn5c_branch2a_mean, bn5c_branch2a_var);
		
		MatrixXd scale5c_branch2a = scale(bn5c_branch2a, scale5c_branch2a_weights, scale5c_branch2a_b);
		
		MatrixXd res5c_branch2a_relu = relu(scale5c_branch2a);
		
		MatrixXd res5c_branch2b;
		float gemm_time_52;
		float offline_time_52;
		std::tie(res5c_branch2b, gemm_time_52, offline_time_52) = convolve(res5c_branch2a_relu, im_size_52, im_height_52, im_width_52, im_depth_52, k_size_52, stride_52, res5c_branch2b_b, p1_52, p2_52, res5c_branch2b_w, output_size_52, mode, res5c_branch2b_min, res5c_branch2b_max, res5c_branch2b_result_min, res5c_branch2b_result_max);
		
		MatrixXd bn5c_branch2b = batchnorm(res5c_branch2b, bn5c_branch2b_mean, bn5c_branch2b_var);
		
		MatrixXd scale5c_branch2b = scale(bn5c_branch2b, scale5c_branch2b_weights, scale5c_branch2b_b);
		
		MatrixXd res5c_branch2b_relu = relu(scale5c_branch2b);
		
		MatrixXd res5c_branch2c;
		float gemm_time_53;
		float offline_time_53;
		std::tie(res5c_branch2c, gemm_time_53, offline_time_53) = convolve(res5c_branch2b_relu, im_size_53, im_height_53, im_width_53, im_depth_53, k_size_53, stride_53, res5c_branch2c_b, p1_53, p2_53, res5c_branch2c_w, output_size_53, mode, res5c_branch2c_min, res5c_branch2c_max, res5c_branch2c_result_min, res5c_branch2c_result_max);
		
		MatrixXd bn5c_branch2c = batchnorm(res5c_branch2c, bn5c_branch2c_mean, bn5c_branch2c_var);
		
		MatrixXd scale5c_branch2c = scale(bn5c_branch2c, scale5c_branch2c_weights, scale5c_branch2c_b);
		
		MatrixXd res5c = eltwise(scale5c_branch2c, res5b_relu);
		
		MatrixXd res5c_relu = relu(res5c);
		
		MatrixXd pool5 = pool(res5c_relu, f_2, s_2, output_width_53, output_height_53, pp1_2, pp2_2, mode_2);
		
		MatrixXd fc1000 = fully_connect(pool5, pool5.rows(), fc1000_weights, fc1000_b);
		
        clock_t run_time_end = clock();

        float run_time = (float) (run_time_end-run_time_start) / CLOCKS_PER_SEC;
		run_time_total += (run_time - offline_time_1 - offline_time_2 - offline_time_3 - offline_time_4 - offline_time_5 - offline_time_6 - offline_time_7 - offline_time_8 - offline_time_9 - offline_time_10 - offline_time_11 - offline_time_12 - offline_time_13 - offline_time_14 - offline_time_15 - offline_time_16 - offline_time_17 - offline_time_18 - offline_time_19 - offline_time_20 - offline_time_21 - offline_time_22 - offline_time_23 - offline_time_24 - offline_time_25 - offline_time_26 - offline_time_27 - offline_time_28 - offline_time_29 - offline_time_30 - offline_time_31 - offline_time_32 - offline_time_33 - offline_time_34 - offline_time_35 - offline_time_36 - offline_time_37 - offline_time_38 - offline_time_39 - offline_time_40 - offline_time_41 - offline_time_42 - offline_time_43 - offline_time_44 - offline_time_45 - offline_time_46 - offline_time_47 - offline_time_48 - offline_time_49 - offline_time_50 - offline_time_51 - offline_time_52 - offline_time_53);
		gemm_time_total += 0.0 + gemm_time_1 + gemm_time_2 + gemm_time_3 + gemm_time_4 + gemm_time_5 + gemm_time_6 + gemm_time_7 + gemm_time_8 + gemm_time_9 + gemm_time_10 + gemm_time_11 + gemm_time_12 + gemm_time_13 + gemm_time_14 + gemm_time_15 + gemm_time_16 + gemm_time_17 + gemm_time_18 + gemm_time_19 + gemm_time_20 + gemm_time_21 + gemm_time_22 + gemm_time_23 + gemm_time_24 + gemm_time_25 + gemm_time_26 + gemm_time_27 + gemm_time_28 + gemm_time_29 + gemm_time_30 + gemm_time_31 + gemm_time_32 + gemm_time_33 + gemm_time_34 + gemm_time_35 + gemm_time_36 + gemm_time_37 + gemm_time_38 + gemm_time_39 + gemm_time_40 + gemm_time_41 + gemm_time_42 + gemm_time_43 + gemm_time_44 + gemm_time_45 + gemm_time_46 + gemm_time_47 + gemm_time_48 + gemm_time_49 + gemm_time_50 + gemm_time_51 + gemm_time_52 + gemm_time_53;
		
		std::string name_1 = "../features/ResNet50/" + mode + "/fc1000_" + std::to_string(i) + ".csv";
		write_to_csv(name_1, fc1000);
    }

    infile.close();

    float avg_run_time = 0.0;
    avg_run_time = run_time_total / im_num;
    cout << "average online run time: " << avg_run_time << endl;

    float avg_gemm_time = 0.0;
    avg_gemm_time = gemm_time_total / im_num;
    cout << "average total time for GEMM: " << avg_gemm_time << endl;

    return 0;
}
