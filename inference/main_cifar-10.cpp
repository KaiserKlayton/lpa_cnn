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

	const int im_height_1 = 32;
	const int im_width_1 = 32;
	const int im_depth_1 = 3;
	const int im_size_1 = im_height_1*im_width_1;
	
	const int k_num_1 = 32;
	const int k_size_1 = 25;
	const int stride_1 = 1;
	const int k_depth_1 = im_depth_1;
	
	const int p1_1 = 2;
	const int p2_1 = 2;
	
	const int output_height_1 = (im_height_1 + 2 * p1_1 - sqrt(k_size_1)) / stride_1 + 1;
	const int output_width_1 = (im_width_1 + 2 * p2_1 - sqrt(k_size_1)) / stride_1 + 1;
	const int output_size_1 = output_height_1 * output_width_1;
	
	MatrixXd conv1_weights = load_csv_arma<MatrixXd>("../weights/cifar-10/conv1_weights.csv");
	Map<MatrixXd> conv1_w(conv1_weights.data(), k_num_1, k_size_1 * k_depth_1);
	const float conv1_min = conv1_w.minCoeff();
	const float conv1_max = conv1_w.maxCoeff();
	
	MatrixXd conv1_result_params = load_csv_arma<MatrixXd>("../features/cifar-10/caffe/result_params.csv");
	const float conv1_result_min = conv1_result_params(0, 0);
	const float conv1_result_max = conv1_result_params(0, 1);
	
	MatrixXd conv1_biases = load_csv_arma<MatrixXd>("../weights/cifar-10/conv1_biases.csv");
	VectorXd conv1_b(Map<VectorXd>(conv1_biases.data(), conv1_biases.cols()*conv1_biases.rows()));
	
	const int f_1 = 3;
	const int s_1 = 2;
	std::string mode_1 = "max";
	
	const int pp1_1 = 0;
	const int pp2_1 = 0;
	
	const int im_height_2 = static_cast<int>(ceil(static_cast<float>(output_height_1 + 2 * pp1_1 - f_1 ) / s_1)) + 1;
	const int im_width_2 = static_cast<int>(ceil(static_cast<float>(output_width_1 + 2 * pp2_1 - f_1) / s_1)) + 1;
	const int im_depth_2 = k_num_1;
	const int im_size_2 = im_height_2 * im_width_2;
	
	const int k_num_2 = 32;
	const int k_size_2 = 25;
	const int stride_2 = 1;
	const int k_depth_2 = im_depth_2;
	
	const int p1_2 = 2;
	const int p2_2 = 2;
	
	const int output_height_2 = (im_height_2 + 2 * p1_2 - sqrt(k_size_2)) / stride_2 + 1;
	const int output_width_2 = (im_width_2 + 2 * p2_2 - sqrt(k_size_2)) / stride_2 + 1;
	const int output_size_2 = output_height_2 * output_width_2;
	
	MatrixXd conv2_weights = load_csv_arma<MatrixXd>("../weights/cifar-10/conv2_weights.csv");
	MatrixXd conv2_w = conv2_weights;
	const float conv2_min = conv2_w.minCoeff();
	const float conv2_max = conv2_w.maxCoeff();
	
	MatrixXd conv2_result_params = load_csv_arma<MatrixXd>("../features/cifar-10/caffe/result_params.csv");
	const float conv2_result_min = conv2_result_params(1, 0);
	const float conv2_result_max = conv2_result_params(1, 1);
	
	MatrixXd conv2_biases = load_csv_arma<MatrixXd>("../weights/cifar-10/conv2_biases.csv");
	VectorXd conv2_b(Map<VectorXd>(conv2_biases.data(), conv2_biases.cols()*conv2_biases.rows()));
	
	const int f_2 = 3;
	const int s_2 = 2;
	std::string mode_2 = "ave";
	
	const int pp1_2 = 0;
	const int pp2_2 = 0;
	
	const int im_height_3 = static_cast<int>(ceil(static_cast<float>(output_height_2 + 2 * pp1_2 - f_2 ) / s_2)) + 1;
	const int im_width_3 = static_cast<int>(ceil(static_cast<float>(output_width_2 + 2 * pp2_2 - f_2) / s_2)) + 1;
	const int im_depth_3 = k_num_2;
	const int im_size_3 = im_height_3 * im_width_3;
	
	const int k_num_3 = 64;
	const int k_size_3 = 25;
	const int stride_3 = 1;
	const int k_depth_3 = im_depth_3;
	
	const int p1_3 = 2;
	const int p2_3 = 2;
	
	const int output_height_3 = (im_height_3 + 2 * p1_3 - sqrt(k_size_3)) / stride_3 + 1;
	const int output_width_3 = (im_width_3 + 2 * p2_3 - sqrt(k_size_3)) / stride_3 + 1;
	const int output_size_3 = output_height_3 * output_width_3;
	
	MatrixXd conv3_weights = load_csv_arma<MatrixXd>("../weights/cifar-10/conv3_weights.csv");
	MatrixXd conv3_w = conv3_weights;
	const float conv3_min = conv3_w.minCoeff();
	const float conv3_max = conv3_w.maxCoeff();
	
	MatrixXd conv3_result_params = load_csv_arma<MatrixXd>("../features/cifar-10/caffe/result_params.csv");
	const float conv3_result_min = conv3_result_params(2, 0);
	const float conv3_result_max = conv3_result_params(2, 1);
	
	MatrixXd conv3_biases = load_csv_arma<MatrixXd>("../weights/cifar-10/conv3_biases.csv");
	VectorXd conv3_b(Map<VectorXd>(conv3_biases.data(), conv3_biases.cols()*conv3_biases.rows()));
	
	const int f_3 = 3;
	const int s_3 = 2;
	std::string mode_3 = "ave";
	
	const int pp1_3 = 0;
	const int pp2_3 = 0;
	
	MatrixXd ip1_weights = load_csv_arma<MatrixXd>("../weights/cifar-10/ip1_weights.csv");
	
	MatrixXd ip1_biases = load_csv_arma<MatrixXd>("../weights/cifar-10/ip1_biases.csv");
	VectorXd ip1_b(Map<VectorXd>(ip1_biases.data(), ip1_biases.cols()*ip1_biases.rows()));
	
	MatrixXd ip2_weights = load_csv_arma<MatrixXd>("../weights/cifar-10/ip2_weights.csv");
	
	MatrixXd ip2_biases = load_csv_arma<MatrixXd>("../weights/cifar-10/ip2_biases.csv");
	VectorXd ip2_b(Map<VectorXd>(ip2_biases.data(), ip2_biases.cols()*ip2_biases.rows()));
	
	const int im_num = 1000;
	
	ifstream infile;
	infile.open("../inputs/cifar-10/production/cifar-10_img_norm_1000.csv");
	
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
		
		MatrixXd pool1 = pool(conv1, f_1, s_1, output_width_1, output_height_1, pp1_1, pp2_1, mode_1);
		
		MatrixXd relu1 = relu(pool1);
		
		MatrixXd conv2;
		float gemm_time_2;
		float offline_time_2;
		std::tie(conv2, gemm_time_2, offline_time_2) = convolve(relu1, im_size_2, im_height_2, im_width_2, im_depth_2, k_size_2, stride_2, conv2_b, p1_2, p2_2, conv2_w, output_size_2, mode, conv2_min, conv2_max, conv2_result_min, conv2_result_max);
		
		MatrixXd relu2 = relu(conv2);
		
		MatrixXd pool2 = pool(relu2, f_2, s_2, output_width_2, output_height_2, pp1_2, pp2_2, mode_2);
		
		MatrixXd conv3;
		float gemm_time_3;
		float offline_time_3;
		std::tie(conv3, gemm_time_3, offline_time_3) = convolve(pool2, im_size_3, im_height_3, im_width_3, im_depth_3, k_size_3, stride_3, conv3_b, p1_3, p2_3, conv3_w, output_size_3, mode, conv3_min, conv3_max, conv3_result_min, conv3_result_max);
		
		MatrixXd relu3 = relu(conv3);
		
		MatrixXd pool3 = pool(relu3, f_3, s_3, output_width_3, output_height_3, pp1_3, pp2_3, mode_3);
		
		MatrixXd ip1 = fully_connect(pool3, pool3.rows(), ip1_weights, ip1_b);
		
		MatrixXd ip2 = fully_connect(ip1, ip1.rows(), ip2_weights, ip2_b);
		
        clock_t run_time_end = clock();

        float run_time = (float) (run_time_end-run_time_start) / CLOCKS_PER_SEC;
		run_time_total += (run_time - offline_time_1 - offline_time_2 - offline_time_3);
		gemm_time_total += 0.0 + gemm_time_1 + gemm_time_2 + gemm_time_3;
		
		std::string name_1 = "../features/cifar-10/" + mode + "/ip2_" + std::to_string(i) + ".csv";
		write_to_csv(name_1, ip2);
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
