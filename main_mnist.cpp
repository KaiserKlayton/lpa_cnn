#include "reader.h"
#include "convolution_layer/convolution.h"
#include "pooling_layer/pooling.h"
#include "fully_connected_layer/fully_connected.h"
#include "relu_layer/relu.h"

using Eigen::IOFormat;
using Eigen::StreamPrecision;
using Eigen::DontAlignCols;

void write_to_csv(string name, MatrixXd matrix) {
    const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
    
    ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
}

int main() 
{
    // INPUT PREP //
    const int im_height = 28; // OPTION
    const int im_width = 28; // OPTION
    const int im_depth = 1; // OPTION
    const int im_size = im_height*im_width;
    MatrixXd train = read_mnist_train();

	  const int im_num = 100; // OPTION
		float total_1 = 0.0;
		float total_2 = 0.0;	
    for(int i=0; i < im_num; i++)  // For every row (image)...
    {   // Parse image with full depth (minus the label).
        MatrixXd image = train.block<1,im_size*im_depth>(i,1); 
            
    // CONVOLUTION 1 //
        // Define Kernels.
        const int k_num = 20; // OPTION
        const int k_size = 25; // OPTION
        const int stride = 1; // OPTION
        const int k_depth = im_depth;
        //const int p1 = ((stride * im_height) - stride - im_height + sqrt(k_size)) / 2;
        //const int p2 = ((stride * im_width) - stride - im_width + sqrt(k_size)) / 2;
        const int p1 = 0; // OPTION
        const int p2 = 0; // OPTION
        const int output_height = (((im_height+(2*p1)) - sqrt(k_size))/stride) + 1;
        const int output_width = (((im_width+(2*p2)) - sqrt(k_size))/stride) + 1;
        const int output_size = output_height * output_width;
        MatrixXd conv1_weights = read_mnist_conv1_weights();
        Map<MatrixXd> w(conv1_weights.data(), k_num, k_size * k_depth);
        // Define Biases.
        MatrixXd conv1_biases = read_mnist_conv1_biases();
        VectorXd conv1_b(Map<VectorXd>(conv1_biases.data(), conv1_biases.cols()*conv1_biases.rows()));
        // Convolve.
				MatrixXd convolved_1;
				double time_1;
        std::tie(convolved_1, time_1) = convolve(image, im_size, im_height, im_width, im_depth, k_size, stride, conv1_b, p1, p2, w, output_size);               
        // Write convolved matrix to file.
        std::string name = "data/mnist/features/conv1_" + std::to_string(i) + ".csv";
        write_to_csv(name, convolved_1);

				//Aggregate time_1.
				total_1 += time_1;

    // POOLING 1 //
        // Define Pooling behaviour.   
        const int f = 2; //OPTION
        const int s = 2; //OPTION
        // Pool.
        MatrixXd pooled_1 = pool(convolved_1, f, s, output_width, output_height);
        // Write pooled matrix to file.
        std::string name2 = "data/mnist/features/pool1_" + std::to_string(i) + ".csv";
        write_to_csv(name2, pooled_1);

     // CONVOLUTION 2 //
        // Input Prep.
        const int im_height_2 = ((output_height - f) / s) + 1;
        const int im_width_2 = ((output_width - f) / s) + 1;
        const int im_depth_2 = pooled_1.rows();
        const int im_size_2 = pooled_1.cols();
        // Define Kernels.
        const int k_num_2 = 50; // OPTION
        const int k_size_2 = 25; // OPTION
        const int stride_2 = 1; // OPTION
        const int k_depth_2 = pooled_1.rows();
        //const int p1_2 = ((stride_2 * im_height_2) - stride_2 - im_height_2 + sqrt(k_size_2)) / 2;
        //const int p2_2 = ((stride_2 * im_width_2) - stride_2 - im_width_2 + sqrt(k_size_2)) / 2;
        const int p1_2 = 0;
        const int p2_2 = 0;        
        const int output_height_2 = (((im_height_2+(2*p1_2)) - sqrt(k_size_2))/stride_2) + 1;
        const int output_width_2 = (((im_width_2+(2*p2_2)) - sqrt(k_size_2))/stride_2) + 1;
        const int output_size_2 = output_height_2 * output_width_2;
        MatrixXd conv2_weights = read_mnist_conv2_weights();
        MatrixXd w_2 = conv2_weights;
        // Define Biases.
        MatrixXd conv2_biases = read_mnist_conv2_biases();
        VectorXd conv2_b(Map<VectorXd>(conv2_biases.data(), conv2_biases.cols()*conv2_biases.rows()));
        // Convolve.      
				MatrixXd convolved_2;
				double time_2;
        std::tie(convolved_2, time_2) = convolve(pooled_1, im_size_2, im_height_2, im_width_2, im_depth_2, k_size_2, stride_2, conv2_b, p1_2, p2_2, w_2, output_size_2);
        // Write convolved matrix to file.
        std::string name_2 = "data/mnist/features/conv2_" + std::to_string(i) + ".csv";
        write_to_csv(name_2, convolved_2);  

			//Aggregate time_2.
				total_2 += time_2;

     // POOLING 2 //
        // Define Pooling behaviour.   
        const int f_2 = 2; //OPTION
        const int s_2 = 2; //OPTION
        // Pool.
        MatrixXd pooled_2 = pool(convolved_2, f_2, s_2, output_width_2, output_height_2);
        // Write pooled matrix to file.
        std::string name3 = "data/mnist/features/pool2_" + std::to_string(i) + ".csv";
        write_to_csv(name3, pooled_2);

   // FULLY CONNECTED 1 //
        // Define Weights.
        MatrixXd fc1_weights = read_mnist_fc1_weights();
        // Define Biases.
        MatrixXd fc1_biases = read_mnist_fc1_biases();
        VectorXd fc1_b(Map<VectorXd>(fc1_biases.data(), fc1_biases.cols()*fc1_biases.rows()));
        // Fully Connect.
        MatrixXd fc1 = fully_connect(pooled_2, pooled_2.rows(), fc1_weights, fc1_b);
        // Write fully connected matrix to file.
        std::string name4 = "data/mnist/features/fc1_" + std::to_string(i) + ".csv";
        write_to_csv(name4, fc1);
        
    // ReLU 1 //
        MatrixXd relud_1 = relu(fc1);
        // Write relud matrix to file.
        std::string name5 = "data/mnist/features/relu1_" + std::to_string(i) + ".csv";
        write_to_csv(name5, relud_1);

   // FULLY CONNECTED 2 //
        // Define Weights.
        MatrixXd fc2_weights = read_mnist_fc2_weights();
        // Define Biases.
        MatrixXd fc2_biases = read_mnist_fc2_biases();
        VectorXd fc2_b(Map<VectorXd>(fc2_biases.data(), fc2_biases.cols()*fc2_biases.rows()));
        // Fully Connect.
        MatrixXd fc2 = fully_connect(relud_1, relud_1.rows(), fc2_weights, fc2_b);
        // Write fully connected matrix to file.
        std::string name6 = "data/mnist/features/fc2_" + std::to_string(i) + ".csv";
        write_to_csv(name6, fc2);

    }
		//Print means of Time1 and Time2.
		float avg_1 = 0.0;		
		avg_1 = total_1 / im_num;
		cout << avg_1 << endl;

		float avg_2 = 0.0;		
		avg_2 = total_2 / im_num;
		cout << avg_2 << endl;

    return 0; 
}
