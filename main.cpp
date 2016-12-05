#include "reader.h"
#include "convolution_layer/convolution.h"
#include "pooling_layer/pooling.h"
#include "relu_layer/relu.h"

using Eigen::IOFormat;
using Eigen::StreamPrecision;
using Eigen::DontAlignCols;

void convolved_to_csv(string name, MatrixXd matrix) {
    const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
    
    ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
}

int main() 
{
    // Define Images (input).
    const int im_num = 100; // OPTION
    const int im_height = 28; // OPTION
    const int im_width = 28; // OPTION
    const int im_depth = 1; // OPTION
    const int im_size = im_height*im_width;
    MatrixXd train = read_mnist_train();

    for(int i=0; i < im_num; i++)  // For every row (image)...
    {   // Parse image with full depth (minus the label).
        MatrixXd image = train.block<1,im_size*im_depth>(i,1); 
            
    // CONVOLUTION 1 //
        // Define Kernels.
        const int k_num = 20; // OPTION
        const int k_size = 25; // OPTION
        const int stride = 1; // OPTION
        const int k_depth = im_depth;
        const int p1 = ((stride * im_height) - stride - im_height + sqrt(k_size)) / 2;
        const int p2 = ((stride * im_width) - stride - im_width + sqrt(k_size)) / 2;
        MatrixXd conv1_weights = read_mnist_conv1_weights();
        Map<MatrixXd> w(conv1_weights.data(), k_num, k_size * k_depth);
        // Define Biases.
        MatrixXd conv1_biases = read_mnist_conv1_biases();
        VectorXd conv1_b(Map<VectorXd>(conv1_biases.data(), conv1_biases.cols()*conv1_biases.rows()));
        // Convolve.        
        MatrixXd convolved = convolve(image, im_size, im_height, im_width, im_depth, k_size, stride, conv1_b, p1, p2, w);
        // Write convolved matrix to file.
        std::string name = "data/mnist/features/conv1_" + std::to_string(i) + ".csv";
        convolved_to_csv(name, convolved);

    // POOLING 1 //
        // Define Pooling behaviour.   
        const int f = 2; //OPTION
        const int s = 2; //OPTION
        // Pool.
        MatrixXd pooled = pool(convolved, f, s, im_width, im_height);
        // Write pooled matrix to file.
        std::string name2 = "data/mnist/features/pool1_" + std::to_string(i) + ".csv";
        convolved_to_csv(name2, pooled);

    }
    return 0;
}
