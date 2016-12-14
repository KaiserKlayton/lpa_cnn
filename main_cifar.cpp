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
    // INPUT PREP //
    const int im_height = 32; // OPTION
    const int im_width = 32; // OPTION
    const int im_depth = 3; // OPTION
    const int im_size = im_height*im_width;
    MatrixXd train = read_cifar10();

    const int im_num = 1; // OPTION
    for(int i=0; i < im_num; i++)  // For every row (image)...
    {   // Parse image with full depth (no labels this time).        
        MatrixXd img = train.block<1,im_size*im_depth>(i,0);
        MatrixXd image = Map<Matrix<double, im_depth, im_size, RowMajor>>(img.data());

    // CONVOLUTION 1 //
        // Define Kernels.
        const int k_num = 32; // OPTION
        const int k_size = 25; // OPTION
        const int stride = 1; // OPTION
        const int k_depth = im_depth;
        //const int p1 = ((stride * im_height) - stride - im_height + sqrt(k_size)) / 2;
        //const int p2 = ((stride * im_width) - stride - im_width + sqrt(k_size)) / 2;
        const int p1 = 2; // OPTION
        const int p2 = 2; // OPTION
        const int output_height = (((im_height+(2*p1)) - sqrt(k_size))/stride) + 1;
        const int output_width = (((im_width+(2*p2)) - sqrt(k_size))/stride) + 1;
        const int output_size = output_height * output_width;
        MatrixXd conv1_weights = read_cifar10_conv1_weights();
        Map<MatrixXd> w(conv1_weights.data(), k_num, k_size * k_depth);
        // Define Biases.
        MatrixXd conv1_biases = read_cifar10_conv1_biases();
        VectorXd conv1_b(Map<VectorXd>(conv1_biases.data(), conv1_biases.cols()*conv1_biases.rows()));       
        // Convolve.        
        MatrixXd convolved = convolve(image, im_size, im_height, im_width, im_depth, k_size, stride, conv1_b, p1, p2, w, output_size);               
        // Write convolved matrix to file.
        std::string name = "data/cifar10/features/conv1_" + std::to_string(i) + ".csv";
        write_to_csv(name, convolved);

    }
    return 0;
}
