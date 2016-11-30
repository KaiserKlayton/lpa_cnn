#include "reader.h"
#include "convolution_layer/convolution.h"

using Eigen::IOFormat;
using Eigen::StreamPrecision;
using Eigen::DontAlignCols;

void convolved_to_csv(string name, MatrixXd matrix) 
{
    const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
    
    ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
}

int main() 
{
    // Images.
    const int im_num = 100; 
    const int im_height = 28; 
    const int im_width = 28; 
    const int im_size = im_height*im_width;
    const int im_depth = 1; 
    MatrixXd train = read_mnist_train();

    // Biases.
    MatrixXd biases = read_mnist_biases();
    VectorXd b(Map<VectorXd>(biases.data(), biases.cols()*biases.rows()));

    // Kernels.
    const int k_num = 20; 
    const int k_size = 25; 
    const int k_depth = im_depth;
    const int stride = 1; 
    const int p1 = ((stride * im_height) - stride - im_height + sqrt(k_size)) / 2;
    const int p2 = ((stride * im_width) - stride - im_width + sqrt(k_size)) / 2;
    MatrixXd weights = read_mnist_weights();
    Map<MatrixXd> w(weights.data(), k_num, k_size * k_depth);

    // For every row (image).
    for(int i=0; i < im_num; i++)   
    {   // Parse image with full depth (minus the label).
        MatrixXd image = train.block<1,im_size*im_depth>(i,1); 

        // im2col for each slice, then concatinate slices.
        MatrixXd im(im_size*im_depth, k_size);
        for (int d=0; d < im_depth ; d++) {
            // Take slice out of im_depth.
            MatrixXd slice = image.block<1,im_size>(0,(im_size+1)*d);
            // Resize slice to be square.
            Map<MatrixXd> box(slice.data(), im_height, im_width);
            // Pad box with 0s.
            MatrixXd padded_box = add_padding(box, im_height, im_width, p1, p2);
            // im2col on particular slice.
            MatrixXd col_slice = im2col(padded_box, k_size, stride);
            // Concatinate col_slice to output 'im'.
            im.block<im_size, k_size>(0,(im_size+1)*d) = col_slice;
        }  

        // GEMM Multiplication Operation w/ Eigen.
        clock_t start = clock();    
        MatrixXd c = im*w.transpose();
        clock_t end = clock();
        double time = (double) (end-start) / CLOCKS_PER_SEC;           
        cout << time << endl;       

        // Add biases.
        c.rowwise() += b.transpose();

        // Reshape back to image.
        MatrixXd convolved = col2im(c);
        
        // ReLU.
        for(int i=0; i < convolved.rows(); i++) {
            for(int j=0; j < convolved.cols(); j++) {
                if (convolved(i,j) < 0) {
                    convolved(i,j) = 0;                
                }            
            }         
        }

        // Write convolved matrix to file.
        std::string name="data/mnist/features/conv1_" + std::to_string(i) + ".csv";
        convolved_to_csv(name, convolved);     
    }
    return 0;
}
