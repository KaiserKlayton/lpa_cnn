#ifndef CONVOLUTION_H
#define CONVOLUTION_H
 
#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using namespace std;

MatrixXd col2im(MatrixXd c);
MatrixXd add_padding(MatrixXd box, int im_height, int im_width, int p1, int p2);
MatrixXd kernel2col(MatrixXd kernel);
MatrixXd im2col(MatrixXd input, int k_size, int stride);
MatrixXd convolve(MatrixXd image, int im_size, int im_height, int im_width, int im_depth, int k_size, int stride, VectorXd b, int p1, int p2, MatrixXd w, int output_size);

#endif
