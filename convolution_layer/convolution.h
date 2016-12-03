#ifndef CONVOLUTION_H
#define CONVOLUTION_H
 
#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using namespace std;

MatrixXd add_padding(MatrixXd box, int im_height, int im_width, int p1, int p2);
MatrixXd im2col(MatrixXd input, int k_size, int stride);
MatrixXd kernel2col(MatrixXd kernel);
MatrixXd col2im(MatrixXd c);
 
#endif
