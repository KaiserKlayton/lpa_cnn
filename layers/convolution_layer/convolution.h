#ifndef CONVOLUTION_H
#define CONVOLUTION_H
 
#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using namespace std;

MatrixXd col2im(const MatrixXd &c);
MatrixXd add_padding(const MatrixXd &box, const int im_height, const int im_width, const int p1, const int p2);
MatrixXd kernel2col(MatrixXd &kernel);
MatrixXd im2col(const MatrixXd &input, const int k_size, const int stride);
std::tuple<MatrixXd, double, double> convolve(const MatrixXd &image, const int im_size, const int im_height, const int im_width, const int im_depth, const int k_size, const int stride, const VectorXd &b, const int p1, const int p2, const MatrixXd &w, const int output_size, std::string mode);

#endif
