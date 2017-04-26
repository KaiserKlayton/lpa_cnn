#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H
 
#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using namespace std;

MatrixXd fully_connect(MatrixXd input, int k_num, MatrixXd weights, VectorXd biases);

#endif
