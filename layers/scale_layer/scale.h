#ifndef SCALE_H
#define SCALE_H

#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using namespace std;

MatrixXd scale(MatrixXd &input, const MatrixXd &scale_weights, const VectorXd &scale_biases);

#endif
