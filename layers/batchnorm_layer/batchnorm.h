#ifndef BATCHNORM_H
#define BATCHNORM_H

#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using namespace std;

MatrixXd batchnorm(MatrixXd &input, const MatrixXd &mean, const MatrixXd &var);

#endif
