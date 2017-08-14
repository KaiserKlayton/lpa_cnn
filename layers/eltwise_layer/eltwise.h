#ifndef ELTWISE_H
#define ELTWISE_H

#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using namespace std;

MatrixXd eltwise(const MatrixXd &input_1, const MatrixXd &input_2);

#endif
