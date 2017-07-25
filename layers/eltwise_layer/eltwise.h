#ifndef ELTWISE_H
#define ELTWISE_H

#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using namespace std;

MatrixXd eltwise(MatrixXd input_1, MatrixXd input_2);

#endif
