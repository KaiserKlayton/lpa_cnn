#ifndef POOLING_H
#define POOLING_H

#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using namespace std;

MatrixXd pool(MatrixXd convolved, int f, int s, int im_width, int im_height);

#endif
