#ifndef POOLING_H
#define POOLING_H

#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using namespace std;

MatrixXd add_pool_padding(MatrixXd box, int im_height, int im_width, int pp1, int pp2);
MatrixXd pool(MatrixXd convolved, int f, int s, int im_width, int im_height, int pp_1, int pp_2);

#endif
