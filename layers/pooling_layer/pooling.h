#ifndef POOLING_H
#define POOLING_H

#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using namespace std;

MatrixXd add_special_padding(const MatrixXd &box, const int im_height, const int im_width, const int pp1, const int pp2);
MatrixXd add_pool_padding(const MatrixXd &box, const int im_height, const int im_width, const int pp1, const int pp2);
MatrixXd pool(const MatrixXd &convolved, const int f, const int s, const int im_width, const int im_height, const int pp_1, const int pp_2, std::string mode);

#endif
