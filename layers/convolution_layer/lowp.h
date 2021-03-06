#ifndef LOWP_H
#define LOWP_H

#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;

std::tuple<MatrixXd, float, float> glp(const int r, const int d, const int c, const MatrixXd &a, const MatrixXd &b, const float w_min, const float w_max, const float result_min, const float result_max);

#endif
