#ifndef LOWP_H
#define LOWP_H
 
#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;

std::tuple<MatrixXd, double, double> glp(const int r, const int d, const int c, const MatrixXd &a, const MatrixXd &b);

#endif
