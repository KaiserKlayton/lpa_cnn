#ifndef LOWP_H
#define LOWP_H
 
#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;

std::pair<MatrixXd, double> glp(int r, int d, int c, MatrixXd a, MatrixXd b);

#endif
