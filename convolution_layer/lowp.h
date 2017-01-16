#ifndef LOWP_H
#define LOWP_H
 
#include <eigen3/Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;

MatrixXd glp(int r, int d, int c, MatrixXd a, MatrixXd b);

#endif