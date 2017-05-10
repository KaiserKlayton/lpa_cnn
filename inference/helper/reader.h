#ifndef WRITER_H
#define WRITER_H

#include <armadillo>
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Map;

template <typename M>
M load_csv_arma(const std::string & path);

#endif
