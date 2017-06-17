#ifndef READER_H
#define READER_H

#include <armadillo>
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Map;

template <typename M>
M load_csv_arma(const std::string & path) {
    arma::mat X;
    X.load(path, arma::csv_ascii);

    return Eigen::Map<const M>(X.memptr(), X.n_rows, X.n_cols);
}

#endif
