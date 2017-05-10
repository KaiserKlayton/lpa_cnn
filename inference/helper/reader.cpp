#include "reader.h"

template <typename M>
M load_csv_arma(const std::string & path) {
    arma::mat X;
    X.load(path, arma::csv_ascii);

    return Eigen::Map<const M>(X.memptr(), X.n_rows, X.n_cols);
}
