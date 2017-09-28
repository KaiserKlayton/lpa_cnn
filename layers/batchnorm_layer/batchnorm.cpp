#include "batchnorm.h"

MatrixXd batchnorm(MatrixXd &input, const MatrixXd &mean, const MatrixXd &var) {
    MatrixXd mean_box = mean.replicate(1, input.cols());
    MatrixXd var_box = var.replicate(1, input.cols());
    MatrixXd epsilon_box = MatrixXd::Constant(input.rows(), input.cols(), .00001);

    // Normalize using global parameters (learned at training from batches).
    MatrixXd result = (input - mean_box).array() / (var_box + epsilon_box).array().sqrt();

    return result;
}
