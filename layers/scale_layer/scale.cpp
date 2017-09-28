#include "scale.h"

MatrixXd scale(MatrixXd &input, const MatrixXd &scale_weights, const VectorXd &scale_biases) {
    MatrixXd scale_weights_box = scale_weights.replicate(1, input.cols());
    MatrixXd scale_biases_box = scale_biases.replicate(1, input.cols());

    // Scale.
    MatrixXd scaled = input.cwiseProduct(scale_weights_box);

    // Add Biases.
    MatrixXd result = scaled + scale_biases_box;

    return result;
}
