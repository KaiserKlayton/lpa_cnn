#include "eltwise.h"

MatrixXd eltwise(const MatrixXd &input_1, const MatrixXd &input_2) {
    MatrixXd output = input_1 + input_2;

    return output;
}
