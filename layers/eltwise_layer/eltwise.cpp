#include "eltwise.h"

MatrixXd eltwise(MatrixXd input_1, MatrixXd input_2) {
    MatrixXd output = input_1 + input_2;

    return output;
}
