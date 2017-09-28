#include "relu.h"

MatrixXd relu(MatrixXd input) {
    for(int i=0; i < input.rows(); i++) {
        for(int j=0; j < input.cols(); j++) {
            if (input(i,j) < 0) {
                input(i,j) = 0;
            }
        }
    }

    return input;
}
