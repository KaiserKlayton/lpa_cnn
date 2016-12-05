#include "relu.h"

MatrixXd relu(MatrixXd convolved) {
    for(int i=0; i < convolved.rows(); i++) {
        for(int j=0; j < convolved.cols(); j++) {
            if (convolved(i,j) < 0) {
                convolved(i,j) = 0;                
            }            
        }         
    }
    return convolved;
}
