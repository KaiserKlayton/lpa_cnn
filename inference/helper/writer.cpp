#include "writer.h"

void write_to_csv(string name, MatrixXd matrix) {
    const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
    
    ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
}
