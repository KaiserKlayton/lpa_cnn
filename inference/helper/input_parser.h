#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H

#include <eigen3/Eigen/Dense>
#include <vector>
#include <fstream>

using namespace Eigen;
using namespace std;

template<typename M>
M load_csv (ifstream& infile, int i) {
    string line, csvItem;
    vector<double> values;
    uint rows = 0;
    int lineNumber = 0;
    int lineNumberSought = i;
    while (getline(infile, line)) {
        if(lineNumber == lineNumberSought) {
            stringstream lineStream(line);
            while (getline(lineStream, csvItem, ',')) {
                values.push_back(stod(csvItem));
            }
            ++rows;
            
            return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
        }
        ++lineNumber;
    }
}

#endif
