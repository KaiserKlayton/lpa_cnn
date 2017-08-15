#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H

#include <eigen3/Eigen/Dense>
#include <vector>
#include <fstream>

using namespace Eigen;
using namespace std;

template<typename M>
M load_csv (ifstream &infile) {
    string line, deleteline, csvItem;
    vector<double> values;
    while (getline(infile, line)) {
        stringstream lineStream(line);
        while (getline(lineStream, csvItem, ',')) {
            values.push_back(stod(csvItem));
        }
        line.replace(line.find(deleteline),deleteline.length(),"");

        return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), 1, values.size());
    }
}

#endif
