#ifndef WRITER_H
#define WRITER_H

#include <eigen3/Eigen/Dense>
#include <fstream>

using Eigen::IOFormat;
using Eigen::FullPrecision;
using Eigen::DontAlignCols;
using Eigen::MatrixXd;
using namespace std;

void write_to_csv(const string name, const MatrixXd matrix);

#endif
