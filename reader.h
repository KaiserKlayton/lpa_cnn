#ifndef READER_H
#define READER_H

#include <armadillo>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::Map;
using namespace std;

template <typename M>
M load_csv_arma(const std::string & path);
template <typename M>
M load_binary_arma(const std::string & path);
MatrixXd read_mnist_train();
MatrixXd read_mnist_test();
MatrixXd read_mnist_weights();
MatrixXd read_mnist_biases();
MatrixXd read_cifar10_batch_1();
MatrixXd read_cifar10_weights();
MatrixXd read_cifar10_biases();

#endif
