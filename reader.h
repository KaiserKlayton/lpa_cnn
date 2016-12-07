#ifndef READER_H
#define READER_H

#include <armadillo>
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Map;

template <typename M>
M load_csv_arma(const std::string & path);
template <typename M>
M load_binary_arma(const std::string & path);
MatrixXd read_mnist_train();
MatrixXd read_mnist_test();
MatrixXd read_mnist_conv2_weights();
MatrixXd read_mnist_conv2_biases();
MatrixXd read_mnist_conv1_weights();
MatrixXd read_mnist_conv1_biases();
MatrixXd read_cifar10_batch_1();
MatrixXd read_cifar10_weights();
MatrixXd read_cifar10_biases();

#endif
