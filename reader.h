#ifndef READER_H
#define READER_H

#include <armadillo>
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Map;

template <typename M>
M load_csv_arma(const std::string & path);
MatrixXd read_mnist_train();
MatrixXd read_mnist_test();
MatrixXd read_mnist_fc2_weights();
MatrixXd read_mnist_fc2_biases();
MatrixXd read_mnist_fc1_weights();
MatrixXd read_mnist_fc1_biases();
MatrixXd read_mnist_conv2_weights();
MatrixXd read_mnist_conv2_biases();
MatrixXd read_mnist_conv1_weights();
MatrixXd read_mnist_conv1_biases();
MatrixXd read_cifar10();
MatrixXd read_cifar10_fc2_weights();
MatrixXd read_cifar10_fc2_biases();
MatrixXd read_cifar10_fc1_weights();
MatrixXd read_cifar10_fc1_biases();
MatrixXd read_cifar10_conv3_weights();
MatrixXd read_cifar10_conv3_biases();
MatrixXd read_cifar10_conv2_weights();
MatrixXd read_cifar10_conv2_biases();
MatrixXd read_cifar10_conv1_weights();
MatrixXd read_cifar10_conv1_biases();
MatrixXd read_VGG_CNN_S_fc8_weights();
MatrixXd read_VGG_CNN_S_fc8_biases();
MatrixXd read_VGG_CNN_S_fc7_weights();
MatrixXd read_VGG_CNN_S_fc7_biases();
MatrixXd read_VGG_CNN_S_fc6_weights();
MatrixXd read_VGG_CNN_S_fc6_biases();
MatrixXd read_VGG_CNN_S_conv5_weights();
MatrixXd read_VGG_CNN_S_conv5_biases();
MatrixXd read_VGG_CNN_S_conv4_weights();
MatrixXd read_VGG_CNN_S_conv4_biases();
MatrixXd read_VGG_CNN_S_conv3_weights();
MatrixXd read_VGG_CNN_S_conv3_biases();
MatrixXd read_VGG_CNN_S_conv2_weights();
MatrixXd read_VGG_CNN_S_conv2_biases();
MatrixXd read_VGG_CNN_S_conv1_weights();
MatrixXd read_VGG_CNN_S_conv1_biases();

#endif
