#include "reader.h"

template <typename M>
M load_csv_arma(const std::string & path) {
    arma::mat X;
    X.load(path, arma::csv_ascii);

    return Eigen::Map<const M>(X.memptr(), X.n_rows, X.n_cols);
}


template <typename M>
M load_binary_arma(const std::string & path) {
    arma::mat X;
    X.load(path, arma::raw_binary);

    return Eigen::Map<const M>(X.memptr(), X.n_rows, X.n_cols);
}


// MNIST //
MatrixXd read_mnist_train() {
    MatrixXd train = load_csv_arma<MatrixXd>("data/mnist/mnist_train_100.csv");

    return train;
}


MatrixXd read_mnist_test() {
    MatrixXd test = load_csv_arma<MatrixXd>("data/mnist/mnist_test_10.csv");

    return test;
}


MatrixXd read_mnist_fc1_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/mnist/fc1_weights.csv");

    return weights;
}

MatrixXd read_mnist_fc1_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/mnist/fc1_biases.csv");

    return biases;
}

MatrixXd read_mnist_conv2_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/mnist/conv2_weights.csv");

    return weights;
}


MatrixXd read_mnist_conv2_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/mnist/conv2_biases.csv");

    return biases;
}


MatrixXd read_mnist_conv1_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/mnist/conv1_weights.csv");

    return weights;
}


MatrixXd read_mnist_conv1_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/mnist/conv1_biases.csv");

    return biases;
}


// CIFAR-10 //
MatrixXd read_cifar10_batch_1() {
    MatrixXd train = load_binary_arma<MatrixXd>("data/cifar10/data_batch_1.bin");

    return train;
}


MatrixXd read_cifar10_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/cifar10/weights.csv");

    return weights;
}


MatrixXd read_cifar10_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/cifar10/biases.csv");

    return biases;
}
