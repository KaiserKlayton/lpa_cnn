#include "reader.h"

template <typename M>
M load_csv_arma(const std::string & path) {
    arma::mat X;
    X.load(path, arma::csv_ascii);

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


MatrixXd read_mnist_fc2_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/mnist/fc2_weights.csv");

    return weights;
}


MatrixXd read_mnist_fc2_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/mnist/fc2_biases.csv");

    return biases;
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
MatrixXd read_cifar10() {
    MatrixXd train = load_csv_arma<MatrixXd>("data/cifar10/cifar10.csv");

    return train;
}


MatrixXd read_cifar10_fc2_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/cifar10/fc2_weights.csv");

    return weights;
}


MatrixXd read_cifar10_fc2_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/cifar10/fc2_biases.csv");

    return biases;
}


MatrixXd read_cifar10_fc1_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/cifar10/fc1_weights.csv");

    return weights;
}


MatrixXd read_cifar10_fc1_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/cifar10/fc1_biases.csv");

    return biases;
}


MatrixXd read_cifar10_conv3_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/cifar10/conv3_weights.csv");

    return weights;
}


MatrixXd read_cifar10_conv3_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/cifar10/conv3_biases.csv");

    return biases;
}


MatrixXd read_cifar10_conv2_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/cifar10/conv2_weights.csv");

    return weights;
}


MatrixXd read_cifar10_conv2_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/cifar10/conv2_biases.csv");

    return biases;
}


MatrixXd read_cifar10_conv1_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/cifar10/conv1_weights.csv");

    return weights;
}


MatrixXd read_cifar10_conv1_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/cifar10/conv1_biases.csv");

    return biases;
}

// VGG_CNN_S //
MatrixXd read_VGG_CNN_S_fc8_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/VGG_CNN_S/fc8_weights.csv");

    return weights;
}

MatrixXd read_VGG_CNN_S_fc8_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/VGG_CNN_S/fc8_biases.csv");

    return biases;
}

MatrixXd read_VGG_CNN_S_fc7_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/VGG_CNN_S/fc7_weights.csv");

    return weights;
}


MatrixXd read_VGG_CNN_S_fc7_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/VGG_CNN_S/fc7_biases.csv");

    return biases;
}


MatrixXd read_VGG_CNN_S_fc6_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/VGG_CNN_S/fc6_weights.csv");

    return weights;
}


MatrixXd read_VGG_CNN_S_fc6_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/VGG_CNN_S/fc6_biases.csv");

    return biases;
}

MatrixXd read_VGG_CNN_S_conv5_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/VGG_CNN_S/conv5_weights.csv");

    return weights;
}


MatrixXd read_VGG_CNN_S_conv5_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/VGG_CNN_S/conv5_biases.csv");

    return biases;
}


MatrixXd read_VGG_CNN_S_conv4_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/VGG_CNN_S/conv4_weights.csv");

    return weights;
}


MatrixXd read_VGG_CNN_S_conv4_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/VGG_CNN_S/conv4_biases.csv");

    return biases;
}

MatrixXd read_VGG_CNN_S_conv3_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/VGG_CNN_S/conv3_weights.csv");

    return weights;
}


MatrixXd read_VGG_CNN_S_conv3_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/VGG_CNN_S/conv3_biases.csv");

    return biases;
}


MatrixXd read_VGG_CNN_S_conv2_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/VGG_CNN_S/conv2_weights.csv");

    return weights;
}


MatrixXd read_VGG_CNN_S_conv2_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/VGG_CNN_S/conv2_biases.csv");

    return biases;
}


MatrixXd read_VGG_CNN_S_conv1_weights() {
    MatrixXd weights = load_csv_arma<MatrixXd>("data/VGG_CNN_S/conv1_weights.csv");

    return weights;
}


MatrixXd read_VGG_CNN_S_conv1_biases() {
    MatrixXd biases = load_csv_arma<MatrixXd>("data/VGG_CNN_S/conv1_biases.csv");

    return biases;
}
