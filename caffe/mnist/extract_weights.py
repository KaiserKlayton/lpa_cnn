#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2016"

import caffe
import numpy as np

def main():
    # Specifiy paths here.
    net = caffe.Net('caffe/mnist/lenet.prototxt', 'caffe/mnist/lenet_iter_10000.caffemodel', caffe.TEST)

    Wconv1 = net.params['conv1'][0].data[...]
    bconv1 = net.params['conv1'][1].data[...]
    Wconv1 = Wconv1.reshape(20,25)

    np.savetxt("data/mnist/conv1_weights.csv", Wconv1, fmt='%f', delimiter=',')
    np.savetxt("data/mnist/conv1_biases.csv", bconv1, fmt='%f', delimiter=',')

    Wconv2 = net.params['conv2'][0].data[...]
    bconv2 = net.params['conv2'][1].data[...]
    Wconv2 = Wconv2.reshape(50,500)

    np.savetxt("data/mnist/conv2_weights.csv", Wconv2, fmt='%f', delimiter=',')
    np.savetxt("data/mnist/conv2_biases.csv", bconv2, fmt='%f', delimiter=',')

    Wfc1 = net.params['ip1'][0].data[...]
    bfc1 = net.params['ip1'][1].data[...]

    np.savetxt("data/mnist/fc1_weights.csv", Wfc1, fmt='%f', delimiter=',')
    np.savetxt("data/mnist/fc1_biases.csv", bfc1, fmt='%f', delimiter=',')

if __name__ == "__main__":
    main()
