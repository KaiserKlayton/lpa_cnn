#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2016"

import caffe
import numpy as np

def main():
    # Specifiy paths here.
    net = caffe.Net('caffe/mnist/lenet.prototxt', 'caffe/mnist/lenet_iter_10000.caffemodel', caffe.TEST)

    W = net.params['conv1'][0].data[...]
    b = net.params['conv1'][1].data[...]

    W = W.reshape(20,25)

    np.savetxt("data/mnist/weights.csv", W, fmt='%f', delimiter=',')
    np.savetxt("data/mnist/biases.csv", b, fmt='%f', delimiter=',')

if __name__ == "__main__":
    main()
