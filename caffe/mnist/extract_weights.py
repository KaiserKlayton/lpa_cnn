#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2016"

import caffe

def main():
    # Specifiy paths here.
    net = caffe.Net('caffe/mnist/lenet.prototxt', 'caffe/mnist/lenet_iter_10000.caffemodel', caffe.TEST)

    W = net.params['conv1'][0].data[...]
    b = net.params['conv1'][1].data[...]

    W.tofile("data/mnist/weights.csv", sep=",")
    b.tofile("data/mnist/biases.csv", sep=",")

if __name__ == "__main__":
    main()
