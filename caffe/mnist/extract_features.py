#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2016"

import numpy as np
import caffe

caffe.set_mode_cpu()

def main():    
    # Specifiy paths here.
    net = caffe.Net('caffe/mnist/lenet.prototxt', 'caffe/mnist/lenet_iter_10000.caffemodel', caffe.TEST)
    
    # Load (1) image into data blob.
    images = np.loadtxt("data/mnist/mnist_train_100.csv", delimiter=",")
    image = images[0,1:785]    
    image =  image.reshape(1,1,28,28)
    net.blobs['data'].data[...] = image

    # Forward propogate.
    out = net.forward()

    # Get features.
    conv1 = net.blobs['conv1'].data[...]
    conv1 = conv1.reshape (20,576)

    pool1 = net.blobs['pool1'].data[...]
    pool1 = pool1.reshape (20,144)

    conv2 = net.blobs['conv2'].data[...]
    conv2 = conv2.reshape (50,64)

    pool2 = net.blobs['pool2'].data[...]
    pool2 = pool2.reshape (50,16)

    fc1 = net.blobs['ip1'].data[...]

    np.savetxt('data/mnist/caffe_conv1_features.csv', conv1, fmt='%f', delimiter=',')
    np.savetxt('data/mnist/caffe_pool1_features.csv', pool1, fmt='%f', delimiter=',')
    np.savetxt('data/mnist/caffe_conv2_features.csv', conv2, fmt='%f', delimiter=',')
    np.savetxt('data/mnist/caffe_pool2_features.csv', pool2, fmt='%f', delimiter=',')
    np.savetxt('data/mnist/caffe_fc1_features.csv', fc1, fmt='%f', delimiter=',')

if __name__ == "__main__":
    main()
