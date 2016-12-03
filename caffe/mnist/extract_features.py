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
    f = net.blobs['relu1'].data[...]
    f = f.reshape (20,784)

    p = net.blobs['pool1'].data[...]
    p = p.reshape (20,196)

    np.savetxt('data/mnist/caffe_relu1_features.csv', f, fmt='%f', delimiter=',')
    np.savetxt('data/mnist/caffe_pool1_features.csv', p, fmt='%f', delimiter=',')

if __name__ == "__main__":
    main()
