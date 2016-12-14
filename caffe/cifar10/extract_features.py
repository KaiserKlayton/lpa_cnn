#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2016"

import numpy as np
import caffe

caffe.set_mode_cpu()

def main():    
    # Specifiy paths here.
    net = caffe.Net('caffe/cifar10/cifar10_quick_train_test.prototxt', 'caffe/cifar10/cifar10_quick_iter_5000.caffemodel', caffe.TEST)
    
    # Load (1) image into data blob.
    def unpickle(file):
    	import cPickle
    	fo = open(file, 'rb')
    	dict = cPickle.load(fo)
    	fo.close()
    	return dict

    images = unpickle("data/cifar10/cifar-10-batches-py/data_batch_1")
    images = images['data']
    image =  images[1,:].transpose()
    image = image.reshape(1,3072)
    image = image.reshape(1,3,32,32)
    net.blobs['data'].data[...] = image

    # Forward propogate.
    out = net.forward()

    # Get features.
    conv1 = net.blobs['conv1'].data[...]
    conv1 = conv1.reshape(32,1024)

    pool1 = net.blobs['pool1'].data[...]
    pool1 = pool1.reshape(32,256)

    conv2 = net.blobs['conv2'].data[...]
    conv2 = conv2.reshape(32,256)

    pool2 = net.blobs['pool2'].data[...]
    pool2 = pool2.reshape(32,64)

    conv3 = net.blobs['conv3'].data[...]
    conv3 = conv3.reshape(64,64)

    pool3 = net.blobs['pool3'].data[...]   
    pool3 = pool3.reshape(64,16)

    fc1 = net.blobs['ip1'].data[...]

    fc2 = net.blobs['ip2'].data[...]

    np.savetxt('data/cifar10/caffe_conv1_features.csv', conv1, fmt='%f', delimiter=',')
    np.savetxt('data/cifar10/caffe_pool1_features.csv', pool1, fmt='%f', delimiter=',')
    np.savetxt('data/cifar10/caffe_conv2_features.csv', conv2, fmt='%f', delimiter=',')
    np.savetxt('data/cifar10/caffe_pool2_features.csv', pool2, fmt='%f', delimiter=',')
    np.savetxt('data/cifar10/caffe_conv3_features.csv', conv3, fmt='%f', delimiter=',')
    np.savetxt('data/cifar10/caffe_pool3_features.csv', pool3, fmt='%f', delimiter=',')
    np.savetxt('data/cifar10/caffe_fc1_features.csv', fc1, fmt='%f', delimiter=',')
    np.savetxt('data/cifar10/caffe_fc2_features.csv', fc2, fmt='%f', delimiter=',')

if __name__ == "__main__":
    main()
