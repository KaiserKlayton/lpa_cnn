#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2016"

import caffe
import numpy as np

def main():
    # Specifiy paths here.
    net = caffe.Net('caffe/cifar10/cifar10_quick_train_test.prototxt', 'caffe/cifar10/cifar10_quick_iter_5000.caffemodel', caffe.TEST)

    Wconv1 = net.params['conv1'][0].data[...]
    bconv1 = net.params['conv1'][1].data[...]   
    Wconv1 = Wconv1.reshape(32,75)

    np.savetxt("data/cifar10/conv1_weights.csv", Wconv1, fmt='%f', delimiter=',')
    np.savetxt("data/cifar10/conv1_biases.csv", bconv1, fmt='%f', delimiter=',')

    Wconv2 = net.params['conv2'][0].data[...]
    bconv2 = net.params['conv2'][1].data[...]
    Wconv2 = Wconv2.reshape(32,800)

    np.savetxt("data/cifar10/conv2_weights.csv", Wconv2, fmt='%f', delimiter=',')
    np.savetxt("data/cifar10/conv2_biases.csv", bconv2, fmt='%f', delimiter=',')

    Wconv3 = net.params['conv3'][0].data[...]
    bconv3 = net.params['conv3'][1].data[...]   
    Wconv3 = Wconv3.reshape(64,800)

    np.savetxt("data/cifar10/conv3_weights.csv", Wconv3, fmt='%f', delimiter=',')
    np.savetxt("data/cifar10/conv3_biases.csv", bconv3, fmt='%f', delimiter=',')

    Wfc1 = net.params['ip1'][0].data[...]
    bfc1 = net.params['ip1'][1].data[...]
  
    np.savetxt("data/cifar10/fc1_weights.csv", Wfc1, fmt='%f', delimiter=',')
    np.savetxt("data/cifar10/fc1_biases.csv", bfc1, fmt='%f', delimiter=',')

    Wfc2 = net.params['ip2'][0].data[...]
    bfc2 = net.params['ip2'][1].data[...]
    
    np.savetxt("data/cifar10/fc2_weights.csv", Wfc2, fmt='%f', delimiter=',')
    np.savetxt("data/cifar10/fc2_biases.csv", bfc2, fmt='%f', delimiter=',')

if __name__ == "__main__":
    main()
