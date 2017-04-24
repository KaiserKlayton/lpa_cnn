#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2017"

import caffe
import numpy as np

def main():
    # Specifiy paths here.
    net = caffe.Net('caffe/VGG_CNN_S/VGG_CNN_S_deploy.prototxt', 'caffe/VGG_CNN_S/VGG_CNN_S.caffemodel', caffe.TEST)
    Wconv1 = net.params['conv1'][0].data[...]
    bconv1 = net.params['conv1'][1].data[...]   
    Wconv1 = Wconv1.reshape(96,147)

    np.savetxt("data/VGG_CNN_S/conv1_weights.csv", Wconv1, fmt='%f', delimiter=',')
    np.savetxt("data/VGG_CNN_S/conv1_biases.csv", bconv1, fmt='%f', delimiter=',')

    Wconv2 = net.params['conv2'][0].data[...]
    bconv2 = net.params['conv2'][1].data[...]
    Wconv2 = Wconv2.reshape(256,2400)

    np.savetxt("data/VGG_CNN_S/conv2_weights.csv", Wconv2, fmt='%f', delimiter=',')
    np.savetxt("data/VGG_CNN_S/conv2_biases.csv", bconv2, fmt='%f', delimiter=',')

    Wconv3 = net.params['conv3'][0].data[...]
    bconv3 = net.params['conv3'][1].data[...]   
    Wconv3 = Wconv3.reshape(512,2304)

    np.savetxt("data/VGG_CNN_S/conv3_weights.csv", Wconv3, fmt='%f', delimiter=',')
    np.savetxt("data/VGG_CNN_S/conv3_biases.csv", bconv3, fmt='%f', delimiter=',')

    Wconv4 = net.params['conv4'][0].data[...]
    bconv4 = net.params['conv4'][1].data[...]
    Wconv4 = Wconv4.reshape(512,4608)

    np.savetxt("data/VGG_CNN_S/conv4_weights.csv", Wconv4, fmt='%f', delimiter=',')
    np.savetxt("data/VGG_CNN_S/conv4_biases.csv", bconv4, fmt='%f', delimiter=',')

    Wconv5 = net.params['conv5'][0].data[...]
    bconv5 = net.params['conv5'][1].data[...]
    Wconv5 = Wconv5.reshape(512,4608)   

    np.savetxt("data/VGG_CNN_S/conv5_weights.csv", Wconv5, fmt='%f', delimiter=',')
    np.savetxt("data/VGG_CNN_S/conv5_biases.csv", bconv5, fmt='%f', delimiter=',')

    Wfc6 = net.params['fc6'][0].data[...]
    bfc6 = net.params['fc6'][1].data[...]
    
    np.savetxt("data/VGG_CNN_S/fc6_weights.csv", Wfc6, fmt='%f', delimiter=',')
    np.savetxt("data/VGG_CNN_S/fc6_biases.csv", bfc6, fmt='%f', delimiter=',')

    Wfc7 = net.params['fc7'][0].data[...]
    bfc7 = net.params['fc7'][1].data[...]

    np.savetxt("data/VGG_CNN_S/fc7_weights.csv", Wfc7, fmt='%f', delimiter=',')
    np.savetxt("data/VGG_CNN_S/fc7_biases.csv", bfc7, fmt='%f', delimiter=',')

    Wfc8 = net.params['fc8'][0].data[...]
    bfc8 = net.params['fc8'][1].data[...]   
    
    np.savetxt("data/VGG_CNN_S/fc8_weights.csv", Wfc8, fmt='%f', delimiter=',')
    np.savetxt("data/VGG_CNN_S/fc8_biases.csv", bfc8, fmt='%f', delimiter=',')

if __name__ == "__main__":
    main()
