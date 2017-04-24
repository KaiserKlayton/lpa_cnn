#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2017"

import numpy as np
import caffe
import sys

caffe.set_mode_cpu()

def main():    
    # Specifiy paths here.
    net = caffe.Net('caffe/VGG_CNN_S/VGG_CNN_S_deploy.prototxt', 'caffe/VGG_CNN_S/VGG_CNN_S.caffemodel', caffe.TEST)
    
    # Load (1) image into data blob.
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open("data/VGG_CNN_S/VGG_mean.binaryproto", 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    image = arr[0]

    net.blobs['data'].data[...] = image

    # Forward propogate.
    out = net.forward()

    # Get features.
    conv1 = net.blobs['conv1'].data[...]
    conv1 = conv1.reshape(96,11881)

    pool1 = net.blobs['pool1'].data[...]
    pool1 = pool1.reshape(96,1369)

    conv2 = net.blobs['conv2'].data[...]
    conv2 = conv2.reshape(256,1089)

    pool2 = net.blobs['pool2'].data[...]
    pool2 = pool2.reshape(256,289)

    conv3 = net.blobs['conv3'].data[...]
    conv3 = conv3.reshape(512,289)

    conv4 = net.blobs['conv4'].data[...]
    conv4 = conv4.reshape(512,289)

    conv5 = net.blobs['conv5'].data[...]
    conv5 = conv5.reshape(512,289)

    pool5 = net.blobs['pool5'].data[...]
    pool5 = pool5.reshape(512,36)

    fc6 = net.blobs['fc6'].data[...]   

    fc7 = net.blobs['fc7'].data[...]   

    fc8 = net.blobs['fc8'].data[...]   

    np.savetxt('data/VGG_CNN_S/caffe_conv1_features.csv', conv1, fmt='%f', delimiter=',')
    np.savetxt('data/VGG_CNN_S/caffe_pool1_features.csv', pool1, fmt='%f', delimiter=',')
    np.savetxt('data/VGG_CNN_S/caffe_conv2_features.csv', conv2, fmt='%f', delimiter=',')
    np.savetxt('data/VGG_CNN_S/caffe_pool2_features.csv', pool2, fmt='%f', delimiter=',')
    np.savetxt('data/VGG_CNN_S/caffe_conv3_features.csv', conv3, fmt='%f', delimiter=',')
    np.savetxt('data/VGG_CNN_S/caffe_conv4_features.csv', conv4, fmt='%f', delimiter=',')
    np.savetxt('data/VGG_CNN_S/caffe_conv5_features.csv', conv5, fmt='%f', delimiter=',')
    np.savetxt('data/VGG_CNN_S/caffe_pool5_features.csv', pool5, fmt='%f', delimiter=',')
    np.savetxt('data/VGG_CNN_S/caffe_fc6_features.csv', fc6, fmt='%f', delimiter=',')
    np.savetxt('data/VGG_CNN_S/caffe_fc7_features.csv', fc7, fmt='%f', delimiter=',')
    np.savetxt('data/VGG_CNN_S/caffe_fc8_features.csv', fc8, fmt='%f', delimiter=',')

if __name__ == "__main__":
    main()
