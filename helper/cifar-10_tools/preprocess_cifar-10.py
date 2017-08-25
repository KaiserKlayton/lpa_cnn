#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2017"

## Converts mean.binaryproto file to mean.npy
## Preps cifar-10 input data (1000) for system by subtracting specified mean.
##

import caffe
import numpy as np
import sys
        
def main():
    MEAN_IMAGE = 'helper/cifar-10_tools/mean.binaryproto'
    
    READ_FILE = 'helper/cifar-10_tools/cifar-10_1000/cifar-10_1000.csv'
    WRITE_FILE = 'helper/cifar-10_tools/mean_image_reduced/cifar-10_img_norm_1000.csv'
    
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(MEAN_IMAGE, 'rb').read()
    blob.ParseFromString(data)
    
    data = np.array(blob.data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    mean_image = arr[0].reshape(1, 3072)
    
    data = np.genfromtxt(READ_FILE, delimiter=',')
    
    images = data[:, 1:]
    labels = data[:, 0].reshape(1000, 1)
     
    result = images - mean_image
    
    results = np.hstack((labels, result))
    np.savetxt(WRITE_FILE, results, fmt='%.10f', delimiter=',')
    
if __name__ == "__main__":
    main()
