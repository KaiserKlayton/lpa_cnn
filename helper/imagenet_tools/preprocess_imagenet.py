#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2017"

## Preps ImageNet input data (1000) for system by subtracting specified mean (image or pixel).
## Adds labels as first column.
## Writes result to file as csv.
##

from PIL import Image
import cv2
import numpy as np
import glob
import os
import sys

def load_image(infile):
    image = Image.open(infile)
    image.load()
    data = np.asarray(image, dtype="float")
    
    return data

def main(option):
    print "Taking " + option

    READ_DIR = 'helper/imagenet_tools/imagenet_1000/'
    LABEL_FILE = 'helper/imagenet_tools/imagenet_1000_labels.csv'

    if option == "mean_pixel":
        WRITE_FILE = 'helper/imagenet_tools/mean_pixel_reduced/imagenet_pxl_norm_1000.csv'
    else:
        MEAN_IMAGE = 'helper/imagenet_tools/imagenet_mean.npy'
        WRITE_FILE = 'helper/imagenet_tools/mean_image_reduced/imagenet_img_norm_1000.csv'

    # Load the mean image
    if option == "mean_pixel":
        pass
    else:
        mean_image = np.load(MEAN_IMAGE)
        mean_image = mean_image.astype(float)

    # Load the labels
    labels = np.genfromtxt(LABEL_FILE, dtype=int, delimiter=',')
    labels = labels[:,0]
    
    index = 0
    result = np.empty((0, 150529))
    for i in sorted(glob.glob(READ_DIR + '*.JPEG')):
        print i

        image = cv2.imread(i)
        image = np.moveaxis(image, 2, 0)
        image = image.astype(float)

        if len(image.shape) == 2:
            image = np.stack((image, ) * 3)

        if option == "mean_pixel":
            image[0,:,:] = np.subtract(image[0,:,:], 103.939)
            image[1,:,:] = np.subtract(image[1,:,:], 116.779)
            image[2,:,:] = np.subtract(image[2,:,:], 123.68)
        elif option == "mean_image":
            image -= mean_image

        flattened_image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2])
        flattened_image = np.insert(flattened_image, 0, labels[index]).reshape(1, 150529)
        result = np.append(result, flattened_image, axis=0)

        index += 1

    results = np.append(labels, result)
    np.savetxt(WRITE_FILE, result, fmt='%i', delimiter=',')
            
if __name__ == "__main__":
    try:
        option = sys.argv[1]
    except IndexError:
        print "Usage: python helper/preprocess_imagenet.py <mean_image> | <mean_pixel>"
        sys.exit(1)

    if option.lower() != "mean_image" and option.lower() != "mean_pixel":
        print "Usage: python helper/preprocess_imagenet.py <mean_image> | <mean_pixel>"
        sys.exit(1)

    main(option)
