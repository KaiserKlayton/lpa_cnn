#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2017"

## Extracts Caffe features from Caffe models. Writes to file at: 'features/<model_name>/caffe/..'.
## REQUIRED: Caffe .prototxt and .caffemodel in: 'models/<model_name>/..'
##           Input data in: 'inputs/<model_name>/production/..'
##

import os
import re
import sys

import numpy as np
import caffe

from helper.extract_architecture import extract_architecture
from helper.unpickler import unpickle

caffe.set_mode_cpu()

def main():    
    dirs = [x[0] for x in os.walk('models/')]

    for d in dirs:
        model_match = re.search("models/(.+)", d)
        if model_match:
            model = model_match.group(1)
        else: 
            continue

        if os.path.exists("features/%s" % model_match.group(1)):
            continue
        
        # Get input data path.
        input_files = os.listdir("inputs/" + model + "/production/")
        input_file_path = "inputs/" + model + "/production/" + input_files[0]
        
        try: 
            input_file_path
        except:
            sys.exit("Error: No input data found...")                

        # Get .prototxt and .caffemodel path.
        for f in os.listdir(d):
            if f.endswith('.prototxt'):
                prototxt_file_path = os.path.join(d, f)
            if f.endswith('.caffemodel'):
                model_file_path = os.path.join(d, f) 
        try: 
            prototxt_file_path
        except:
            sys.exit("Error: No suitable Caffe .prototxt found...")    
        try: 
            model_file_path
        except:
            sys.exit("Error: No suitable .caffemodel file found...")    

        # Extract architecture and parameters.
        architecture = extract_architecture(prototxt_file_path)
        a = architecture

        # Define caffe net.
        net = caffe.Net(prototxt_file_path, model_file_path, caffe.TEST)

        # Load (1) image into data blob.
        images = np.genfromtxt(input_file_path, delimiter=",")
        tick = 0
        for image in images:
            image = image[1:len(image)]
            image =  image.reshape(1,a['shape']['d'], a['shape']['w'], a['shape']['h'])
            net.blobs['data'].data[...] = image

            # Forward propogate (1) image.
            out = net.forward()

            # Extract and write features for each relevant layer.
            for key in a:
                if key == "shape" or a[key]['type'] == "relu":
                    continue

                blob = net.blobs[key].data[...]

                if len(blob.shape) == 4:
                    blob = blob.reshape(blob.shape[0]*blob.shape[1], blob.shape[2]*blob.shape[3]) 
                elif len(blob.shape) == 3:
                    blob = blob.reshape(blob.shape[0], blob.shape[1]*blob.shape[2])
                else:
                    pass

                if not os.path.exists(os.path.join('features', model)):
                    os.makedirs(os.path.join('features', model, "caffe"))
                    os.makedirs(os.path.join('features', model, "eigen"))
                    os.makedirs(os.path.join('features', model, "gemmlowp"))

                if key == a.keys()[-1]:
                    np.savetxt(os.path.join('features', model, "caffe", key + "_%s" % tick + ".csv"), blob, fmt='%.10f', delimiter=',')

            tick += 1
            
if __name__ == "__main__":
    main()
