#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2017"

## Extracts Caffe weights from Caffe models. Writes to file at: 'weights/<model_name>/..'.
## REQUIRED: Caffe .prototxt and .caffemodel in: 'models/<model_name>/..'
##

import os
import re
import sys

import numpy as np
import caffe
import cPickle

from helper.extract_architecture import extract_architecture

caffe.set_mode_cpu()

def main():
    dirs = [x[0] for x in os.walk('models/')]

    for d in dirs:
        model_match = re.search("models/(.+)", d)
        if model_match:
            model = model_match.group(1)
        else: 
            continue
            
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

        # Extract and write weights for each relevant layer.
        for key in architecture:
            if key == "shape" or "relu" in key or "pool" in key:
                continue
   
            weight_blob = net.params[key][0].data[...]
            bias_blob = net.params[key][1].data[...]

            if len(weight_blob.shape) == 4:
                weight_blob = weight_blob.reshape(weight_blob.shape[0], weight_blob.shape[1]*weight_blob.shape[2]*weight_blob.shape[3]) 
            elif len(weight_blob.shape) == 3:
                weight_blob = weight_blob.reshape(weight_blob.shape[0], weight_blob.shape[1]*weight_blob.shape[2])
            else:
                pass

            if not os.path.exists(os.path.join('weights', model)):
                os.makedirs(os.path.join('weights', model))
            
            np.savetxt(os.path.join('weights', model, key+"_weights.csv"), weight_blob, fmt='%.10f', delimiter=',')
            np.savetxt(os.path.join('weights', model, key+"_biases.csv"), bias_blob, fmt='%.10f', delimiter=',')

if __name__ == "__main__":
  main()
