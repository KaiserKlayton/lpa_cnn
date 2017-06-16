#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2017"

## Extracts Caffe features from Caffe models. Writes to file at: 'features/<model_name>/caffe/..'.
## REQUIRED: Caffe .prototxt and .caffemodel in: 'models/<model_name>/..'
##           Input data in: 'inputs/<model_name>/..'
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

        # Get input data path.
        input_files = os.listdir("inputs/" + model + "/production/")
        if (len(input_files) > 1):
            for i in range(len(input_files)):
                if input_files[i].endswith('.csv'):
                    continue
                else:
                    input_file_path = "inputs/" + model + "/production/" + input_files[i]
        else:
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
        if input_file_path.endswith('.binaryproto'):
            data = open(input_file_path, 'rb').read()
            blob = caffe.proto.caffe_pb2.BlobProto()
            blob.ParseFromString(data)
            arr = np.array(caffe.io.blobproto_to_array(blob))
            image = arr[0]
            net.blobs['data'].data[...] = image
        elif input_file_path.endswith('.csv'):
            images = np.loadtxt(input_file_path, delimiter=",")
            image = images[0,:]    
            image =  image.reshape(1,a['shape']['d'], a['shape']['w'], a['shape']['h'])
            net.blobs['data'].data[...] = image
        elif input_file_path.endswith('.pkl'):
            images = unpickle(input_file_path)
            images = images['data']
            image =  images[0,:].transpose()
            image = image.reshape(1,a['shape']['d'] * a['shape']['w'] * a['shape']['h'])
            image = image.reshape(1,a['shape']['d'], a['shape']['w'], a['shape']['h'])
            net.blobs['data'].data[...] = image
            
        # Forward propogate (1) image.
        out = net.forward()

        # Extract and write features for each relevant layer.
        for key in architecture:
            if key == "shape" or "relu" in key:
                continue
   
            blob = net.blobs[key].data[...]

            if len(blob.shape) == 4:
                blob = blob.reshape(blob.shape[0]*blob.shape[1], blob.shape[2]*blob.shape[3]) 
            elif len(blob.shape) == 3:
                blob = blob.reshape(blob.shape[0], blob.shape[1]*blob.shape[2])
            else:
                pass

            np.savetxt(os.path.join('features', model, "caffe", key+".csv"), blob, fmt='%.10f', delimiter=',')

if __name__ == "__main__":
    main()
