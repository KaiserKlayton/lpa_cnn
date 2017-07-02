#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2017"

## Function that returns model architecture and parameters as a (ordered) Python dictionary.
## REQUIRED: Caffe .prototxt located in: 'models/<model_name>/..'.
##

import os
import re
import sys

from collections import OrderedDict

def extract_architecture(d):
    layer_types = ['conv', 'pool', 'relu', 'ip', 'fc']
    param_types = ['num_output', 'pad', 'kernel_size', 'stride']
    special_types = ['shape', 'input_dim']
    shape_dims = ['n','d','w','h']

    architecture = OrderedDict()
    architecture['shape'] = {}

    prototxt_file = open(d,'r').readlines()
    for l in prototxt_file:
        for s in special_types:
            if s in l:
                if s == 'shape':
                    param_match = re.search(s + ': { dim: ([0-9]+) dim: ([0-9]+) dim: ([0-9]+) dim: ([0-9]+)', l)
                    if param_match:
                        architecture['shape']['n'] = int(param_match.group(1))
                        architecture['shape']['d'] = int(param_match.group(2))
                        architecture['shape']['w'] = int(param_match.group(3))
                        architecture['shape']['h'] = int(param_match.group(4))                           
                elif s == 'input_dim':
                    param_match = re.search(s + ': ([0-9]+)', l)
                    if param_match:
                        architecture['shape'][shape_dims.pop(0)] = int(param_match.group(1))
                else:
                    sys.exit("Unknown shape format")

        layer_match = re.search('name: "(([a-z]+)[0-9]+_*[0-9]*)"', l)
        if layer_match:
            if layer_match.group(2) not in layer_types:
                sys.exit("Unknown layer type: %s") % layer_match.group(2)
            layer = layer_match.group(1)
            architecture[layer] = {}
          
        for p in param_types:
            if p in l:
                param_match = re.search(p + ': ([0-9]+)', l)
                if param_match:
                    value = int(param_match.group(1))
                    architecture[layer][p] = value

#    # Deal with pooling layers that aren't kernel_size=2
#    for key in architecture:
#        if "pool" in key:
#            for j in architecture[key]:
#                if j == "kernel_size":
#                    if architecture[key][j] != 2:
#                        sys.exit("pooling layer with filter size != 2. Edit in .prototxt and try again.")

    return architecture
