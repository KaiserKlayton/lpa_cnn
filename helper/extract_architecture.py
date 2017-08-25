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
    layer_types = ['convolution', 'pooling', 'relu', 'eltwise', 'innerproduct']
    param_types = ['num_output', 'pad', 'kernel_size', 'stride', 'bias_term', 'pool']
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

        layer_match = re.search('\s+name: "*(.+)"*', l)
        if layer_match:
            layer_name = layer_match.group(1).replace('"', '')

        if not "filler" in l:
            type_match = re.search('type: "*(.+)"*', l)
            if type_match:
                layer_type = type_match.group(1).lower().replace('_', '').replace('"', '')
                if layer_type in layer_types:
                    architecture[layer_name] = {}
                    architecture[layer_name]['type'] = layer_type

        for p in param_types:
            if p in l:
                ave_match = re.search(p + ': (AVE)', l)
                max_match = re.search(p + ': (MAX)', l)
                param_match = re.search(p + ': ([0-9]+)', l)
                bias_param_match = re.search(p + ': ([a-z]+)', l)
                if ave_match:
                    value = ave_match.group(1).lower()
                    architecture[layer_name][p] = value
                if max_match:
                    value = max_match.group(1).lower()
                    architecture[layer_name][p] = value
                if param_match:
                    value = int(param_match.group(1))
                    architecture[layer_name][p] = value
                if bias_param_match:
                    value = bias_param_match.group(1).lower()
                    architecture[layer_name][p] = value

    return architecture
