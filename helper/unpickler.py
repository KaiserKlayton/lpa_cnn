#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2017"

## Function that unpickles a pickled object and returns it as a dict.
##

import cPickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()

    return dict
