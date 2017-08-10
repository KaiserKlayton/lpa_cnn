#!/usr/bin/env Rscript

## Copyright: 2017
## Author: C. Clayton Violand

## Runs the experiment routine over all installed models.
## Results are in results/.
##
## REQUIREMENTS:
##  -models/<model_name>/<model_name.caffemodel>
##  -models/<model_name>/<model_name.prototxt>
##  -inputs/<model_name>/production/<input_file_name.csv>
##
## IT CALLS:
##  -extract_caffe_weights.py
##  -extract_caffe_features.py
##  -generate_cpp.py
##
##  *for every installed model (called from inference/)*...
##  -make -f Makefile.<model_name>
##  -lpa_cnn.out
##

library(rPython)
library(Rcpp)
