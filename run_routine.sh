#!/bin/bash
# ------------------------------------------------------------------
# [C. Clayton Violand] run_routine.sh
#
# Runs the experiment routine over all installed models.
# Results are in results/.
#
# REQUIREMENTS:
#  -models/<model_name>/<model_name.caffemodel>
#  -models/<model_name>/<model_name.prototxt>
#  -inputs/<model_name>/production/<input_file_name.csv>
#
# IT CALLS:
#
# FROM ~:
#  -extract_caffe_weights.py
#  -extract_caffe_features.py
#  -generate_cpp.py
#
# THEN (for each configuration) FROM inference/:
#  -make -f Makefile.<model_name>
#  -./lpa_cnn.out eigen
#  -./lpa_cnn.out gemmlowp
#
# THEN ONCE AGAIN FROM ~:
#  -compile_results.R
#
# USAGE (from lpa_cnn/):
#  $ bash run_routine.sh
# ------------------------------------------------------------------

set -e

echo """
#########################################
## LPA_CNN | C. Clayton Violand | 2017 ##
#########################################
"""

# Get installed model names.
echo "successfully located the following installed models:"
echo "----------------------------------------------------"

cd models

MODELS=(*/)
for m in "${MODELS[@]}"
do
    :
    echo ${m%/}
done
echo

# Define arithmetic modes.
declare -a ARITHMETIC_MODES
ARITHMETIC_MODES=(eigen gemmlowp)

cd ..

# Extract caffe weights for each installed model.
echo "extracting model weights..."
eval python extract_caffe_weights.py
echo "DONE extracting model weights DONE"
echo "----------------------------------"

# Extract model features for (1) image for each installed model.
echo "extracting model features..."
eval python extract_caffe_features.py
echo "DONE extracting model features DONE"
echo "-----------------------------------"

# Generate cpp inference files for each model.
echo "generating cpp inference files..."
eval python generate_cpp.py
echo "DONE generating cpp inference files DONE"
echo "----------------------------------------"

cd inference

for m in "${MODELS[@]}"
do
    :
    echo "compiling inference file for" ${m%/}
    echo "..."
    # Call make on models' Makefile.
    eval make -f Makefile.${m%/}
    echo "..."
    echo "DONE compiling inference file for" ${m%/} "DONE"
    echo "-----------------------------------------------"

    for a in "${ARITHMETIC_MODES[@]}"
    do
        :
        # Run executable
        echo "running inference on" ${m%/} "w/" $a
        echo "..."
        eval ./lpa_cnn.out $a
        echo "..."
        echo "DONE running inference on" ${m%/} "w/" $a "DONE"
        echo "-----------------------------------------------"
    done
    
done

# Return to home directory.
cd ..

# Compile the results!
echo "compiling the results..."
eval Rscript helper/compile_results.R
echo "compiling the results COMPLETE"
echo "------------------------------"
