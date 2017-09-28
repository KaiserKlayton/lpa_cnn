# lpa_cnn
Low Precision Arithmetic for Convolutional Neural Network Inference

## **Dependencies**

gcc 5.4 w/ Eigen 3 & Armadillo

Python 2.7 w/ NumPy & PIL

R w/ gtools & stringr & dplyr & tidyr

Caffe (see `Setup` below for installation)

## **Setup**

Install Caffe as `caffe/` (in root directory), following the guide @ `https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-CPU-Only/`.

Have the following files in place for each desired model:

      models/<model_name>/<model_name.caffemodel>
      models/<model_name>/<model_name.prototxt>
      
adjusting the .prototxt input layer to receive one image (i.e. `1 x <depth> x <width> x <height>`).

Have the following input file in place for each installed model:

      inputs/<model_name>/production/<input_file_name.csv>
      
having the form:
  
      <img_0_label><img_0_channel_1>...<img_0_channel_2><img_0_channel_3>
      <img_1_label><img_1_channel_1>...<img_1_channel_2><img_1_channel_3>
      ...

## **Reproduction**

To run experiments with the installed models, call `$ bash run_routine.sh`.

Results are written to `results/`.
