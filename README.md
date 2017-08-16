# lpa_cnn
Low Precision Arithmetic for Convolutional Neural Network Inference

## **Dependencies**

gcc 5.4 w/ Eigen 3 & Armadillo

Python 2.7 w/ NumPy & PIL

R

## **Setup**

Add caffe Python module directory to `$PYTHONPATH`.

Have the following files in place for each desired model:

      models/<model_name>/<model_name.caffemodel>
      models/<model_name>/<model_name.prototxt>
      
Have the following input file in place for each installed model:

      inputs/<model_name>/production/<input_file_name.csv>
      
having the form:
  
      <img_0_label><img_0_channel_1>...<img_0_channel_2><img_0_channel_3>
      <img_1_label><img_1_channel_1>...<img_1_channel_2><img_1_channel_3>
      ...

## **Reproduction**

To reproduce the experiments with the installed models, call `$ bash run_routine.sh`. 

Results are written to `results/`.
