# lpa_cnn
Low Precision Arithmetic for Convolutional Neural Network Inference

-Dependencies-
1. Python 2.7 w/ numpy & pandas
2. R w/ rPython, Rcpp
3. gcc 5.4
4. Eigen 3

-Setup-
1. Add caffe Python module directory to $PYTHONPATH.
2. Have the following files in place for each desired model:
    models/<model_name>/<model_name.caffemodel>
    models/<model_name>/<model_name.prototxt>
3. Have the following input file in place for each installed model:
    inputs/<model_name>/production/<input_file_name.csv>
   having the form:
    <img_0_label><img_0_channel_1>...<img_0_channel_2><img_0_channel_3>
    <img_1_label><img_1_channel_1>...<img_1_channel_2><img_1_channel_3>
    ...

-Reproduction-
To reproduce the experiments with the installed models, call run/run_routine.py. 
Results are written to results/.
