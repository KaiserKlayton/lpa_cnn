# **lpa_cnn**

Low Precision Arithmetic For Convolutional Neural Network Inference

`lpa_cnn` is a benchmarking tool for comparing accuracies and speeds of convolutional neural networks run with different arithmetic precision modes for the convolutions. The first mode is the baseline Caffe implentation, the second is floating point arithmetic with eigen, and the third is quantized mode, which uses integer airthmetic through gemmlowp.

### **Dependencies**

gcc 5.4 w/ Eigen 3 & Armadillo

Python 2.7 w/ NumPy & PIL

R w/ gtools & stringr

Caffe (see `Setup` below for installation)

### **Setup**

Install Caffe as `caffe/` (in root directory), following the guide @ https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-CPU-Only/.

Have the following files in place for each desired model:

      models/<model_name>/<model_name.caffemodel>
      models/<model_name>/<model_name.prototxt>

adjusting the .prototxt input layer to receive one image as follows:

      1 x <depth> x <width> x <height>

Have the following input file in place for each installed model:

      inputs/<model_name>/production/<input_file_name.csv>

having the form:

      <img_0_label><img_0_channel_1>...<img_0_channel_2><img_0_channel_3>
      <img_1_label><img_1_channel_1>...<img_1_channel_2><img_1_channel_3>
      ...

### **Reproduction**

To run experiments with the installed models, call `$ bash run_routine.sh`.

Results are written to `results/`.

### **Installing new models**

A great resource for finding new Caffe models is Model Zoo @ https://github.com/BVLC/caffe/wiki/Model-Zoo

To install a new model, follow the `Setup` directions above, providing an appropriate and consistent model name as `<model_name>`.

`NOTE` that when preparing .prototxt files, `lpa_cnn` supports the following parameters:

      layer_types = ['convolution', 'pooling', 'relu', 'eltwise', 'innerproduct', 'scale', 'batchnorm']
      param_types = ['num_output', 'pad', 'kernel_size', 'stride', 'bias_term', 'pool']
      special_types = ['shape', 'input_dim']
      shape_dims = ['n','d','w','h']

`NOTE` that batch processing is not supported.
