all:
	g++ -O3 -march=native -std=c++11 main.cpp reader.cpp convolution_layer/convolution.cpp pooling_layer/pooling.cpp -o lpa_cnn.out
