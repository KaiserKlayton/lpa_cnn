CC=g++
CFLAGS=-c -O3 -march=native -std=c++11
SOURCES=main.cpp reader.cpp convolution_layer/convolution.cpp pooling_layer/pooling.cpp fully_connected_layer/fully_connected.cpp relu_layer/relu.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=lpa_cnn.out

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
