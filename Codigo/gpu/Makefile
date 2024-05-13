CC := g++
NVCC := nvcc
CFLAGS := -O3
CUDAFLAGS := -m64 -gencode arch=compute_50,code=sm_50
INCLUDES := -I /usr/local/include/opencv4
LIBS := -L /usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
CXX_FLAGS := $(CFLAGS) $(INCLUDES) $(LIBS)
PATH_LEER_IMG = CNN/auxiliar
PATH_CONV = CNN/capas/convolutional
PATH_FLAT = CNN/capas/flatten
PATH_MAXPOOL = CNN/capas/pooling_max
PATH_FULLY = CNN/capas/fully_connected
PATH_CNN = CNN

# CPP -------------------------------------
main: main.o leer_imagenes.o convolutional.o flatten.o poolingMax.o fullyconnected.o CNN.o
	$(CC) main.cpp leer_imagenes.o convolutional.o flatten.o poolingMax.o fullyconnected.o CNN.o $(INCLUDES) $(LIBS) -o main -lcudart
	
main.o: main.cpp 
	$(CC) -c main.cpp $(INCLUDES) $(LIBS) -o main.o 

# OPENCV -------------------------------------
leer_imagenes.o: $(PATH_LEER_IMG)/leer_imagenes.cpp $(PATH_LEER_IMG)/leer_imagenes.h
	$(CC) -c $(PATH_LEER_IMG)/leer_imagenes.cpp  $(INCLUDES) $(LIBS) -o leer_imagenes.o 

# GPU -------------------------------------
CNN.o: $(PATH_CNN)/CNN.cu $(PATH_CNN)/CNN.h
	$(NVCC) -c $(CUDAFLAGS) $(PATH_CNN)/CNN.cu $(INCLUDES) $(LIBS) -o CNN.o
	
convolutional.o: $(PATH_CONV)/convolutional.cu $(PATH_CONV)/convolutional.h
	$(NVCC) -c $(CUDAFLAGS)  $(PATH_CONV)/convolutional.cu -o convolutional.o

flatten.o: $(PATH_FLAT)/flatten.cu $(PATH_FLAT)/flatten.h
	$(NVCC) -c $(CUDAFLAGS)  $(PATH_FLAT)/flatten.cu -o flatten.o

poolingMax.o: $(PATH_MAXPOOL)/poolingMax.cu $(PATH_MAXPOOL)/poolingMax.h
	$(NVCC) -c $(CUDAFLAGS)  $(PATH_MAXPOOL)/poolingMax.cu -o poolingMax.o

fullyconnected.o: $(PATH_FULLY)/fullyconnected.cu $(PATH_FULLY)/fullyconnected.h
	$(NVCC) -c $(CUDAFLAGS)  $(PATH_FULLY)/fullyconnected.cu -o fullyconnected.o

clean:
	rm -f *.o main

