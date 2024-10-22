CC := g++
NVCC := nvcc
CFLAGS := -O3
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SMXX    := -gencode arch=compute_70,code=compute_70
CUDAFLAGS   ?= -O3 $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SMXX)
CUDA_INCLUDE := -I /usr/local/cuda/include
CUDA_LIB := -L /usr/local/cuda/lib64 -lcudart
INCLUDES := -I /usr/include/opencv4 $(CUDA_INCLUDE)
LIBS := -L /usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs $(CUDA_LIB)
CXX_FLAGS := $(CFLAGS) $(INCLUDES) $(LIBS)
PATH_LEER_IMG = CNN/auxiliar
PATH_CONV = CNN/capas/convolutional
PATH_FLAT = CNN/capas/flatten
PATH_MAXPOOL = CNN/capas/pooling_max
PATH_FULLY = CNN/capas/fully_connected
PATH_CNN = CNN
PATH_OBJ = obj
PATH_BIN = bin

# CPP -------------------------------------
$(PATH_BIN)/main: $(PATH_OBJ)/main.o $(PATH_OBJ)/leer_imagenes.o $(PATH_OBJ)/convolutional.o $(PATH_OBJ)/flatten.o $(PATH_OBJ)/poolingMax.o $(PATH_OBJ)/fullyconnected.o $(PATH_OBJ)/CNN.o
	$(CC) main.cpp $(PATH_OBJ)/leer_imagenes.o $(PATH_OBJ)/convolutional.o $(PATH_OBJ)/flatten.o $(PATH_OBJ)/poolingMax.o $(PATH_OBJ)/fullyconnected.o $(PATH_OBJ)/CNN.o $(CXX_FLAGS) -o $(PATH_BIN)/main -lcudnn
	
$(PATH_OBJ)/main.o: main.cpp 
	$(CC) -c main.cpp $(CXX_FLAGS) -o $(PATH_OBJ)/main.o 

# OPENCV -------------------------------------
$(PATH_OBJ)/leer_imagenes.o: $(PATH_LEER_IMG)/leer_imagenes.cpp $(PATH_LEER_IMG)/leer_imagenes.h
	$(CC) -c $(PATH_LEER_IMG)/leer_imagenes.cpp  $(CXX_FLAGS) -o $(PATH_OBJ)/leer_imagenes.o 

# GPU -------------------------------------
$(PATH_OBJ)/CNN.o: $(PATH_CNN)/CNN.cu $(PATH_CNN)/CNN.h
	$(NVCC) -c $(CUDAFLAGS) $(INCLUDES) $(LIBS) $(PATH_CNN)/CNN.cu -o $(PATH_OBJ)/CNN.o -lcudnn

	
$(PATH_OBJ)/convolutional.o: $(PATH_CONV)/convolutional.cu $(PATH_CONV)/convolutional.h
	$(NVCC) -c $(CUDAFLAGS)  $(PATH_CONV)/convolutional.cu $(INCLUDES) -o $(PATH_OBJ)/convolutional.o

$(PATH_OBJ)/flatten.o: $(PATH_FLAT)/flatten.cu $(PATH_FLAT)/flatten.h
	$(NVCC) -c $(CUDAFLAGS)  $(PATH_FLAT)/flatten.cu $(INCLUDES) -o $(PATH_OBJ)/flatten.o

$(PATH_OBJ)/poolingMax.o: $(PATH_MAXPOOL)/poolingMax.cu $(PATH_MAXPOOL)/poolingMax.h
	$(NVCC) -c $(CUDAFLAGS)  $(PATH_MAXPOOL)/poolingMax.cu $(INCLUDES) -o $(PATH_OBJ)/poolingMax.o

$(PATH_OBJ)/fullyconnected.o: $(PATH_FULLY)/fullyconnected.cu $(PATH_FULLY)/fullyconnected.h
	$(NVCC) -c $(CUDAFLAGS)  $(PATH_FULLY)/fullyconnected.cu $(INCLUDES) -o $(PATH_OBJ)/fullyconnected.o

pool: $(PATH_MAXPOOL)/poolingMax.cu $(PATH_MAXPOOL)/poolingMax.h
	$(NVCC) $(CUDAFLAGS)  $(PATH_MAXPOOL)/poolingMax.cu $(INCLUDES) -o pool
	
run: $(PATH_BIN)/main
	./$(PATH_BIN)/main

clean:
	rm -f $(PATH_OBJ)/*.o $(PATH_BIN)/main

