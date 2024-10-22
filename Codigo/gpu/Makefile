CC := g++
NVCC := nvcc
CFLAGS := -O3
CUDAFLAGS := -O3 -m64 -gencode arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets
INCLUDES := -I /usr/local/include/opencv4
LIBS := -L /usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
CXX_FLAGS := $(CFLAGS) $(INCLUDES) $(LIBS)
PATH_LEER_IMG = CNN/auxiliar
PATH_CONV = CNN/capas/convolutional
PATH_MAXPOOL = CNN/capas/pooling_max
PATH_FULLY = CNN/capas/fully_connected
PATH_CNN = CNN
PATH_OBJ = obj
PATH_BIN = bin

# CPP -------------------------------------
$(PATH_BIN)/main: $(PATH_OBJ)/main.o $(PATH_OBJ)/leer_imagenes.o $(PATH_OBJ)/convolutional.o $(PATH_OBJ)/poolingMax.o $(PATH_OBJ)/fullyconnected.o $(PATH_OBJ)/CNN.o
	$(CC) main.cpp $(PATH_OBJ)/leer_imagenes.o $(PATH_OBJ)/convolutional.o $(PATH_OBJ)/poolingMax.o $(PATH_OBJ)/fullyconnected.o $(PATH_OBJ)/CNN.o $(CXX_FLAGS) -o $(PATH_BIN)/main -lcudart
	
$(PATH_OBJ)/main.o: main.cpp 
	$(CC) -c main.cpp $(CXX_FLAGS) -o $(PATH_OBJ)/main.o 

# OPENCV -------------------------------------
$(PATH_OBJ)/leer_imagenes.o: $(PATH_LEER_IMG)/leer_imagenes.cpp $(PATH_LEER_IMG)/leer_imagenes.h
	$(CC) -c $(PATH_LEER_IMG)/leer_imagenes.cpp  $(CXX_FLAGS) -o $(PATH_OBJ)/leer_imagenes.o 

# GPU -------------------------------------
$(PATH_OBJ)/CNN.o: $(PATH_CNN)/CNN.cu $(PATH_CNN)/CNN.h
	$(NVCC) -c $(CUDAFLAGS) $(PATH_CNN)/CNN.cu $(INCLUDES) $(LIBS) -o $(PATH_OBJ)/CNN.o
	
$(PATH_OBJ)/convolutional.o: $(PATH_CONV)/convolutional.cu $(PATH_CONV)/convolutional.h
	$(NVCC) -c $(CUDAFLAGS)  $(PATH_CONV)/convolutional.cu -o $(PATH_OBJ)/convolutional.o

$(PATH_OBJ)/poolingMax.o: $(PATH_MAXPOOL)/poolingMax.cu $(PATH_MAXPOOL)/poolingMax.h
	$(NVCC) -c $(CUDAFLAGS)  $(PATH_MAXPOOL)/poolingMax.cu -o $(PATH_OBJ)/poolingMax.o

$(PATH_OBJ)/fullyconnected.o: $(PATH_FULLY)/fullyconnected.cu $(PATH_FULLY)/fullyconnected.h
	$(NVCC) -c $(CUDAFLAGS)  $(PATH_FULLY)/fullyconnected.cu -o $(PATH_OBJ)/fullyconnected.o

run: $(PATH_BIN)/main
	./$(PATH_BIN)/main

clean:
	rm -f $(PATH_OBJ)/*.o $(PATH_BIN)/main

