CC := g++
CFLAGS := -O3
INCLUDES := -I /usr/local/include/opencv4
LIBS := -L /usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
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
	$(CC) main.cpp $(PATH_OBJ)/leer_imagenes.o $(PATH_OBJ)/convolutional.o $(PATH_OBJ)/flatten.o $(PATH_OBJ)/poolingMax.o $(PATH_OBJ)/fullyconnected.o $(PATH_OBJ)/CNN.o $(CXX_FLAGS) -o $(PATH_BIN)/main -lcudart
	
$(PATH_OBJ)/main.o: main.cpp 
	$(CC) -c main.cpp $(CXX_FLAGS) -o $(PATH_OBJ)/main.o 

# OPENCV -------------------------------------
$(PATH_OBJ)/leer_imagenes.o: $(PATH_LEER_IMG)/leer_imagenes.cpp $(PATH_LEER_IMG)/leer_imagenes.h
	$(CC) -c $(PATH_LEER_IMG)/leer_imagenes.cpp  $(CXX_FLAGS) -o $(PATH_OBJ)/leer_imagenes.o 

# GPU -------------------------------------
$(PATH_OBJ)/CNN.o: $(PATH_CNN)/CNN.cpp $(PATH_CNN)/CNN.h
	$(CC) -c $(CXX_FLAGS) $(PATH_CNN)/CNN.cpp $(INCLUDES) $(LIBS) -o $(PATH_OBJ)/CNN.o
	
$(PATH_OBJ)/convolutional.o: $(PATH_CONV)/convolutional.cpp $(PATH_CONV)/convolutional.h
	$(CC) -c $(CXX_FLAGS)  $(PATH_CONV)/convolutional.cpp -o $(PATH_OBJ)/convolutional.o

$(PATH_OBJ)/flatten.o: $(PATH_FLAT)/flatten.cpp $(PATH_FLAT)/flatten.h
	$(CC) -c $(CXX_FLAGS)  $(PATH_FLAT)/flatten.cpp -o $(PATH_OBJ)/flatten.o

$(PATH_OBJ)/poolingMax.o: $(PATH_MAXPOOL)/poolingMax.cpp $(PATH_MAXPOOL)/poolingMax.h
	$(CC) -c $(CXX_FLAGS)  $(PATH_MAXPOOL)/poolingMax.cpp -o $(PATH_OBJ)/poolingMax.o

$(PATH_OBJ)/fullyconnected.o: $(PATH_FULLY)/fullyconnected.cpp $(PATH_FULLY)/fullyconnected.h
	$(CC) -c $(CXX_FLAGS)  $(PATH_FULLY)/fullyconnected.cpp -o $(PATH_OBJ)/fullyconnected.o

run: $(PATH_BIN)/main
	./$(PATH_BIN)/main

clean:
	rm -f $(PATH_OBJ)/*.o $(PATH_BIN)/main

