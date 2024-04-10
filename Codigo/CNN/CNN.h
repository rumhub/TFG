#include <iostream>
#include "capas/convolutional/convolutional.cpp"
#include "capas/flatten/flatten.cpp"
#include "capas/fully_connected/fullyconnected.cpp"
#include "capas/pooling_max/poolingMax.cpp"
#include "auxiliar/leer_imagenes.cpp"
#include "auxiliar/auxiliar.cpp"

#include <vector>

#define THREAD_NUM 8

class CNN
{
    private:
        Convolutional * convs;   // Capas convolucionales
        PoolingMax * plms;          // Capas MaxPool
        FullyConnected *fully;        // Red Fullyconnected
        Flatten * flat;             // Capa flatten
        vector<vector<vector<vector<float>>>> train_imgs;   // Imágenes de entrenamiento
        vector<vector<vector<vector<float>>>> outputs;      // Imágenes con las dimensiones del output de capas conv y pooling
        vector<vector<float>> train_labels;             // Etiqueta de cada imagen de training
        vector<int> padding;
        float lr;                           // Learning rate
        int n_capas_conv;

    public:
        CNN(const vector<vector<int>> &capas_conv, const vector<vector<int>> &tams_pool, const vector<int> &padding, vector<int> &capas_fully, const vector<vector<vector<float>>> &input, const float &lr);

        void leer_imagenes();
        void leer_imagenes_mnist(const int n_imagenes, const int n_clases);
        void leer_imagenes_cifar10(const int n_imagenes, const int n_clases);
        
        void mostrar_arquitectura();

        void train(int epocas, int mini_batch);
        void accuracy(vector<vector<float>> &fully_a, vector<vector<float>> &fully_z);

        // DEBUG ----------------------------------------
        void prueba();
};