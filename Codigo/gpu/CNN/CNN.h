#ifndef CNN_INCLUDED
#define CNN_INCLUDED

#include <iostream>
#include "capas/convolutional/convolutional.h"
#include "capas/flatten/flatten.h"
#include "capas/fully_connected/fullyconnected.h"
#include "capas/pooling_max/poolingMax.h"
//#include "auxiliar/leer_imagenes.h"
#include "auxiliar/auxiliar.h"

#include <vector>

class CNN
{
    private:
        Convolutional * convs;      // Capas convolucionales
        PoolingMax * plms;          // Capas MaxPool
        FullyConnected *fully;      // Red Fullyconnected
        Flatten * flat;             // Capa flatten
        vector<vector<vector<vector<float>>>> train_imgs;   // Imágenes de entrenamiento
        vector<vector<vector<vector<float>>>> test_imgs;   // Imágenes de test
        vector<vector<vector<vector<float>>>> outputs;      // Imágenes con las dimensiones del output de capas conv y pooling
        vector<vector<float>> train_labels;             // Etiqueta de cada imagen de training
        vector<vector<float>> test_labels;             // Etiqueta de cada imagen de test
        vector<int> padding;                        // Nivel de padding en cada capa convolucional
        float lr;                           // Learning rate o Tasa ded Aprendizaje
        int n_capas_conv;                   // Número de capas convolucionales

    public:
        // Constructor
        CNN(const vector<vector<int>> &capas_conv, const vector<vector<int>> &tams_pool, const vector<int> &padding, vector<int> &capas_fully, const vector<vector<vector<float>>> &input, const float &lr);

        // Leer imágenes de entrada
        void leer_imagenes();
        void leer_imagenes_mnist(const int n_imagenes, const int n_clases);
        void leer_imagenes_cifar10(const int &n_imagenes_train, const int &n_imagenes_test, const int n_clases);
        
        // Mostrar arquitectura
        void mostrar_arquitectura();

        // Entrenar
        void train(int epocas, int mini_batch);
        
        // Evaluar el modelo
        void evaluar_modelo();
        void evaluar_modelo_en_test();

        // Modificar imagen
        void padding_interno(vector<vector<vector<float>>> &input, const int &pad);
};

#endif
