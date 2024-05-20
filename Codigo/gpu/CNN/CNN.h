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
#include <random>
#include <chrono>

using namespace std;
using namespace std::chrono;

class CNN
{
    private:
        Convolutional * convs;      // Capas convolucionales
        PoolingMax * plms;          // Capas MaxPool
        FullyConnected *fully;      // Red Fullyconnected
        Flatten * flat;             // Capa flatten
        vector<vector<vector<vector<float>>>> h_train_imgs;   // Imágenes de entrenamiento
        vector<vector<vector<vector<float>>>> test_imgs;   // Imágenes de test
        vector<vector<vector<vector<float>>>> outputs;      // Imágenes con las dimensiones del output de capas conv y pooling
        vector<vector<float>> h_train_labels;             // Etiqueta de cada imagen de training
        vector<vector<float>> test_labels;             // Etiqueta de cada imagen de test
        int *padding = nullptr;                        // Nivel de padding en cada capa convolucional
        float lr;                           // Learning rate o Tasa ded Aprendizaje
        int n_capas_conv;                   // Número de capas convolucionales

    public:
        // Constructor
        CNN(int *capas_conv, int n_capas_conv, int *tams_pool, int *padding, int *capas_fully, int n_capas_fully, int C, int H, int W, const float &lr);
        ~CNN(){free(padding);}

        // Mostrar arquitectura
        void mostrar_arquitectura();

        // Entrenar
        void train(int epocas, int mini_batch);
        
        // Evaluar el modelo
        void evaluar_modelo();
        void evaluar_modelo_en_test();

        // Modificar imagen
        void padding_interno(vector<vector<vector<float>>> &input, const int &pad);

        void set_train(const vector<vector<vector<vector<float>>>> &x, const vector<vector<float>> &y);
        void set_test(const vector<vector<vector<vector<float>>>> &x, const vector<vector<float>> &y){this->test_imgs = x; this->test_labels = y;};
};

#endif
