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
        Convolutional * convs = nullptr;      // Capas convolucionales
        PoolingMax * plms = nullptr;          // Capas MaxPool
        FullyConnected *fully = nullptr;      // Red Fullyconnected
        Flatten * flat = nullptr;             // Capa flatten
        vector<vector<vector<vector<float>>>> test_imgs;   // Imágenes de test
        float *train_labels = nullptr;             // Etiqueta de cada imagen de training
        vector<vector<float>> test_labels;             // Etiqueta de cada imagen de test
        int *padding = nullptr;                        // Nivel de padding en cada capa convolucional
        float lr;                           // Learning rate o Tasa ded Aprendizaje
        int n_capas_conv;                   // Número de capas convolucionales
        int n_imagenes;
        int n_clases;
        int max_H, max_W, max_C;
        int *i_conv_out = nullptr;
        int *i_conv_in = nullptr;
        int *i_plm_out = nullptr;
        int *i_plm_in = nullptr;
        int *i_w = nullptr;
        int *i_b = nullptr;


        // Punteros device -------------------------------
        float *flat_outs_gpu = nullptr;
        float *d_flat_outs = nullptr;
        float *d_flat_outs_T = nullptr;
        float *d_img_in = nullptr;
        float *d_img_in_copy = nullptr;
        float *d_img_out = nullptr;
        float *d_conv_a = nullptr;
        float *d_train_labels = nullptr;             // Etiqueta de cada imagen de training
        float *d_train_imgs = nullptr;

    public:
        // Constructor
        CNN(int *capas_conv, int n_capas_conv, int *tams_pool, int *padding, int *capas_fully, int n_capas_fully, int C, int H, int W, const float &lr, const int n_datos);
        ~CNN(){free(train_labels); free(padding); free(i_conv_out); free(i_conv_in); free(i_plm_out); free(i_plm_in); free(i_w);
               free(i_b);
               cudaFree(d_img_in); cudaFree(d_img_in_copy); cudaFree(d_img_out); cudaFree(d_conv_a); cudaFree(d_flat_outs);
               cudaFree(d_train_labels); cudaFree(d_train_imgs);};

        // Mostrar arquitectura
        void mostrar_arquitectura();

        // Entrenar
        void train(int epocas, int mini_batch);

        // Evaluar el modelo
        void evaluar_modelo();
        void evaluar_modelo_en_test();

        // Modificar imagen
        void padding_interno(vector<vector<vector<float>>> &input, const int &pad);

        void padding_interno_ptr(float *input, int C, int H, int W, const int &pad);


        void set_train(float *x, float *y, int n_imgs, int n_clases, int C, int H, int W);
        void set_test(const vector<vector<vector<vector<float>>>> &x, const vector<vector<float>> &y){this->test_imgs = x; this->test_labels = y;};

        // Debug
        void mostrar_ptr(float *x, int C, int H, int W);
};

#endif
