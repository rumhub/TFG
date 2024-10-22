#ifndef CNN_INCLUDED
#define CNN_INCLUDED

#include <iostream>
#include "capas/convolutional/convolutional.h"
#include "capas/flatten/flatten.h"
#include "capas/fully_connected/fullyconnected.h"
#include "capas/pooling_max/poolingMax.h"
//#include "auxiliar/leer_imagenes.h"
#include "auxiliar/auxiliar.h"
#include <cudnn.h>
#include <cuda_runtime.h>

#include <vector>
#include <random>
#include <chrono>

using namespace std;
using namespace std::chrono;

class CNN
{
    private:

        // Descriptores cuDNN
        cudnnHandle_t cudnnHandle;
        cudnnTensorDescriptor_t dataTensor;
        cudnnTensorDescriptor_t *convBiasTensor = nullptr;
        cudnnTensorDescriptor_t *convGradWTensor = nullptr;
        cudnnTensorDescriptor_t *convOutTensor = nullptr;
        cudnnTensorDescriptor_t *convATensor = nullptr;
        cudnnTensorDescriptor_t *poolOutTensor = nullptr;
        cudnnConvolutionDescriptor_t *convDesc = nullptr;
        cudnnPoolingDescriptor_t *poolDesc = nullptr;
        cudnnFilterDescriptor_t *convFilterDesc = nullptr;
        cudnnActivationDescriptor_t *activation = nullptr;

        // Gradientes
        float *d_dpool = nullptr;
        float *d_dconv = nullptr;
        float *d_dconv_a = nullptr;
        float *d_dconv_a_copy = nullptr;
        float *d_dkernel = nullptr;

        Convolutional * convs = nullptr;      // Capas convolucionales
        PoolingMax * plms = nullptr;          // Capas MaxPool
        FullyConnected *fully = nullptr;      // Red Fullyconnected
        Flatten * flat = nullptr;             // Capa flatten
        vector<vector<vector<vector<float>>>> test_imgs;   // Imágenes de test
        vector<vector<float>> test_labels;             // Etiqueta de cada imagen de test
        int *padding = nullptr;                        // Nivel de padding en cada capa convolucional
        float lr;                           // Learning rate o Tasa de Aprendizaje
        int n_capas_conv;                   // Número de capas convolucionales
        int n_imagenes;                     // Número de imágenes
        int n_clases;                       // Número de clases
        int mini_batch;                     // Tamaño del minibatch
        int n_batches;                      // Número de batches
        int max_H, max_W, max_C;            // Dimensiones máximas de una imagen 3D

        // Índices
        int *i_conv_out = nullptr;
        int *i_conv_in = nullptr;
        int *i_plm_out = nullptr;
        int *i_plm_in = nullptr;
        int *i_w = nullptr;
        int *i_b = nullptr;
        int *indices = nullptr;

        // Batches
        int *batch = nullptr;           // Batches
        int *tam_batches = nullptr;     // Tamaño de los batches


        // Punteros device -------------------------------
        float *flat_outs_gpu = nullptr;
        float *d_flat_outs = nullptr;
        float *d_flat_outs_T = nullptr;
        float *d_img_in = nullptr;
        float *d_img_in_copy = nullptr;
        float *d_img_out = nullptr;
        float *d_conv_a_eval = nullptr;
        float *d_train_labels = nullptr;             // Etiqueta de cada imagen de training
        float *d_train_imgs = nullptr;

        float *d_grad_x_fully = nullptr;
        float *d_y_batch = nullptr;
        float *d_flat_outs_batch = nullptr;
        float *d_plms_outs = nullptr;
        float *d_plms_in_copys = nullptr;
        float *d_conv_grads_w = nullptr;
        float *d_conv_grads_w_total = nullptr;
        float *d_conv_grads_bias = nullptr;
        float *d_convs_outs = nullptr;
        float *d_conv_a = nullptr;
        int *d_indices = nullptr;
        int *d_batch = nullptr;

        // Bloques y Grids
        dim3 block_1D;
        dim3 block_2D;
        dim3 grid_1D;
        dim3 grid_2D;

        // Tamaños
        int tam_in_convs, tam_out_convs, tam_in_pools, tam_out_pools, tam_kernels_conv,
            tam_flat_out, n_bias_conv;

    public:
        // Constructor
        CNN(int *capas_conv, int n_capas_conv, int *tams_pool, int *padding, int *capas_fully, int n_capas_fully, int C, int H, int W, const float &lr, const int n_datos, const int mini_batch);
        
        // Destructor
        ~CNN(){free(padding); free(i_conv_out); free(i_conv_in); free(i_plm_out); free(i_plm_in); free(i_w);
               free(i_b); free(indices); free(batch); free(tam_batches);
               cudaFree(d_img_in); cudaFree(d_img_in_copy); cudaFree(d_img_out); cudaFree(d_conv_a_eval); cudaFree(d_flat_outs);
               cudaFree(d_train_labels); cudaFree(d_train_imgs); cudaFree(d_grad_x_fully); cudaFree(d_y_batch); cudaFree(d_flat_outs_batch);
               cudaFree(d_plms_outs); cudaFree(d_plms_in_copys); cudaFree(d_conv_grads_w); cudaFree(d_conv_grads_w_total); cudaFree(d_conv_grads_bias); 
               cudaFree(d_convs_outs); cudaFree(d_conv_a); cudaFree(d_indices); cudaFree(d_batch);
               destruir_handles(); free(convDesc); free(poolDesc); 
               cudaFree(d_dpool); cudaFree(d_dconv); cudaFree(d_dconv_a); 
               cudaFree(d_dconv_a_copy); cudaFree(d_dkernel);};

        // Mostrar arquitectura
        void mostrar_arquitectura();

        // Entrenar
        void train(int epocas, int mini_batch);

        // Evaluar el modelo
        void evaluar_modelo();

        // Establecer conjunto de entrenamiento y de test
        void set_train(float *x, float *y, int n_imgs, int n_clases, int C, int H, int W);

        // Manejo de handles
        void crear_handles(int mini_batch);
        void destruir_handles();
};

#endif
