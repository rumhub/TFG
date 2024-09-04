#ifndef CONVOLUTIONAL_H
#define CONVOLUTIONAL_H

#include <vector>
#include <math.h>
#include <iostream>
#include <chrono>
#include "random"
#include "omp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

#define BLOCK_SIZE 8

class Convolutional
{
    private:
        // Kernel de pesos
        int n_kernels;          // Número de kernels 3D
        int kernel_fils;        // Filas del kernel de pesos
        int kernel_cols;        // Columnas del kernel de pesos

        // Dimensiones de la imagen de entrada
        int C;              // Canales de profundidad de la imgen de entrada
        int H;              // Filas de la imagen de entrada (por canal de profundidad)
        int W;              // Columnas de la imagen de entrada (por canal de profundidad)

        // Dimensiones de la imagen de salida
        int H_out;          // Filas de la imagen de salida (por canal de profundidad)
        int W_out;          // Filas de la imagen de salida (por canal de profundidad)

        // Dimensiones de la imagen de salida con padding
        int pad;            // Padding aplicado sobre la imagen de salida
        int H_out_pad;      // Filas de la imagen de salida con padding (por canal de profundidad)
        int W_out_pad;      // Columnas de la imagen de salida (por canal de profundidad)

        float lr;                               // Learning Rate o Tasa de Aprendizaje

        // Tamaño de bloque
        dim3 block;
        dim3 block_1D;

        // Tamaño de grid
        dim3 grid;          // Grid para realizar la propagación hacia delante

        // Dimensiones de los volúmenes "desenrrollados"
        int fils_input_unroll, cols_input_unroll;                       // Dimensiones de la entrada 'desenrrollada'
        int fils_w, cols_w;                                           // Dimensiones de los pesos como matriz 2D

        // Tamaños de los volúmenes "desenrrollados"
        int bytes_input_unroll;          // Espacio para la entrada 'desenrrollada'
        int bytes_output;              // Espacio para la salida
        int bytes_output_pad;              // Espacio para la salida
        int bytes_w;              // Espacio para los pesos  como matriz 2D

        int fils_output_unroll;
        int cols_output_unroll;
        int fils_matriz_pesos;
        int cols_matriz_pesos;
        int fils_input_back_unroll;
        int cols_input_back_unroll;
        int bytes_output_unroll;
        int bytes_matriz_pesos;
        int bytes_input_back_unroll;
        int bytes_bias;

        // Punteros device
        float *d_input_unroll = nullptr;        // Volumen de entrada 'desenrrollado'
        float *d_w = nullptr;
        float *d_bias = nullptr;

        float * d_output_unroll = nullptr;
        float *d_matriz_pesos = nullptr;
        float *d_input_back_unroll = nullptr;
        float * d_output_centrado = nullptr;
        float * d_output_pad = nullptr;
        float *d_input_back_unroll_T = nullptr;
        float *d_output_unroll_T = nullptr;
        float *d_sum_local = nullptr;

        size_t smem;            // Memoria compartida requerida por el kernel
        size_t smem_1D;
        float *d_max = nullptr;
        float *d_min = nullptr;

    public:

        // Constructores
        Convolutional(int n_kernels, int kernel_fils, int kernel_cols, int C, int H, int W, float lr, int pad);
        Convolutional(){};
        
        // Propagación hacia delante
        void forwardPropagation_vectores_externos(float *input, float *output, float *a);
        
        // Retropropagación
        void backPropagation_vectores_externos(float *input, float *output, float *a, float *grad_w, float *grad_bias);
        
        // Generar pesos
        void generar_pesos_ptr(float *w);
        
        // Actualizar pesos
        void actualizar_grads_vectores_externos(float *grad_w, float *grad_bias, int mini_batch);
        
        // Escalar pesos
        void escalar_pesos_vectores_externos(float clip_value);
        
        // Aplicar padding
        void aplicar_padding_ptr(float *imagen_3D, int C, int H, int W, int pad);
        
        // Copiar de una capa convolucional a otra
        void copiar(const Convolutional & conv);

        // Destructor
        ~Convolutional();

        // Funciones de activación
        float activationFunction(float x);
        float deriv_activationFunction(float x);

        // Gets
        int get_kernel_fils(){return this->kernel_fils;};
        int get_kernel_cols(){return this->kernel_cols;};
        int get_n_kernels(){return this->n_kernels;};
        int get_H(){return this->H;};
        int get_W(){return this->W;};
        int get_C(){return this->C;};
        int get_H_out(){return this->H_out;};
        int get_W_out(){return this->W_out;};
        int get_cols_input_unroll(){return this->cols_input_unroll;};
        float * get_dw(){return this->d_w;};

        // Debug
        void checkCudaErrors(cudaError_t err);
};

#endif
