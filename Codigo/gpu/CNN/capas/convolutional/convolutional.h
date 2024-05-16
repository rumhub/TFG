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
        vector<vector<vector<vector<float>>>> w;    // Pesos de la capa convolucional
        vector<vector<vector<float>>> a;            // Valor de cada neurona antes de aplicar la función de activación

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

        vector<float> bias;                     // Un bias por filtro, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
        float lr;                               // Learning Rate o Tasa de Aprendizaje

        // Tamaño de bloque
        dim3 block;

        // Tamaño de grid
        dim3 grid_forward;          // Grid para realizar la propagación hacia delante
        dim3 grid_grad_w;           // Grid para calcular el gradiente respecto a los pesos
        dim3 grid_grad_input;       // Grid para calcular el gradiente respecto a la entrada


        // Dimensiones de los volúmenes "desenrrollados"
        int fils_input_unroll, cols_input_unroll;                       // Dimensiones de la entrada 'desenrrollada'
        int fils_w, cols_w;                                           // Dimensiones de los pesos como matriz 2D

        // Tamaños de los volúmenes "desenrrollados"
        int bytes_input_unroll;          // Espacio para la entrada 'desenrrollada'
        int bytes_output;              // Espacio para la salida
        int bytes_w;              // Espacio para los pesos  como matriz 2D

        // CPU -------------------
        float *h_input_unroll = nullptr;        // Volumen de entrada 'desenrrollado'
        float *output_pad = nullptr;
        float *grad_w_it = nullptr;
        float *h_output_unroll = nullptr;
        float *h_matriz_pesos = nullptr;
        float *h_input_back_unroll = nullptr;

        int fils_output_unroll;
        int cols_output_unroll;
        int fils_matriz_pesos;
        int cols_matriz_pesos;
        int fils_input_back_unroll; 
        int cols_input_back_unroll;
        int bytes_output_unroll;
        int bytes_matriz_pesos;
        int bytes_input_back_unroll;
 
        // GPU -------------------
        // Punteros device
        float *d_input_unroll = nullptr;        // Volumen de entrada 'desenrrollado'
        float *d_a = nullptr; 
        float *d_w = nullptr; 

        float * d_output_unroll = nullptr; 
        float *d_matriz_pesos = nullptr; 
        float *d_input = nullptr; 
        float *d_input_back_unroll = nullptr; 
        float *d_output = nullptr; 
        float *d_grad_w = nullptr;


        float *w_ptr = nullptr;
        float *bias_ptr = nullptr;
        size_t smem;            // Memoria compartida requerida por el kernel

    public:

        // GPU ------------------------------------------
        Convolutional(int n_kernels, int kernel_fils, int kernel_cols, int C, int H, int W, float lr);
        void forwardPropagationGEMM(float *input, float *output, float *a);
        void backPropagationGEMM(float *input, float *output, float *a, float *grad_w, float *grad_bias);
        void generar_pesos_ptr();
        ~Convolutional(){cudaFree(d_input_unroll); cudaFree(d_a); cudaFree(d_w); cudaFree(w_ptr); cudaFree(bias_ptr); free(h_input_unroll); free(output_pad); free(grad_w_it);
                        free(h_output_unroll); free(h_matriz_pesos); free(h_input_back_unroll); cudaFree(d_output_unroll); cudaFree(d_matriz_pesos); cudaFree(d_input);
                        cudaFree(d_input_back_unroll);};

        // CPU ------------------------------------------

        // Constructores
        Convolutional(int n_kernels, int kernel_fils, int kernel_cols, const vector<vector<vector<float>>> &input, float lr);
        Convolutional(){};

        // Funciones de activación
		float activationFunction(float x);
        float deriv_activationFunction(float x);

        // Propagación hacia delante
        void forwardPropagation(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &a);

        // Retropropagación
        void backPropagation(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> output, const vector<vector<vector<float>>> &a, vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias);
        

        // Modificación de parámetros
        void generar_pesos();
        void reset_gradients(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias);
        void actualizar_grads(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias);
        void escalar_pesos(float clip_value);
        void matrizTranspuesta(float* matrix, int rows, int cols);
        void unroll(int C, int n, int K, float *X, float *X_unroll);
        void unroll_1dim(int C, int H, int W, int K, float *X, float *X_unroll);
        void unroll_3dim(int C, int H, int W, int K, float *X, float *X_unroll);

        // Aplicar padding
        void aplicar_padding(vector<vector<vector<float>>> &imagen_3D, int pad);

        // Gets
        int get_kernel_fils(){return this->kernel_fils;};
        int get_kernel_cols(){return this->kernel_cols;};
        int get_n_kernels(){return this->n_kernels;};
        vector<vector<vector<vector<float>>>> get_pesos(){return this->w;};
        vector<float> get_bias(){return this->bias;};
        // https://calvinfeng.gitbook.io/machine-learning-notebook/supervised-learning/convolutional-neural-network/convolution_operation

        // Debug
        void set_w(const vector<vector<vector<vector<float>>>> &w_){this->w = w_;};
        void set_b(const vector<float> &b){this->bias = b;};
        void printMatrix_3D(float* matrix, int C, int n);
        void printMatrix(float* matrix, int h, int w);
        void printMatrix_vector(const vector<vector<vector<float>>> &X);
        void multiplicarMatrices(float* m1, int rows1, int cols1, float* m2, int cols2, float* result);
        void printMatrix_4D(float* matrix, int F, int C, int n);
};

#endif