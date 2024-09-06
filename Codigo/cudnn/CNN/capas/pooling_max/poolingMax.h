#ifndef MAXPOOL_H
#define MAXPOOL_H

#include <vector>
#include <math.h>
#include <iostream>
#include <chrono>
#include "random"
#include "omp.h"
#include <stdlib.h>
#include <stdio.h>
#include <cfloat>
#include <limits>

using namespace std;

class PoolingMax
{
    private:
        // Kernel de pesos
        int kernel_fils;        // Filas del kernel de pesos
        int kernel_cols;        // Columnas del kernel de pesos

        // Dimensiones de la imagen de entrada
        int C;              // Canales de profundidad de la imgen de entrada
        int H;              // Filas de la imagen de entrada (por canal de profundidad)
        int W;              // Columnas de la imagen de entrada (por canal de profundidad)

        // Dimensiones de la imagen de salida
        int H_out;          // Filas de la imagen de salida (por canal de profundidad)
        int W_out;          // Filas de la imagen de salida (por canal de profundidad)

        // Padding
        int pad;            // Cantidad de padding a aplicar sobre la imagen de salida

        // Bytes requeridos por cada imagen
        int bytes_input;    // Bytes de la imagen de entrada
        int bytes_output;   // Bytes de la imagen de salida

        // Tamaño de bloque
        dim3 block;

        // Tamaño de grid
        dim3 grid;

        // Posiciones en GPU
        float *d_input = nullptr;       // Imagen de entrada
        float *d_input_copy = nullptr;      // Copia de la imagen de entrada
        float *d_output = nullptr;      // Imagen de salida
        bool liberar_memoria;

    public:
        PoolingMax(int kernel_fils, int kernel_cols, vector<vector<vector<float>>> &input);
        PoolingMax(int kernel_fils, int kernel_cols, int C, int H, int W);
        PoolingMax(){liberar_memoria = false;};
        void copiar(const PoolingMax & plm);

        ~PoolingMax(){if(this->liberar_memoria){cudaFree(d_input); cudaFree(d_input_copy); cudaFree(d_output);}};

        // CPU -------------------------------------------
        // Aplica padding a un conjunto de imágenes 2D
        void forwardPropagation(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad);
        void backPropagation(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad_output);

        // GPU ------------------------------------------
        void forwardPropagationGPU(float *input, float *output, float *input_copy);
        void forwardPropagation_vectores_externos(float *input, float *output, float *input_copy);
        void backPropagationGPU(float *input, float *output, float *input_copy);
        void backPropagation_vectores_externos(float *input, float *output, float *input_copy);

        // Comunes ------------------------------------------
        void mostrar_tam_kernel();
        int get_kernel_fils(){return this->kernel_fils;};
        int get_kernel_cols(){return this->kernel_cols;};
        int get_image_canales(){return this->C;};
        int get_bytes_input(){return this->bytes_input;};
        int get_bytes_output(){return this->bytes_output;};
        int get_pad(){return this->pad;};

        // GPU
        float * get_d_input(){return this->d_input;};
        float * get_d_input_copy(){return this->d_input_copy;};
        float * get_d_output(){return this->d_output;};
        dim3 get_block(){return this->block;};
        dim3 get_grid(){return this->grid;};
        int get_H(){return this->H;};
        int get_W(){return this->W;};
        int get_C(){return this->C;};
        int get_H_out(){return this->H_out;};
        int get_W_out(){return this->W_out;};
};

#endif
