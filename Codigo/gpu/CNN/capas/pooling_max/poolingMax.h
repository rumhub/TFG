#ifndef MAXPOOL_H
#define MAXPOOL_H

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
#include <cfloat>
#include <limits>

using namespace std;

//#define BLOCK_SIZE 32

class PoolingMax
{
    private:
        int kernel_fils;
        int kernel_cols;
        int image_fils;
        int image_cols;
        int image_canales;
        int n_filas_eliminadas;
        
    public:
        PoolingMax(int kernel_fils, int kernel_cols, vector<vector<vector<float>>> &input);
        PoolingMax(){};

        // Aplica padding a un conjunto de imágenes 2D
        void forwardPropagation(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad);

        void forwardPropagationGPU(float *input, float *output, float *input_copy, const int &pad, const int &C, const int &H, const int &W);

        void backPropagation(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad_output);
        
        void backPropagationGPU(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad_output);

        void mostrar_tam_kernel();

        int get_kernel_fils(){return this->kernel_fils;};
        int get_kernel_cols(){return this->kernel_cols;};
        int get_image_canales(){return this->image_canales;};

};


// https://www.linkedin.com/pulse/implementation-from-scratch-forward-back-propagation-layer-coy-ulloa

#endif
