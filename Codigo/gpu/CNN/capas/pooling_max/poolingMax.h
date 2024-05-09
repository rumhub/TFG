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

using namespace std;

#define TILE_DIM 32
#define BLOCK_SIZE 32

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

        void forwardPropagationGPU(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad);

        void backPropagation(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad_output);

        void mostrar_tam_kernel();

        int get_kernel_fils(){return this->kernel_fils;};
        int get_kernel_cols(){return this->kernel_cols;};
        int get_image_canales(){return this->image_canales;};

};


// https://www.linkedin.com/pulse/implementation-from-scratch-forward-back-propagation-layer-coy-ulloa

__global__ void maxpool(int C, int H, int W, int K, float *X, float *Y, int pad)
{
    // Memoria compartida dinámica
	//extern __shared__ float sdata[];

    int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);

    // Convertir de índices de hebra a índices de matriz 
  	int H_out = H / K + 2*pad, W_out = W /K + 2*pad,
        iy_Y = threadIdx.y + blockIdx.y * blockDim.y, ix_Y = threadIdx.x + blockIdx.x * blockDim.x,     // Coordenadas respecto a la mtriz de salida Y
        iy_X = iy_Y *K, ix_X = ix_Y *K,                                     // Coordenadas respecto a la matriz de entrada X
        idX = iy_X*W + ix_X, idY = (iy_Y+pad)*W_out + (ix_Y+pad),
        tam_capaY = H_out * W_out,
        tam_capaX = H * W;

    float max = FLT_MIN;

    if(iy_Y < H_out-2*pad && ix_Y < W_out-2*pad)    // -2*pad para quitar el padding. Como solo hay padding en la salida, usamos las mismas hebras que si no hubiera ningún padding
    for(int c=0; c<C; c++)
    {
        max = FLT_MIN;
        for(int i=0; i<K; i++)
            for(int j=0; j<K; j++)
                if(max < X[idX + i*W + j] && iy_X + i < H && ix_X + j < W)
                    max = X[idX +i*W +j];

        // Establecer valor del píxel "IdY" de salida
        Y[idY] = max;

        // Actualizar índice para siguiente capa
        idY += tam_capaY;
        idX += tam_capaX;
    }
}