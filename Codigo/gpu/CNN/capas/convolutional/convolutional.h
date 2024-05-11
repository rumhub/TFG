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

#define TILE_DIM 8
#define BLOCK_SIZE 8

class Convolutional
{
    private:
        vector<vector<vector<vector<float>>>> w;    // Pesos de la capa convolucional
        vector<vector<vector<float>>> a;            // Valor de cada neurona antes de aplicar la función de activación

        int n_kernels;                          // Número de kernels 3D
        int kernel_fils;                        // Número de filas por kernel
        int kernel_cols;                        // Número ded columnas por kernel
        int kernel_depth;                       // Número de canales de profundidad por kernel

        vector<float> bias;                     // Un bias por filtro, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
        float lr;                               // Learning Rate o Tasa de Aprendizaje

    public:

        // Constructores
        Convolutional(int n_kernels, int kernel_fils, int kernel_cols, const vector<vector<vector<float>>> &input, float lr);
        Convolutional(){};

        // Funciones de activación
		float activationFunction(float x);
        float deriv_activationFunction(float x);

        // Propagación hacia delante
        void forwardPropagation(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &a);

        void forwardPropagationGEMM(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &a);

        // Retropropagación
        void backPropagation(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> output, const vector<vector<vector<float>>> &a, vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias, const int &pad);
        
        void backPropagationGEMM(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> output, const vector<vector<vector<float>>> &a, vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias, const int &pad);

        // Modificación de parámetros
        void generar_pesos();
        void reset_gradients(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias);
        void actualizar_grads(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias);
        void escalar_pesos(float clip_value, vector<float> &maxs, vector<float> &mins);
        void matrizTranspuesta(float* matrix, int rows, int cols);
        void unroll(int C, int n, int K, float *X, float *X_unroll);

        // Aplicar padding
        void aplicar_padding(vector<vector<vector<float>>> &imagen_3D, int pad);

        // Gets
        int get_kernel_fils(){return this->kernel_fils;};
        int get_kernel_cols(){return this->kernel_cols;};
        int get_kernel_depth(){return this->kernel_depth;};
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


/*
    Emplea tiles. Un tile por bloque. Usa memoria compartida
*/
__global__ void multiplicarMatricesGPU(int M, int N, int K, const float *A, const float *B, float *C)
{
    // Memoria compartida dinámica
	extern __shared__ float sdata[];

    // Convertir de índices de hebra a índices de matriz 
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x, 
        idA = iy*K + ix, idB = iy*N + ix, id_tile = threadIdx.y * blockDim.x + threadIdx.x, iy_tile_B = iy, ix_tile_A = ix;;
    int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);
    int n_tiles = (K + blockDim.x - 1) / blockDim.x;

    // Punteros a A y B
    float *sA = sdata,
          *sB = sdata + blockDim.x * blockDim.y;

    float sum = 0.0f;    

    int lim = blockDim.x;

    // Si tam_bloque > tam_A
    if(lim > K)
        lim = K;

    /*
        Para multiplicar A(MxK) x B(KxN) hay que multiplicar una fila de A x una columna de B
        Es decir, multiplicar KxK elementos y sumarlos
        Un tile es más pequeño que K -> Dividir K en tiles e iterar sobre ellos
    */
    for(int tile=0; tile < n_tiles; ++tile)
    {
        idA = iy*K + tile * blockDim.x + threadIdx.x;
        idB = (tile * blockDim.x + threadIdx.y)*N + ix;
       
        // Cargar submatrices de A y B en memoria compartida (tamaño tilex x tiley)
        // Cada hebra carga en memoria compartida un elemento de A y otro de B
        (iy < M && tile * blockDim.x + threadIdx.x < K) ? sA[id_tile] = A[idA] : sA[id_tile] = 0.0;
        (tile * blockDim.x + threadIdx.y < K && ix < N) ? sB[id_tile] = B[idB] : sB[id_tile] = 0.0;

        // Sincronizar hebras
        __syncthreads();

        // Realizar multiplicación matricial
        if(iy < M && ix < N)
        {
            // Si última iteración
            if(tile == n_tiles -1)
                lim = K - tile * blockDim.x;

            // Cada hebra calcula una posición de C (una fila de A * una columna de B)
            for (int i = 0; i < lim; i++) 
                sum += sA[threadIdx.y*blockDim.x + i] * sB[threadIdx.x + i*blockDim.x];
        }

        // Sincronizar hebras
        __syncthreads();
    }

    if(iy < M && ix < N)
        C[iy*N + ix] = sum;
}
