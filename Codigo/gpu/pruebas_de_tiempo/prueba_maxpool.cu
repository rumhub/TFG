#include <iostream>
#include <chrono>
#include "random"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <stdio.h>
#include "cuda_profiler_api.h"
#include <cfloat>
using namespace std;
using namespace std::chrono;

#define BLOCKSIZE 8

/*
    Volumen de entrada X(CxHxW)
    Volumen de salida Y(Cx(H/2)x(W/2))
    Tamaño de ventana K
*/
__global__ void maxpool(int C, int H, int W, int H_out, int W_out, int K, float *X, float *Y)
{
    // Memoria compartida dinámica
	//extern __shared__ float sdata[];

    int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);

    // Convertir de índices de hebra a índices de matriz 
  	int iy_Y = threadIdx.y + blockIdx.y * blockDim.y, ix_Y = threadIdx.x + blockIdx.x * blockDim.x,     // Coordenadas respecto a la mtriz de salida Y
        iy_X = iy_Y *K, ix_X = ix_Y *K,                                     // Coordenadas respecto a la matriz de entrada X
        idX = iy_X*W + ix_X, idY = iy_Y*W_out + ix_Y,
        tamY = C*H_out*W_out, tam_capaY = H_out * W_out,
        tamX = C*H*W, tam_capaX = H * W;

    float max = -1;

    if(iy_Y < H_out && ix_Y < W_out)
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

bool comprobarResultados(float *C1, float *C2, int M, int N)
{
    bool correcto = true;
    float epsilon = 0.000000001;
    for(int i=0; i<M; i++)
        for(int j=0; j<N; j++)
            if(abs(C1[i*N +j] - C2[i*N +j]) > epsilon)
            {
                correcto = false;
                //cout << C1[i*N +j] << " vs " << C2[i*N +j] << endl;
            }
    
    return correcto;
}

void printMatrix(float* matrix, int h, int w_) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w_; j++) 
            cout << matrix[i * w_ + j] << " ";
        cout << endl;
    }
}


int main()
{
    // Imagen de entrada X(CxHxW)
    int C = 2, H=6, W=6, K=3, H_out=H/K, W_out = W / K,
        bytes_X = C*H*W * sizeof(float),
        bytes_Y = C*H_out*W_out * sizeof(float);

    // Medidas de tiempo
    auto ini = high_resolution_clock::now();
    auto fin = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(fin - ini);
    
    // Reserva memoria en host
    float *h_X = (float*)malloc(bytes_X),
          *h_Y = (float*)malloc(bytes_Y);

    // Reserva de memoria en device
    float *d_X, *d_Y;
    cudaMalloc((void **) &d_X, bytes_X);
    cudaMalloc((void **) &d_Y, bytes_Y);

    // Inicializar las matrices ----------------
    // Inicializar X
    for(int i=0; i<C; i++)
        for(int j=0; j<H; j++)
            for(int k=0; k<W; k++)
                h_X[i*H*W + j*W + k] = i+j+k;
                //h_A[i*K + j] = (rand() % 100) + 1;

    // Inicializar Y
    for(int i=0; i<C; i++)
        for(int j=0; j<H_out; j++)
            for(int k=0; k<W_out; k++)
                h_Y[i*H_out*W_out + j*W_out + k] = 0.0;
                //h_A[i*K + j] = (rand() % 100) + 1;

    cout << "--------- X ---------" << endl;
    for(int i=0; i<C; i++)
    {
        for(int j=0; j<H; j++)
        {
            for(int k=0; k<W; k++)
                cout << h_X[i*H*W + j*W + k] << " ";
            cout << endl;
        }
        cout << endl;
    }



    // Copiar matrices A y B de CPU a GPU
    cudaMemcpy(d_X, h_X, bytes_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, bytes_Y, cudaMemcpyHostToDevice);
    
    // Maxpool en GPU
    dim3 block(BLOCKSIZE, BLOCKSIZE);
    //dim3 block(K, K);
    dim3 grid(ceil( (float)(W + block.x -1) / block.x), ceil((float)(H + block.y -1) / block.y));
    
    cout << "Grid de (" << grid.x << "x" << grid.y << ") " << endl;
    cout << "Cada bloque es de " << block.x << "x" << block.y << endl;

    int max_H_block = block.x;
    if(max_H_block > H)
        max_H_block = H;
    int n_iteraciones = max_H_block / K;
    size_t smem_tile = (K*K * n_iteraciones) *sizeof(float);
    cout << "Men: " << n_iteraciones << endl;
    ini = high_resolution_clock::now();
    maxpool<<<grid, block, smem_tile>>>(C, H, W, H_out, W_out, K, d_X, d_Y);
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    // Copiar resultados de GPU a CPU
    cudaMemcpy(h_Y, d_Y, bytes_Y, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_X, d_X, bytes_X, cudaMemcpyDeviceToHost);

    
    cout << " ------------------- DESPUÉS -------------------" << endl;
    /*
    cout << "--------- X ---------" << endl;
    for(int i=0; i<C; i++)
    {
        for(int j=0; j<H; j++)
        {
            for(int k=0; k<W; k++)
                cout << h_X[i*H*W + j*W + k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    */
    cout << "--------- Y ---------" << endl;
    for(int i=0; i<C; i++)
    {
        for(int j=0; j<H_out; j++)
        {
            for(int k=0; k<H_out; k++)
                cout << h_Y[i*H_out*H_out + j*H_out + k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    
    // Liberar memoria
    free(h_X); free(h_Y);
    cudaFree(d_X); cudaFree(d_Y);


    return 0;
}

