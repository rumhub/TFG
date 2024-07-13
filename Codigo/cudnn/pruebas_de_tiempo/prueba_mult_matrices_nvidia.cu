#include <iostream>
#include <chrono>
#include "random"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <stdio.h>
#include "cuda_profiler_api.h"
using namespace std;
using namespace std::chrono;

#define w 32
#define TILE_DIM w
#define BLOCK_SIZE TILE_DIM

// https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf

__global__ void simpleMultiply(float *a, float* b, float *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < TILE_DIM; i++) {
        sum += a[row*TILE_DIM+i] * b[i*N+col];
    }

    c[row*N+col] = sum;
}

__global__ void coalescedMultiply(float *a, float* b, float *c, int N)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    __syncwarp();

    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* b[i*N+col];
    }
    c[row*N+col] = sum;
}

__global__ void sharedABMultiply(float *a, float* b, float *c, int N)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
    bTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
    __syncthreads();

    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
    }
    c[row*N+col] = sum;
}

bool comprobarResultados(float *C1, float *C2, int M, int N)
{
    bool correcto = true;
    float epsilon = 0.000000001;
    for(int i=0; i<M; i++)
        for(int j=0; j<N; j++)
            if(abs(C1[i*N +j] - C2[i*N +j]) > epsilon)
                correcto = false;
    
    return correcto;
}

void printMatrix(float* matrix, int h, int w_) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w_; j++) 
            cout << matrix[i * w_ + j] << " ";
        cout << endl;
    }
}

void multiplyMatrices(float* m1, int rows1, int cols1, float* m2, int cols2, float* result) {
    for (int i = 0; i < rows1; i++) 
        for (int j = 0; j < cols2; j++) {
            result[i * cols2 + j] = 0.0f;

            for (int k = 0; k < cols1; k++) 
                result[i * cols2 + j] += m1[i * cols1 + k] * m2[k * cols2 + j];
            
        }
}

int main()
{
    // A = Mxw, B = wxN, C = MxN
    int M = 32, N= 32,
        bytes_A = M*w * sizeof(float),
        bytes_B = w*N * sizeof(float),
        bytes_C = M*N * sizeof(float);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N/w, M/w);

    // Medidas de tiempo
    auto ini = high_resolution_clock::now();
    auto fin = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(fin - ini);
    
    // Reserva memoria en host
    float *h_A = (float*)malloc(bytes_A),
          *h_B = (float*)malloc(bytes_B),
          *C_cpu = (float*)malloc(bytes_C),
          *h_C_gpu_1 = (float*)malloc(bytes_C),
          *h_C_gpu_2 = (float*)malloc(bytes_C),
          *h_C_gpu_3 = (float*)malloc(bytes_C);

    // Reserva de memoria en device
    float *d_A, *d_B, *d_C_gpu_1, *d_C_gpu_2, *d_C_gpu_3;
    cudaMalloc((void **) &d_A, bytes_A);
    cudaMalloc((void **) &d_B, bytes_B);
    cudaMalloc((void **) &d_C_gpu_1, bytes_C);
    cudaMalloc((void **) &d_C_gpu_2, bytes_C);
    cudaMalloc((void **) &d_C_gpu_3, bytes_C);

    // Inicializar las matrices ----------------
    // Inicializar A
    for(int i=0; i<M; i++)
        for(int j=0; j<w; j++)
            h_A[i*w + j] = 2.0;

    // Inicializar B
    for(int i=0; i<w; i++)
        for(int j=0; j<N; j++)
            h_B[i*N + j] = 3.0;

    // Copiar matrices A y B de CPU a GPU
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    // Multiplicar las matrices en CPU
    ini = high_resolution_clock::now();
    multiplyMatrices(h_A, M, w, h_B, N, C_cpu);
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);

    // Mostrar tiempo
    cout << "Tiempo CPU: " << duration.count() << " (us)" << endl;

    // ----------------------------------------- Multiplicar las matrices en GPU -----------------------------------------

    // Método simple -----------------------------------------
    ini = high_resolution_clock::now();
    //cudaProfilerStart();
    simpleMultiply<<<grid, block>>>(d_A, d_B, d_C_gpu_1, N);
    cudaDeviceSynchronize();
    //cudaProfilerStop();
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);
    
    // Copiar resultados de GPU a CPU
    cudaMemcpy(h_C_gpu_1, d_C_gpu_1, bytes_C, cudaMemcpyDeviceToHost);

    // Mostrar resultado
    cout << "Tiempo GPU método simple: " << duration.count() << " (us)" << endl;

    
    // Método memoria compartida con coalescencia -----------------------------------------
    ini = high_resolution_clock::now();
    coalescedMultiply<<<grid, block>>>(d_A, d_B, d_C_gpu_2, N);
    cudaDeviceSynchronize();
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);
    
    // Copiar resultados de GPU a CPU
    cudaMemcpy(h_C_gpu_2, d_C_gpu_2, bytes_C, cudaMemcpyDeviceToHost);

    // Mostrar tiempo
    cout << "Tiempo GPU método memoria compartida para coalescencia: " << duration.count() << " (us)" << endl;

    // Método multiplicación compartida -----------------------------------------
    ini = high_resolution_clock::now();
    sharedABMultiply<<<grid, block>>>(d_A, d_B, d_C_gpu_3, N);
    cudaDeviceSynchronize();
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);
    
    // Copiar resultados de GPU a CPU
    cudaMemcpy(h_C_gpu_3, d_C_gpu_3, bytes_C, cudaMemcpyDeviceToHost);

    // Mostrar tiempo
    cout << "Tiempo GPU método de multiplicación compartida: " << duration.count() << " (us)" << endl;

    // Comprobar resultados
    if(comprobarResultados(C_cpu, h_C_gpu_1, M, N) && comprobarResultados(h_C_gpu_1, h_C_gpu_2, M, N) && comprobarResultados(h_C_gpu_1, h_C_gpu_3, M, N))
        cout << "Todo correcto!" << endl;
    else
        cout << "Hay errores" << endl;

    

    // Liberar memoria
    free(h_A); free(h_B); free(C_cpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_gpu_1); cudaFree(d_C_gpu_2); cudaFree(d_C_gpu_3);


    return 0;
}