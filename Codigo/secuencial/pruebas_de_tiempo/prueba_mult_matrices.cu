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

#define BLOCKSIZE 32

/*
    Para multiplicar A(MxK) x B(KxN) se necesita un bloque de (MxN).
    Cada hebra calcula un elemento de la matriz de salida C, multiplicando una fila de A * una columna de B.
    Usa memoria compartida.
    No usar este método, se queda muy rápido sin memoria por usar bloques tan grandes. No puedes
    multiplicar A(50x50) x B(50x50).
*/
__global__ void multiplicarMatrices2(int M, int N, int K, const float *A, const float *B, float *C)
{
    // Memoria compartida dinámica
	extern __shared__ float sdata[];

    // Convertir de índices de hebra a índices de matriz 
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x, 
        idA = iy*K + ix, idB = iy*N + ix;
    //int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);
    
    float sum = 0.0f;    

    // Punteros a A y B
    float *sA = sdata,
          *sB = sdata + blockDim.y*K;
    
    // Cada hebra carga en memoria compartida un elemento de A y otro de B
    if(iy < M && ix < K)
        sA[idA] = A[idA];

    if(iy < K && ix < N)
        sB[idB] = B[idB];

    // Sincronizar hebras
    __syncthreads();

    /*
    if(tid == 0)
    {
        
        printf(" --------- A --------- \n");
        for(int i=0; i<M; i++)
        {
            for(int j=0; j<K; j++)
                printf("%f ", A[i*K +j]);
            printf("\n");
        }
        

        printf(" --------- sA --------- \n");
        for(int i=0; i<M; i++)
        {
            for(int j=0; j<K; j++)
                printf("%f ", sA[i*K +j]);
            printf("\n");
        }

        /*
        printf(" --------- B --------- \n");
        for(int i=0; i<K; i++)
        {
            for(int j=0; j<N; j++)
                printf("%f ", B[i*N +j]);
            printf("\n");
        }
        

        printf(" --------- sB --------- \n");
        for(int i=0; i<K; i++)
        {
            for(int j=0; j<N; j++)
                printf("%f ", sB[i*N +j]);
            printf("\n");
        }
        
    }
    */
    
    // Multiplicación matricial
    if(iy < M && ix < N)
    {
        // Cada hebra calcula una posición de C (una fila de A * una columna de B)
        for (int i = 0; i < K; i++) 
            sum += sA[iy*K + i] * sB[ix + i*N];

        C[iy*N + ix] = sum;
    }
    //__syncthreads();
    
    /*
    if(tid == 0)
    {
        printf(" --------- C --------- \n");
        for(int i=0; i<M; i++)
        {
            for(int j=0; j<N; j++)
                printf("%f ", C[i*N +j]);
            printf("\n");
        }
    }
    */
  
}

/*
    Emplea tiles. Un tile por bloque. No usa memoria compartida
*/
__global__ void multiplicarMatrices3(int M, int N, int K, const float *A, const float *B, float *C)
{
    // Memoria compartida dinámica
	extern __shared__ float sdata[];

    // Convertir de índices de hebra a índices de matriz 
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;    
    
    // Multiplicación matricial
    if(iy < M && ix < N && threadIdx.y < blockDim.y && threadIdx.x < blockDim.x)
    {
        // Cada hebra calcula una posición de C (una fila de A * una columna de B)
        for (int i = 0; i < K; i++) 
            sum += A[iy*K + i] * B[ix + i*N];

        C[iy*N + ix] = sum;
    }
    

    /*
    // Mostrar Tiles
    if(iy < M && ix < N && threadIdx.y < blockDim.y && threadIdx.x < blockDim.x)
    {
        C[iy*N + ix] = blockIdx.x + blockIdx.y;
    }
    */
}

/*
    Emplea tiles. Un tile por bloque. Usa memoria compartida
*/
__global__ void multiplicarMatrices4(int M, int N, int K, const float *A, const float *B, float *C)
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

__global__ void multiplicarMatrices(int M, int N, int K, const float *A, const float *B, float *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

   if(row < M && col < N)
   {
        // Una hebra hace una fila de A * una col de B
        for (int i = 0; i < K; i++) {
            sum += A[row*K + i] * B[col + i*N];
        }

        C[row*N + col] = sum;
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
    // A = MxK, B = KxN, C = MxN
    int M = 2000, K=2000, N=2000,
        bytes_A = M*K * sizeof(float),
        bytes_B = K*N * sizeof(float),
        bytes_C = M*N * sizeof(float);

    dim3 block(N, M);
    dim3 grid(ceil( (float)(N + block.x -1) / block.x), ceil((float)(M + block.y -1) / block.y));

    //cout << "Grid de (" << grid.x << "x" << grid.y << ") " << endl;
    //cout << "Cada bloque es de " << block.x << "x" << block.y << endl;
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
          *h_C_gpu_3 = (float*)malloc(bytes_C),
          *h_C_gpu_4 = (float*)malloc(bytes_C);

    // Reserva de memoria en device
    float *d_A, *d_B, *d_C_gpu_1, *d_C_gpu_2, *d_C_gpu_3, *d_C_gpu_4;
    cudaMalloc((void **) &d_A, bytes_A);
    cudaMalloc((void **) &d_B, bytes_B);
    cudaMalloc((void **) &d_C_gpu_1, bytes_C);
    cudaMalloc((void **) &d_C_gpu_2, bytes_C);
    cudaMalloc((void **) &d_C_gpu_3, bytes_C);
    cudaMalloc((void **) &d_C_gpu_4, bytes_C);

    // Inicializar las matrices ----------------
    // Inicializar A
    for(int i=0; i<M; i++)
        for(int j=0; j<K; j++)
            h_A[i*K + j] = i+j;
            //h_A[i*K + j] = i;
            //h_A[i*K + j] = (rand() % 100) + 1;

    // Inicializar B
    for(int i=0; i<K; i++)
        for(int j=0; j<N; j++)
            h_B[i*N + j] = i+j;
            //h_B[i*N + j] = j;
            //h_B[i*N + j] = (rand() % 100) + 1;

    // Copiar matrices A y B de CPU a GPU
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);
    
    // Multiplicar las matrices en CPU
    ini = high_resolution_clock::now();
    multiplyMatrices(h_A, M, K, h_B, N, C_cpu);
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);
    
    //printMatrix(C_cpu, M, N);

    // Mostrar tiempo
    cout << "Tiempo CPU: " << duration.count() << " (us)" << endl;
    
    // ----------------------------------------- Multiplicar las matrices en GPU -----------------------------------------
    /*
    // Método simple -----------------------------------------
    ini = high_resolution_clock::now();
    multiplicarMatrices<<<grid, block>>>(M, N, K, d_A, d_B, d_C_gpu_1);
    cudaDeviceSynchronize();
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);
    
    // Copiar resultados de GPU a CPU
    cudaMemcpy(h_C_gpu_1, d_C_gpu_1, bytes_C, cudaMemcpyDeviceToHost);

    //printMatrix(h_C_gpu_1, M, N);
    
    // Mostrar resultado
    cout << "Tiempo GPU método simple: " << duration.count() << " (us)" << endl;
    

    // Método con memoria compartida -----------------------------------------
    
    size_t smem = (block.x * K + block.y * K) *sizeof(float);
    ini = high_resolution_clock::now();
    multiplicarMatrices2<<<grid, block, smem>>>(M, N, K, d_A, d_B, d_C_gpu_2);
    cudaDeviceSynchronize();
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);
    
    // Copiar resultados de GPU a CPU
    cudaMemcpy(h_C_gpu_2, d_C_gpu_2, bytes_C, cudaMemcpyDeviceToHost);

    //printMatrix(h_C_gpu_2, M, N);

    
    // Mostrar resultado
    cout << "Tiempo GPU método con memoria compartida: " << duration.count() << " (us)" << endl;
    */

    // Método sin memoria compartida con tiles -----------------------------------------
    dim3 block_tile(BLOCKSIZE, BLOCKSIZE);
    dim3 grid_tile(ceil( (float)(N + block_tile.x -1) / block_tile.x), ceil((float)(M + block_tile.y -1) / block_tile.y));
    
    cout << "Grid de (" << grid_tile.x << "x" << grid_tile.y << ") " << endl;
    cout << "Cada bloque es de " << block_tile.x << "x" << block_tile.y << endl;
    ini = high_resolution_clock::now();
    multiplicarMatrices3<<<grid_tile, block_tile>>>(M, N, K, d_A, d_B, d_C_gpu_3);
    cudaDeviceSynchronize();
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);
    
    // Copiar resultados de GPU a CPU
    cudaMemcpy(h_C_gpu_3, d_C_gpu_3, bytes_C, cudaMemcpyDeviceToHost);

    //printMatrix(h_C_gpu_3, M, N);
    
    // Mostrar resultado
    cout << "Tiempo GPU método sin memoria compartida con tiles: " << duration.count() << " (us)" << endl;


    // Método con memoria compartida con tiles -----------------------------------------
    // Cada tile ocupa (BLOCKSIZExBLOCKSIZE), cada bloque almacena en memoria compartida un tile de A y otro de B
    // Es decir, 2 tiles
    size_t smem_tile = (2*block_tile.x * block_tile.y+1) *sizeof(float);
    ini = high_resolution_clock::now();
    multiplicarMatrices4<<<grid_tile, block_tile, smem_tile>>>(M, N, K, d_A, d_B, d_C_gpu_4);
    cudaDeviceSynchronize();
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);
    //cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess) 
    //    printf("Error: %s\n", cudaGetErrorString(err));

    
    // Copiar resultados de GPU a CPU
    cudaMemcpy(h_C_gpu_4, d_C_gpu_4, bytes_C, cudaMemcpyDeviceToHost);

    //printMatrix(h_C_gpu_4, M, N);
    
    // Mostrar resultado
    cout << "Tiempo GPU método con memoria compartida con tiles: " << duration.count() << " (us)" << endl;

    // Comprobar resultados
    //if(comprobarResultados(C_cpu, h_C_gpu_1, M, N) && comprobarResultados(C_cpu, h_C_gpu_2, M, N) && comprobarResultados(C_cpu, h_C_gpu_3, M, N) && comprobarResultados(C_cpu, h_C_gpu_4, M, N))
    if(comprobarResultados(h_C_gpu_3, h_C_gpu_4, M, N))
        cout << "Todo correcto!" << endl;
    else
        cout << "Hay errores" << endl;
    
    

    // Liberar memoria
    free(h_A); free(h_B); free(C_cpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_gpu_1); cudaFree(d_C_gpu_2); cudaFree(d_C_gpu_3); cudaFree(d_C_gpu_4);


    return 0;
}