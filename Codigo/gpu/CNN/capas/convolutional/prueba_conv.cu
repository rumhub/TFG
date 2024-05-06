#include "convolutional.cpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>         // Para valores random
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

// https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf
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

// Function to print a matrix
void printMatrix_vector(const vector<vector<vector<float>>> &X) {
    for (int i = 0; i < X.size(); i++) {
        for (int j = 0; j < X[i].size(); j++) {
            for (int k = 0; k < X[i][j].size(); k++) 
                cout << X[i][j][k] << " ";
            cout << endl;
        }
        cout << endl;
    }
}

void printMatrix_3D(float* matrix, int C, int n) {
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) 
                cout << matrix[i*n*n +j*n +k] << " ";
            cout << endl;
        }
        cout << endl;
    }
}

void printMatrix_4D(float* matrix, int F, int C, int n) {
    for (int f = 0; f < F; f++) {
        for (int i = 0; i < C; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) 
                    cout << matrix[f*C*n*n + i*n*n +j*n +k] << " ";
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }
}

void unroll(int C, int n, int K, float *X, float *X_unroll){
    int H_out = n-K+1;
    int W_out = n-K+1;
    int w_base;
    int W = H_out * W_out;

    for(int c=0; c<C; c++)
    {
        w_base = c * (K*K);
        for(int p=0; p<K; p++)
            for(int q=0; q<K; q++)
                for(int h=0; h<H_out; h++){
                    int h_unroll = w_base + p*K + q;
                    for(int w=0; w < W_out; w++){
                        int w_unroll = h * W_out + w;
                        X_unroll[h_unroll*W + w_unroll] = X[c*n*n + (h+p)*n + (w+q)];
                    }
                }

        
    } 
}

void printMatrix(float* matrix, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) 
            cout << matrix[i * w + j] << " ";
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

void transposeMatrix(float* matrix, int rows, int cols) {
    // Allocate a new matrix to hold the transposed data
    float* transposedMatrix = (float*)malloc(cols * rows * sizeof(float));
    
    if (transposedMatrix == nullptr) {
        cerr << "Memory allocation for transposed matrix failed" << endl;
        exit(1);
    }

    // Transpose the matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Copy the element from original matrix at position (i, j) to
            // transposed matrix at position (j, i)
            transposedMatrix[j * rows + i] = matrix[i * cols + j];
        }
    }

    // Copy the transposed data back to the original matrix
    // This requires the original matrix to be able to hold the new dimensions
    // i.e., it must have space for 'cols * rows' elements
    for (int i = 0; i < cols * rows; i++) {
        matrix[i] = transposedMatrix[i];
    }

    // Free the allocated memory for the transposed matrix
    free(transposedMatrix);
}

int main()
{
    // -----------------------------------------------------------------------------------------------------
    // Método estándar
    // -----------------------------------------------------------------------------------------------------
    int n_kernels = 2, K=2, H=3, W=H, H_out = H-K+1, W_out = W-K+1;
    vector<vector<vector<float>>> input = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}}, a = input;
    vector<vector<vector<float>>> output(n_kernels);

    // Crear volumen de salida
    vector<vector<float>> aux_2D(H_out);
    vector<float> aux_1D(W_out);

    for(int i=0; i<aux_2D.size(); i++)
        aux_2D[i] = aux_1D;
    
    for(int i=0; i<output.size(); i++)
        output[i] = aux_2D;

    // Kernel de pesos
    vector<vector<vector<vector<float>>>> w = {{{{1, 1}, {1,1}}, {{1, 1}, {1,1}}}, {{{2, 2}, {2,2}}, {{2, 2}, {2,2}}}};
    // Vector de sesgos
    vector<float> bias(n_kernels, 0.0);

    Convolutional conv(n_kernels, K, K, input, 0.01);
    conv.set_w(w);
    conv.set_b(bias);

    cout << " -------------------------- Método Estándar -------------------------- " << endl;
    cout << "Input" << endl;
    printMatrix_vector(input);

    conv.forwardPropagation(input, output, a);

    cout << "Ouput" << endl;
    printMatrix_vector(output);


    // -----------------------------------------------------------------------------------------------------
    // Método GEMM
    // -----------------------------------------------------------------------------------------------------
    // n = Size of the matrix (n x n)
    int n = 3, C = 2, fils_X_unroll = K*K*C, cols_X_unroll = H_out*W_out, fils_W = n_kernels, cols_W = K*K*C,
    bytes_X = n * n * C * sizeof(float), bytes_X_unroll = fils_X_unroll * cols_X_unroll *sizeof(float), bytes_kernel_W = cols_W*fils_W * sizeof(float),
    bytes_kernel_bias = n_kernels * sizeof(float), bytes_result = cols_X_unroll * fils_W * sizeof(float);

    // Reserva de memoria en host
    float* X = (float*)malloc(bytes_X);
    float* h_X_unroll = (float*)malloc(bytes_X_unroll);
    float *h_kernel_W = (float *) malloc(bytes_kernel_W);
    float *h_kernel_bias = (float *) malloc(bytes_kernel_bias);
    float *h_result = (float*)malloc(bytes_result);

    // Reserva de memoria en device
    float *d_X_unroll, *d_kernel_W, *d_kernel_bias, *d_result;
    cudaMalloc((void **) &d_X_unroll, bytes_X_unroll);
    cudaMalloc((void **) &d_kernel_W, bytes_kernel_W);
    cudaMalloc((void **) &d_kernel_bias, bytes_kernel_bias);
    cudaMalloc((void **) &d_result, bytes_result);


    // Inicializar matriz de entrada X
    int cont = 1;
    for (int i = 0; i < C; i++) 
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                X[i*n*n +j*n +k] = cont++;
            }
        }

    // Inicializar matriz de pesos 
    for (int f = 0; f < n_kernels; f++) 
        for (int i = 0; i < C; i++) 
            for (int j = 0; j < K; j++) {
                for (int k = 0; k < K; k++) {
                    h_kernel_W[f*C*K*K + i*K*K + j*K + k] = f+1;
                }
            }
    
    // Inicializar vector de sesgos
    for (int f = 0; f < n_kernels; f++) 
        h_kernel_bias[f] = 0.0;

    cout << " -------------------------- Método GEMM -------------------------- " << endl;

    cout << "Input" << endl;
    printMatrix_3D(X, C, n);
    
    unroll(C, n, K, X, h_X_unroll);

    // Paso de valores de CPU a GPU
    cudaMemcpy(d_X_unroll, h_X_unroll, bytes_X_unroll, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_W, h_kernel_W, bytes_kernel_W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_bias, h_kernel_bias, bytes_kernel_bias, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, bytes_result, cudaMemcpyHostToDevice);

    // Factores de escala
    float alpha = 1.0f;     // Calcular: c = (alpha*a) *b + (beta*c)
    float beta = 0.0f;

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    //multiplyMatrices(h_kernel_W, n_kernels, K*K*C, h_X_unroll, H_out*W_out, h_result); 
    
    // (m X n) * (n X k) = (m X k)
    // Formato: handle, operation, operation, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
    cublasSgemm(handle, 
    CUBLAS_OP_T, CUBLAS_OP_T,       // Operaciones con matrices transpuestas
    fils_W, cols_X_unroll, cols_W,  // ax, by, ay
    &alpha, 
    d_kernel_W, cols_W,              // d_a, ay
    d_X_unroll, cols_X_unroll,        // d_b, by
    &beta, 
    d_result, n_kernels);           // d_c, ax
    
    // Paso de GPU a CPU
    cudaMemcpy(h_result, d_result, bytes_result, cudaMemcpyDeviceToHost);

    // Calcular la transpuesta para obtener la estructura original
    transposeMatrix(h_result, cols_X_unroll, fils_W);

    // Mostrar resultados obtenidos con cuBLAS
    cout << "Output" << endl;
    printMatrix_3D(h_result, n_kernels, H_out);

    // Liberar memoria
    free(X); free(h_X_unroll); free(h_kernel_W); free(h_kernel_bias); free(h_result);
    cudaFree(d_X_unroll); cudaFree(d_kernel_W); cudaFree(d_kernel_bias); cudaFree(d_result);
    
    return 0;
}