#include "poolingMax.h"
#include "../../auxiliar/auxiliar.cpp"

using namespace std;

void checkCudaErrors_pool(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}

/*
    @brief      Propagación hacia delante en capa de agrupación máxima (PoolingMax)
    @C          Número de canales de profundidad de la entrada
    @H          Número de filas de la entrada
    @W          Número de columnas de la entrada
    @K          Tamaño de la ventana de agrupación máxima
    @X          Entrada
    @X_copy     Copia de la entrada
    @Y          Salida
    @pad        Padding o Relleno
*/
__global__ void maxpool_forward(int C, int H, int W, int K, float *X, float *X_copy, float *Y, int pad)
{
    // Convertir de índices de hebra a índices de matriz
  	int H_out = H / K + 2*pad, W_out = W /K + 2*pad,
        iy_Y = threadIdx.y + blockIdx.y * blockDim.y, ix_Y = threadIdx.x + blockIdx.x * blockDim.x,     // Coordenadas respecto a la mtriz de salida Y
        iy_X = iy_Y *K, ix_X = ix_Y *K,                                     // Coordenadas respecto a la matriz de entrada X
        idX = iy_X*W + ix_X, idY = (iy_Y+pad)*W_out + (ix_Y+pad),
        tam_capaY = H_out * W_out,
        tam_capaX = H * W;

    float max = FLT_MIN;
    int pos_max = idX;

    // Inicializar output
    if(iy_Y < H_out && ix_Y < W_out)
      Y[idY] = 0.0;

    if(iy_Y < H_out-2*pad && ix_Y < W_out-2*pad)    // -2*pad para quitar el padding. Como solo hay padding en la salida, usamos las mismas hebras que si no hubiera ningún padding
    {
        for(int c=0; c<C; c++)
        {

            max = FLT_MIN;
            for(int i=0; i<K; i++)
                for(int j=0; j<K; j++)
                {
                    // Inicializar input_copy a 0
                    if(iy_X + i < H && ix_X + j < W)
                        X_copy[idX +i*W +j] = 0.0;

                    if(max < X[idX + i*W + j] && iy_X + i < H && ix_X + j < W)
                    {
                        max = X[idX +i*W +j];
                        pos_max = idX +i*W +j;
                    }
                }

            __syncthreads();
            
            // Establecer valor del píxel "IdY" de salida
            Y[idY] = max;
            
            // Establecer posición del máximo
            if(pos_max < C*H*W)
                X_copy[pos_max] = 1.0;
            
            // Actualizar índice para siguiente capa
            idY += tam_capaY;
            idX += tam_capaX;
        }
    }
}

/*
    @brief      Retropropagación en capa de agrupación máxima (PoolingMax)
    @C          Número de canales de profundidad de la entrada
    @H          Número de filas de la entrada
    @W          Número de columnas de la entrada
    @K          Tamaño de la ventana de agrupación máxima
    @X          Entrada
    @X_copy     Copia de la entrada
    @Y          Salida
    @pad        Padding o Relleno
*/
__global__ void maxpool_back(int C, int H, int W, int K, float *X, float *X_copy, float *Y, int pad)
{
    //int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);

    // Convertir de índices de hebra a índices de matriz
  	int H_out = H / K + 2*pad, W_out = W /K + 2*pad,
        iy_Y = threadIdx.y + blockIdx.y * blockDim.y, ix_Y = threadIdx.x + blockIdx.x * blockDim.x,     // Coordenadas respecto a la mtriz de salida Y
        iy_X = iy_Y *K, ix_X = ix_Y *K,                                     // Coordenadas respecto a la matriz de entrada X
        idX = iy_X*W + ix_X, idY = (iy_Y+pad)*W_out + (ix_Y+pad),
        tam_capaY = H_out * W_out,
        tam_capaX = H * W;

    if(iy_Y < H_out-2*pad && ix_Y < W_out-2*pad)    // -2*pad para quitar el padding. Como solo hay padding en la salida, usamos las mismas hebras que si no hubiera ningún padding
    for(int c=0; c<C; c++)
    {
        for(int i=0; i<K; i++)
            for(int j=0; j<K; j++)
                if(iy_X + i < H && ix_X + j < W)
                    if(X_copy[idX +i*W +j] != 0)
                        X[idX +i*W +j] = Y[idY];
                    else
                        X[idX +i*W +j] = 0.0;

        // Actualizar índice para siguiente capa
        idY += tam_capaY;
        idX += tam_capaX;
    }
}

/*
    @brief          Constructor
    @kernel_fils    Filas de la ventana de agrupación máxima
    @kernel_cols    Columnas de la ventana de agrupación máxima
    @C          Número de canales de profundidad de la entrada
    @H          Número de filas de la entrada
    @W          Número de columnas de la entrada
    @pad        Padding o Relleno
*/
PoolingMax::PoolingMax(int kernel_fils, int kernel_cols, int C, int H, int W, int pad)
{

    // Kernel
    this->kernel_fils = kernel_fils;
    this->kernel_cols = kernel_cols;

    // Liberar mmemoria o no
    this->liberar_memoria = true;

    // Dimensiones de la imagen de entrada
    this->C = C;
    this->H = H;
    this->W = W;

    // Dimensiones de la imagen de salida
    this->H_out = H/this->kernel_fils +2*pad;
    this->W_out = W/this->kernel_cols +2*pad;

    // Padding
    this->pad = pad;

    // Bytes necesarios
    this->bytes_input = C*H*W * sizeof(float);
    this->bytes_output = C*H_out*W_out * sizeof(float);

    // Tamaño de bloque
    this->block.x = kernel_fils;
    this->block.y = kernel_cols;

    // Tamaño de grid
    this->grid.x = ceil( (float)(W + block.x -1) / block.x);
    this->grid.y = ceil((float)(H + block.y -1) / block.y);

    // Reserva memoria en device
    cudaMalloc((void **) &d_input, bytes_input);
    cudaMalloc((void **) &d_input_copy, bytes_input);
    cudaMalloc((void **) &d_output, bytes_output);

    if(this->H % kernel_fils != 0 || this->W % kernel_cols != 0)
        cout << "Warning. Las dimensiones del volumen de entrada(" << this->H << ") no son múltiplos del kernel max_pool(" << kernel_fils << "). \n";

};

/*
    @brief  Copia información de una capa de agrupación máxima a otra
    @plm    Capa de agrupación máxima de la que copiar la información
*/
void PoolingMax::copiar(const PoolingMax & plm)
{

    // Kernel
    this->kernel_fils = plm.kernel_fils;
    this->kernel_cols = plm.kernel_cols;

    // Liberar mmemoria o no
    this->liberar_memoria = plm.liberar_memoria;

    // Dimensiones de la imagen de entrada
    this->C = plm.C;
    this->H = plm.H;
    this->W = plm.W;

    // Dimensiones de la imagen de salida
    this->H_out = plm.H_out;
    this->W_out = plm.W_out;

    // Padding
    this->pad = plm.pad;

    // Bytes necesarios
    this->bytes_input = plm.bytes_input;
    this->bytes_output = plm.bytes_output;

    // Tamaño de bloque
    this->block.x = plm.block.x;
    this->block.y = plm.block.y;

    // Tamaño de grid
    this->grid.x = plm.grid.x;
    this->grid.y = plm.grid.y;

    // Reserva memoria en device
    cudaMalloc((void **) &d_input, bytes_input);
    cudaMalloc((void **) &d_input_copy, bytes_input);
    cudaMalloc((void **) &d_output, bytes_output);

    if(this->H % kernel_fils != 0 || this->W % kernel_cols != 0)
        cout << "Warning. Las dimensiones del volumen de entrada(" << this->H << ") no son múltiplos del kernel max_pool(" << kernel_fils << "). \n";
}

/*
    @brief          Propagación hacia delante en capa de agrupación máxima
    @input          Entrada
    @output         Salida
    @input_copy     Copia de la entrada
*/
void PoolingMax::forwardPropagation_vectores_externos(float *input, float *output, float *input_copy)
{
    // Realizar MaxPool
    maxpool_forward<<<grid, block>>>(C, H, W, kernel_fils, input, input_copy, d_output, pad);

    cudaMemcpy(output, d_output, bytes_output, cudaMemcpyDeviceToDevice);
    checkCudaErrors_pool(cudaGetLastError());
};

/*
    @brief          Retropropagación en capa de agrupación máxima
    @input          Entrada
    @output         Salida
    @input_copy     Copia de la entrada
*/
void PoolingMax::backPropagation_vectores_externos(float *input, float *output, float *input_copy)
{
    // Realizar MaxPool
    maxpool_back<<<grid, block>>>(C, H, W, kernel_fils, input, input_copy, output, pad);
};

/*
    @brief  Muestra el tamaño de la ventana de agrupación máxima
*/
void PoolingMax::mostrar_tam_kernel()
{
    cout << "Estructura kernel "<< this->kernel_fils << "x" << this->kernel_cols << "x" << this->C << endl;
}
