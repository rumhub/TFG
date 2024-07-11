#include "poolingMax.h"
#include "../../auxiliar/auxiliar.cpp"

using namespace std;

void checkCudaErrors_pool(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}

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

PoolingMax::PoolingMax(int kernel_fils, int kernel_cols, vector<vector<vector<float>>> &input)
{
    this->kernel_fils = kernel_fils;
    this->kernel_cols = kernel_cols;
    this->C = input.size();
    this->H = input[0].size();
    this->W = input[0][0].size();
    this->liberar_memoria = false;

    if(H % kernel_fils != 0 || W % kernel_cols != 0)
        cout << "Warning. Las dimensiones del volumen de entrada(" << H << ") no son múltiplos del kernel max_pool(" << kernel_fils << "). \n";
};

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


// Idea de input_copy --> Inicializar a 0. Cada kernel se quedará solo con 1 valor, pues lo pones a 1 en input_copy para luego saber cuál era al hacer backpropagation
// Suponemos que H y W son múltiplos de K
// Es decir, suponemos que tanto el ancho como el alto de la imagen de entrada "input" son múltiplos del tamaño del kernel a aplicar
void PoolingMax::forwardPropagation(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad)
{
    int M = input.size(), K=kernel_fils, n_veces_fils = input[0].size() / K , n_veces_cols = input[0][0].size() / K;
    float max;

    if(input_copy.size() != input.size() || input_copy[0].size() != input[0].size() || input_copy[0][0].size() != input[0][0].size())
    {
        cout << "Error. En la capa Max_Pool no coinciden las dimensiones de input e input_copy. " << endl;
        exit(-1);
    }

    // Inicializamos input_copy a 0
    for(int i=0; i<input_copy.size(); ++i)
    for(int j=0; j<input_copy[0].size(); ++j)
        for(int k=0; k<input_copy[0][0].size(); ++k)
            input_copy[i][j][k] = 0.0;

    int i_m, j_m;
    bool encontrado;

    for(int m=0; m<M; ++m) // Para cada canal
        for(int h=0; h<n_veces_fils; ++h) // Se calcula el nº de desplazamientos del kernel a lo largo del volumen de entrada 3D
            for(int w=0; w<n_veces_cols; ++w)
            {
                max = numeric_limits<float>::min();
                encontrado = false;
                // Para cada subregión, realizar el pooling
                for(int p=0; p<K; ++p)
                    for(int q=0; q<K; ++q)
                    {
                        if(input[m][h*K + p][w*K + q] > max)
                        {
                            max = input[m][h*K + p][w*K + q];
                            i_m = h*K + p;
                            j_m = w*K + q;
                            encontrado = true;
                        }

                    }

                output[m][pad+h][pad+w] = max;

                if(encontrado)
                    input_copy[m][i_m][j_m] = 1.0;
            }
};


void PoolingMax::forwardPropagationGPU(float *input, float *output, float *input_copy)
{
    // Copiar datos de CPU a GPU
    cudaMemcpy(d_input, input, bytes_input, cudaMemcpyHostToDevice);

    // Realizar MaxPool
    maxpool_forward<<<grid, block>>>(C, H, W, kernel_fils, d_input, d_input_copy, d_output, pad);

    cudaMemcpy(output, d_output, bytes_output, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_copy, d_input_copy, bytes_input, cudaMemcpyDeviceToHost);
};

void PoolingMax::forwardPropagation_vectores_externos(float *input, float *output, float *input_copy)
{
    // Realizar MaxPool
    maxpool_forward<<<grid, block>>>(C, H, W, kernel_fils, input, input_copy, d_output, pad);

    cudaMemcpy(output, d_output, bytes_output, cudaMemcpyDeviceToDevice);
    checkCudaErrors_pool(cudaGetLastError());
};

void PoolingMax::backPropagationGPU(float *input, float *output, float *input_copy)
{
    // Inicializar input a 0
    for(int i=0; i<C*H*W; i++)
        input[i] = 0.0;

    // Copiar datos de CPU a GPU
    cudaMemcpy(d_input, input, bytes_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_copy, input_copy, bytes_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, bytes_output, cudaMemcpyHostToDevice);

    // Realizar MaxPool
    maxpool_back<<<grid, block>>>(C, H, W, kernel_fils, d_input, d_input_copy, d_output, pad);

    cudaMemcpy(input, d_input, bytes_input, cudaMemcpyDeviceToHost);
};

void PoolingMax::backPropagation_vectores_externos(float *input, float *output, float *input_copy)
{
    // Realizar MaxPool
    maxpool_back<<<grid, block>>>(C, H, W, kernel_fils, input, input_copy, output, pad);
};

void PoolingMax::backPropagation(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad_output)
{
    int n_canales = this->C, n_veces_fils = this->H / kernel_fils, n_veces_cols = this->W / kernel_cols;
    float max = 0.0;
    int output_fil = 0, output_col = 0;

    // Inicializar imagen de entrada a 0
    for(int i=0; i<input.size(); ++i)
        for(int j=0; j<input[0].size(); ++j)
            for(int k=0; k<input[0][0].size(); ++k)
                input[i][j][k] = 0.0;

    // Para cada imagen 2D
    for(int t=0; t<n_canales; ++t)
    {
        output_fil = 0;
        for(int i=0; i<n_veces_fils*kernel_fils; i+=kernel_fils, ++output_fil)
        {
            output_col = 0;
            // En este momento imagenes_2D[t][i][j] contiene la casilla inicial del kernel en la imagen t para realizar el pooling
            // Casilla inicial = esquina superior izquierda
            for(int j=0; j<n_veces_cols*kernel_cols; j+=kernel_cols, ++output_col)
            {
                max = output[t][pad_output + output_fil][pad_output + output_col];

                // Para cada subregión, realizar el pooling
                for(int k=i; k<(i+kernel_fils)  && k<this->H; ++k)
                {
                    for(int h=j; h<(j+kernel_cols) && h<this->W; ++h)
                    {
                        // Si es el valor máximo, dejarlo como estaba
                        if(input_copy[t][k][h] != 0)
                            input[t][k][h] = max;
                        else
                            input[t][k][h] = 0.0;
                    }
                }
            }

        }
    }

};

void PoolingMax::mostrar_tam_kernel()
{
    cout << "Estructura kernel "<< this->kernel_fils << "x" << this->kernel_cols << "x" << this->C << endl;
}

/*
void aplicar_padding(vector<vector<vector<float>>> &imagen_3D, int pad)
{
    vector<vector<vector<float>>> imagen_3D_aux;
    vector<vector<float>> imagen_aux;
    vector<float> fila_aux;

    // Por cada imagen
    for(int i=0; i<imagen_3D.size();i++)
    {
        // Añadimos padding superior
        for(int j=0; j<imagen_3D[i].size() + pad*2; j++) // pad*2 porque hay padding tanto a la derecha como a la izquierda
            fila_aux.push_back(0.0);

        for(int k=0; k<pad; k++)
            imagen_aux.push_back(fila_aux);

        fila_aux.clear();

        // Padding lateral (izquierda y derecha)
        // Por cada fila de cada imagen
        for(int j=0; j<imagen_3D[i].size(); j++)
        {
            // Añadimos padding lateral izquierdo
            for(int t=0; t<pad; t++)
                fila_aux.push_back(0.0);

            // Dejamos casillas centrales igual que en la imagen original
            for(int k=0; k<imagen_3D[i][j].size(); k++)
                fila_aux.push_back(imagen_3D[i][j][k]);

            // Añadimos padding lateral derecho
            for(int t=0; t<pad; t++)
                fila_aux.push_back(0.0);

            // Añadimos fila construida a la imagen
            imagen_aux.push_back(fila_aux);
            fila_aux.clear();
        }

        // Añadimos padding inferior
        fila_aux.clear();

        for(int j=0; j<imagen_3D[i].size() + pad*2; j++) // pad*2 porque hay padding tanto a la derecha como a la izquierda
            fila_aux.push_back(0.0);

        for(int k=0; k<pad; k++)
            imagen_aux.push_back(fila_aux);

        fila_aux.clear();

        // Añadimos imagen creada al conjunto de imágenes
        imagen_3D_aux.push_back(imagen_aux);
        imagen_aux.clear();
    }

    imagen_3D = imagen_3D_aux;
};


// https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/pooling_layer
int main()
{
    int C=2, H=8, W=8, K=2, H_out = H/K, W_out = W/K;
    vector<vector<vector<float>>> input_cpu(C, vector<vector<float>>(H, vector<float>(W, 0))), input_copy_cpu = input_cpu;
    vector<vector<vector<float>>> output_cpu(C, vector<vector<float>>(H_out, vector<float>(W_out, 0)));
    float *input_gpu = (float *)malloc(H*C*W * sizeof(float)),
          *input_copy_gpu = (float *)malloc(H*C*W * sizeof(float));

    // Input
    for(int i=0; i<C; i++)
        for(int j=0; j<H; j++)
            for(int k=0; k<W; k++)
            {
                input_cpu[i][j][k] = (rand() % 100) + 1;
                input_gpu[i*H*W + j*W + k] = input_cpu[i][j][k];
                input_copy_gpu[i*H*W + j*W + k] = input_cpu[i][j][k];
            }
    input_copy_cpu = input_cpu;


    int pad = 1;    // Padding se mete en capas convolucionales. Por tanto, si metemos padding de pad, antes de la capa pooling_max (input) hay que quitarlo al hacer backprop

    // Output
    int H_out_pad = H/K + 2*pad, W_out_pad = W/K + 2*pad;
    float *output_gpu = (float *)malloc(H_out_pad*C*W_out_pad * sizeof(float));

    for(int i=0; i<C; i++)
        for(int j=0; j<H_out_pad; j++)
            for(int k=0; k<W_out_pad; k++)
            {
                output_gpu[i*H_out_pad*W_out_pad + j*W_out_pad + k] = 0.0;
            }


    Aux *aux = new Aux();

    cout << "--------------- SIMULACIÓN GRADIENTE ------------------ " << endl;

    // Aplicamos padding al output
    // Output viene ya con las dimensiones tras aplicar padding
    aplicar_padding(output_cpu, pad);

    cout << "------------ Imagen inicial: ------------" << endl;
    aux->mostrar_imagen(input_cpu);

    PoolingMax plm1(K, K, input_cpu);

    plm1.forwardPropagation(input_cpu, output_cpu, input_copy_cpu, pad);

    cout << "Output \n";
    aux->mostrar_imagen(output_cpu);


    cout << "------------ Pooling Max, Back Propagation: ------------" << endl;

    // Cambiamos el output porque contendrá un gradiente desconocido
    for(int i=0; i<output_cpu.size(); i++)
        for(int j=0; j<output_cpu[0].size(); j++)
            for(int k=0; k<output_cpu[0][0].size(); k++)
                output_cpu[i][j][k] = 9;

    //cout << "--- Output modificado ---- \n";
    //aux->mostrar_imagen(output);

    plm1.backPropagation(input_cpu, output_cpu, input_copy_cpu, pad);

    cout << "Input\n";
    aux->mostrar_imagen(input_cpu);



    // ---------------- GPU --------------------------
    PoolingMax plm_gpu(K, K, C, H, W, pad);

    // FORWARD CAPA MAXPOOL ---------------------------------------------------------------------------------------------------------------
    // Inicializar input_copy a 0
    for(int i=0; i<C*H*W; i++)
        input_copy_gpu[i] = 2.0;

    /*
    cudaMemcpy(plm_gpu.get_d_input(), input_gpu, plm_gpu.get_bytes_input(), cudaMemcpyHostToDevice);

    // Realizar MaxPool
    maxpool_forward<<<plm_gpu.get_grid(), plm_gpu.get_block()>>>(C, H, W, K, plm_gpu.get_d_input(), plm_gpu.get_d_input_copy(), plm_gpu.get_d_output(), plm_gpu.get_pad());

    cudaMemcpy(output_gpu, plm_gpu.get_d_output(), plm_gpu.get_bytes_output(), cudaMemcpyDeviceToHost);
    cudaMemcpy(input_copy_gpu, plm_gpu.get_d_input_copy(), plm_gpu.get_bytes_input(), cudaMemcpyDeviceToHost);
    // FORWARD CAPA MAXPOOL ---------------------------------------------------------------------------------------------------------------
    */
    /*
    cout << " ------------------ GPU ---------------------" << endl;
    plm_gpu.forwardPropagationGPU(input_gpu, output_gpu, input_copy_gpu);

    cout << "Input copy\n";
    aux->mostrar_imagen3D(input_copy_gpu, C, H, W);

    cout << "Output\n";
    aux->mostrar_imagen3D(output_gpu, C, H_out_pad, W_out_pad);


    cout << "------------ Back Propagation: ------------" << endl;
    // Cambiamos el output porque contendrá un gradiente desconocido
    for(int i=0; i<C*H_out_pad*W_out_pad; i++)
        output_gpu[i] = 9.0;

    // BACK CAPA MAXPOOL ---------------------------------------------------------------------------------------
    // Inicializar input a 0
    for(int i=0; i<C*H*W; i++)
        input_gpu[i] = 0.0;

    // Copiar datos de CPU a GPU
    cudaMemcpy(plm_gpu.get_d_input(), input_gpu, plm_gpu.get_bytes_input(), cudaMemcpyHostToDevice);
    cudaMemcpy(plm_gpu.get_d_input_copy(), input_copy_gpu, plm_gpu.get_bytes_input(), cudaMemcpyHostToDevice);
    cudaMemcpy(plm_gpu.get_d_output(), output_gpu, plm_gpu.get_bytes_output(), cudaMemcpyHostToDevice);

    // Realizar MaxPool
    maxpool_back<<<plm_gpu.get_grid(), plm_gpu.get_block()>>>(C, H, W, K, plm_gpu.get_d_input(), plm_gpu.get_d_input_copy(), plm_gpu.get_d_output(), plm_gpu.get_pad());

    cudaMemcpy(input_gpu, plm_gpu.get_d_input(), plm_gpu.get_bytes_input(), cudaMemcpyDeviceToHost);
    // BACK CAPA MAXPOOL ---------------------------------------------------------------------------------------

    //plm_gpu.backPropagationGPU(input_gpu, output_gpu, input_copy_gpu);

    cout << "Input\n";
    aux->mostrar_imagen3D(input_gpu, C, H, W);



    // Comprobación de errores --------------------------
    bool correcto = true;

    // Input
    for(int i=0; i<C; i++)
        for(int j=0; j<H; j++)
            for(int k=0; k<W; k++)
            {
                if(input_cpu[i][j][k] != input_gpu[i*H*W + j*W +k])
                    correcto = false;
            }

    // Output
    for(int i=0; i<C; i++)
        for(int j=0; j<H_out_pad; j++)
            for(int k=0; k<W_out_pad; k++)
            {
                if(output_gpu[i*H_out_pad*W_out_pad + j*W_out_pad + k] != output_cpu[i][j][k])
                    correcto = false;
            }


    if(correcto)
        cout << "Todo correcto!" << endl;
    else
        cout << "Hay errores" << endl;


    free(input_gpu); free(input_copy_gpu); free(output_gpu);

    return 0;
}
*/
