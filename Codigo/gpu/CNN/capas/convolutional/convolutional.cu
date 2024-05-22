#include "convolutional.h"

using namespace std::chrono;



/*
    Emplea tiles. Un tile por bloque. Usa memoria compartida
*/
__global__ void multiplicarMatricesGPU(int M, int N, int K, const float *A, const float *B, float *C)
{
    // Memoria compartida dinámica
	extern __shared__ float sdata[];

    // Convertir de índices de hebra a índices de matriz 
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x, 
        idA = iy*K + ix, idB = iy*N + ix, id_tile = threadIdx.y * blockDim.x + threadIdx.x;
    //int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);
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

/*
    CONSTRUCTOR de la clase Convolutional
    --------------------------------------
  
    @n_kernels      Número de kernels a aplicar
    @kernel_fils    Número de filas por kernel
    @kernel_cols    Número de columnas por kernel
    @input          Volumen 3D de entrada
    @lr             Learning Rate o Tasa de Aprendizaje
*/
/*
Convolutional::Convolutional(int n_kernels, int kernel_fils, int kernel_cols, const vector<vector<vector<float>>> &input, float lr)
{
    this->n_kernels = n_kernels;
    this->C = input.size();
    this->kernel_fils = kernel_fils;
    this->kernel_cols = kernel_cols;
    this->lr = lr;
    this->w.clear();

    vector<vector<vector<vector<float>>>> pesos_por_kernel;
    vector<vector<vector<float>>> pesos_totales;    // Representa los pesos de un kernel (los pesos de todas sus dimensiones)
    vector<vector<float>> pesos_kernel_2D;      // Representa los pesos de un kernel 2D
    vector<float> pesos_fila;               // Representa los pesos de una fila de un kernel 2D. Podría verse como un kernel 1D (un array)
    
    // Crear estructura de los pesos
    for(int f=0; f< n_kernels; ++f) // Por cada kernel
    {
        for(int k=0; k< C; ++k)
        {
            for(int i=0; i<kernel_fils; ++i)
            {
                for(int j=0; j<kernel_cols; ++j)
                {
                    pesos_fila.push_back(0.0);
                }
                pesos_kernel_2D.push_back(pesos_fila);
                pesos_fila.clear();
            }
            pesos_totales.push_back(pesos_kernel_2D);
            pesos_kernel_2D.clear();
        }
        this->w.push_back(pesos_totales);
        pesos_totales.clear();
    }     

    // Inicializar pesos mediante Inicialización He
    this->generar_pesos();

    // Bias    
    // Un bias por filtro, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
    for(int i=0; i<n_kernels; i++)
        this->bias.push_back(0.0);
    
};
*/

/*
    CONSTRUCTOR de la clase Convolutional
    --------------------------------------
  
    @n_kernels      Número de kernels a aplicar
    @kernel_fils    Número de filas por kernel
    @kernel_cols    Número de columnas por kernel
    @C              Número de canales de profundiad de la entrada
    @lr             Learning Rate o Tasa de Aprendizaje
*/
Convolutional::Convolutional(int n_kernels, int kernel_fils, int kernel_cols, int C, int H, int W, float lr)
{
    // Establecer device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    //cout << "Estableciendo dispotisivo para GPU...\n";
    //printf("Usando el dispositivo %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // Kernels de pesos
    this->n_kernels = n_kernels;
    this->kernel_fils = kernel_fils;
    this->kernel_cols = kernel_cols;

    // Imagen de entrada
    this->C = C;
    this->H = H;
    this->W = W;

    // Imagen de salida
    this->H_out = H - kernel_fils + 1;
    this->W_out = W - kernel_cols + 1;

    // Dimensiones de los volúmenes "desenrrollados" ------------------
    // Dimensiones de la entrada 'desenrrollada'
    this->fils_input_unroll = kernel_fils*kernel_cols*C; 
    this->cols_input_unroll = H_out * W_out;
    
    // Dimensiones de los pesos como matriz 2D
    this->fils_w = this->n_kernels;
    this->cols_w = kernel_fils*kernel_cols*C;
    
    // Tamaños de los volúmenes "desenrrollados"
    this->bytes_input_unroll = fils_input_unroll * cols_input_unroll *sizeof(float);    // Espacio para la entrada 'desenrrollada'
    this->bytes_output = cols_input_unroll * fils_w *sizeof(float);              // Espacio para la salida
    this->bytes_w = fils_w * cols_w * sizeof(float);
    
    // Learning Rate
    this->lr = lr;
    
    // Pesos
    this->w_ptr = (float *)malloc(fils_w * cols_w * sizeof(float));
    
    // Inicializar pesos mediante Inicialización He
    this->generar_pesos_ptr();

    // Bias    
    this->bias_ptr = (float *)malloc(this->n_kernels * sizeof(float));

    // Un bias por filtro, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
    for(int i=0; i<n_kernels; i++)
        this->bias_ptr[i] = 0.0;    

    // CPU -------------------
    this->h_input_unroll = (float*)malloc(bytes_input_unroll);  // Volumen de entrada 'desenrrollado'
    
    // GPU -------------------------
    // Tamaño de bloque
    this->block.x = BLOCK_SIZE;
    this->block.y = BLOCK_SIZE;

    // Memoria compartida a nivel de bloque
    this->smem = (2*block.x * block.y) *sizeof(float);

    // Punteros device
    cudaMalloc((void **) &d_input_unroll, bytes_input_unroll);
    cudaMalloc((void **) &d_a, bytes_output);
    cudaMalloc((void **) &d_w, bytes_w);


    this->pad = kernel_fils-1; 
    this->H_out_pad = H_out +2*pad; 
    this->W_out_pad = W_out + 2*pad;

    this->output_pad = (float *)malloc(this->n_kernels * H_out_pad * W_out_pad * sizeof(float));
    this->grad_w_it = (float *)malloc(this->n_kernels*this->C*this->kernel_fils*this->kernel_cols * sizeof(float));



    this->fils_output_unroll = kernel_fils*kernel_cols*n_kernels; 
    this->cols_output_unroll = H*W; 
    this->fils_matriz_pesos = C; 
    this->cols_matriz_pesos = kernel_fils*kernel_cols*n_kernels;
    this->fils_input_back_unroll = H_out * W_out; 
    this->cols_input_back_unroll = kernel_fils*kernel_cols*C;
    this->bytes_output_unroll = fils_output_unroll * cols_output_unroll * sizeof(float);
    this->bytes_matriz_pesos = fils_matriz_pesos * cols_matriz_pesos * sizeof(float);
    this->bytes_input_back_unroll = fils_input_unroll * cols_input_unroll * sizeof(float);

    this->h_output_unroll = (float *)malloc(bytes_output_unroll);
    this->h_matriz_pesos = (float *)malloc(bytes_matriz_pesos);
    this->h_input_back_unroll = (float *)malloc(bytes_input_back_unroll);
    


    // Tamaños de los grids ---------------------------------------
    // Tamaño de grid para propagación hacia delante
    this->grid_forward.x = (cols_input_unroll  + BLOCK_SIZE -1) / BLOCK_SIZE; 
    this->grid_forward.y = (fils_w + BLOCK_SIZE -1) / BLOCK_SIZE;

    // Tamaño de grid para calcular el gradiente respecto a los pesos
    this->grid_grad_w.x = (cols_input_back_unroll  + BLOCK_SIZE -1) / BLOCK_SIZE;
    this->grid_grad_w.y = (this->n_kernels + BLOCK_SIZE -1) / BLOCK_SIZE;

    // Tamao del grid para calcular el gradiente respecto a la entrada
    this->grid_grad_input.x = (cols_output_unroll  + BLOCK_SIZE -1) / BLOCK_SIZE;
    this->grid_grad_input.y = (C + BLOCK_SIZE -1) / BLOCK_SIZE;


    // Reserva de memoria en device
    cudaMalloc((void **) &d_output_unroll, bytes_output_unroll);
    cudaMalloc((void **) &d_matriz_pesos, bytes_matriz_pesos);
    cudaMalloc((void **) &d_input, C*H*W * sizeof(float));
    cudaMalloc((void **) &d_output, n_kernels*H_out*W_out * sizeof(float));
    cudaMalloc((void **) &d_input_back_unroll, bytes_input_back_unroll);
    cudaMalloc((void **) &d_grad_w, this->n_kernels*C*kernel_fils*kernel_cols * sizeof(float));
};


void Convolutional::copiar(const Convolutional & conv)
{
    // Establecer device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    //cout << "Estableciendo dispotisivo para GPU...\n";
    //printf("Usando el dispositivo %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // Kernels de pesos
    this->n_kernels = conv.n_kernels;
    this->kernel_fils = conv.kernel_fils;
    this->kernel_cols = conv.kernel_cols;

    // Imagen de entrada
    this->C = conv.C;
    this->H = conv.H;
    this->W = conv.W;

    // Imagen de salida
    this->H_out = conv.H_out;
    this->W_out = conv.W_out;

    // Dimensiones de los volúmenes "desenrrollados" ------------------
    // Dimensiones de la entrada 'desenrrollada'
    this->fils_input_unroll = conv.fils_input_unroll; 
    this->cols_input_unroll = conv.cols_input_unroll;
    
    // Dimensiones de los pesos como matriz 2D
    this->fils_w = conv.fils_w;
    this->cols_w = conv.cols_w;
    
    // Tamaños de los volúmenes "desenrrollados"
    this->bytes_input_unroll = conv.bytes_input_unroll;    // Espacio para la entrada 'desenrrollada'
    this->bytes_output = conv.bytes_output;              // Espacio para la salida
    this->bytes_w = conv.bytes_w;
    
    // Learning Rate
    this->lr = conv.lr;
    
    // Pesos
    this->w_ptr = (float *)malloc(fils_w * cols_w * sizeof(float));
    
    // Inicializar pesos mediante Inicialización He
    this->generar_pesos_ptr();

    // Bias    
    this->bias_ptr = (float *)malloc(this->n_kernels * sizeof(float));

    // Un bias por filtro, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
    for(int i=0; i<n_kernels; i++)
        this->bias_ptr[i] = 0.0;    

    // CPU -------------------
    this->h_input_unroll = (float*)malloc(bytes_input_unroll);  // Volumen de entrada 'desenrrollado'
    
    // GPU -------------------------
    // Tamaño de bloque
    this->block.x = BLOCK_SIZE;
    this->block.y = BLOCK_SIZE;

    // Memoria compartida a nivel de bloque
    this->smem = (2*block.x * block.y) *sizeof(float);

    // Punteros device
    cudaMalloc((void **) &d_input_unroll, bytes_input_unroll);
    cudaMalloc((void **) &d_a, bytes_output);
    cudaMalloc((void **) &d_w, bytes_w);


    this->pad = conv.pad; 
    this->H_out_pad = conv.H_out_pad; 
    this->W_out_pad = conv.W_out_pad;

    this->output_pad = (float *)malloc(this->n_kernels * H_out_pad * W_out_pad * sizeof(float));
    this->grad_w_it = (float *)malloc(this->n_kernels*this->C*this->kernel_fils*this->kernel_cols * sizeof(float));



    this->fils_output_unroll = conv.fils_output_unroll; 
    this->cols_output_unroll = conv.cols_output_unroll; 
    this->fils_matriz_pesos = conv.fils_matriz_pesos; 
    this->cols_matriz_pesos = conv.cols_matriz_pesos;
    this->fils_input_back_unroll = conv.fils_input_back_unroll; 
    this->cols_input_back_unroll = conv.cols_input_back_unroll;
    this->bytes_output_unroll = conv.bytes_output_unroll;
    this->bytes_matriz_pesos = conv.bytes_matriz_pesos;
    this->bytes_input_back_unroll = conv.bytes_input_back_unroll;

    this->h_output_unroll = (float *)malloc(bytes_output_unroll);
    this->h_matriz_pesos = (float *)malloc(bytes_matriz_pesos);
    this->h_input_back_unroll = (float *)malloc(bytes_input_back_unroll);
    


    // Tamaños de los grids ---------------------------------------
    // Tamaño de grid para propagación hacia delante
    this->grid_forward.x = conv.grid_forward.x; 
    this->grid_forward.y = conv.grid_forward.y;

    // Tamaño de grid para calcular el gradiente respecto a los pesos
    this->grid_grad_w.x = conv.grid_grad_w.x;
    this->grid_grad_w.y = conv.grid_grad_w.y;

    // Tamao del grid para calcular el gradiente respecto a la entrada
    this->grid_grad_input.x = conv.grid_grad_input.x;
    this->grid_grad_input.y = conv.grid_grad_input.y;


    // Reserva de memoria en device
    cudaMalloc((void **) &d_output_unroll, bytes_output_unroll);
    cudaMalloc((void **) &d_matriz_pesos, bytes_matriz_pesos);
    cudaMalloc((void **) &d_input, C*H*W * sizeof(float));
    cudaMalloc((void **) &d_output, n_kernels*H_out*W_out * sizeof(float));
    cudaMalloc((void **) &d_input_back_unroll, bytes_input_back_unroll);
    cudaMalloc((void **) &d_grad_w, this->n_kernels*C*kernel_fils*kernel_cols * sizeof(float));
}


Convolutional::~Convolutional()
{
    if(h_input_unroll != nullptr)   // Si no se creó una instancia de la clase mediante constructor por defecto
    {
        free(h_input_unroll); free(output_pad); free(grad_w_it); free(h_output_unroll); free(h_matriz_pesos); free(h_input_back_unroll); 
        free(w_ptr); free(bias_ptr); cudaFree(d_input_unroll); cudaFree(d_a); cudaFree(d_w); cudaFree(d_output_unroll);
        cudaFree(d_matriz_pesos); cudaFree(d_input); cudaFree(d_input_back_unroll); cudaFree(d_output); cudaFree(d_grad_w);
    }
    cout << "Destructor: ";
    checkCudaErrors(cudaGetLastError());

};

/*
    @brief      Inicializa los pesos de la capa convolucional según la inicialización He
    @return     Se modifica w (los pesos de la capa)
*/
void Convolutional::generar_pesos_ptr() 
{
    // Inicialización He
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> distribution(0.0, sqrt(2.0 / (this->n_kernels * this->C * this->kernel_fils * this->kernel_fils)));

    for(int i=0; i<n_kernels; ++i)
        for(int j=0; j<C; ++j)
            for(int k=0; k<kernel_fils; ++k)
                for(int p=0; p<kernel_cols; ++p)
                    this->w_ptr[i*C*kernel_fils*kernel_cols + j*kernel_fils*kernel_cols + k*kernel_cols + p ] = distribution(gen);
                    //this->w_ptr[i*C*kernel_fils*kernel_cols + j*kernel_fils*kernel_cols + k*kernel_cols + p ] = 1.0;
                    //this->w_ptr[i*this->kernel_depth*this->kernel_fils*this->kernel_cols + j*this->kernel_fils*this->kernel_cols + k*this->kernel_cols + p ] = distribution(gen);



}


/*
    @brief      Inicializa los pesos de la capa convolucional según la inicialización He
    @return     Se modifica w (los pesos de la capa)
*/
void Convolutional::generar_pesos() 
{
    // Inicialización He
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> distribution(0.0, sqrt(2.0 / (this->n_kernels * this->C * this->kernel_fils * this->kernel_fils)));

    for(int i=0; i<this->w.size(); ++i)
        for(int j=0; j<this->w[0].size(); ++j)
            for(int k=0; k<this->w[0][0].size(); ++k)
                for(int p=0; p<this->w[0][0][0].size(); ++p)
                    this->w[i][j][k][p] = 1.0;
                    //this->w[i][j][k][p] = distribution(gen);
}

/*
    @brief      Función de activación ReLU
    @x          Valor sobre el cual aplicar ReLU
    @return     @x tras aplicar ReLU sobre él
*/
float Convolutional::activationFunction(float x)
{
	// RELU
	return (x > 0.0) ? x : 0;
};

/*
    @brief      Derivada de la función de activación ReLU
    @x          Valor sobre el cual aplicar la derivada de ReLU
    @return     @x tras aplicar la derivada de ReLU sobre él
*/
float Convolutional::deriv_activationFunction(float x)
{
    float result = 0.0;

    if(x > 0)
        result = 1;
    
    return result;
}

/*
    @brief      Aplicar padding a un volumen de entrada 3D
    @imagen_3D  Volumen 3D de entrada
    @pad        Nivel de padding a aplicar
    @return     Se añaden @pad niveles de padding al volumen de entrada @imagen_3D
*/
void Convolutional::aplicar_padding(vector<vector<vector<float>>> &imagen_3D, int pad)
{
    vector<vector<vector<float>>> imagen_3D_aux;
    vector<vector<float>> imagen_aux;
    vector<float> fila_aux;

    // Por cada imagen
    for(int i=0; i<imagen_3D.size(); ++i)
    {
        // Añadimos padding superior
        for(int j=0; j<imagen_3D[i].size() + pad*2; ++j) // pad*2 porque hay padding tanto a la derecha como a la izquierda
            fila_aux.push_back(0.0);
        
        for(int k=0; k<pad; ++k)
            imagen_aux.push_back(fila_aux);
        
        fila_aux.clear();

        // Padding lateral (izquierda y derecha)
        // Por cada fila de cada imagen
        for(int j=0; j<imagen_3D[i].size(); ++j)
        {
            // Añadimos padding lateral izquierdo
            for(int t=0; t<pad; ++t)
                fila_aux.push_back(0.0);

            // Dejamos casillas centrales igual que en la imagen original
            for(int k=0; k<imagen_3D[i][j].size(); ++k)
                fila_aux.push_back(imagen_3D[i][j][k]);
            
            // Añadimos padding lateral derecho
            for(int t=0; t<pad; ++t)
                fila_aux.push_back(0.0);
            
            // Añadimos fila construida a la imagen
            imagen_aux.push_back(fila_aux);
            fila_aux.clear();
        }
        
        // Añadimos padding inferior
        fila_aux.clear();

        for(int j=0; j<imagen_3D[i].size() + pad*2; ++j) // pad*2 porque hay padding tanto a la derecha como a la izquierda
            fila_aux.push_back(0.0);
        
        for(int k=0; k<pad; ++k)
            imagen_aux.push_back(fila_aux);
        
        fila_aux.clear();
        
        // Añadimos imagen creada al conjunto de imágenes
        imagen_3D_aux.push_back(imagen_aux);
        imagen_aux.clear();
    }

    imagen_3D = imagen_3D_aux;
};


/*
    @brief      Propagación hacia delante a lo largo de toda la capa convolucional
    @input      Volumen de entrada 3D
    @output     Volumen de salida 3D
    @a          Valor de las neuronas antes de aplicar la función de activación
    @return     Se modifica @output y @a
*/
void Convolutional::forwardPropagation(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &a)
{
    // https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
    vector<vector<vector<float>>> input_copy;
    vector<vector<float>> conv_imagen;
    vector<float> conv_fila;
    
    // nº de kernels, nº de "capas 2D" por kernel, nº de filas del kernel, nº de columnas del kernel a aplicar
    int M = this->w.size(), depth_k = this->w[0].size(), K = this->w[0][0].size();

    // nº de "capas 2D",   nº de filas del volumen de entrada
    int C = input.size(), fils_img = input[0].size();

    // nº de veces que se deslizará el kernel sobre el volumen de entrada input
    // Suponemos nº filas = nº columnas
    int n_veces = fils_img - K + 1;

    if(C != depth_k)
    {
        cout << "Error. La profundidad del volumen del entrada no coincide con la profundidad de los kernels proporcionados. " << endl;
        exit(-1);
    }
    
    //a = output;

    // Por cada kernel M 
    for(int img_out=0; img_out<M; ++img_out)
        for(int i=0; i<n_veces; ++i)    
            for(int j=0; j<n_veces; ++j)  
            {
                a[img_out][i][j] = 0.0;
                
                // Realizar convolución 3D
                for(int c=0; c<C; ++c)
                    for(int i_k=0; i_k<K; ++i_k)
                        for(int j_k=0; j_k<K; ++j_k)
                            a[img_out][i][j] += input[c][i+i_k][j+j_k] * this->w[img_out][c][i_k][j_k];                            

                // Sumamos bias a la suma ponderada obtenida
                a[img_out][i][j] += this->bias[img_out];


                // Aplicamos función de activación
                output[img_out][i][j] = activationFunction(a[img_out][i][j]);
            }
};


void Convolutional::checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }else
    {
        cout << "Todo correcto!" << endl;
    }
}

/*
    @brief      Propagación hacia delante a lo largo de toda la capa convolucional
    @input      Volumen de entrada 3D
    @output     Volumen de salida 3D
    @a          Valor de las neuronas antes de aplicar la función de activación
    @return     Se modifica @output y @a
*/
void Convolutional::forwardPropagationGEMM(float *input, float *output, float *a)
{
    this->unroll(C, H, kernel_fils, input, h_input_unroll);    
    
    // Copiar de CPU a GPU
    cudaMemcpy(d_input_unroll, h_input_unroll, bytes_input_unroll, cudaMemcpyHostToDevice);

    cudaMemcpy(d_w, this->w_ptr, bytes_w, cudaMemcpyHostToDevice);
    
    // Multiplicación de matrices
    multiplicarMatricesGPU<<<grid_forward, block, smem>>>(fils_w, cols_input_unroll, cols_w, d_w, d_input_unroll, d_a);
    
    // Paso de GPU a CPU
    cudaMemcpy(a, d_a, bytes_output, cudaMemcpyDeviceToHost);
    
    // Sumar bias
    for(int i = 0; i < this->n_kernels; i++) 
        for(int j = 0; j < H_out; j++)
            for(int k = 0; k < W_out; k++)
                a[i*H_out*W_out +j*W_out +k] +=  this->bias_ptr[i];

    // Aplicar función de activación
    for(int i = 0; i < this->n_kernels; i++) 
        for(int j = 0; j < H_out; j++)
            for(int k = 0; k < W_out; k++)
                output[i*H_out*W_out +j*W_out +k] = activationFunction(a[i*H_out*W_out +j*W_out +k]);
};


void Convolutional::unroll(int C, int n, int K, float *X, float *X_unroll){
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

void Convolutional::unroll_1dim(int C, int H, int W, int K, float *X, float *X_unroll){

    int H_out = H - K+1, W_out = W -K +1;
    int cont = 0;

    for(int i=0; i<C; i++)
    {
        for(int k=0; k<W_out; k++)
        {
            for(int j=0; j<H_out; j++)
            {
                // Guardar K*K elementos de "convolución"
                for(int ky=0; ky < K; ky++)
                    for(int kx=0; kx<K; kx++)
                    {

                        
                        X_unroll[cont++] = X[i*H*W + (j+ky)*W + (k+kx)];
                        //X_unroll[cont++] = X[i*H*W + (j+ky)*W + (k+kx)];
                        //cout << X[i*H*W + (j+ky)*W + (k+kx)] << " ";
                    }
                //cout << endl;
            }
        }
    }        
    //cout << " ---------- " << endl;
}

void Convolutional::unroll_3dim(int C, int H, int W, int K, float *X, float *X_unroll){

        // Calculate the size of output
        int H_out = H - K + 1;
        int W_out = W - K + 1;

        int outputIndex = 0;
        for (int i = 0; i < H_out; ++i) {
            for (int j = 0; j < W_out; ++j) {
                int inputIndex = 0;
                for (int d = 0; d < C; ++d) {
                    for (int ki = 0; ki < K; ++ki) {
                        for (int kj = 0; kj < K; ++kj) {
                            X_unroll[outputIndex * (K * K * C) + inputIndex++] = X[d * (H * W) + (i + ki) * W + (j + kj)];
                        }
                    }
                }
                ++outputIndex;
            }
        }

}


void Convolutional::printMatrix_3D(float* matrix, int C, int n) {
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) 
                cout << matrix[i*n*n +j*n +k] << " ";
            cout << endl;
        }
        cout << endl;
    }
}

void Convolutional::printMatrix(float* matrix, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) 
            cout << matrix[i * w + j] << " ";
        cout << endl;
    }
}

void Convolutional::matrizTranspuesta(float* matrix, int rows, int cols)
{
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

/*
    @brief      Establece el valor de todos gradientes de la capa convolucional a 0.0
    @grad_w     Gradientes respecto a pesos
    @grad_bias  Gradientes respecto a sesgos
    @return     Se modifica tantp @grad_w como @grad_bias
*/
void Convolutional::reset_gradients(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias)
{
    // Reset gradientes -------------------------------------------------------------------
    
    // Reset gradiende del bias
    for(int k=0; k<this->n_kernels; ++k)
        grad_bias[k] = 0.0;

    // Reset del gradiente de cada peso
    for(int i=0; i<this->w.size(); ++i)
        for(int j=0; j<this->w[0].size(); ++j)
            for(int k=0; k<this->w[0][0].size(); ++k)
                for(int l=0; l<this->w[0][0][0].size(); ++l)
                    grad_w[i][j][k][l] = 0.0;
                
}


/*
    @brief      Retropropagación de la capa convolucional
    @input      Volumen 3D de entrada de la capa
    @output     Volumen 3D de salida de la capa
    @a          Valor de las neuronas antes de aplicar la función de activación
    @grad_w     Gradientes respecto a pesos
    @grad_bias  Gradientes respecto a sesgos
    @pad        Nivel de padding que se aplicó anteriormente   
*/
void Convolutional::backPropagation(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> output, const vector<vector<vector<float>>> &a, vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias)
{
    // https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
    vector<vector<vector<float>>> grad_input = input;
    vector<vector<float>> conv_imagen;
    vector<float> conv_fila;
    
    // nº de kernels, nº de filas del kernel, nº de columnas del kernel a aplicar
    int F = this->w.size(), K = this->w[0][0].size(), C = input.size();

    // Inicializar input a 0
    for(int i=0; i<input.size(); ++i)
        for(int j=0; j<input[0].size(); ++j)    
            for(int k=0; k<input[0][0].size(); ++k)
                grad_input[i][j][k] = 0.0;

    // Realizar derivada Y_out/Y_in
    for(int i=0; i<output.size(); ++i)
        for(int j=0; j<output[0].size(); ++j)    
            for(int k=0; k<output[0][0].size(); ++k)
                output[i][j][k] = output[i][j][k] * deriv_activationFunction(a[i][j][k]);
    
    // Suponemos nº filas = nº columnas
    int H = input[0].size(), W = input[0][0].size();
    int H_out = H-K+1, W_out = W-K+1;

    // Convolución entre output y pesos    
    for(int f=0; f<F; ++f)  // Por cada filtro f
    {
        // Gradiente respecto a entrada
        for(int i=0; i<H; ++i)    
            for(int j=0; j<W; ++j)    
                for(int c=0; c<C; ++c)  // Convolución entre salida y pesos invertidos
                    for(int i_k=0; i_k<K; ++i_k)
                        for(int j_k=0; j_k<K; ++j_k)
                            if(i-i_k >= 0 && j-j_k >= 0 && i-i_k < H_out && j-j_k < W_out)
                                grad_input[c][i][j] += output[f][i-i_k][j-j_k] * this->w[f][c][K -1 - i_k][K -1 - j_k];                            
        
        // Gradiente respecto a pesos
        for(int i=0; i<H_out; ++i)    
            for(int j=0; j<W_out; ++j)  
                for(int c=0; c<C; ++c)  // Convolución entre entrada y salida
                    for(int i_k=0; i_k<this->kernel_fils; ++i_k)
                        for(int j_k=0; j_k<this->kernel_cols; ++j_k)
                            grad_w[f][c][i_k][j_k] += input[c][i + i_k][j + j_k] * output[f][i][j];
    }
    

    // Gradiente respecto a entrada
    for(int i=0; i<input.size(); ++i)
        for(int j=0; j<input[0].size(); ++j)    
            for(int k=0; k<input[0][0].size(); ++k)
                input[i][j][k] = grad_input[i][j][k];

    // Calcular el gradiente del bias
    for(int i=0; i<output.size(); ++i)
        for(int j=0; j<output[0].size(); ++j)    
            for(int k=0; k<output[0][0].size(); ++k)
                grad_bias[i] += output[i][j][k];
};



void Convolutional::aplicar_padding_ptr(float *imagen_3D, int C, int H, int W, int pad)
{   

    for(int c_=0; c_<C; c_++)
    {
        // Traslado vertical en "pad unidades"
        for(int p=0; p<pad; p++)
            for(int j=H-1; j>0; j--)
                for(int k=0; k<W; k++)
                    imagen_3D[c_*H*W + j*W + k] = imagen_3D[c_*H*W + (j-1)*W + k];
            

        // Inicializar a 0.0 las "pad" primeras filas
        for(int i=0; i<pad; i++)
            for(int j=0; j<W; j++)
                imagen_3D[c_*H*W + i*W + j] = 0.0;

        // Inicializar a 0.0 las "pad" últimas filas
        for(int i=H-1; i>H-1-pad; i--)
            for(int j=0; j<W; j++)
                imagen_3D[c_*H*W + i*W + j] = 0.0;

        // Traslado horizontal en "pad unidades"
        for(int p=0; p<pad; p++)
            for(int j=0; j<H; j++)
                for(int k=W-1; k>0; k--)
                    imagen_3D[c_*H*W + j*W + k] = imagen_3D[c_*H*W + j*W + k-1];
            

        // Inicializar a 0.0 las "pad" primeras columnas
        for(int i=0; i<H; i++)
            for(int j=0; j<pad; j++)
                imagen_3D[c_*H*W + i*W + j] = 0.0;   

        // Inicializar a 0.0 las "pad" últimas columnas
        for(int i=0; i<H; i++)
            for(int j=W-1; j>W-1-pad; j--)
                imagen_3D[c_*H*W + i*W + j] = 0.0;   

    }
};


/*
    @brief      Retropropagación de la capa convolucional
    @input      Volumen 3D de entrada de la capa
    @output     Volumen 3D de salida de la capa
    @a          Valor de las neuronas antes de aplicar la función de activación
    @grad_w     Gradientes respecto a pesos
    @grad_bias  Gradientes respecto a sesgos
    @pad        Nivel de padding que se aplicó anteriormente   
*/
void Convolutional::backPropagationGEMM(float *input, float *output, float *a, float *grad_w, float *grad_bias)
{
    // https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
    // Realizar derivada Y_out/Y_in
    int pos;
    for(int i=0; i<this->n_kernels; ++i)
        for(int j=0; j<H_out; ++j)    
            for(int k=0; k<W_out; ++k)
            {
                pos = i*H_out*W_out + j*W_out + k;
                output[pos] = output[pos] * deriv_activationFunction(a[pos]);
            }

    for(int i=0; i<this->n_kernels; ++i)
        for(int j=0; j<H_out; ++j)    
            for(int k=0; k<W_out; ++k)
                output_pad[i*H_out_pad*W_out_pad + j*W_out_pad + k] = output[i*H_out*W_out + j*W_out + k];

    aplicar_padding_ptr(output_pad, this->n_kernels, H_out_pad, W_out_pad, pad);

    // "Desenrrollar" imágenes
    unroll_3dim(n_kernels, H_out_pad, W_out_pad, kernel_fils, output_pad, h_output_unroll);    
    unroll_1dim(C, H, W, H_out, input, h_input_back_unroll);
    matrizTranspuesta(h_input_back_unroll, kernel_fils*kernel_cols*C, H_out*W_out);
    matrizTranspuesta(h_output_unroll, cols_output_unroll, fils_output_unroll);

    // Concatenar los pesos de todos los kernels para una misma capa de profundidad C
    for(int j = 0; j < C; j++)
        for(int i = 0; i < this->n_kernels; i++) 
            for(int kx = kernel_fils-1; kx >= 0; kx--)
                for(int ky = kernel_cols-1; ky >=0; ky--)
                    h_matriz_pesos[j*this->n_kernels*kernel_fils*kernel_cols + i*kernel_fils*kernel_cols + kx*kernel_cols + ky] = this->w_ptr[i*C*kernel_fils*kernel_cols + j*kernel_fils*kernel_cols + kx*kernel_cols + ky];
        
    // Copiar datos de CPU a GPU
    cudaMemcpy(d_matriz_pesos, h_matriz_pesos, bytes_matriz_pesos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_unroll, h_output_unroll, bytes_output_unroll, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_back_unroll, h_input_back_unroll, bytes_input_back_unroll, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, n_kernels*H_out*W_out * sizeof(float), cudaMemcpyHostToDevice);

    // Multiplicación de matrices
    multiplicarMatricesGPU<<<grid_grad_w, block, smem>>>(this->n_kernels, cols_input_back_unroll, H_out * W_out, d_output, d_input_back_unroll, d_grad_w);    // Gradiente respecto a pesos
    multiplicarMatricesGPU<<<grid_grad_input, block, smem>>>(fils_matriz_pesos, cols_output_unroll, cols_matriz_pesos, d_matriz_pesos, d_output_unroll, d_input);  // Gradiente respecto a entrada

    // Paso de GPU a CPU
    cudaMemcpy(grad_w_it, d_grad_w, this->n_kernels*C*kernel_fils*kernel_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(input, d_input, C*H*W * sizeof(float), cudaMemcpyDeviceToHost);

    // Sumar gradiente respecto a pesos
    for(int i=0; i<this->n_kernels; i++)
        for(int j=0; j<C; j++)
            for(int kx=0; kx<kernel_fils; kx++)
                for(int ky=0; ky<kernel_cols; ky++)
                    grad_w[i*C*kernel_fils*kernel_cols + j*kernel_fils*kernel_cols + kx*kernel_cols + ky] += grad_w_it[i*C*kernel_fils*kernel_cols + j*kernel_fils*kernel_cols + kx*kernel_cols + ky]; 

    // Calcular el gradiente del bias
    for(int i=0; i<n_kernels; ++i)
        for(int j=0; j<H_out; ++j)    
            for(int k=0; k<W_out; ++k)
                grad_bias[i] += output[i*H_out*W_out + j*W_out + k];
};


void Convolutional::printMatrix_4D(float* matrix, int F, int C, int n) {
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

void Convolutional::multiplicarMatrices(float* m1, int rows1, int cols1, float* m2, int cols2, float* result) {
    for (int i = 0; i < rows1; i++) 
        for (int j = 0; j < cols2; j++) {
            result[i * cols2 + j] = 0.0f;

            for (int k = 0; k < cols1; k++) 
                result[i * cols2 + j] += m1[i * cols1 + k] * m2[k * cols2 + j];
            
        }
}

void Convolutional::printMatrix_vector(const vector<vector<vector<float>>> &X) {
    for (int i = 0; i < X.size(); i++) {
        for (int j = 0; j < X[i].size(); j++) {
            for (int k = 0; k < X[i][j].size(); k++) 
                cout << X[i][j][k] << " ";
            cout << endl;
        }
        cout << endl;
    }
}


/*
    @brief          Escalar los pesos para evitar que los gradientes "exploten"
    @clip_value     Valor a emplear para realizar el "clip" o escalado
    @return         Se actualizan los valores de w (pesos de la capa)
*/
void Convolutional::escalar_pesos_ptr(float clip_value)
{
    // Calculate the maximum and minimum values of weights
    float max = this->w_ptr[0], min = this->w_ptr[0];

    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                for(int l=0; l<this->w[i][j][k].size(); l++)
                {
                    if(max < this->w[i][j][k][l])
                        max = this->w[i][j][k][l];
                    
                    if(min > this->w[i][j][k][l])
                        min = this->w[i][j][k][l];
                }

    // Perform gradient clipping
    float scaling_factor = clip_value / std::max(std::abs(max), std::abs(min));
    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                for(int l=0; l<this->w[i][j][k].size(); l++)
                    this->w[i][j][k][l] = std::max(std::min(this->w[i][j][k][l] * scaling_factor, clip_value), -clip_value);
}

/*
    @brief          Escalar los pesos para evitar que los gradientes "exploten"
    @clip_value     Valor a emplear para realizar el "clip" o escalado
    @return         Se actualizan los valores de w (pesos de la capa)
*/
void Convolutional::escalar_pesos(float clip_value)
{
    // Calculate the maximum and minimum values of weights
    float max = this->w[0][0][0][0], min = this->w[0][0][0][0];

    for(int i=0; i<this->n_kernels; i++)
        for(int j=0; j<this->C; j++)
            for(int k=0; k<this->H; k++)
                for(int l=0; l<this->W; l++)
                {
                    if(max < this->w_ptr[i*C*H*W + j*H*W + k*W + l])
                        max = this->w[i][j][k][l];
                    
                    if(min > this->w_ptr[i*C*H*W + j*H*W + k*W + l])
                        min = this->w_ptr[i*C*H*W + j*H*W + k*W + l];
                }

    // Perform gradient clipping
    float scaling_factor = clip_value / std::max(std::abs(max), std::abs(min));
    for(int i=0; i<this->n_kernels; i++)
        for(int j=0; j<this->C; j++)
            for(int k=0; k<this->H; k++)
                for(int l=0; l<this->W; l++)
                    this->w_ptr[i*C*H*W + j*H*W + k*W + l] = std::max(std::min(this->w_ptr[i*C*H*W + j*H*W + k*W + l] * scaling_factor, clip_value), -clip_value);
}


/*
    @brief          Actualizar los pesos y sesgos de la capa
    @grad_w         Gradiente de cada peso de la capa
    @grad_b         Gradiente de cada sesgo de la capa
    @return         Se actualizar los valores de w y bias (pesos y sesgos de la capa)
*/
void Convolutional::actualizar_grads_ptr(float *grad_w, float *grad_bias)
{
    // Actualizar pesos
    for(int i=0; i<n_kernels; ++i)
        for(int j=0; j<C; ++j)
            for(int k=0; k<kernel_fils; ++k)
                for(int p=0; p<kernel_cols; ++p)
                    this->w_ptr[i*C*kernel_fils*kernel_cols + j*kernel_fils*kernel_cols + k*kernel_cols + p] -= this->lr * grad_w[i*C*kernel_fils*kernel_cols + j*kernel_fils*kernel_cols + k*kernel_cols + p];

    // Actualizar bias
    for(int i=0; i<this->n_kernels; i++)
        this->bias_ptr[i] -= this->lr * grad_bias[i];
}


/*
    @brief          Actualizar los pesos y sesgos de la capa
    @grad_w         Gradiente de cada peso de la capa
    @grad_b         Gradiente de cada sesgo de la capa
    @return         Se actualizar los valores de w y bias (pesos y sesgos de la capa)
*/
void Convolutional::actualizar_grads(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias)
{
    // Actualizar pesos
    for(int j=0; j<this->w.size(); j++)
        for(int k=0; k<this->w[j].size(); k++)
            for(int p=0; p<this->w[j][k].size(); p++)
                for(int l=0; l<this->w[j][k][p].size(); l++)
                    this->w[j][k][p][l] -= this->lr * grad_w[j][k][p][l];
    
    // Actualizar bias
    for(int i=0; i<this->bias.size(); i++)
        this->bias[i] -= this->lr * grad_bias[i];
    
}



// https://towardsdatascience.com/forward-and-backward-propagation-in-convolutional-neural-networks-64365925fdfa
// https://colab.research.google.com/drive/13MLFWdi3uRMZB7UpaJi4wGyGAZ9ftpPD?authuser=1#scrollTo=FEFgOKF4gGv2


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
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

/*
int main()
{
    // -----------------------------------------------------------------------------------------------------
    // Método estándar
    // -----------------------------------------------------------------------------------------------------    
    auto ini = high_resolution_clock::now();
    auto fin = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(fin - ini);
    int n_kernels = 26, K=5, H=50, W=50, H_out = H-K+1, W_out = W-K+1, pad = 0, C=100;   // C=9
    //int n_kernels = 2, K=2, H=3, W=3, H_out = H-K+1, W_out = W-K+1, C=2;   // C=9
    vector<vector<vector<float>>> a_cpu;

    vector<vector<vector<float>>> input_cpu(C, vector<vector<float>>(H, vector<float>(W, 0)));
    vector<vector<vector<float>>> output_cpu(n_kernels, vector<vector<float>>(H_out, vector<float>(W_out, 0)));

    float *input_gpu = (float *)malloc(H*C*W * sizeof(float));

    for(int i=0; i<C; i++)
        for(int j=0; j<H; j++)
            for(int k=0; k<W; k++)
            {
                //input_cpu[i][j][k] = (float) (j+k) / (H*W);
                input_cpu[i][j][k] = 1.0;
                input_gpu[i*H*W + j*W + k] = input_cpu[i][j][k];
            }
                
    a_cpu = output_cpu;
    
    // Output
    int H_out_pad = H-K + 1, W_out_pad = W-K + 1;
    float *output_gpu = (float *)malloc(n_kernels*H_out_pad*W_out_pad * sizeof(float)),
          *a_gpu = (float *)malloc(n_kernels*H_out_pad*W_out_pad * sizeof(float));

    // Inicializar output a 0
    for(int i=0; i<n_kernels; i++)
        for(int j=0; j<H_out; j++)
            for(int k=0; k<W_out; k++)
                output_gpu[i*H_out*W_out + j*W_out + k] = 0.0;
    
    // Gradiente respecto a pesos
    vector<vector<vector<vector<float>>>> grad_w_cpu(n_kernels, vector<vector<vector<float>>>(C, vector<vector<float>>(K, vector<float>(K, 0.0))));
    float *grad_w_gpu = (float *)malloc(n_kernels * C * K * K * sizeof(float));

    // Inicializar gradientes respecto a pesos a 0.0
    for(int i=0; i<n_kernels * C * K * K; i++)
        grad_w_gpu[i] = 0.0;

    // Gradiente respecto a bias
    vector<float> grad_bias_cpu(n_kernels);
    float *grad_bias_gpu = (float *)malloc(n_kernels * sizeof(float));

    for(int i=0; i<n_kernels; i++)
    {
        grad_bias_cpu[i] = 0.0;
        grad_bias_gpu[i] = 0.0;
    }
        


    // Crear capa convolucional
    Convolutional conv(n_kernels, K, K, input_cpu, 0.1);

    // Establecer device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cout << "Estableciendo dispotisivo para GPU...\n";
    printf("Usando el dispositivo %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    cout << " -------------------------- Método Estándar -------------------------- " << endl;
    //cout << "Input" << endl;
    //printMatrix_vector(input);
    
    ini = high_resolution_clock::now();
    conv.forwardPropagation(input_cpu, output_cpu, a_cpu);

    //cout << "Ouput" << endl;
    //printMatrix_vector(output_cpu);
    
    /*
    //cout << "-- Backprop --" << endl;
    
    
    // Inicializar gradiente de pesos
    for(int i = 0; i < n_kernels; i++) 
        for(int j = 0; j < C; j++)
            for(int kx = 0; kx < K; kx++)
                for(int ky = 0; ky < K; ky++)
                    grad_w[i][j][kx][ky] = 0.0;

    grad_w2 = grad_w;
    */
    /*
    //cout << "Input" << endl;
    conv.backPropagation(input_cpu, output_cpu, a_cpu, grad_w_cpu, grad_bias_cpu);
    //printMatrix_vector(input_cpu);
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);
    
    // Mostrar resultado
    cout << "Tiempo CPU: " << duration.count() << " (us)" << endl;

    /*
    cout << "Gradientes de pesos" << endl;
    // Mostrar gradientes de pesos
    for(int i = 0; i < n_kernels; i++) 
    {
        for(int j = 0; j < C; j++)
        {
            for(int kx = 0; kx < K; kx++)
            {
                for(int ky = 0; ky < K; ky++)
                    cout << grad_w_cpu[i][j][kx][ky] << " ";
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }
    */
    /*
    cout << " -------------------------- Método GEMM -------------------------- " << endl;
    Convolutional conv_gpu(n_kernels, K, K, C, H, W, 0.1);
    
    //cout << "Input" << endl;
    //printMatrix_vector(input_gpu);
    ini = high_resolution_clock::now();
    conv_gpu.forwardPropagationGEMM(input_gpu, output_gpu, a_gpu);
    /*
    cout << "Ouput" << endl;
    // Inicializar output a 0
    for(int i=0; i<n_kernels; i++)
    {
        for(int j=0; j<H_out_pad; j++)
        {
            for(int k=0; k<W_out_pad; k++)
                cout << output_gpu[i*H_out_pad*W_out_pad + j*W_out_pad + k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    */
    /*
    //cout << "-- Backprop --" << endl;
    conv_gpu.backPropagationGEMM(input_gpu, output_gpu, a_gpu, grad_w_gpu, grad_bias_gpu);
    cudaDeviceSynchronize();
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);
    
    // Mostrar resultado
    cout << "Tiempo GPU: " << duration.count() << " (us)" << endl;
    //cout << "Input" << endl;
    //printMatrix_vector(input_gpu);
    
    /*
    for(int i=0; i<C; i++)
    {
        for(int j=0; j<H; j++)
        {
            for(int k=0; k<W; k++)
                cout << input_gpu[i*H*W + j*W + k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    

    
    cout << "Gradientes de pesos" << endl;
    // Mostrar gradientes de pesos
    for(int i = 0; i < n_kernels; i++) 
    {
        for(int j = 0; j < C; j++)
        {
            for(int kx = 0; kx < K; kx++)
            {
                for(int ky = 0; ky < K; ky++)
                    cout << grad_w_gpu[i*C*K*K + j*K*K + kx*K] << " ";
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }
    */
    /*
   // Comprobar resultados
    bool correcto = true;
    float epsilon = 0000000.1;
    int n_errores = 0;
    float err_medio_input = 0.0, err_medio_w = 0.0;

    for(int i=0; i<n_kernels; i++)
        for(int j=0; j<H_out; j++)
            for(int k=0; k<W_out; k++)
                if(abs(output_cpu[i][j][k] - output_gpu[i*H_out*W_out + j*W_out + k]) > epsilon)
                {
                    correcto = false;
                    cout << abs(output_cpu[i][j][k] - output_gpu[i*H_out*W_out + j*W_out + k]) << "output" << endl;
                    n_errores++;
                }

    for(int i=0; i<C; i++)
        for(int j=0; j<H; j++)
            for(int k=0; k<W; k++)
                if(abs(input_cpu[i][j][k] - input_gpu[i*H*W + j*W + k]) > epsilon)
                {
                    correcto = false;
                    //cout << abs(input[i][j][k] - input_gpu[i][j][k]) << " input. " << input[i][j][k] << " vs " << input_gpu[i][j][k] << endl;
                    n_errores++;
                    err_medio_input += abs(input_cpu[i][j][k] - input_gpu[i*H*W + j*W + k]);
                }


    for(int i = 0; i < n_kernels; i++) 
        for(int j = 0; j < C; j++)
            for(int kx = 0; kx < K; kx++)
                for(int ky = 0; ky < K; ky++)
                if(abs(grad_w_cpu[i][j][kx][ky] - grad_w_gpu[i*C*K*K + j*K*K + kx*K + ky]) > epsilon)
                {
                    correcto = false;
                    //cout << abs(grad_w[i][j][kx][ky] - grad_w2[i][j][kx][ky]) << " pesos " << endl;
                    n_errores++;
                    err_medio_w += abs(grad_w_cpu[i][j][kx][ky] - grad_w_gpu[i*C*K*K + j*K*K + kx*K + ky]);
                }



    if(correcto)
        cout << "Todo correcto" << endl;
    else
    {
        cout << "Incorrecto (" << n_errores << " errores) " << endl;
        cout << "Error medio input: " << err_medio_input / C*H*W << endl;
        cout << "Error medio w: " << err_medio_w / n_kernels*C*K*K << endl;
    }
    

    free(input_gpu); free(output_gpu); free(a_gpu);

    return 0;
}
*/