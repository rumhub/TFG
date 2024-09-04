#include "CNN.h"

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}

/*
    @brief Calcula la matriz transpuesta
    @X      Matriz de entrada a transponer
    @Y      Matriz salida transpuesta
    @rows   Filas
    @cols   Columnas
*/
__global__ void transpuesta_flat_outs(float* X, float *Y, int rows, int cols)
{
		int i = blockIdx.x * blockDim.x + threadIdx.x; // tid = threadIdx.x

    // Cada hebra se encarga de una fila
    if(i < rows)
        for (int j = 0; j < cols; j++)
            Y[j * rows + i] = X[i * cols + j];
}

/*
    @brief  Inicializa los índices a 0
    @N      Número de índices a inicializar
*/
__global__ void inicializar_a_0(float *indices, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // tid = threadIdx.x

    // Cada hebra se encarga de un dato
    if(i < N)
        indices[i] = 0.0;
}

/*
    @brief      Actualiza un batch
    @indices    Indices a emplear en la actualización
    @N      Número de índices a emplear
*/
__global__ void actualizar_batch(int *batch, int *indices, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // tid = threadIdx.x

    // Cada hebra se encarga de un dato
    if(i < N)
        batch[i] = indices[i];
}

/*
    @brief          Actualiza las etiquetas de un batch
    @y_batch        Etiquetas a actualizar
    @batch          Batch a emplear
    @train_labels   Etiquetas de los datos de entrenamiento del batch
    @tam_batch      Tamaño del batch
    @n_clases       Número de clases
*/
__global__ void actualizar_etiquetas_batch(float *y_batch, int *batch, float *train_labels, int tam_batch, int n_clases)
{
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;

    // Cada hebra se encarga de un dato
    if(iy < tam_batch && ix < n_clases)
        y_batch[iy*n_clases + ix] = train_labels[batch[iy]*n_clases + ix];
}

/*
    @brief          Suma dos matrices
    @C              Número de canales de profundidad 
    @H              Número de filas 
    @W              Número de columnas
    @X              Primera matriz a sumar
    @Y              Segunda maatriz a sumar y posterior matriz resultado
*/
__global__ void sumar_matrices(int C, int H, int W, float *X, float *Y)
{
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x, ic=0;

    // Cada hebra se encarga de un dato
    if(iy < H && ix < W)
    {
        for(int i=0; i<C; i++)
        {
            Y[iy*W + ix + ic] += X[iy*W + ix + ic];

            ic += H*W;
        }
    }
}

// Error checking macro
#define checkCUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "CUDNN error: " << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

/*
    @brief          Muestra un tensor
    @C              Número de canales de profundidad 
    @H              Número de filas 
    @W              Número de columnas
    @X              Tensor a mostrar
    @Y              Puntero device del tensor a mostrar
*/
void mostrar_tensor(int C, int H, int W, float *X, float *d_X)
{
    cudaMemcpy(X, d_X, C*H*W * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i<C; i++)
    {
        for(int j=0; j<H; j++)
        {
            for(int k=0; k<W; k++)
                cout << X[i*H*W + j*W + k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
}

/*
    CONSTRUCTOR de la clase CNN
    --------------------------------------

    @capas_conv     Indica el número de capas convolucionales, así como la estructura de cada una. Habrá "capas_conv.size()" capas convolucionales, y la estructura de la capa i
                    vendrá dada por capas_conv[i]. De esta forma. capas_conv[i] = {3, 2, 2} corresponde a un kernel 3x2x2, por ejemplo.
    @tams_pool      Indica el número de capas de agrupación, así como la estructura de cada una. tams_pool[i] = {2,2} indica un tamaño de ventana de agrupamiento  de 2x2.
    @padding        Indica el nivel de padding de cada capa convolucional. padding[i] corresponde con el nivel de padding a aplicar en la capa capas_conv[i].
    @capas_fully    Vector de enteros que indica el número de neuronas por capa dentro de la capa totalmente conectada. Habrá capas.size() capas y cada una contendrá capas[i] neuronas.
    @input          Volumen 3D de entrada. Se tendrán en cuenta sus dimensiones para crear las estructuras necesarias y permitir un posterior entrenamiento con volúmenes de iguales dimensiones.
    @lr             Learning Rate o Tasa de Aprendizaje
*/
CNN::CNN(int *capas_conv, int n_capas_conv, int *tams_pool, int *padding, int *capas_fully, int n_capas_fully, int C, int H, int W, const float &lr, const int n_datos, const int mini_batch)
{
    int * i_capas_conv = nullptr;
    int * i_capas_pool = nullptr;

    this->mini_batch = mini_batch;

    // Inicializar tamaño de bloque 1D
    this->block_1D.x = 32;
    this->block_1D.y = 1;

    // Inicializar tamaño de bloque 2D
    this->block_2D.x = 32;
    this->block_2D.y = 32;

    // Inicializar tamaño de grid 1D
    this->grid_1D.y = 1;

    // Ejemplo de uso, capas_conv[0] = {16, 3, 3}

    if(C <= 0)
    {
        cout << "ERROR. Hay que proporcionar un input a la red. \n";
        exit(-1);
    }

    this->n_capas_conv = n_capas_conv;
    this->lr = lr;
    this->convs = new Convolutional[this->n_capas_conv];
    this->plms = new PoolingMax[this->n_capas_conv];
    this->padding = (int *)malloc(n_capas_conv * sizeof(int));
    this->n_clases = capas_fully[n_capas_fully-1];
    this->max_H = 0;
    this->max_W = 0;
    this->max_C = 0;
    this->i_conv_out = (int *)malloc(n_capas_conv * sizeof(int));
    this->i_conv_in = (int *)malloc(n_capas_conv * sizeof(int));
    this->i_plm_out = (int *)malloc(n_capas_conv * sizeof(int));
    this->i_plm_in = (int *)malloc(n_capas_conv * sizeof(int));
    this->i_w = (int *)malloc(n_capas_conv * sizeof(int));
    this->i_b = (int *)malloc(n_capas_conv * sizeof(int));

    for(int i=0; i<n_capas_conv; i++)
        this->padding[i] = padding[i];

    if(max_C < C)
        max_C = C;

    if(max_H < H)
        max_H = H;

    if(max_W < W)
        max_W = W;

    // Inicializar capas convolucionales y maxpool --------------------------------------------
    for(int i=0; i<n_capas_conv; ++i)
    {
        i_capas_conv = capas_conv + 3*i;
        i_capas_pool = tams_pool + 2*i;

        // Capas convolucionales ------------------------------------------------
        //                  nºkernels          filas_kernel      cols_kernel
        Convolutional conv(i_capas_conv[0], i_capas_conv[1], i_capas_conv[2], C, H, W, lr, padding[i]);
        this->convs[i].copiar(conv);

        // H_out = H - K + 1
        C = this->convs[i].get_n_kernels();
        H = this->convs[i].get_H_out();
        W = this->convs[i].get_W_out();

        if(max_C < C)
            max_C = C;

        if(max_H < H)
            max_H = H;

        if(max_W < W)
            max_W = W;

        // Capas MaxPool -----------------------------------------------------------
        //           filas_kernel_pool  cols_kernel_pool
        PoolingMax plm(i_capas_pool[0], i_capas_pool[1], C, H, W);
        this->plms[i].copiar(plm);

        // H_out = H / K + 2*pad
        H = this->plms[i].get_H_out();
        W = this->plms[i].get_W_out();

        if(max_C < C)
            max_C = C;

        if(max_H < H)
            max_H = H;

        if(max_W < W)
            max_W = W;
    }

    
    // Inicializar capa fullyconnected -----------------------------------------
    int *capas_fully_ptr = (int *)malloc((n_capas_fully+1) * sizeof(int));

    capas_fully_ptr[0] = C*H*W;

    for(int i=1; i<n_capas_fully+1; i++)
        capas_fully_ptr[i] = capas_fully[i-1];

    
    this->fully = new FullyConnected(capas_fully_ptr, n_capas_fully+1, lr, n_datos*n_clases);

    // Reserva de espacio para posteriores operaciones
    int tam_img_max = max_C*max_H*max_W;
    //int tam_img_max = max_C*max_H*max_W;

    // Reserva de memoria en device
    checkCudaErrors(cudaMalloc((void **) &d_img_in, tam_img_max * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_img_in_copy, tam_img_max * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_img_out, tam_img_max * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_conv_a_eval, tam_img_max * sizeof(float)));


    int i_out_c = 0, i_in_c = 0, i_out_p = 0, i_in_p = 0, i_w_ = 0, i_b_ = 0;
    for(int i=0; i<n_capas_conv; i++)
    {
        // Convolucional
        this->i_conv_in[i] = i_in_c;
        i_in_c += this->convs[i].get_C() * this->convs[i].get_H() * this->convs[i].get_W();

        this->i_conv_out[i] = i_out_c;
        i_out_c += this->convs[i].get_n_kernels() * this->convs[i].get_cols_input_unroll();

        // Agrupación máxima
        this->i_plm_in[i] = this->i_conv_out[i];

        this->i_plm_out[i] = i_out_p;
        i_out_p += this->plms[i].get_C() * this->plms[i].get_H_out() * this->plms[i].get_W_out();

        i_w[i] = i_w_;
        i_w_ += this->convs[i].get_n_kernels() * this->convs[i].get_C() * this->convs[i].get_kernel_fils() * this->convs[i].get_kernel_cols();

        i_b[i] = i_b_;
        i_b_ += this->convs[i].get_n_kernels();
    }



    tam_in_convs = 0; tam_out_convs = 0; tam_out_pools = 0; tam_kernels_conv = 0;
    tam_flat_out = this->plms[this->n_capas_conv-1].get_C() * this->plms[this->n_capas_conv-1].get_H_out() * this->plms[this->n_capas_conv-1].get_W_out();
    n_bias_conv = 0;

    for(int i=0; i<this->n_capas_conv; i++)
    {
        tam_kernels_conv += this->convs[i].get_n_kernels() * this->convs[i].get_C() * this->convs[i].get_kernel_fils() * this->convs[i].get_kernel_cols();
        tam_in_convs += this->convs[i].get_C() * this->convs[i].get_H() * this->convs[i].get_W();
        tam_out_convs += this->convs[i].get_n_kernels() * this->convs[i].get_cols_input_unroll();
        tam_out_pools += this->plms[i].get_C() * this->plms[i].get_H_out() * this->plms[i].get_W_out();
        n_bias_conv += this->convs[i].get_n_kernels();
    }
    tam_in_pools = tam_out_convs;

    // Reserva de memoria en device
    checkCudaErrors(cudaMalloc((void **) &d_grad_x_fully, mini_batch* this->fully->get_capas()[0] * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_y_batch, mini_batch*n_clases * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_flat_outs_batch, mini_batch* tam_flat_out * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_plms_outs, mini_batch * tam_out_pools * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_conv_grads_w, tam_kernels_conv * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_conv_grads_w_total, tam_kernels_conv * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_conv_grads_bias, n_bias_conv * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_convs_outs, mini_batch * tam_out_convs * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_plms_in_copys, mini_batch * tam_out_convs * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_conv_a, mini_batch * tam_out_convs * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_batch, mini_batch * sizeof(int)));

    // Liberar memoria
    free(capas_fully_ptr);
    

    // ------------------- CUDNN ---------------------------------
    // Tensor
    this->convBiasTensor = new cudnnTensorDescriptor_t[mini_batch * this->n_capas_conv];
    this->poolOutTensor = new cudnnTensorDescriptor_t[mini_batch * this->n_capas_conv];
    this->convOutTensor = new cudnnTensorDescriptor_t[mini_batch * this->n_capas_conv];
    this->convATensor = new cudnnTensorDescriptor_t[mini_batch * this->n_capas_conv];
    this->convGradWTensor = new cudnnTensorDescriptor_t[mini_batch * this->n_capas_conv];

    // Desc
    this->convDesc = new cudnnConvolutionDescriptor_t[mini_batch * this->n_capas_conv];
    this->poolDesc = new cudnnPoolingDescriptor_t[mini_batch * this->n_capas_conv];
    this->convFilterDesc = new cudnnFilterDescriptor_t[mini_batch * this->n_capas_conv];
    this->activation = new cudnnActivationDescriptor_t[mini_batch * this->n_capas_conv];

    cout << "Max: " << mini_batch * this->n_capas_conv << endl;

    // Gradientes
    checkCudaErrors(cudaMalloc((void **) &d_dpool, tam_img_max * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_dconv, tam_img_max * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_dconv_a, tam_img_max * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_dconv_a_copy, tam_img_max * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_dkernel, tam_img_max * sizeof(float)));

    crear_handles(mini_batch);

    // ------------------- CUDNN ---------------------------------


    checkCudaErrors(cudaGetLastError());
}

/*
    @brief          Crea los handles de cuDNN
    @mini_batch     Tamaño del minibatch
*/
void CNN::crear_handles(int mini_batch)
{
    int n_imgs = 1;
    
        
    // Datos de entrenamiento
    checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
    checkCUDNN(cudnnSetTensor4dDescriptor(dataTensor,
                                    CUDNN_TENSOR_NCHW,
                                    CUDNN_DATA_FLOAT,
                                    n_imgs,  // batch size
                                    this->convs[0].get_C(),  // channels
                                    this->convs[0].get_H(), // height (reduced for simplicity)
                                    this->convs[0].get_W()  // width (reduced for simplicity)
    ));

    for(int m=0; m<mini_batch; m++)
    {
        for(int i=0; i<this->n_capas_conv; i++)
        {
            // Tensor
            checkCUDNN(cudnnCreateTensorDescriptor(&convBiasTensor[m*n_capas_conv + i]));
            checkCUDNN(cudnnCreateTensorDescriptor(&poolOutTensor[m*n_capas_conv + i]));
            checkCUDNN(cudnnCreateTensorDescriptor(&convOutTensor[m*n_capas_conv + i]));
            checkCUDNN(cudnnCreateTensorDescriptor(&convATensor[m*n_capas_conv + i]));
            checkCUDNN(cudnnCreateTensorDescriptor(&convGradWTensor[m*n_capas_conv + i]));


            // Desc
            checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc[m*n_capas_conv + i]));
            checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc[m*n_capas_conv + i]));
            checkCUDNN(cudnnCreateFilterDescriptor(&convFilterDesc[m*n_capas_conv + i]));
            checkCUDNN(cudnnCreateActivationDescriptor(&activation[m*n_capas_conv + i]));
        }

        // Establecer dimensiones de los tensores
        for(int i=0; i<this->n_capas_conv; i++)
        {
            // Tensor
            checkCUDNN(cudnnSetTensor4dDescriptor(convBiasTensor[m*n_capas_conv + i],   // cudnnTensorDescriptor_t
                                        CUDNN_TENSOR_NCHW,      // cudnnTensorFormat_t
                                        CUDNN_DATA_FLOAT,       // cudnnDataType_t
                                        1,                  // n 
                                        this->convs[i].get_n_kernels(), // c
                                        1,                      // h
                                        1));                     // w

            checkCUDNN(cudnnSetTensor4dDescriptor(poolOutTensor[m*n_capas_conv + i],       // cudnnTensorDescriptor_t
                                        CUDNN_TENSOR_NCHW,      // cudnnTensorFormat_t
                                        CUDNN_DATA_FLOAT,   // cudnnDataType_t
                                        n_imgs,         // n 
                                        this->plms[i].get_C(),  // c
                                        this->plms[i].get_H_out(),  // h
                                        this->plms[i].get_W_out())); // w

            checkCUDNN(cudnnSetTensor4dDescriptor(convOutTensor[m*n_capas_conv + i],
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n_imgs,  // batch size
                                        this->convs[i].get_n_kernels(), // channels
                                        this->convs[i].get_H_out(), // height (calculated manually)
                                        this->convs[i].get_W_out()  // width (calculated manually)
            ));

            checkCUDNN(cudnnSetTensor4dDescriptor(convATensor[m*n_capas_conv + i],
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        n_imgs,  // batch size
                                        this->convs[i].get_n_kernels(), // channels
                                        this->convs[i].get_H_out(), // height (calculated manually)
                                        this->convs[i].get_W_out()  // width (calculated manually)
            ));

            checkCUDNN(cudnnSetTensor4dDescriptor(convGradWTensor[m*n_capas_conv + i],   // cudnnTensorDescriptor_t
                                        CUDNN_TENSOR_NCHW,      // cudnnTensorFormat_t
                                        CUDNN_DATA_FLOAT,       // cudnnDataType_t
                                        this->convs[i].get_n_kernels(),                  // n 
                                        this->convs[i].get_C(), // c
                                        this->convs[i].get_kernel_fils(),                      // h
                                        this->convs[i].get_kernel_cols()));                     // w



            // Desc
            checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc[m*n_capas_conv + i],         // cudnnPoolingDescriptor_t
                                        CUDNN_POOLING_MAX,       // cudnnPoolingMode_t
                                        CUDNN_PROPAGATE_NAN,         // cudnnNanPropagation_t
                                        this->plms[i].get_kernel_fils(), this->plms[i].get_kernel_cols(),  // windowHeight, windowWidth
                                        0, 0,    // verticalPadding, horizontalPadding
                                        this->plms[i].get_kernel_fils(), this->plms[i].get_kernel_cols()));      // verticalStride, horizontalStride

            checkCUDNN(cudnnSetFilter4dDescriptor(convFilterDesc[m*n_capas_conv + i],
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        this->convs[i].get_n_kernels(), // out_channels
                                        this->convs[i].get_C(),  // in_channels
                                        this->convs[i].get_kernel_fils(),  // kernel height
                                        this->convs[i].get_kernel_cols()   // kernel width
            ));

            checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc[m*n_capas_conv + i],
                                        this->padding[i], this->padding[i],  // padding
                                        1, 1,  // strides
                                        1, 1,  // dilation
                                        CUDNN_CROSS_CORRELATION,
                                        CUDNN_DATA_FLOAT
            ));

            checkCUDNN(cudnnSetActivationDescriptor(activation[m*n_capas_conv + i], 
                                                CUDNN_ACTIVATION_RELU,
                                                CUDNN_PROPAGATE_NAN, 
                                                0.0));

        }    

    }

    checkCUDNN(cudnnCreate(&cudnnHandle));
}


/*
    @brief  Destruye los handles
*/
void CNN::destruir_handles()
{
    cudnnDestroyTensorDescriptor(dataTensor);

    for(int i=0; i<this->n_capas_conv; i++)
    {
        // Tensor
        cudnnDestroyTensorDescriptor(convBiasTensor[i]);
        cudnnDestroyTensorDescriptor(poolOutTensor[i]);
        cudnnDestroyTensorDescriptor(convOutTensor[i]);
        cudnnDestroyTensorDescriptor(convATensor[i]);
    
        // Desc
        cudnnDestroyConvolutionDescriptor(convDesc[i]);
        cudnnDestroyPoolingDescriptor(poolDesc[i]);
        cudnnDestroyFilterDescriptor(convFilterDesc[i]);
    }
    
}

/*
    @brief  Muestra la arquitectura de la red
*/
void CNN::mostrar_arquitectura()
{
    cout << "\n-----------Arquitectura de la red-----------\n";
    cout << "Padding por capa: ";
    for(int i=0; i<this->n_capas_conv-1; i++)
        cout << this->padding[i] << ", ";
    cout << this->padding[this->n_capas_conv-1];
    cout << endl;

    for(int i=0; i<this->n_capas_conv; i++)
    {
        cout << "Dimensiones de entrada a " << this->convs[i].get_n_kernels() << " kernels " << this->convs[i].get_kernel_fils() << "x" << this->convs[i].get_kernel_cols() << " convolucionales: " << this->convs[i].get_C() << "x" << this->convs[i].get_H() << "x" << this->convs[i].get_W() << endl;
        cout << "Dimensiones de entrada a un kernel " << this->plms[i].get_kernel_fils() << "x" << this->plms[i].get_kernel_cols() << " MaxPool: " << this->plms[i].get_C() << "x" << this->plms[i].get_H() << "x" << this->plms[i].get_W() << endl;
    }

    // Volúmen de salida de la última capa MaxPool
    cout << "Dimensiones de salida de un kernel " << this->plms[this->n_capas_conv-1].get_kernel_fils() << "x" << this->plms[this->n_capas_conv-1].get_kernel_cols() << " MaxPool: " << this->plms[this->n_capas_conv-1].get_C() << "x" << this->plms[this->n_capas_conv-1].get_H_out() << "x" << this->plms[this->n_capas_conv-1].get_W_out() << endl;

    // Capas totalmente conectadas
    int * capas = this->fully->get_capas();

    cout << "Capas totalmente concetadas: ";
    for(int i=0; i<this->fully->get_n_capas()-1; i++)
        cout << capas[i] << ", ";
    cout << capas[this->fully->get_n_capas()-1];

    cout << endl;
}

/*
    @brief      Establece el conjunto de entrenamiento
    @x          Imágenes de entrada
    @y          Etiquetas de las imágenes (one-hot)
    @n_imgs     Número de imágenes
    @n_clases   Número de clases
    @C          Número de canales de profundidad por imagen de entrada
    @H          Número de filas por imagen de entrada
    @W          Número de columnas por imagen de entrada
*/
void CNN::set_train(float *x, float *y, int n_imgs, int n_clases, int C, int H, int W)
{
    n_imgs -= 1;
    H += 2*this->padding[0];
    W += 2*this->padding[0];
    this->n_imagenes = n_imgs * n_clases;
    const int M = this->n_imagenes / mini_batch;
    n_batches = M;
    if(this->n_imagenes % mini_batch != 0)
        n_batches++;

    int tam_flat_out = this->plms[this->n_capas_conv-1].get_C() * this->plms[this->n_capas_conv-1].get_H_out() * this->plms[this->n_capas_conv-1].get_W_out();

    this->flat_outs_gpu = (float *)malloc(this->n_imagenes* tam_flat_out * sizeof(float));

    indices = (int *)malloc(this->n_imagenes * sizeof(int));
    batch = (int *)malloc(this->mini_batch * sizeof(int));
    tam_batches = (int *)malloc(n_batches * sizeof(int));
    checkCudaErrors(cudaMalloc((void **) &d_indices, this->n_imagenes * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &this->d_flat_outs, this->n_imagenes* tam_flat_out * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &this->d_flat_outs_T, this->n_imagenes* tam_flat_out * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &this->d_train_labels, this->n_imagenes* n_clases * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &this->d_train_imgs, n_imagenes*C*H*W * sizeof(float)));

    if(this->n_clases != n_clases)
        cout << "\n\nError. Número de clases distinto al establecido previamente en la arquitectura de la red. " << this->n_clases << " != " << n_clases << endl << endl;

    cudaMemcpy(d_train_labels, y, this->n_imagenes* n_clases * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_imgs, x, n_imagenes*C*H*W * sizeof(float), cudaMemcpyHostToDevice);


    // Inicializar tamaño de mini-batches
    for(int i=0; i<M; ++i)
        tam_batches[i] = mini_batch;

    // Último batch puede tener distinto tamaño al resto
    if(this->n_imagenes % mini_batch != 0)
        tam_batches[n_batches-1] = this->n_imagenes % mini_batch;    
}

/*
    @brief      Desordena un vector de elementos
    @vec        Vector a desordenar
    @tam_vec    Tamaño del vector a desordenar
    @rng        random_device mt19937
*/
void shuffle(int *vec, int tam_vec, mt19937& rng) {
    for (int i = tam_vec - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng);
        std::swap(vec[i], vec[j]);
    }
}

/*
    @brief          Entrenamiento mediante descenso del gradiente estocástico (SGD)
    @epocas         Número de épocas
    @mini_batch     Tamaño del minibatch
*/
void CNN::train(int epocas, int mini_batch)
{
    // ------------------------------
    auto ini_prueba = high_resolution_clock::now();
    auto fin_prueba = high_resolution_clock::now();
    auto duration_prueba = duration_cast<milliseconds>(fin_prueba - ini_prueba);
    // ------------------------------

    auto ini = high_resolution_clock::now();
    auto fin = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(fin - ini);

    int n=this->n_imagenes;

    float *d_img_conv_out = nullptr;
    float *d_img_conv_a = nullptr;
    float *d_img_plms_out = nullptr;
    float *d_img_plms_in_copy = nullptr;
    float *d_img_flat_out = nullptr;
    float *d_img_grad_x_fully = nullptr;
    float *d_img_grad_w_conv = nullptr;
    float *d_img_grad_w_conv_total = nullptr;
    float *d_img_grad_b_conv = nullptr;


    int tam_ini = this->convs[0].get_C() * this->convs[0].get_H() * this->convs[0].get_W();

    std::random_device rd;
    std::mt19937 g(rd());
    const float alpha = 1.0f, beta = 0.0f;

    dim3 grid(2,2);
    dim3 block(32, 32);

    //-------------------------------------------------
    // Inicializar índices
    //-------------------------------------------------
    // Inicializar vector de índices
    for(int i=0; i<n; ++i)
        indices[i] = i;

    // -------------------------------------------------------------------------------------------
    int C = this->convs[0].get_C(), H = this->convs[0].get_H(), W = this->convs[0].get_W();
    int n_k = this->convs[0].get_n_kernels(), H_out = this->convs[0].get_H_out(), W_out = this->convs[0].get_W_out();
    float *conv_in = (float*)malloc(128*H*W * sizeof(float));
    float *conv_out = (float*)malloc(128*H_out*W_out * sizeof(float));
    float *conv_a = (float*)malloc(128*H_out*W_out * sizeof(float));
    float *pool_out = (float*)malloc(128*H_out*W_out * sizeof(float));
    // --------------------------------------------------------------------------------------------

    int k1;
    for(int ep=0; ep<epocas; ++ep)
    {
        
        // Desordenar vector de índices
        shuffle(indices, n, g);
        cudaMemcpy(d_indices, indices, n * sizeof(int), cudaMemcpyHostToDevice);

        ini = high_resolution_clock::now();
        
        // ForwardPropagation de cada batch -----------------------------------------------------------------------
        for(int i=0; i<n_batches; ++i)
        {
            
            // Establecer tamaño de grids 1D y 2D
            grid_1D.x = (tam_batches[i]  + block_1D.x -1) / block_1D.x;
            grid_2D.x = (n_clases  + block_2D.x -1) / block_2D.x;
            grid_2D.y = (tam_batches[i]  + block_2D.y -1) / block_2D.y;

            // Crear batch ----------------------
            for(int j=0; j<tam_batches[i]; j++)
                batch[j] = indices[mini_batch*i + j];

            actualizar_batch<<<grid_1D, block_1D>>>(d_batch, d_indices + mini_batch*i, tam_batches[i]);
            actualizar_etiquetas_batch<<<grid_2D, block_2D>>>(d_y_batch, d_batch, d_train_labels, tam_batches[i], n_clases);
            
            
            for(int img=0; img<tam_batches[i]; ++img)
            {

                cudaMemcpy(this->d_img_in, d_train_imgs + tam_ini*batch[img], tam_ini * sizeof(float), cudaMemcpyDeviceToDevice);
                d_img_conv_out = d_convs_outs + img*tam_out_convs + i_conv_out[0];
                d_img_conv_a = d_conv_a + img*tam_out_convs + i_conv_out[0];

                checkCUDNN(cudnnConvolutionForward(
                    cudnnHandle,    // handle
                    &alpha,         // alpha
                    dataTensor,         // xDesc
                    this->d_img_in,         // x
                    convFilterDesc[img * this->n_capas_conv + 0],          // wDesc
                    this->convs[0].get_dw(),        // w
                    convDesc[img * this->n_capas_conv + 0],                // convDesc
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,   // algo
                    nullptr,                            // workSpace 
                    0,                              // workSpaceSizeInBytes
                    &beta,                              // beta
                    convATensor[img * this->n_capas_conv + 0],                     // yDesc
                    d_img_conv_a                        // y
                ));

                checkCUDNN(cudnnActivationForward(cudnnHandle,  // handle 
                                                activation[img * this->n_capas_conv + 0],  // activationDesc
                                                &alpha,         // alpha
                                                convATensor[img * this->n_capas_conv + 0],      // xDesc
                                                d_img_conv_a,            // x
                                                &beta,          // beta
                                                convOutTensor[img * this->n_capas_conv + 0],      // yDesc
                                                d_img_conv_out));      // y

                d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[0];
                d_img_plms_in_copy = d_plms_in_copys + img*tam_out_convs + i_conv_out[0];

                C = this->plms[0].get_C();
                H = this->plms[0].get_H();
                W = this->plms[0].get_W();
                H_out = this->plms[0].get_H_out();
                W_out = this->plms[0].get_W_out();
                
                checkCUDNN(cudnnPoolingForward(
                    cudnnHandle,                          // handle
                    poolDesc[img * this->n_capas_conv + 0],                 // poolingDesc
                    &alpha,                         // alpha
                    convOutTensor[img * this->n_capas_conv + 0],             // xDesc
                    d_img_conv_out,                  // x
                    &beta,                              // beta
                    poolOutTensor[img * this->n_capas_conv + 0],         // yDesc
                    d_img_plms_out               // y
                ));

                // Resto de capas convolucionales y maxpool ----------------------------
                for(int j=1; j<this->n_capas_conv; ++j)
                {
                    // Capa convolucional
                    d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[j-1];
                    d_img_conv_out = d_convs_outs + img*tam_out_convs + i_conv_out[j];
                    d_img_conv_a = d_conv_a + img*tam_out_convs + i_conv_out[j];

                    checkCUDNN(cudnnConvolutionForward(
                        cudnnHandle,    // handle
                        &alpha,         // alpha
                        poolOutTensor[img * this->n_capas_conv + j-1],         // xDesc
                        d_img_plms_out,         // x
                        convFilterDesc[img * this->n_capas_conv + j],          // wDesc
                        this->convs[j].get_dw(),        // w
                        convDesc[img * this->n_capas_conv + j],                // convDesc
                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,   // algo
                        nullptr,                            // workSpace 
                        0,                              // workSpaceSizeInBytes
                        &beta,                              // beta
                        convATensor[img * this->n_capas_conv + j],                     // yDesc
                        d_img_conv_a                        // y
                    ));

                    checkCUDNN(cudnnActivationForward(cudnnHandle,  // handle 
                                                    activation[img * this->n_capas_conv + j],  // activationDesc
                                                    &alpha,         // alpha
                                                    convATensor[img * this->n_capas_conv + j],      // xDesc
                                                    d_img_conv_a,            // x
                                                    &beta,          // beta
                                                    convOutTensor[img * this->n_capas_conv + j],      // yDesc
                                                    d_img_conv_out));      // y

                    // Capa MaxPool
                    d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[j];
                    d_img_plms_in_copy = d_plms_in_copys + img*tam_in_pools + i_plm_in[j];

                    checkCUDNN(cudnnPoolingForward(
                        cudnnHandle,                          // handle
                        poolDesc[img * this->n_capas_conv + j],                 // poolingDesc
                        &alpha,                         // alpha
                        convOutTensor[img * this->n_capas_conv + j],             // xDesc
                        d_img_conv_out,                  // x
                        &beta,                              // beta
                        poolOutTensor[img * this->n_capas_conv + j],         // yDesc
                        d_img_plms_out               // y
                    ));
                }
                
                // Capa flatten
                d_img_flat_out = d_flat_outs_batch + img*tam_flat_out;
                cudaMemcpy(d_img_flat_out, d_img_plms_out, tam_flat_out * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            
            
            // ---------------------------------------------------------------------------------------------------------------------------
            // Capa totalmente conectada
            // ---------------------------------------------------------------------------------------------------------------------------
            // Realizar propagación hacia delante y hacia detrás en la capa totalmente conectada
            transpuesta_flat_outs<<<grid_1D, block_1D>>>(d_flat_outs_batch, d_flat_outs_T, tam_batches[i], tam_flat_out);
            this->fully->set_train_gpu(d_flat_outs_T, d_y_batch, tam_batches[i]);
            this->fully->train_vectores_externos(d_grad_x_fully);
            this->fully->actualizar_parametros_gpu();
            this->fully->escalar_pesos_GEMM(2);



            // ---------------------------------------------------------------------------------------------------------------------------
            // Capas convolucionales, de agrupación y aplanado
            // ---------------------------------------------------------------------------------------------------------------------------

            // ----------------------------------------------
            // ----------------------------------------------
            // BackPropagation ------------------------------
            // ----------------------------------------------
            // ----------------------------------------------
            ini_prueba = high_resolution_clock::now();

            // Inicializar gradientes a 0
            grid_1D.x = (tam_kernels_conv + block_1D.x -1) / block_1D.x;
            inicializar_a_0<<<grid_1D, block_1D>>>(d_conv_grads_w_total, tam_kernels_conv);

            grid_1D.x = (n_bias_conv + block_1D.x -1) / block_1D.x;
            inicializar_a_0<<<grid_1D, block_1D>>>(d_conv_grads_bias, n_bias_conv);

            
            // Cálculo de gradientes respecto a cada parámetro
            for(int img=0; img<tam_batches[i]; ++img)
            {
                // Última capa, su output no tiene padding
                int i_c=this->n_capas_conv-1;

                cudaMemcpy(this->d_img_in, d_train_imgs + tam_ini*batch[img], tam_ini * sizeof(float), cudaMemcpyDeviceToDevice);

                // Usar grad_x_fully[img] en vez de plms_outs[img][i_c] en la última capa MaxPool
                d_img_grad_x_fully = d_grad_x_fully + img*this->fully->get_capas()[0];

                // Capa MaxPool
                d_img_conv_out = d_convs_outs + img*tam_out_convs + i_conv_out[i_c];
                d_img_plms_in_copy = d_plms_in_copys + img*tam_in_pools + i_plm_in[i_c];

                d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[i_c];
                cudaMemcpy(d_dpool, d_img_grad_x_fully, this->plms[i_c].get_C()*this->plms[i_c].get_H_out()*this->plms[i_c].get_W_out() * sizeof(float), cudaMemcpyDeviceToDevice);

                // Perform the maxpool backward pass
                checkCUDNN(cudnnPoolingBackward(
                    cudnnHandle,                       // handle
                    poolDesc[img * this->n_capas_conv + i_c],                 // poolingDesc
                    &alpha,                             // *alpha
                    poolOutTensor[img * this->n_capas_conv + i_c],             // yDesc
                    d_img_grad_x_fully, //d_img_plms_out,                      // *y
                    poolOutTensor[img * this->n_capas_conv + i_c],             // dyDesc
                    d_dpool,                            // *dy
                    convOutTensor[img * this->n_capas_conv + i_c],             // xDesc
                    d_img_conv_out,                      // *xData
                    &beta,                              // *beta
                    convOutTensor[img * this->n_capas_conv + i_c],             // dxDesc
                    d_dconv                             // *dx
                ));

                // Capa convolucional
                d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[i_c-1];
                d_img_conv_a = d_conv_a + img*tam_out_convs + i_conv_out[i_c];
                d_img_grad_w_conv = d_conv_grads_w + i_w[i_c];
                d_img_grad_w_conv_total = d_conv_grads_w_total + i_w[i_c];
                d_img_grad_b_conv = d_conv_grads_bias + i_b[i_c];

                // Perform the ReLU backward pass
                checkCUDNN(cudnnActivationBackward(
                    cudnnHandle,                                    // handle
                    activation[img * this->n_capas_conv + i_c],     // activationDesc
                    &alpha,                                             // *alpha
                    convOutTensor[img * this->n_capas_conv + i_c],      // yDesc
                    d_img_conv_out,                                            // *y
                    convOutTensor[img * this->n_capas_conv + i_c],      // dyDesc
                    d_dconv,                                            // *dy
                    convATensor[img * this->n_capas_conv + i_c],        // xDesc
                    d_img_conv_a,                                       // *x
                    &beta,                                              // *beta
                    convATensor[img * this->n_capas_conv + i_c],        // dxDesc
                    d_dconv_a                                           // *dx
                ));

                if(this->n_capas_conv > 1)
                {
                    checkCUDNN(cudnnConvolutionBackwardFilter(
                        cudnnHandle,                                        // handle
                        &alpha,                                             // *alpha
                        poolOutTensor[img * this->n_capas_conv + i_c-1],    // xDesc
                        d_img_plms_out,                                     // *x
                        convATensor[img * this->n_capas_conv + i_c],        // dyDesc
                        d_dconv_a,                                          // *dy
                        convDesc[img * this->n_capas_conv + i_c],           // convDesc
                        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,                // algo
                        nullptr, 0,                                         // *workSpace, workSpaceSizeInBytes
                        &beta,                                              // *beta
                        convFilterDesc[img * this->n_capas_conv + i_c],     // dwDesc
                        d_img_grad_w_conv                                   // *dw
                    ));
                }else
                {
                    checkCUDNN(cudnnConvolutionBackwardFilter(
                        cudnnHandle,
                        &alpha,
                        poolOutTensor[img * this->n_capas_conv + i_c-1],
                        this->d_img_in,
                        convATensor[img * this->n_capas_conv + i_c],
                        d_dconv_a,
                        convDesc[img * this->n_capas_conv + i_c],
                        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                        nullptr, 0,
                        &beta,
                        convFilterDesc[img * this->n_capas_conv + i_c],
                        d_img_grad_w_conv
                    ));  
                }
 
                // Perform the convolution backward pass
                checkCUDNN(cudnnConvolutionBackwardData(
                    cudnnHandle,                // handle
                    &alpha,                     // *alpha
                    convFilterDesc[img * this->n_capas_conv + i_c],          // wDesc
                    this->convs[i_c].get_dw(),    // *w
                    convATensor[img * this->n_capas_conv + i_c],             // dyDesc
                    d_dconv_a,                  // *dy
                    convDesc[img * this->n_capas_conv + i_c],                // convDesc
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,  // algo
                    nullptr, 0,                         // *workSpace, workSpaceSizeInBytes
                    &beta,                              // *beta
                    poolOutTensor[img * this->n_capas_conv + i_c-1],                 // dxDesc        
                    d_img_plms_out                      // *dx
                ));

                // Sumar gradientes de pesos al total acumulado
                grid.x = (this->convs[i_c].get_W() + block.x -1) / block.x;
                grid.y = (this->convs[i_c].get_H() + block.x -1) / block.x;
                sumar_matrices<<<grid, block>>>(this->convs[i_c].get_C(), this->convs[i_c].get_H(), this->convs[i_c].get_W(), d_img_grad_w_conv, d_img_grad_w_conv_total);

                for(int j=this->n_capas_conv-2; j>=1; j--)
                {
                    C = this->plms[j].get_C();
                    H = this->plms[j].get_H();
                    W = this->plms[j].get_W();
                    H_out = this->plms[j].get_H_out();
                    W_out = this->plms[j].get_W_out();

                    // Capa MaxPool
                    d_img_conv_out = d_convs_outs + img*tam_out_convs + i_conv_out[j];
                    d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[j];
                    d_img_plms_in_copy = d_plms_in_copys + img*tam_in_pools + i_plm_in[j];

                    cudaMemcpy(d_dpool, d_img_plms_out, C*H_out*W_out * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(d_img_plms_out, d_dconv_a_copy, C*H_out*W_out * sizeof(float), cudaMemcpyDeviceToDevice);

                    checkCUDNN(cudnnPoolingBackward(
                        cudnnHandle,                       // handle
                        poolDesc[img * this->n_capas_conv + j],                 // poolingDesc
                        &alpha,                             // *alpha
                        poolOutTensor[img * this->n_capas_conv + j],             // yDesc
                        d_img_plms_out,                      // *y
                        poolOutTensor[img * this->n_capas_conv + j],             // dyDesc
                        d_dpool,                            // *dy
                        convOutTensor[img * this->n_capas_conv + j],             // xDesc
                        d_img_conv_out,                      // *xData
                        &beta,                              // *beta
                        convOutTensor[img * this->n_capas_conv + j],             // dxDesc
                        d_dconv                             // *dx
                    ));

                    // Capa convolucional
                    d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[j-1];
                    d_img_conv_a = d_conv_a + img*tam_out_convs + i_conv_out[j];
                    d_img_grad_w_conv = d_conv_grads_w + i_w[j];
                    d_img_grad_w_conv_total = d_conv_grads_w_total + i_w[j];
                    d_img_grad_b_conv = d_conv_grads_bias + i_b[j];

                    checkCUDNN(cudnnActivationBackward(
                        cudnnHandle,
                        activation[img * this->n_capas_conv + j],
                        &alpha,
                        convOutTensor[img * this->n_capas_conv + j],
                        d_img_conv_out,
                        convOutTensor[img * this->n_capas_conv + j],
                        d_dconv,
                        convATensor[img * this->n_capas_conv + j],
                        d_img_conv_a,
                        &beta,
                        convATensor[img * this->n_capas_conv + j],
                        d_dconv_a
                    ));

                    checkCUDNN(cudnnConvolutionBackwardFilter(
                        cudnnHandle,
                        &alpha,
                        poolOutTensor[img * this->n_capas_conv + j-1],
                        d_img_plms_out,
                        convATensor[img * this->n_capas_conv + j],
                        d_dconv_a,
                        convDesc[img * this->n_capas_conv + j],
                        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                        nullptr, 0,
                        &beta,
                        convFilterDesc[img * this->n_capas_conv + j],
                        d_img_grad_w_conv
                    ));

                    checkCUDNN(cudnnConvolutionBackwardData(
                        cudnnHandle,                // handle
                        &alpha,                     // *alpha
                        convFilterDesc[img * this->n_capas_conv + j],          // wDesc
                        this->convs[j].get_dw(),    // *w
                        convATensor[img * this->n_capas_conv + j],             // dyDesc
                        d_dconv_a,                  // *dy
                        convDesc[img * this->n_capas_conv + j],                // convDesc
                        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,  // algo
                        nullptr, 0,                         // *workSpace, workSpaceSizeInBytes
                        &beta,                              // *beta
                        poolOutTensor[img * this->n_capas_conv + j-1],                 // dxDesc        
                        d_img_plms_out                      // *dx
                    ));

                    // Sumar gradientes de pesos al total acumulado
                    grid.x = (this->convs[j].get_W() + block.x -1) / block.x;
                    grid.y = (this->convs[j].get_H() + block.x -1) / block.x;
                    sumar_matrices<<<grid, block>>>(this->convs[j].get_C(), this->convs[j].get_H(), this->convs[j].get_W(), d_img_grad_w_conv, d_img_grad_w_conv_total);
                }

                if(this->n_capas_conv >1)
                {
                    C = this->plms[0].get_C();
                    H = this->plms[0].get_H();
                    W = this->plms[0].get_W();
                    H_out = this->plms[0].get_H_out();
                    W_out = this->plms[0].get_W_out();

                    d_img_conv_out = d_convs_outs + img*tam_out_convs + i_conv_out[0];
                    d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[0];
                    d_img_plms_in_copy = d_plms_in_copys + img*tam_in_pools + i_plm_in[0];

                    cudaMemcpy(d_dpool, d_img_plms_out, C*H_out*W_out * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(d_img_plms_out, d_dconv_a_copy, C*H_out*W_out * sizeof(float), cudaMemcpyDeviceToDevice);

                    checkCUDNN(cudnnPoolingBackward(
                        cudnnHandle,                       // handle
                        poolDesc[img * this->n_capas_conv + 0],                 // poolingDesc
                        &alpha,                             // *alpha
                        poolOutTensor[img * this->n_capas_conv + 0],             // yDesc
                        d_img_plms_out,                      // *y
                        poolOutTensor[img * this->n_capas_conv + 0],             // dyDesc
                        d_dpool,                            // *dy
                        convOutTensor[img * this->n_capas_conv + 0],             // xDesc
                        d_img_conv_out,                      // *xData
                        &beta,                              // *beta
                        convOutTensor[img * this->n_capas_conv + 0],             // dxDesc
                        d_dconv                             // *dx
                    ));


                    d_img_conv_a = d_conv_a + img*tam_out_convs + i_conv_out[0];
                    d_img_grad_w_conv = d_conv_grads_w + i_w[0];
                    d_img_grad_w_conv_total = d_conv_grads_w_total + i_w[0];
                    d_img_grad_b_conv = d_conv_grads_bias + i_b[0];
                
                    checkCUDNN(cudnnActivationBackward(
                        cudnnHandle,
                        activation[img * this->n_capas_conv + 0],
                        &alpha,
                        convOutTensor[img * this->n_capas_conv + 0],
                        d_img_conv_out,
                        convOutTensor[img * this->n_capas_conv + 0],
                        d_dconv,
                        convATensor[img * this->n_capas_conv + 0],
                        d_img_conv_a,
                        &beta,
                        convATensor[img * this->n_capas_conv + 0],
                        d_dconv_a
                    ));

                    cudaMemcpy(d_dconv_a_copy, d_dconv_a, C*H*W * sizeof(float), cudaMemcpyDeviceToDevice);

                    checkCUDNN(cudnnConvolutionBackwardFilter(
                        cudnnHandle,
                        &alpha,
                        dataTensor,
                        d_img_in,
                        convATensor[img * this->n_capas_conv + 0],
                        d_dconv_a,
                        convDesc[img * this->n_capas_conv + 0],
                        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                        nullptr, 0,
                        &beta,
                        convFilterDesc[img * this->n_capas_conv + 0],
                        d_img_grad_w_conv
                    ));
                
                    // Sumar gradientes de pesos al total acumulado
                    grid.x = (this->convs[0].get_W() + block.x -1) / block.x;
                    grid.y = (this->convs[0].get_H() + block.x -1) / block.x;
                    sumar_matrices<<<grid, block>>>(this->convs[0].get_C(), this->convs[0].get_H(), this->convs[0].get_W(), d_img_grad_w_conv, d_img_grad_w_conv_total);
                }
            }

            // Actualizar parámetros --------------------------------------------------------------------
            // Actualizar parámetros de capas convolucionales
            for(int j=0; j<this->n_capas_conv; ++j)
            {
                d_img_grad_w_conv_total = d_conv_grads_w_total + i_w[j];
                d_img_grad_b_conv = d_conv_grads_bias + i_b[j];
                this->convs[j].actualizar_grads_vectores_externos(d_img_grad_w_conv_total, d_img_grad_b_conv, tam_batches[i]);
            }

            // Actualizar parámetros de capas convolucionales
            for(int j=0; j<this->n_capas_conv; ++j)
                this->convs[j].escalar_pesos_vectores_externos(2);
        }
        
        
        evaluar_modelo();
        cudaDeviceSynchronize();

        fin = high_resolution_clock::now();
        duration = duration_cast<seconds>(fin - ini);

        cout << "Época: " << ep << ",                                           " << duration.count() << " (s)" << endl;
        
        checkCudaErrors(cudaGetLastError());
    }
}


/*
    @brief  Evalúa el modelo sobre los datos de entrenamiento. Las medidas de evaluación son Accuracy y Entropía Cruzada
*/
void CNN::evaluar_modelo()
{
    int tam_flat_out = this->plms[this->n_capas_conv-1].get_C() * this->plms[this->n_capas_conv-1].get_H_out() * this->plms[this->n_capas_conv-1].get_W_out(),
        tam_ini = this->convs[0].get_C() * this->convs[0].get_H() * this->convs[0].get_W();

    dim3 block_1D(32, 1);
    dim3 grid_1D((this->n_imagenes  + block_1D.x -1) / block_1D.x, 1);
    const float alpha = 1.0f, beta = 0.0f;

    // Realizar la propagación hacia delante
    for(int img=0; img<this->n_imagenes; ++img)
    {
        // Copiar imagen de entrenamiento en img_in
        cudaMemcpy(this->d_img_in, d_train_imgs + img*tam_ini, tam_ini * sizeof(float), cudaMemcpyDeviceToDevice);

        // Capas convolucionales y maxpool ----------------------------
        for(int i=0; i<this->n_capas_conv; ++i)
        {
            // Capa convolucional
            if(i==0)
            {
                checkCUDNN(cudnnConvolutionForward(
                    cudnnHandle,    // handle
                    &alpha,         // alpha
                    dataTensor,         // xDesc
                    this->d_img_in,         // x
                    convFilterDesc[i],          // wDesc
                    this->convs[i].get_dw(),        // w
                    convDesc[i],                // convDesc
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,   // algo
                    nullptr,                            // workSpace 
                    0,                              // workSpaceSizeInBytes
                    &beta,                              // beta
                    convATensor[i],                     // yDesc
                    this->d_conv_a_eval                        // y
                ));
            }else
            {
                checkCUDNN(cudnnConvolutionForward(
                    cudnnHandle,    // handle
                    &alpha,         // alpha
                    poolOutTensor[i-1],         // xDesc
                    this->d_img_in,         // x
                    convFilterDesc[i],          // wDesc
                    this->convs[i].get_dw(),        // w
                    convDesc[i],                // convDesc
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,   // algo
                    nullptr,                            // workSpace 
                    0,                              // workSpaceSizeInBytes
                    &beta,                              // beta
                    convATensor[i],                     // yDesc
                    this->d_conv_a_eval                        // y
                ));
            }


            checkCUDNN(cudnnActivationForward(cudnnHandle,  // handle 
                                            activation[i],  // activationDesc
                                            &alpha,         // alpha
                                            convATensor[i],      // xDesc
                                            this->d_conv_a_eval,            // x
                                            &beta,          // beta
                                            convOutTensor[i],      // yDesc
                                            this->d_img_out));      // y

            // Capa MaxPool
            checkCUDNN(cudnnPoolingForward(
                cudnnHandle,                          // handle
                poolDesc[i],                 // poolingDesc
                &alpha,                         // alpha
                convOutTensor[i],             // xDesc
                this->d_img_out,                  // x
                &beta,                              // beta
                poolOutTensor[i],         // yDesc
                this->d_img_in               // y
            ));

        }

        // Capa flatten
        cudaMemcpy(d_flat_outs + img*tam_flat_out, this->d_img_in, tam_flat_out * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    transpuesta_flat_outs<<<grid_1D, block_1D>>>(d_flat_outs, d_flat_outs_T, this->n_imagenes, tam_flat_out);
    this->fully->set_train_gpu(d_flat_outs_T, this->d_train_labels, n_imagenes);
    this->fully->evaluar_modelo_GEMM();

    // Realizar media y obtener valores finales
    checkCudaErrors(cudaGetLastError());
}
