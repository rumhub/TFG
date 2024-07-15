#include "CNN.h"

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}

__global__ void transpuesta_flat_outs(float* X, float *Y, int rows, int cols)
{
		int i = blockIdx.x * blockDim.x + threadIdx.x; // tid = threadIdx.x

    // Cada hebra se encarga de una fila
    if(i < rows)
        for (int j = 0; j < cols; j++)
            Y[j * rows + i] = X[i * cols + j];
}

__global__ void inicializar_indices(int* indices, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // tid = threadIdx.x

    // Cada hebra se encarga de un dato
    if(i < N)
        indices[i] = i;
}

__global__ void inicializar_a_0(float *indices, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // tid = threadIdx.x

    // Cada hebra se encarga de un dato
    if(i < N)
        indices[i] = 0.0;
}


__global__ void actualizar_batch(int *batch, int *indices, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // tid = threadIdx.x

    // Cada hebra se encarga de un dato
    if(i < N)
        batch[i] = indices[i];
}


__global__ void actualizar_etiquetas_batch(float *y_batch, int *batch, float *train_labels, int tam_batch, int n_clases)
{
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;

    // Cada hebra se encarga de un dato
    if(iy < tam_batch && ix < n_clases)
        y_batch[iy*n_clases + ix] = train_labels[batch[iy]*n_clases + ix];
}


// Error checking macro
#define checkCUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "CUDNN error: " << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
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

    // Padding de la primera capa
    H += 2*padding[0];
    W += 2*padding[0];

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
        Convolutional conv(i_capas_conv[0], i_capas_conv[1], i_capas_conv[2], C, H, W, lr);
        this->convs[i].copiar(conv);

        // H_out = H - K + 1
        C = i_capas_conv[0];
        H = H - i_capas_conv[1] + 1;
        W = W - i_capas_conv[2] + 1;

        if(max_C < C)
            max_C = C;

        if(max_H < H)
            max_H = H;

        if(max_W < W)
            max_W = W;

        // Capas MaxPool -----------------------------------------------------------
        int pad_sig = 0;    // Padding de la siguiente capa convolucional
        if(this->n_capas_conv > i+1)
            pad_sig = this->padding[i+1];
        //           filas_kernel_pool  cols_kernel_pool
        PoolingMax plm(i_capas_pool[0], i_capas_pool[1], C, H, W, pad_sig);
        this->plms[i].copiar(plm);

        // H_out = H / K + 2*pad
        H = H / i_capas_pool[0] + 2*pad_sig;
        W = W / i_capas_pool[0] + 2*pad_sig;

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
    checkCudaErrors(cudaMalloc((void **) &d_conv_grads_bias, n_bias_conv * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_convs_outs, mini_batch * tam_out_convs * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_plms_in_copys, mini_batch * tam_out_convs * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_conv_a, mini_batch * tam_out_convs * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_batch, mini_batch * sizeof(int)));

    // Liberar memoria
    free(capas_fully_ptr);
    

    // ------------------- CUDNN ---------------------------------
    // Tensor
    this->convBiasTensor = new cudnnTensorDescriptor_t[this->n_capas_conv];
    this->poolOutTensor = new cudnnTensorDescriptor_t[this->n_capas_conv];
    this->convOutTensor = new cudnnTensorDescriptor_t[this->n_capas_conv];
    this->convATensor = new cudnnTensorDescriptor_t[this->n_capas_conv];
    
    // Desc
    this->convDesc = new cudnnConvolutionDescriptor_t[this->n_capas_conv];
    this->poolDesc = new cudnnPoolingDescriptor_t[this->n_capas_conv];
    this->convFilterDesc = new cudnnFilterDescriptor_t[this->n_capas_conv];
    this->activation = new cudnnActivationDescriptor_t[this->n_capas_conv];

    crear_handles(mini_batch);

    // ------------------- CUDNN ---------------------------------


    checkCudaErrors(cudaGetLastError());
}

void CNN::crear_handles(int mini_batch)
{
    mini_batch = 1;
    
        
    // Datos de entrenamiento
    checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
    checkCUDNN(cudnnSetTensor4dDescriptor(dataTensor,
                                    CUDNN_TENSOR_NCHW,
                                    CUDNN_DATA_FLOAT,
                                    mini_batch,  // batch size
                                    this->convs[0].get_C(),  // channels
                                    this->convs[0].get_H(), // height (reduced for simplicity)
                                    this->convs[0].get_W()  // width (reduced for simplicity)
    ));

    
    checkCUDNN(cudnnCreate(&cudnnHandle));
    for(int i=0; i<this->n_capas_conv; i++)
    {
        // Tensor
        checkCUDNN(cudnnCreateTensorDescriptor(&convBiasTensor[i]));
        checkCUDNN(cudnnCreateTensorDescriptor(&poolOutTensor[i]));
        checkCUDNN(cudnnCreateTensorDescriptor(&convOutTensor[i]));
        checkCUDNN(cudnnCreateTensorDescriptor(&convATensor[i]));

        // Desc
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc[i]));
        checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc[i]));
        checkCUDNN(cudnnCreateFilterDescriptor(&convFilterDesc[i]));
        checkCUDNN(cudnnCreateActivationDescriptor(&activation[i]));
    }

    // Establecer dimensiones de los tensores
    for(int i=0; i<this->n_capas_conv; i++)
    {
        // Tensor
        checkCUDNN(cudnnSetTensor4dDescriptor(convBiasTensor[i],   // cudnnTensorDescriptor_t
                                    CUDNN_TENSOR_NCHW,      // cudnnTensorFormat_t
                                    CUDNN_DATA_FLOAT,       // cudnnDataType_t
                                    1,                  // n 
                                    this->convs[i].get_n_kernels(), // c
                                    1,                      // h
                                    1));                     // w

        checkCUDNN(cudnnSetTensor4dDescriptor(poolOutTensor[i],       // cudnnTensorDescriptor_t
                                    CUDNN_TENSOR_NCHW,      // cudnnTensorFormat_t
                                    CUDNN_DATA_FLOAT,   // cudnnDataType_t
                                    mini_batch,         // n 
                                    this->plms[i].get_C(),  // c
                                    this->plms[i].get_H_out(),  // h
                                    this->plms[i].get_W_out())); // w

        checkCUDNN(cudnnSetTensor4dDescriptor(convOutTensor[i],
                                    CUDNN_TENSOR_NCHW,
                                    CUDNN_DATA_FLOAT,
                                    mini_batch,  // batch size
                                    this->convs[i].get_n_kernels(), // channels
                                    this->convs[i].get_H_out(), // height (calculated manually)
                                    this->convs[i].get_W_out()  // width (calculated manually)
        ));

        checkCUDNN(cudnnSetTensor4dDescriptor(convATensor[i],
                                    CUDNN_TENSOR_NCHW,
                                    CUDNN_DATA_FLOAT,
                                    mini_batch,  // batch size
                                    this->convs[i].get_n_kernels(), // channels
                                    this->convs[i].get_H_out(), // height (calculated manually)
                                    this->convs[i].get_W_out()  // width (calculated manually)
        ));


        // Desc
        checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc[i],         // cudnnPoolingDescriptor_t
                                    CUDNN_POOLING_MAX,       // cudnnPoolingMode_t
                                    CUDNN_PROPAGATE_NAN,         // cudnnNanPropagation_t
                                    this->plms[i].get_kernel_fils(), this->plms[i].get_kernel_cols(),  // windowHeight, windowWidth
                                    0, 0,    // verticalPadding, horizontalPadding
                                    1, 1));      // verticalStride, horizontalStride

        checkCUDNN(cudnnSetFilter4dDescriptor(convFilterDesc[i],
                                    CUDNN_DATA_FLOAT,
                                    CUDNN_TENSOR_NCHW,
                                    this->convs[i].get_n_kernels(), // out_channels
                                    this->convs[i].get_C(),  // in_channels
                                    this->convs[i].get_kernel_fils(),  // kernel height
                                    this->convs[i].get_kernel_cols()   // kernel width
        ));

        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc[i],
                                    0, 0,  // padding
                                    1, 1,  // strides
                                    1, 1,  // dilation
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT
        ));

        checkCUDNN(cudnnSetActivationDescriptor(activation[i], 
                                            CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 
                                            0.0));

    }    
}

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


void shuffle(int *vec, int tam_vec, mt19937& rng) {
    for (int i = tam_vec - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng);
        std::swap(vec[i], vec[j]);
    }
}

void CNN::prueba_cudnn()
{
    // Set up tensor descriptors for input
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1,  // batch size
        3,  // channels
        4, // height (reduced for simplicity)
        4  // width (reduced for simplicity)
    ));

    // Set up tensor descriptors for convolution output
    cudnnTensorDescriptor_t conv_output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&conv_output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        conv_output_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1,  // batch size
        2, // channels
        2, // height (calculated manually)
        2  // width (calculated manually)
    ));

    // Set up convolution descriptor
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(
        kernel_descriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        2, // out_channels
        3,  // in_channels
        3,  // kernel height
        3   // kernel width
    ));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
        convolution_descriptor,
        0, 0,  // padding
        1, 1,  // strides
        1, 1,  // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    // Allocate memory for tensors
    float h_input[1 * 3 * 4 * 4] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    float h_kernel[2 * 3 * 3 * 3] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,

        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
    };

    float* d_input;
    float* d_conv_output;
    float* d_kernel;
    cudaMalloc(&d_input, sizeof(h_input));
    cudaMalloc(&d_conv_output, 1 * 2 * 2 * 2 * sizeof(float));
    cudaMalloc(&d_kernel, sizeof(h_kernel));

    cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    // Perform the convolution forward pass
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(
        cudnnHandle,
        &alpha,
        dataTensor,
        this->d_img_in,
        convFilterDesc[0],
        this->convs[0].get_dw(),        
        convDesc[0],

        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        nullptr, 0,
        &beta,
        convOutTensor[0],
        //d_conv_output
        //d_img_conv_out
        d_convs_outs
    ));

                /*
        const float alpha = 1.0f, beta = 0.0f;
                checkCUDNN(cudnnConvolutionForward(
                    cudnnHandle,
                    &alpha,
                    dataTensor,
                    this->d_img_in,
                    convFilterDesc[0],
                    this->convs[0].get_dw(),
                    convDesc[0],
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                    nullptr, 0,
                    &beta,
                    convOutTensor[0],
                    d_img_conv_out
                ));
                */

    /*
    // Perform the convolution forward pass
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(
        cudnnHandle,
        &alpha,
        input_descriptor,
        d_input,
        kernel_descriptor,
        d_kernel,
        convolution_descriptor,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        nullptr, 0,
        &beta,
        conv_output_descriptor,
        d_conv_output
    ));
    */

    // Copy the result back to the host
    float h_conv_output[1 * 2 * 2 * 2];
    cudaMemcpy(h_conv_output, d_conv_output, sizeof(h_conv_output), cudaMemcpyDeviceToHost);

    // Print the convolution output
    std::cout << "Convolution output tensor:" << std::endl;

    // Print the maxpool output
    std::cout << "Maxpool output tensor:" << std::endl;

    // Print the fully connected layer output
    std::cout << "Fully connected layer output:" << std::endl;

    // Clean up
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(conv_output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudaFree(d_input);
    cudaFree(d_conv_output);
    cudaFree(d_kernel);
}

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
    float *d_img_grad_b_conv = nullptr;


    int tam_ini = this->convs[0].get_C() * this->convs[0].get_H() * this->convs[0].get_W();

    std::random_device rd;
    std::mt19937 g(rd());
    const float alpha = 1.0f, beta = 0.0f;


    //-------------------------------------------------
    // Inicializar índices
    //-------------------------------------------------
    // Inicializar vector de índices
    for(int i=0; i<n; ++i)
        indices[i] = i;

    // int * indices2 = (int*)malloc(n*sizeof(int));

    for(int ep=0; ep<epocas; ++ep)
    {
        
        // Desordenar vector de índices
        //shuffle(indices, n, g);
        cudaMemcpy(d_indices, indices, n * sizeof(int), cudaMemcpyHostToDevice);

        // cudaMemcpy(indices2, d_indices, n * sizeof(int), cudaMemcpyDeviceToHost);

        // for(int i=0; i<n; i++)
        //     if(abs(indices[i] - indices2[i]) > 0.001)
        //         cout << "Error. " << indices[i] << " != " << indices2[i] << endl;

        

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
                //this->convs[0].forwardPropagation_vectores_externos(this->d_img_in, d_img_conv_out, d_img_conv_a);

                // ----------------------------------------------------
                int C = this->convs[0].get_C(), H = this->convs[0].get_H(), W = this->convs[0].get_W();
                int n_k = this->convs[0].get_n_kernels(), H_out = this->convs[0].get_H_out(), W_out = this->convs[0].get_W_out();
                float *conv_in = (float*)malloc(64*H*W * sizeof(float));
                float *conv_out = (float*)malloc(64*H_out*W_out * sizeof(float));
                float *conv_a = (float*)malloc(64*H_out*W_out * sizeof(float));

                // Perform the convolution forward pass
                checkCUDNN(cudnnConvolutionForward(
                    cudnnHandle,    // handle
                    &alpha,         // alpha
                    dataTensor,         // xDesc
                    this->d_img_in,         // x
                    convFilterDesc[0],          // wDesc
                    this->convs[0].get_dw(),        // w
                    convDesc[0],                // convDesc
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,   // algo
                    nullptr,                            // workSpace 
                    0,                              // workSpaceSizeInBytes
                    &beta,                              // beta
                    convATensor[0],                     // yDesc
                    d_img_conv_a                        // y
                ));

                checkCUDNN(cudnnActivationForward(cudnnHandle,  // handle 
                                                activation[0],  // activationDesc
                                                &alpha,         // alpha
                                                convATensor[0],      // xDesc
                                                d_img_conv_a,            // x
                                                &beta,          // beta
                                                convOutTensor[0],      // yDesc
                                                d_img_conv_out));      // y

                //prueba_cudnn();


                cudaMemcpy(conv_in, this->d_img_in, tam_ini * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(conv_out, d_img_conv_out, n_k*H_out*W_out * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(conv_a, d_img_conv_a, n_k*H_out*W_out * sizeof(float), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                checkCudaErrors(cudaGetLastError());

                cout << "conv_in" << endl;
                for(int i=0; i<C; i++)
                {
                    for(int j=0; j<H; j++)
                    {
                        for(int k=0; k<W; k++)
                            cout << conv_in[i*H*W + j*W + k] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                cout << "conv_a" << endl;
                for(int i=0; i<n_k; i++)
                {
                    for(int j=0; j<H_out; j++)
                    {
                        for(int k=0; k<W_out; k++)
                            cout << conv_a[i*H_out*W_out + j*W_out + k] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                cout << "conv_out" << endl;
                for(int i=0; i<n_k; i++)
                {
                    for(int j=0; j<H_out; j++)
                    {
                        for(int k=0; k<W_out; k++)
                            cout << conv_out[i*H_out*W_out + j*W_out + k] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }

                // ----------------------------------------------------

                d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[0];
                d_img_plms_in_copy = d_plms_in_copys + img*tam_out_convs + i_conv_out[0];
                //this->plms[0].forwardPropagation_vectores_externos(d_img_conv_out, d_img_plms_out, d_img_plms_in_copy);


                // ----------------------------------------
                C = this->plms[0].get_C();
                H = this->plms[0].get_H();
                W = this->plms[0].get_W();
                H_out = this->plms[0].get_H_out();
                W_out = this->plms[0].get_W_out();
                
                float *pool_out = (float*)malloc(64*H_out*W_out * sizeof(float));
                // Capa MaxPool con cudnn
                checkCUDNN(cudnnPoolingForward(
                    cudnnHandle,                          // handle
                    poolDesc[0],                 // poolingDesc
                    &alpha,                         // alpha
                    convOutTensor[0],             // xDesc
                    d_img_conv_out,                  // x
                    &beta,                              // beta
                    poolOutTensor[0],         // yDesc
                    d_img_plms_out               // y
                ));
                cudaMemcpy(pool_out, d_img_plms_out, C*H_out*W_out * sizeof(float), cudaMemcpyDeviceToHost);


                cout << "pool_out" << endl;
                for(int i=0; i<C; i++)
                {
                    for(int j=0; j<H_out; j++)
                    {
                        for(int k=0; k<W_out; k++)
                            cout << pool_out[i*H_out*W_out + j*W_out + k] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }

                // ----------------------------------------

                // Resto de capas convolucionales y maxpool ----------------------------
                for(int j=1; j<this->n_capas_conv; ++j)
                {
                    // Capa convolucional
                    d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[j-1];
                    d_img_conv_out = d_convs_outs + img*tam_out_convs + i_conv_out[j];
                    d_img_conv_a = d_conv_a + img*tam_out_convs + i_conv_out[j];
                    //this->convs[j].forwardPropagation_vectores_externos(d_img_plms_out, d_img_conv_out, d_img_conv_a);

                    // Perform the convolution forward pass
                    checkCUDNN(cudnnConvolutionForward(
                        cudnnHandle,    // handle
                        &alpha,         // alpha
                        poolOutTensor[j-1],         // xDesc
                        d_img_plms_out,         // x
                        convFilterDesc[j],          // wDesc
                        this->convs[j].get_dw(),        // w
                        convDesc[j],                // convDesc
                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,   // algo
                        nullptr,                            // workSpace 
                        0,                              // workSpaceSizeInBytes
                        &beta,                              // beta
                        convATensor[j],                     // yDesc
                        d_img_conv_a                        // y
                    ));

                    checkCUDNN(cudnnActivationForward(cudnnHandle,  // handle 
                                                    activation[j],  // activationDesc
                                                    &alpha,         // alpha
                                                    convATensor[j],      // xDesc
                                                    d_img_conv_a,            // x
                                                    &beta,          // beta
                                                    convOutTensor[j],      // yDesc
                                                    d_img_conv_out));      // y
                    // --------------------------------------
                    cudaMemcpy(conv_in, d_img_plms_out, tam_out_pools * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(conv_out, d_img_conv_out, tam_out_convs * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(conv_a, d_img_conv_a, tam_out_convs * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();
                    checkCudaErrors(cudaGetLastError());

                    C = this->convs[j].get_C(), H = this->convs[j].get_H(), W = this->convs[j].get_W();
                    n_k = this->convs[j].get_n_kernels(), H_out = this->convs[j].get_H_out(), W_out = this->convs[j].get_W_out();
                    
                    cout << "conv_in" << endl;
                    for(int i=0; i<C; i++)
                    {
                        for(int j=0; j<H; j++)
                        {
                            for(int k=0; k<W; k++)
                                cout << conv_in[i*H*W + j*W + k] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    cout << "conv_a" << endl;
                    for(int i=0; i<n_k; i++)
                    {
                        for(int j=0; j<H_out; j++)
                        {
                            for(int k=0; k<W_out; k++)
                                cout << conv_a[i*H_out*W_out + j*W_out + k] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    cout << "conv_out" << endl;
                    for(int i=0; i<n_k; i++)
                    {
                        for(int j=0; j<H_out; j++)
                        {
                            for(int k=0; k<W_out; k++)
                                cout << conv_out[i*H_out*W_out + j*W_out + k] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }

                    // --------------------------------------

                    // Capa MaxPool
                    d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[j];
                    d_img_plms_in_copy = d_plms_in_copys + img*tam_in_pools + i_plm_in[j];
                    //this->plms[j].forwardPropagation_vectores_externos(d_img_conv_out, d_img_plms_out, d_img_plms_in_copy);

                    // ----------------------------
                    C = this->plms[j].get_C();
                    H = this->plms[j].get_H();
                    W = this->plms[j].get_W();
                    H_out = this->plms[j].get_H_out();
                    W_out = this->plms[j].get_W_out();

                    checkCUDNN(cudnnPoolingForward(
                        cudnnHandle,                          // handle
                        poolDesc[j],                 // poolingDesc
                        &alpha,                         // alpha
                        convOutTensor[j],             // xDesc
                        d_img_conv_out,                  // x
                        &beta,                              // beta
                        poolOutTensor[j],         // yDesc
                        d_img_plms_out               // y
                    ));
                    cudaMemcpy(pool_out, d_img_plms_out, C*H_out*W_out * sizeof(float), cudaMemcpyDeviceToHost);


                    cout << "pool_out" << endl;
                    for(int i=0; i<C; i++)
                    {
                        for(int j=0; j<H_out; j++)
                        {
                            for(int k=0; k<W_out; k++)
                                cout << pool_out[i*H_out*W_out + j*W_out + k] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    int k1;
                    cin >> k1;
                    // ----------------------------
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

            //cout << " ----------- BACKPROP ----------- " << endl;
            // Inicializar gradientes a 0
            grid_1D.x = (tam_kernels_conv + block_1D.x -1) / block_1D.x;
            inicializar_a_0<<<grid_1D, block_1D>>>(d_conv_grads_w, tam_kernels_conv);

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
                this->plms[i_c].backPropagation_vectores_externos(d_img_conv_out, d_img_grad_x_fully, d_img_plms_in_copy);

                // Capa convolucional
                d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[i_c-1];
                d_img_conv_a = d_conv_a + img*tam_out_convs + i_conv_out[i_c];
                d_img_grad_w_conv = d_conv_grads_w + i_w[i_c];
                d_img_grad_b_conv = d_conv_grads_bias + i_b[i_c];

                if(this->n_capas_conv > 1)
                    this->convs[i_c].backPropagation_vectores_externos(d_img_plms_out, d_img_conv_out, d_img_conv_a, d_img_grad_w_conv, d_img_grad_b_conv);
                else
                    this->convs[i_c].backPropagation_vectores_externos(this->d_img_in, d_img_conv_out, d_img_conv_a, d_img_grad_w_conv, d_img_grad_b_conv);

                for(int j=this->n_capas_conv-2; j>=1; j--)
                {
                    // Capa MaxPool
                    d_img_conv_out = d_convs_outs + img*tam_out_convs + i_conv_out[j];
                    d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[j];
                    d_img_plms_in_copy = d_plms_in_copys + img*tam_in_pools + i_plm_in[j];
                    this->plms[j].backPropagation_vectores_externos(d_img_conv_out, d_img_plms_out, d_img_plms_in_copy);

                    // Capa convolucional
                    d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[j-1];
                    d_img_conv_a = d_conv_a + img*tam_out_convs + i_conv_out[j];
                    d_img_grad_w_conv = d_conv_grads_w + i_w[j];
                    d_img_grad_b_conv = d_conv_grads_bias + i_b[j];
                    this->convs[j].backPropagation_vectores_externos(d_img_plms_out, d_img_conv_out, d_img_conv_a, d_img_grad_w_conv, d_img_grad_b_conv);
                }


                if(this->n_capas_conv >1)
                {
                    d_img_conv_out = d_convs_outs + img*tam_out_convs + i_conv_out[0];
                    d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[0];
                    d_img_plms_in_copy = d_plms_in_copys + img*tam_in_pools + i_plm_in[0];
                    this->plms[0].backPropagation_vectores_externos(d_img_conv_out, d_img_plms_out, d_img_plms_in_copy);

                    d_img_conv_a = d_conv_a + img*tam_out_convs + i_conv_out[0];
                    d_img_grad_w_conv = d_conv_grads_w + i_w[0];
                    d_img_grad_b_conv = d_conv_grads_bias + i_b[0];
                    this->convs[0].backPropagation_vectores_externos(d_img_in, d_img_conv_out, d_img_conv_a, d_img_grad_w_conv, d_img_grad_b_conv);
                }
            }

            // Actualizar parámetros --------------------------------------------------------------------
            // Actualizar parámetros de capas convolucionales
            for(int j=0; j<this->n_capas_conv; ++j)
            {
                d_img_grad_w_conv = d_conv_grads_w + i_w[j];
                d_img_grad_b_conv = d_conv_grads_bias + i_b[j];
                this->convs[j].actualizar_grads_vectores_externos(d_img_grad_w_conv, d_img_grad_b_conv, tam_batches[i]);
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
    
    //evaluar_modelo_en_test();

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

    // Realizar la propagación hacia delante
    for(int img=0; img<this->n_imagenes; ++img)
    {
        // Copiar imagen de entrenamiento en img_in
        cudaMemcpy(this->d_img_in, d_train_imgs + img*tam_ini, tam_ini * sizeof(float), cudaMemcpyDeviceToDevice);

        // Capas convolucionales y maxpool ----------------------------
        for(int i=0; i<this->n_capas_conv; ++i)
        {
            // Capa convolucional
            this->convs[i].forwardPropagation_vectores_externos(this->d_img_in, this->d_img_out, this->d_conv_a_eval);

            // Capa MaxPool
            this->plms[i].forwardPropagation_vectores_externos(this->d_img_out, this->d_img_in, this->d_img_in_copy);
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

/*
    @brief  Evalúa el modelo sobre los datos de test. Las medidas de evaluación son Accuracy y Entropía Cruzada
*/
void CNN::evaluar_modelo_en_test()
{
    /*
    int n=this->test_imgs.size();
    double t1, t2;
    vector<vector<vector<float>>> img_in, img_out, img_in_copy, conv_a;

    vector<float> flat_out;
    float acc ,entr;

    vector<vector<float>> flat_outs(n);

    // Inicialización de parámetros
    //t1 = omp_get_wtime();
    acc = 0.0;
    entr = 0.0;


    // Popagación hacia delante
    for(int img=0; img<n; img++)
    {
        img_in = this->test_imgs[img];

        // Capas convolucionales y maxpool ----------------------------
        for(int i=0; i<this->n_capas_conv; ++i)
        {
            // Capa convolucional
            img_out = this->outputs[i*2];
            conv_a = img_out;
            this->convs[i].forwardPropagation(img_in, img_out, conv_a);
            img_in = img_out;

            // Capa MaxPool
            img_out = this->outputs[i*2+1];
            img_in_copy = img_in;

            int pad_sig = 0;    // Padding de la siguiente capa convolucional
            if(this->n_capas_conv > i+1)
                pad_sig = this->padding[i+1];

            this->plms[i].forwardPropagation(img_in, img_out, img_in_copy, pad_sig);
            img_in = img_out;
        }

        // Capa de aplanado
        (*this->flat).forwardPropagation(img_out, flat_out);

        flat_outs[img] = flat_out;
    }

    // Cada hebra obtiene el accuracy y la entropía cruzada sobre una porción de imágenes
    acc = (*this->fully).accuracy(flat_outs,this->test_labels);
    entr = (*this->fully).cross_entropy(flat_outs, this->test_labels);

    // Realizar media y obtener valores finales
    acc = acc / n * 100;
    entr = -entr / n;

    //t2 = omp_get_wtime();

    cout << "\n------------- RESULTADOS EN TEST --------------- " << endl;
    cout << "Accuracy: " << acc << " %,  ";


    cout << "Entropía cruzada: " << entr << ",         " << endl << endl;
    //cout << "Entropía cruzada: " << entr << ",         " << t2 - t1 << " (s) " << endl << endl;
    */
}

/*
int main()
{
    //vector<vector<int>> capas_conv = {{3, 3, 3}, {3, 5, 5}}, tams_pool = {{2, 2}, {2, 2}};
    int C=2, H=10, W=10, n_capas_fully = 2, n_capas_conv = 2, n_imagenes = 5, n_clases = 4;
    int *capas_fully = (int *)malloc(n_capas_fully * sizeof(int)),
        *capas_conv = (int *)malloc(n_capas_conv*3 * sizeof(int)),
        *capas_pool = (int *)malloc(n_capas_conv*2 * sizeof(int)),
        *padding = (int *)malloc(n_capas_conv * sizeof(int));

    float *X = (float *)malloc(n_imagenes*C*H*W * sizeof(float)),
        *Y = (float *)malloc(n_imagenes*n_clases * sizeof(float));

    float lr = 0.0001;
    int i=0;
    capas_fully[0] = 2;
    capas_fully[1] = n_clases;

    // Primera capa convolucional
    capas_conv[i*3 +0] = 3;      // 4 kernels
    capas_conv[i*3 +1] = 3;      // kernels de 3 filas
    capas_conv[i*3 +2] = 3;      // kernels de 2 columnas

    i = 1;
    // Segunda capa convolucional
    capas_conv[i*3 +0] = 3;      // 7 kernels
    capas_conv[i*3 +1] = 3;      // kernels de 5 filas
    capas_conv[i*3 +2] = 3;      // kernels de 5 columnas

    i=0;
    // Primera capa MaxPool
    capas_pool[i*2 +0] = 2;      // kernels de 2 filas
    capas_pool[i*2 +1] = 2;      // kernels de 2 columnas

    i = 1;
    // Segunda capa MaxPool
    capas_pool[i*2 +0] = 2;      // kernels de 2 filas
    capas_pool[i*2 +1] = 2;      // kernels de 2 columnas

    // Padding
    padding[0] = 0;
    padding[1] = 0;

    // Input
    for(int i=0; i<n_imagenes*C*H*W; i++)
        X[i] = i;

    // Etiquetas
    for(int i=0; i<n_imagenes; i++)
        for(int j=0; j<n_clases; j++)
            Y[i*n_clases + j] = 0.0;

    // Poner que todas las imágenes pertecen a la clase 1, por ejemplo
    for(int i=0; i<n_imagenes; i++)
        Y[i*n_clases + 1] = 1.0;

    CNN cnn(capas_conv, n_capas_conv, capas_pool, padding, capas_fully, n_capas_fully, C, H, W, lr);
    //CNN cnn(capas_conv, n_capas_conv, capas_pool, padding, capas_fully, n_capas_fully, C, H-2*padding[0], W-2*padding[0], lr);
    cnn.mostrar_arquitectura();
    cnn.set_train(X, Y, n_imagenes, n_clases, C, H, W);
    //cnn.evaluar_modelo();
    cnn.train(10, 2);

    free(capas_fully); free(capas_conv); free(capas_pool); free(padding);
    return 0;
}
*/
