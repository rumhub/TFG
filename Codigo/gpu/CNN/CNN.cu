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
    @brief  Inicializa los índices
    @N      Número de índices a inicializar
*/
__global__ void inicializar_indices(int* indices, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // tid = threadIdx.x

    // Cada hebra se encarga de un dato
    if(i < N)
        indices[i] = i;
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
    CONSTRUCTOR de la clase CNN
    --------------------------------------

    @capas_conv     Indica el número de capas convolucionales, así como la estructura de cada una. Habrá "capas_conv.size()" capas convolucionales, y la estructura de la capa i
                    vendrá dada por capas_conv[i]. De esta forma. capas_conv[i] = {3, 2, 2} corresponde a un kernel 3x2x2, por ejemplo.
    @tams_pool      Indica el número de capas de agrupación, así como la estructura de cada una. tams_pool[i] = {2,2} indica un tamaño de ventana de agrupamiento  de 2x2.
    @padding        Indica el nivel de padding de cada capa convolucional. padding[i] corresponde con el nivel de padding a aplicar en la capa capas_conv[i].
    @capas_fully    Vector de enteros que indica el número de neuronas por capa dentro de la capa totalmente conectada. Habrá capas.size() capas y cada una contendrá capas[i] neuronas.
    @n_capas_fully  Número de capas totalmente conectadas
    @C              Número de canales de profundidad por imagen de entrada
    @H              Número de filas por imagen de entrada
    @W              Número de columnas por imagen de entrada
    @lr             Learning Rate o Tasa de Aprendizaje
    @n_datos        Número de imágenes de entrada
    @mini_batch     Tamaño del minibatch
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
    

    //checkCudaErrors(cudaGetLastError());
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
    @brief      Establece el conjunto de entrenamiento y test
    @x          Imágenes de entrada
    @y          Etiquetas de las imágenes (one-hot)
    @n_clases   Número de clases
    @C          Número de canales de profundidad por imagen de entrada
    @H          Número de filas por imagen de entrada
    @W          Número de columnas por imagen de entrada
*/
void CNN::set_train(float *x_train, float *y_train, float *x_test, float *y_test, int n_imgs, int n_clases, int C, int H, int W)
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

    checkCudaErrors(cudaMalloc((void **) &this->d_test_labels, this->n_imagenes* n_clases * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &this->d_test_imgs, n_imagenes*C*H*W * sizeof(float)));

    if(this->n_clases != n_clases)
        cout << "\n\nError. Número de clases distinto al establecido previamente en la arquitectura de la red. " << this->n_clases << " != " << n_clases << endl << endl;

    cudaMemcpy(d_train_labels, y_train, this->n_imagenes* n_clases * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_imgs, x_train, n_imagenes*C*H*W * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_test_labels, y_test, this->n_imagenes* n_clases * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_imgs, x_test, n_imagenes*C*H*W * sizeof(float), cudaMemcpyHostToDevice);

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
    float *d_img_grad_b_conv = nullptr;


    int tam_ini = this->convs[0].get_C() * this->convs[0].get_H() * this->convs[0].get_W();

    std::random_device rd;
    std::mt19937 g(rd());


    //-------------------------------------------------
    // Inicializar índices
    //-------------------------------------------------
    // Inicializar vector de índices
    for(int i=0; i<n; ++i)
        indices[i] = i;

    cout << "\n ------------ ENTRENAMIENTO ------------ " << endl;


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
                this->convs[0].forwardPropagation_vectores_externos(this->d_img_in, d_img_conv_out, d_img_conv_a);

                d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[0];
                d_img_plms_in_copy = d_plms_in_copys + img*tam_out_convs + i_conv_out[0];
                this->plms[0].forwardPropagation_vectores_externos(d_img_conv_out, d_img_plms_out, d_img_plms_in_copy);

                // Resto de capas convolucionales y maxpool ----------------------------
                for(int j=1; j<this->n_capas_conv; ++j)
                {
                    // Capa convolucional
                    d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[j-1];
                    d_img_conv_out = d_convs_outs + img*tam_out_convs + i_conv_out[j];
                    d_img_conv_a = d_conv_a + img*tam_out_convs + i_conv_out[j];
                    this->convs[j].forwardPropagation_vectores_externos(d_img_plms_out, d_img_conv_out, d_img_conv_a);

                    // Capa MaxPool
                    d_img_plms_out = d_plms_outs + img*tam_out_pools + i_plm_out[j];
                    d_img_plms_in_copy = d_plms_in_copys + img*tam_in_pools + i_plm_in[j];
                    this->plms[j].forwardPropagation_vectores_externos(d_img_conv_out, d_img_plms_out, d_img_plms_in_copy);
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
            this->fully->train_GEMM(d_grad_x_fully);
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
    
    evaluar_modelo_en_test();

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
    int tam_flat_out = this->plms[this->n_capas_conv-1].get_C() * this->plms[this->n_capas_conv-1].get_H_out() * this->plms[this->n_capas_conv-1].get_W_out(),
        tam_ini = this->convs[0].get_C() * this->convs[0].get_H() * this->convs[0].get_W();

    dim3 block_1D(32, 1);
    dim3 grid_1D((this->n_imagenes  + block_1D.x -1) / block_1D.x, 1);

    cout << " --------- TEST --------- " << endl;

    // Realizar la propagación hacia delante
    for(int img=0; img<this->n_imagenes; ++img)
    {
        // Copiar imagen de entrenamiento en img_in
        cudaMemcpy(this->d_img_in, d_test_imgs + img*tam_ini, tam_ini * sizeof(float), cudaMemcpyDeviceToDevice);

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
