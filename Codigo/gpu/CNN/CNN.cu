#include "CNN.h"

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }else
    {
        cout << "Todo correcto!" << endl;
    }
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
CNN::CNN(int *capas_conv, int n_capas_conv, int *tams_pool, int *padding, int *capas_fully, int n_capas_fully, int C, int H, int W, const float &lr, const int mini_batch)
{
    int * i_capas_conv = nullptr;
    int * i_capas_pool = nullptr;

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


    this->fully = new FullyConnected(capas_fully_ptr, n_capas_fully+1, lr, mini_batch);

    // Reserva de espacio para posteriores operaciones
    int tam_img_max = max_C*max_H*max_W;
    //int tam_img_max = max_C*max_H*max_W;

    this->img_in = (float *)malloc(tam_img_max * sizeof(float));
    this->img_out = (float *)malloc(tam_img_max * sizeof(float)); 
    this->img_in_copy = (float *)malloc(tam_img_max * sizeof(float)); 
    this->conv_a = (float *)malloc(tam_img_max * sizeof(float));
    this->a_ptr = (float *)malloc(this->fully->get_n_neuronas() * sizeof(float));
    this->grad_a_ptr = (float *)malloc(this->fully->get_n_neuronas() * sizeof(float));
    this->z_ptr = (float *)malloc(this->fully->get_n_neuronas() * sizeof(float)); 

    // Borrar ----------------------------------------------------------------------------------------------------------------------------------------------------
    // Borrar ----------------------------------------------------------------------------------------------------------------------------------------------------
    for(int i=0; i<tam_img_max; i++)
    {
        this->img_in[i] = 1.0;
        this->img_out[i] = 2.0;
        this->conv_a[i] = 3.0;
    }
    // Borrar ----------------------------------------------------------------------------------------------------------------------------------------------------
    // Borrar ----------------------------------------------------------------------------------------------------------------------------------------------------

    int i_out_c = 0, i_in_c = 0, i_out_p = 0, i_in_p = 0, i_w_ = 0, i_b_ = 0;
    for(int i=0; i<n_capas_conv; i++)
    {
        // Convolucional
        this->i_conv_in[i] = i_in_c;
        i_in_c += this->convs[i].get_C() * this->convs[i].get_H() * this->convs[i].get_W(); 

        this->i_conv_out[i] = i_out_c;
        i_out_c += this->convs[i].get_n_kernels() * this->convs[i].get_H_out() * this->convs[i].get_W_out();   

        // Agrupación máxima
        this->i_plm_in[i] = i_in_p;
        i_in_p += this->plms[i].get_C() * this->plms[i].get_H() * this->plms[i].get_W(); 

        this->i_plm_out[i] = i_out_p;
        i_out_p += this->plms[i].get_C() * this->plms[i].get_H_out() * this->plms[i].get_W_out();  

        i_w[i] = i_w_;
        i_w_ += this->convs[i].get_n_kernels() * this->convs[i].get_C() * this->convs[i].get_kernel_fils() * this->convs[i].get_kernel_cols();

        i_b[i] = i_b_;
        i_b_ += this->convs[i].get_n_kernels();
    }

    // Liberar memoria
    free(capas_fully_ptr);
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

void CNN::mostrar_train_img(int n_img)
{
    int C = this->convs[0].get_C(),
        H = this->convs[0].get_H(),
        W = this->convs[0].get_W();
    float *img_train = this->train_imgs + n_img*C*H*W;

    cout << "\nImagen " << n_img << endl;
    for(int j=0; j<C; j++)
    {
        for(int k=0; k<H; k++)
        {
            for(int p=0; p<W; p++)
                cout << img_train[j*H*W + k*W + p] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
}


void CNN::set_train(float *x, float *y, int n_imgs, int n_clases, int C, int H, int W)
{
    n_imgs -= 1;
    H += 2*this->padding[0];
    W += 2*this->padding[0];
    this->n_imagenes = n_imgs * n_clases;
    this->train_imgs = (float *)malloc(n_imagenes*C*H*W * sizeof(float));
    this->train_labels = (float *)malloc(n_imagenes*n_clases * sizeof(float));

    int tam_flat_out = this->plms[this->n_capas_conv-1].get_C() * this->plms[this->n_capas_conv-1].get_H_out() * this->plms[this->n_capas_conv-1].get_W_out();
    this->flat_outs = (float *)malloc(this->n_imagenes* tam_flat_out * sizeof(float));
    if(this->n_clases != n_clases)
        cout << "\n\nError. Número de clases distinto al establecido previamente en la arquitectura de la red. " << this->n_clases << " != " << n_clases << endl << endl;

    for(int i=0; i<n_imagenes*C*H*W; i++)
        train_imgs[i] = x[i];
    
    for(int i=0; i<n_imagenes*n_clases; i++)
        train_labels[i] = y[i];

    /*
    // Mostrar imágenes
    cout << "\nX\n";
    for(int i=0; i<this->n_imagenes; i++)
    {
        for(int j=0; j<C; j++)
        {
            for(int k=0; k<H; k++)
            {
                for(int p=0; p<W; p++)
                    cout << x[i*C*H*W + j*H*W + k*W + p] << " ";
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;

    cout << "\nY\n";
    for(int i=0; i<this->n_imagenes; i++)
    {
        for(int j=0; j<n_clases; j++)
            cout << y[i*n_clases + j] << " ";
        cout << endl;
    }
    cout << endl;
    */
}

/*
    @brief  Aplica padding sobre una imagen sin aumentar su tamaño
    @input  Imagen sobre la cual aplicar padding
    @pad    Nivel de padding a aplicar
    @return Imagen @input con padding interno aplicado
*/
void CNN::padding_interno_ptr(float *input, int C, int H, int W, const int &pad)
{
    for(int i=0; i<C; ++i)
    {
        // Primeras "pad" filas se igualan a 0.0
        for(int j=0; j<pad; ++j)
            for(int k=0; k<H; ++k)
            input[i*H*W + j*W + k] = 0.0; 

        // Últimas "pad" filas se igualan a 0.0
        for(int j=H-1; j>=H - pad; j--)
            for(int k=0; k<H; ++k)
            input[i*H*W + j*W + k] = 0.0; 
        
        // Por cada fila
        for(int k=0; k<H; ++k)
        {
            // Primeras "pad" casillas se igualan a 0.0
            for(int j=0; j<pad; ++j)
                input[i*H*W + j*W + k] = 0.0;

            // Últimas "pad" casillas se igualan a 0.0
            for(int j=W-1; j>=W - pad; j--)
                input[i*H*W + j*W + k] = 0.0;
        }
    }    
}

/*
    @brief  Aplica padding sobre una imagen sin aumentar su tamaño
    @input  Imagen sobre la cual aplicar padding
    @pad    Nivel de padding a aplicar
    @return Imagen @input con padding interno aplicado
*/
void CNN::padding_interno(vector<vector<vector<float>>> &input, const int &pad)
{
    for(int i=0; i<input.size(); ++i)
    {
        // Primeras "pad" filas se igualan a 0.0
        for(int j=0; j<pad; ++j)
            for(int k=0; k<input[i].size(); ++k)
            input[i][j][k] = 0.0; 

        // Últimas "pad" filas se igualan a 0.0
        for(int j=input[i].size()-1; j>=input[i].size() - pad; j--)
            for(int k=0; k<input[i].size(); ++k)
            input[i][j][k] = 0.0; 
        
        // Por cada fila
        for(int k=0; k<input[i].size(); ++k)
        {
            // Primeras "pad" casillas se igualan a 0.0
            for(int j=0; j<pad; ++j)
                input[i][k][j] = 0.0;

            // Últimas "pad" casillas se igualan a 0.0
            for(int j=input[i][k].size()-1; j>=input[i][k].size() - pad; j--)
                input[i][k][j] = 0.0;
        }
    }    
}


void shuffle(int *vec, int tam_vec, mt19937& rng) {
    for (int i = tam_vec - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng);
        std::swap(vec[i], vec[j]);
    }
}


/*
                cout << "Input" << endl;
                for(int i=0; i<this->convs[0].get_C(); i++)
                {
                    for(int j=0; j<this->convs[0].get_H(); j++)
                    {
                        for(int k=0; k<this->convs[0].get_W(); k++)
                            cout << img_train[i*this->convs[0].get_H()*this->convs[0].get_W() + j*this->convs[0].get_W() + k] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                int pepe;
                cin >> pepe;

                cout << "Output" << endl;
                for(int i=0; i<this->convs[0].get_n_kernels(); i++)
                {
                    for(int j=0; j<this->convs[0].get_H_out(); j++)
                    {
                        for(int k=0; k<this->convs[0].get_W_out(); k++)
                            cout << img_conv_out[i*this->convs[0].get_H_out()*this->convs[0].get_W_out() + j*this->convs[0].get_W_out() + k] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                cin >> pepe;
*/


void CNN::train(int epocas, int mini_batch)
{
    
    auto ini = high_resolution_clock::now();
    auto fin = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(fin - ini);

    int n=this->n_imagenes;
    int C, H_out, W_out;
   
    int tam_in_convs = 0, tam_out_convs = 0, tam_in_pools = 0, tam_out_pools = 0, tam_kernels_conv = 0, 
        tam_flat_out = this->plms[this->n_capas_conv-1].get_C() * this->plms[this->n_capas_conv-1].get_H_out() * this->plms[this->n_capas_conv-1].get_W_out(),
        n_bias_conv = 0;

    for(int i=0; i<this->n_capas_conv; i++)
    {
        tam_kernels_conv += this->convs[i].get_n_kernels() * this->convs[i].get_C() * this->convs[i].get_kernel_fils() * this->convs[i].get_kernel_cols(); 
        tam_in_convs += this->convs[i].get_C() * this->convs[i].get_H() * this->convs[i].get_W(); 
        tam_out_convs += this->convs[i].get_n_kernels() * this->convs[i].get_H_out() * this->convs[i].get_W_out(); 
        tam_out_pools += this->plms[i].get_C() * this->plms[i].get_H_out() * this->plms[i].get_W_out(); 
        tam_in_pools += this->plms[i].get_C() * this->plms[i].get_H() * this->plms[i].get_W(); 
        n_bias_conv += this->convs[i].get_n_kernels();
    }


    float *grad_x_fully = (float *)malloc(mini_batch* this->fully->get_capas()[0] * sizeof(float)), // Capa totalmente conectada
          *grads_bias_fully = (float *)malloc(this->fully->get_n_neuronas() * sizeof(float)),
          *grads_pesos_fully = (float *)malloc(this->fully->get_n_pesos() * sizeof(float)),
          *flat_outs_batch = (float *)malloc(mini_batch* tam_flat_out * sizeof(float)),                   // Capa de aplanado
          *plms_outs = (float *)malloc(mini_batch * tam_out_pools * sizeof(float)),                 // Capa de agrupación máxima
          *plms_in_copys = (float *)malloc(mini_batch * tam_in_pools* sizeof(float)), 
          *conv_grads_w = (float *)malloc(tam_kernels_conv * sizeof(float)),                        // Capa convolucional
          *conv_grads_bias = (float *)malloc(n_bias_conv * sizeof(float)),
          *convs_outs = (float *)malloc(mini_batch * tam_out_convs * sizeof(float)), 
          *conv_a = (float *)malloc(mini_batch * tam_out_convs * sizeof(float)); 

    float *img_train = nullptr;
    float *img_conv_out = nullptr;
    float *img_conv_a = nullptr;
    float *img_plms_out = nullptr;
    float *img_plms_in_copy = nullptr;
    float *img_flat_out = nullptr;
    float *img_grad_x_fully = nullptr;
    float *img_grad_w_conv = nullptr;
    float *img_grad_b_conv = nullptr;


    float *y_batch = (float *)malloc(mini_batch*n_clases * sizeof(float));

    // -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------
    /*
    int cont = 0, cont_b = 0;
    for(int i=0; i<this->n_capas_conv; i++)
    {
        int n_K = this->convs[i].get_n_kernels(), C = this->convs[i].get_C(), H = this->convs[i].get_kernel_fils(), W = this->convs[i].get_kernel_cols();
        img_grad_w_conv = conv_grads_w + i_w[i];
        img_grad_b_conv = conv_grads_bias + i_b[i];
        
        for(int n=0; n<n_K; n++)
        {
            // Bias
            img_grad_b_conv[n] = cont_b++;

            // Conv
            for(int j=0; j<C; j++)
                for(int k=0; k<H; k++)
                    for(int p=0; p<W; p++)
                        img_grad_w_conv[n*C*H*W + j*H*W + k*W + p] = cont++;
        }

    }

    // Mostrar gradientes de pesos
    for(int i=0; i<this->n_capas_conv; i++)
    {
        int n_K = this->convs[i].get_n_kernels(), C = this->convs[i].get_C(), H = this->convs[i].get_kernel_fils(), W = this->convs[i].get_kernel_cols();
        img_grad_w_conv = conv_grads_w + i_w[i];
        img_grad_b_conv = conv_grads_bias + i_b[i];
        
        cout << "Capa conv " << i <<  ", " << n_K << "x" << C << "x" << H << "x" << W << endl;
        for(int n=0; n<n_K; n++)
        {
            cout << "(" << i << ", " << n << ") Bias: " << img_grad_b_conv[n] << endl;
            for(int j=0; j<C; j++)
            {
                for(int k=0; k<H; k++)
                {
                    for(int p=0; p<W; p++)
                        cout << img_grad_w_conv[n*C*H*W + j*H*W + k*W + p] << " ";
                    cout << endl;
                }
                cout << endl;
            }
            cout << endl;
        }
    }
    */


    // -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------


    const int M = n / mini_batch;
    int pad_sig, C_ini = this->convs[0].get_C(), H_ini = this->convs[0].get_H(), W_ini = this->convs[0].get_W(), tam_ini = C_ini*H_ini*W_ini;

    std::random_device rd;
    std::mt19937 g(rd());

    int n_batches = M;
    if(n % mini_batch != 0)
        n_batches++;
    int *indices = (int *)malloc(n * sizeof(int)),
        *batch = (int *)malloc(mini_batch * sizeof(int)),
        *tam_batches = (int *)malloc(n_batches * sizeof(int));
    
    //-------------------------------------------------
    // Inicializar índices
    //-------------------------------------------------
    // Inicializar vector de índices
    for(int i=0; i<n; ++i)
        indices[i] = i;

    // Inicializar tamaño de mini-batches
    for(int i=0; i<M; ++i)
        tam_batches[i] = mini_batch;
    
    // Último batch puede tener distinto tamaño al resto
    if(n % mini_batch != 0)
        tam_batches[n_batches-1] = n % mini_batch;   

    
    int k1;
    for(int ep=0; ep<epocas; ++ep)
    {
        
        ini = high_resolution_clock::now();

        // Desordenar vector de índices
        shuffle(indices, n, g);

        

    // ---------------------------------------------------------------
    // ---------------------------------------------------------------
    for(int i=0; i<mini_batch; i++)
        for(int j=0; j<n_clases; j++)
            y_batch[i*n_clases + j] = train_labels[indices[i]*n_clases + j];

    cout << "LABELS" << endl;
    for(int i=0; i<mini_batch; i++)
    {
        for(int j=0; j<n_clases; j++)
            cout << train_labels[indices[i]*n_clases + j] << " ";
        cout << endl;
    }
    cout << endl;

    cout << "Y" << endl;
    for(int i=0; i<mini_batch; i++)
    {
        for(int j=0; j<n_clases; j++)
            cout << y_batch[i*n_clases + j] << " ";
        cout << endl;
    }
    cout << endl;

    int k2;
    cin >> k2;
    // ---------------------------------------------------------------
    // ---------------------------------------------------------------




        // ForwardPropagation de cada batch -----------------------------------------------------------------------
        for(int i=0; i<n_batches; ++i)
        {
            
            // Crear el batch para cada hebra ----------------------
            for(int j=0; j<tam_batches[i]; j++)
                batch[j] = indices[mini_batch*i + j];   

            
            for(int img=0; img<tam_batches[i]; ++img)
                for(int j=0; j<this->n_capas_conv; ++j)
                {
                    pad_sig = 0;    // Padding de la siguiente capa convolucional
                    if(this->n_capas_conv > j+1)
                        pad_sig = this->padding[j+1];

                    img_plms_out = plms_outs + img*tam_out_pools + i_plm_out[j];
                    padding_interno_ptr(img_plms_out, this->plms[j].get_C(), this->plms[j].get_H_out(), this->plms[j].get_W_out(), pad_sig);
                }
            
            
            // ---------------------------------------------------------------------------------------
            for(int img=0; img<tam_batches[i]; ++img)
            {
                
                // Primera capa convolucional y maxpool -----------------------
                img_train = this->train_imgs + tam_ini*batch[img];
                img_conv_out = convs_outs + img*tam_out_convs + i_conv_out[0];
                img_conv_a = conv_a + img*tam_out_convs + i_conv_out[0];
                this->convs[0].forwardPropagationGEMM(img_train, img_conv_out, img_conv_a);

                /*
                cout << "Input" << endl;
                for(int i_=0; i_<this->convs[0].get_C(); i_++)
                {
                    for(int j_=0; j_<this->convs[0].get_H(); j_++)
                    {
                        for(int k_=0; k_<this->convs[0].get_W(); k_++)
                            cout << img_train[i_*this->convs[0].get_H()*this->convs[0].get_W() + j_*this->convs[0].get_W() + k_] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                //cin >> k1;

                cout << "Input de verdad " << endl;
                mostrar_train_img(img);

                //cin >> k1;
                
                cout << "Conv A 0 " << endl;
                for(int i_=0; i_<this->convs[0].get_n_kernels(); i_++)
                {
                    for(int j_=0; j_<this->convs[0].get_H_out(); j_++)
                    {
                        for(int k_=0; k_<this->convs[0].get_W_out(); k_++)
                            cout << img_conv_a[i_*this->convs[0].get_H_out()*this->convs[0].get_W_out() + j_*this->convs[0].get_W_out() + k_] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                //cin >> k1;

                cout << "Output 0 " << endl;
                for(int i_=0; i_<this->convs[0].get_n_kernels(); i_++)
                {
                    for(int j_=0; j_<this->convs[0].get_H_out(); j_++)
                    {
                        for(int k_=0; k_<this->convs[0].get_W_out(); k_++)
                            cout << img_conv_out[i_*this->convs[0].get_H_out()*this->convs[0].get_W_out() + j_*this->convs[0].get_W_out() + k_] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                //cin >> k1;
                */
                
                img_plms_out = plms_outs + img*tam_out_pools + i_plm_out[0];
                img_plms_in_copy = plms_in_copys + img*tam_in_pools + i_plm_in[0];
                this->plms[0].forwardPropagationGPU(img_conv_out, img_plms_out, img_plms_in_copy);
                
                /*
                cout << "Input MaxPool 0" << endl;
                for(int i_=0; i_<this->convs[0].get_n_kernels(); i_++)
                {
                    for(int j_=0; j_<this->convs[0].get_H_out(); j_++)
                    {
                        for(int k_=0; k_<this->convs[0].get_W_out(); k_++)
                            cout << img_conv_out[i_*this->convs[0].get_H_out()*this->convs[0].get_W_out() + j_*this->convs[0].get_W_out() + k_] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                //cin >> k1;


                cout << "PLM_in_copys 0" << endl;
                for(int i_=0; i_<this->plms[0].get_C(); i_++)
                {
                    for(int j_=0; j_<this->plms[0].get_H(); j_++)
                    {
                        for(int k_=0; k_<this->plms[0].get_W(); k_++)
                            cout << img_plms_in_copy[i_*this->plms[0].get_H()*this->plms[0].get_W() + j_*this->plms[0].get_W() + k_] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                //cin >> k1;

                
                cout << "Output MaxPool 0" << endl;
                for(int i_=0; i_<this->plms[0].get_C(); i_++)
                {
                    for(int j_=0; j_<this->plms[0].get_H_out(); j_++)
                    {
                        for(int k_=0; k_<this->plms[0].get_W_out(); k_++)
                            cout << img_plms_out[i_*this->plms[0].get_H_out()*this->plms[0].get_W_out() + j_*this->plms[0].get_W_out() + k_] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                //cin >> k1;
                */
                
                
                // Resto de capas convolucionales y maxpool ----------------------------
                for(int j=1; j<this->n_capas_conv; ++j)
                {
                    
                    // Capa convolucional 
                    img_plms_out = plms_outs + img*tam_out_pools + i_plm_out[j-1];
                    img_conv_out = convs_outs + img*tam_out_convs + i_conv_out[j];
                    img_conv_a = conv_a + img*tam_out_convs + i_conv_out[j];
                    this->convs[j].forwardPropagationGEMM(img_plms_out, img_conv_out, img_conv_a);

                    /*
                    cout << "Input Conv: " << j << ", " << this->plms[j-1].get_C() << "x" << this->plms[j-1].get_H_out() << "x" << this->plms[j-1].get_W_out() << endl;
                    for(int i_=0; i_<this->plms[j-1].get_C(); i_++)
                    {
                        for(int j_=0; j_<this->plms[j-1].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->plms[j-1].get_W_out(); k_++)
                                cout << img_plms_out[i_*this->plms[j-1].get_H_out()*this->plms[j-1].get_W_out() + j_*this->plms[j-1].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    //cin >> k1;

                    cout << "Conv A: " << j << ", " << this->convs[j].get_n_kernels() << "x" << this->convs[j].get_H_out() << "x" << this->convs[j].get_W_out() << endl;
                    for(int i_=0; i_<this->convs[j].get_n_kernels(); i_++)
                    {
                        for(int j_=0; j_<this->convs[j].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->convs[j].get_W_out(); k_++)
                                cout << img_conv_a[i_*this->convs[j].get_H_out()*this->convs[j].get_W_out() + j_*this->convs[j].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    //cin >> k1;

                    cout << "Output Conv: " << j << ", " << this->convs[j].get_n_kernels() << "x" << this->convs[j].get_H_out() << "x" << this->convs[j].get_W_out() << endl;
                    for(int i_=0; i_<this->convs[j].get_n_kernels(); i_++)
                    {
                        for(int j_=0; j_<this->convs[j].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->convs[j].get_W_out(); k_++)
                                cout << img_conv_out[i_*this->convs[j].get_H_out()*this->convs[j].get_W_out() + j_*this->convs[j].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    //cin >> k1;
                    */
                    
                    
                    // Capa MaxPool 
                    img_plms_out = plms_outs + img*tam_out_pools + i_plm_out[j];
                    img_plms_in_copy = plms_in_copys + img*tam_in_pools + i_plm_in[j];
                    this->plms[j].forwardPropagationGPU(img_conv_out, img_plms_out, img_plms_in_copy);
                    
                    /*
                    cout << "Input MaxPool" << j << endl;
                    for(int i_=0; i_<this->convs[j].get_n_kernels(); i_++)
                    {
                        for(int j_=0; j_<this->convs[j].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->convs[j].get_W_out(); k_++)
                                cout << img_conv_out[i_*this->convs[j].get_H_out()*this->convs[j].get_W_out() + j_*this->convs[j].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    //cin >> k1;

                    cout << "PLM_in_copys " << j << endl;
                    for(int i_=0; i_<this->plms[j].get_C(); i_++)
                    {
                        for(int j_=0; j_<this->plms[j].get_H(); j_++)
                        {
                            for(int k_=0; k_<this->plms[j].get_W(); k_++)
                                cout << img_plms_in_copy[i_*this->plms[j].get_H()*this->plms[j].get_W() + j_*this->plms[j].get_W() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    //cin >> k1;

                    cout << "Output MaxPool " << j << endl;
                    for(int i_=0; i_<this->plms[j].get_C(); i_++)
                    {
                        for(int j_=0; j_<this->plms[j].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->plms[j].get_W_out(); k_++)
                                cout << img_plms_out[i_*this->plms[j].get_H_out()*this->plms[j].get_W_out() + j_*this->plms[j].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    //cin >> k1;
                    */
                }  

                
                // Copiar salida de último MaxPool en flatten
                img_flat_out = flat_outs_batch + img*tam_flat_out; 
                C = this->plms[this->n_capas_conv-1].get_C();
                H_out = this->plms[this->n_capas_conv-1].get_H_out();
                W_out = this->plms[this->n_capas_conv-1].get_W_out();
                //img_plms_out = plms_outs + img*tam_out_pools + i_plm_out[this->n_capas_conv-1];

                
                for(int i_=0; i_<C; i_++)    
                    for(int j_=0; j_<H_out; j_++)    
                        for(int k_=0; k_<W_out; k_++)
                            img_flat_out[i_*H_out*W_out + j_*W_out + k_] = img_plms_out[i_*H_out*W_out + j_*W_out + k_];
                
                /*
                cout << "Input Flatten (img: " << img << ") " << endl;
                for(int i_=0; i_<C; i_++)
                {
                    for(int j_=0; j_<H_out; j_++)
                    {
                        for(int k_=0; k_<W_out; k_++)
                            cout << img_flat_out[i_*H_out*W_out + j_*W_out + k_] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                //cin >> k1;


                cout << "Outputput Flatten (img: " << img << ") " << endl;
                for(int i_=0; i_<tam_flat_out; i_++)
                {
                    cout << img_flat_out[i_] << " ";
                }
                cout << endl;
                cout << endl;

                //cin >> k1;
                */
            }

            /*
            cout << "Output Flatten Total: " << endl;
            for(int i_=0; i_<tam_batches[i]; i_++)
            {
                for(int j_=0; j_<tam_flat_out; j_++)
                    cout << flat_outs_batch[i_*tam_flat_out + j_] << " ";
                cout << endl;
            }
            cout << endl;
            cout << endl;

           //cin >> k1;
           */
           

            
            
            
            // ---------------------------------------------------------------------------------------------------------------------------
            // Capa totalmente conectada
            // ---------------------------------------------------------------------------------------------------------------------------

            // Inicializar gradientes de pesos
            for(int i_=0; i_<this->fully->get_n_pesos(); i_++)
                grads_pesos_fully[i_] = 0.0;
            
            // Inicializar gradientes de sesgos
            for(int i_=0; i_<this->fully->get_n_neuronas(); i_++)
                grads_bias_fully[i_] = 0.0;
            
            // Realizar propagación hacia delante y hacia detrás en la capa totalmente conectada
            this->fully->train_ptr(flat_outs_batch, this->train_labels, batch, tam_batches[i], grads_pesos_fully, grads_bias_fully, grad_x_fully, a_ptr, z_ptr, grad_a_ptr);
            
            /*
            cout << "Output Flatten Total: " << endl;
            for(int i_=0; i_<tam_flat_out*tam_batches[i]; i_++)
            {
                cout << flat_outs_batch[i_] << " ";
            }
            cout << endl;
            cout << endl;

            //cin >> k1;
            */
            int n_clases = this->fully->get_capas()[ this->fully->get_n_capas() - 1];
            
            /*
            cout << "Train labels: " << endl;
            for(int i_=0; i_<n_imagenes; i_++)
            {
                for(int j_=0; j_<n_clases; j_++)
                    cout << train_labels[i_*n_clases + j_] << " ";
                cout << endl;
            }
            cout << endl;
            //cin >> k1;

            cout << "Batch: " << endl;
            for(int i_=0; i_<tam_batches[i]; i_++)
                cout << batch[i_] << " ";
            cout << endl;
            cout << endl;  
            //cin >> k1;
            
            cout << "Nº imágenes batch: " << tam_batches[i] << endl;

            cout << "Pesos fully: " << endl;
            */
            float *w_fully = this->fully->get_pesos_ptr();

            int n_capas_fully = this->fully->get_n_capas(),
                * capas_fully = this->fully->get_capas(),
                *i_w_ptr = this->fully->get_i_w_ptr(),
                *i_capa_fully = this->fully->get_i_capa();

            /*
            // Mostrar pesos
            cout << "Pesos fully" << endl;
            for(int i=0; i<n_capas_fully-1; i++)
            {
                cout << "(" << capas_fully[i] << "x" << capas_fully[i+1] << ") " << endl;
                
                cout << "Capa " << i << endl;
                for(int j=0; j<capas_fully[i]; j++)   // Por cada neurona de la capa actual
                {
                    for(int k=0; k<capas_fully[i+1]; k++)     // Por cada neurona de la siguiente capa
                        cout << w_fully[i_w_ptr[i] + j*capas_fully[i+1] + k] << " ";
                    
                    cout << endl;
                }
                cout << endl;
                
            }
            cout << endl;

           //cin >> k1;
            
            cout << "Grad pesos" << endl;
            for(int i=0; i<n_capas_fully-1; i++)
            {
                cout << "(" << capas_fully[i] << "x" << capas_fully[i+1] << ") " << endl;
                
                cout << "Capa " << i << endl;
                for(int j=0; j<capas_fully[i]; j++)   // Por cada neurona de la capa actual
                {
                    for(int k=0; k<capas_fully[i+1]; k++)     // Por cada neurona de la siguiente capa
                        cout << grads_pesos_fully[i_w_ptr[i] + j*capas_fully[i+1] + k] << " ";
                    
                    cout << endl;
                }
                cout << endl;
                
            }
            cout << endl;

           //cin >> k1;

            /*
            // Mostrar neuronas
            cout << "Grad Bias" << endl;
            for(int i=0; i<n_capas_fully; i++)
            {
                cout << "Capa " << i << endl;
                for(int j=0; j<capas_fully[i]; j++)
                    cout << grads_bias_fully[i_capa_fully[i] + j] << " ";
                cout << endl;
                cout << endl;
            }
            cout << endl;

            //cin >> k1;

            
            cout << "Grad X Fully" << endl;
            cout << tam_batches[i] <<  "x" << this->fully->get_capas()[0] << endl << endl;
            for(int i_=0; i_<tam_batches[i]; i_++)
            {
                cout << "Capa " << i_ << endl;
                for(int j=0; j<this->fully->get_capas()[0]; j++)
                    cout << grad_x_fully[i_*this->fully->get_capas()[0] + j] << " ";
                cout << endl;
            }
            cout << endl;

            //cin >> k1;
            */
            
            // ----------------------------------------------
            // Pesos de la capa totalmente conectada
            // ----------------------------------------------
            // ----------------------------------------------
            // Realizar la media de los gradientes respecto a cada peso
            for(int i_=0; i_<this->fully->get_n_pesos(); i_++)
                grads_pesos_fully[i_] /= tam_batches[i];
            
            /*
            cout << "Grad pesos (después media): " << endl;
            for(int i=0; i<n_capas_fully-1; i++)
            {
                cout << "(" << capas_fully[i] << "x" << capas_fully[i+1] << ") " << endl;
                
                cout << "Capa " << i << endl;
                for(int j=0; j<capas_fully[i]; j++)   // Por cada neurona de la capa actual
                {
                    for(int k=0; k<capas_fully[i+1]; k++)     // Por cada neurona de la siguiente capa
                        cout << grads_pesos_fully[i_w_ptr[i] + j*capas_fully[i+1] + k] << " ";
                    
                    cout << endl;
                }
                cout << endl;
                
            }
            cout << endl;

           //cin >> k1;
           */

            // ----------------------------------------------
            // Bias o Sesgos de la capa totalmente conectada
            // ----------------------------------------------
            // ----------------------------------------------
            // Realizar la media de los gradientes respecto a cada sesgo
            for(int i_=0; i_<this->fully->get_n_neuronas(); i_++)
                grads_bias_fully[i_] /= tam_batches[i];

            /*
            cout << "Grad Bias (después media)" << endl;
            for(int i=0; i<n_capas_fully; i++)
            {
                cout << "Capa " << i << endl;
                for(int j=0; j<capas_fully[i]; j++)
                    cout << grads_bias_fully[i_capa_fully[i] + j] << " ";
                cout << endl;
                cout << endl;
            }
            cout << endl;

           //cin >> k1;
           */


                        
            // Actualizar parámetros de capas totalmente conectadas 
            this->fully->actualizar_parametros_ptr(grads_pesos_fully, grads_bias_fully);
            this->fully->escalar_pesos_ptr(2);
            
            w_fully = this->fully->get_pesos_ptr();
            
            /*
            cout << "Pesos fully (después actualización y escalado):" << endl;
            for(int i=0; i<n_capas_fully-1; i++)
            {
                cout << "(" << capas_fully[i] << "x" << capas_fully[i+1] << ") " << endl;
                
                cout << "Capa " << i << endl;
                for(int j=0; j<capas_fully[i]; j++)   // Por cada neurona de la capa actual
                {
                    for(int k=0; k<capas_fully[i+1]; k++)     // Por cada neurona de la siguiente capa
                        cout << w_fully[i_w_ptr[i] + j*capas_fully[i+1] + k] << " ";
                    
                    cout << endl;
                }
                cout << endl;
                
            }
            cout << endl;

            //cin >> k1;
            

            float *bias_fully = this->fully->get_bias_ptr();
            /*
            cout << "Bias (después actualización y escalado):" << endl;
            for(int i=0; i<n_capas_fully; i++)
            {
                cout << "Capa " << i << endl;
                for(int j=0; j<capas_fully[i]; j++)
                    cout << bias_fully[i_capa_fully[i] + j] << " ";
                cout << endl;
                cout << endl;
            }
            cout << endl;

            //cin >> k1;
            */
            
            
            // ---------------------------------------------------------------------------------------------------------------------------
            // Capas convolucionales, de agrupación y aplanado
            // ---------------------------------------------------------------------------------------------------------------------------

            // ----------------------------------------------
            // ----------------------------------------------
            // BackPropagation ------------------------------
            // ----------------------------------------------
            // ----------------------------------------------

            //cout << " ----------- BACKPROP ----------- " << endl;
            // Inicializar gradientes a 0
            for(int i_=0; i_<tam_kernels_conv; i_++)
                conv_grads_w[i_] = 0.0;
            
            for(int i_=0; i_<n_bias_conv; i_++)
                conv_grads_bias[i_] = 0.0;
            
            // Cálculo de gradientes respecto a cada parámetro 
            for(int img=0; img<tam_batches[i]; ++img)
            {
                
                // Realizar una copia de la imagen "img" de entrenamiento
                //img_in = this->train_imgs + tam_ini*batch[img];
                for(int i_=0; i_<C_ini; i_++)
                    for(int j_=0; j_<H_ini; j_++)
                        for(int k_=0; k_<W_ini; k_++)
                            this->img_in[i_*H_ini*W_ini + j_*W_ini + k_] = this->train_imgs[i_*H_ini*W_ini + j_*W_ini + k_ + tam_ini*batch[img]]; 
                
                // Última capa, su output no tiene padding
                int i_c=this->n_capas_conv-1;
                
                // Usar grad_x_fully[img] en vez de plms_outs[img][i_c] en la última capa MaxPool
                img_grad_x_fully = grad_x_fully + img*this->fully->get_capas()[0];
                

                // -------------------------------------------------------------------------------------
                // -------------------------------------------------------------------------------------
                
                C = this->plms[this->n_capas_conv-1].get_C();
                H_out = this->plms[this->n_capas_conv-1].get_H_out();
                W_out = this->plms[this->n_capas_conv-1].get_W_out();
                /*
                cout << "Input Flatten (img: " << img << ") " << endl;
                for(int i_=0; i_<C; i_++)
                {
                    for(int j_=0; j_<H_out; j_++)
                    {
                        for(int k_=0; k_<W_out; k_++)
                            cout << img_grad_x_fully[i_*H_out*W_out + j_*W_out + k_] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                //cin >> k1;
                */

                // -------------------------------------------------------------------------------------
                // -------------------------------------------------------------------------------------
                                
                // Capa MaxPool 
                img_conv_out = convs_outs + img*tam_out_convs + i_conv_out[i_c];
                img_plms_in_copy = plms_in_copys + img*tam_in_pools + i_plm_in[i_c];
                this->plms[i_c].backPropagationGPU(img_conv_out, img_grad_x_fully, img_plms_in_copy);

                /*
                cout << "Input MaxPool b: " << i_c << endl;
                for(int i_=0; i_<this->convs[i_c].get_n_kernels(); i_++)
                {
                    for(int j_=0; j_<this->convs[i_c].get_H_out(); j_++)
                    {
                        for(int k_=0; k_<this->convs[i_c].get_W_out(); k_++)
                            cout << img_conv_out[i_*this->convs[i_c].get_H_out()*this->convs[i_c].get_W_out() + j_*this->convs[i_c].get_W_out() + k_] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                //cin >> k1;

                cout << "PLM_in_copys b: " << i_c << endl;
                for(int i_=0; i_<this->plms[i_c].get_C(); i_++)
                {
                    for(int j_=0; j_<this->plms[i_c].get_H(); j_++)
                    {
                        for(int k_=0; k_<this->plms[i_c].get_W(); k_++)
                            cout << img_plms_in_copy[i_*this->plms[i_c].get_H()*this->plms[i_c].get_W() + j_*this->plms[i_c].get_W() + k_] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                //cin >> k1;
                */


                
                // Capa convolucional 
                img_plms_out = plms_outs + img*tam_out_pools + i_plm_out[i_c-1];
                img_conv_out = convs_outs + img*tam_out_convs + i_conv_out[i_c];
                img_conv_a = conv_a + img*tam_out_convs + i_conv_out[i_c];
                img_grad_w_conv = conv_grads_w + i_w[i_c];
                img_grad_b_conv = conv_grads_bias + i_b[i_c];

                
                if(this->n_capas_conv > 1)
                    this->convs[i_c].backPropagationGEMM(img_plms_out, img_conv_out, img_conv_a, img_grad_w_conv, img_grad_b_conv);
                else
                    this->convs[i_c].backPropagationGEMM(img_in, img_conv_out, img_conv_a, img_grad_w_conv, img_grad_b_conv);
                
                /*
                cout << "Output MaxPool " << i_c-1 << endl;
                for(int i_=0; i_<this->plms[i_c-1].get_C(); i_++)
                {
                    for(int j_=0; j_<this->plms[i_c-1].get_H_out(); j_++)
                    {
                        for(int k_=0; k_<this->plms[i_c-1].get_W_out(); k_++)
                            cout << img_plms_out[i_*this->plms[i_c-1].get_H_out()*this->plms[i_c-1].get_W_out() + j_*this->plms[i_c-1].get_W_out() + k_] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                //cin >> k1;

                cout << "Conv A: " << i_c << ", " << this->convs[i_c].get_n_kernels() << "x" << this->convs[i_c].get_H_out() << "x" << this->convs[i_c].get_W_out() << endl;
                for(int i_=0; i_<this->convs[i_c].get_n_kernels(); i_++)
                {
                    for(int j_=0; j_<this->convs[i_c].get_H_out(); j_++)
                    {
                        for(int k_=0; k_<this->convs[i_c].get_W_out(); k_++)
                            cout << img_conv_a[i_*this->convs[i_c].get_H_out()*this->convs[i_c].get_W_out() + j_*this->convs[i_c].get_W_out() + k_] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                //cin >> k1;

                cout << "Conv Out b: " << i_c << endl;
                for(int i_=0; i_<this->convs[i_c].get_n_kernels(); i_++)
                {
                    for(int j_=0; j_<this->convs[i_c].get_H_out(); j_++)
                    {
                        for(int k_=0; k_<this->convs[i_c].get_W_out(); k_++)
                            cout << img_conv_out[i_*this->convs[i_c].get_H_out()*this->convs[i_c].get_W_out() + j_*this->convs[i_c].get_W_out() + k_] << " ";
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

                //cin >> k1;
                */
                //cout << "Grad Pesos Conv b: " << i_c << endl;
                int n_kernels = this->convs[i_c].get_n_kernels(), K = this->convs[i_c].get_kernel_fils();
                C = this->convs[i_c].get_C();
                /*
                for(int i=0; i<n_kernels; ++i)
                {
                    for(int j=0; j<C; ++j)
                    {
                        for(int k=0; k<K; ++k)
                        {
                            for(int p=0; p<K; ++p)
                                cout << img_grad_w_conv[i*C*K*K + j*K*K + k*K + p] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;

               //cin >> k1;


                cout << "Grad Bias : " << i_c << endl;
                for(int i=0; i<this->convs[i_c].get_n_kernels(); ++i)
                {
                    cout << img_grad_b_conv[i] << " ";
                }
                cout << endl;
                

               //cin >> k1;
               */

                
                for(int j=this->n_capas_conv-2; j>=1; j--)
                {
                    // Capa MaxPool 
                    img_conv_out = convs_outs + img*tam_out_convs + i_conv_out[j];
                    img_plms_out = plms_outs + img*tam_out_pools + i_plm_out[j];
                    img_plms_in_copy = plms_in_copys + img*tam_in_pools + i_plm_in[j];
                    this->plms[j].backPropagationGPU(img_conv_out, img_plms_out, img_plms_in_copy);
                    
                    /*
                    cout << "Output MaxPool " << j << endl;
                    for(int i_=0; i_<this->plms[j].get_C(); i_++)
                    {
                        for(int j_=0; j_<this->plms[j].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->plms[j].get_W_out(); k_++)
                                cout << img_plms_out[i_*this->plms[j].get_H_out()*this->plms[j].get_W_out() + j_*this->plms[j].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    cout << "PLM_in_copys b: " << j << endl;
                    for(int i_=0; i_<this->plms[j].get_C(); i_++)
                    {
                        for(int j_=0; j_<this->plms[j].get_H(); j_++)
                        {
                            for(int k_=0; k_<this->plms[j].get_W(); k_++)
                                cout << img_plms_in_copy[i_*this->plms[j].get_H()*this->plms[j].get_W() + j_*this->plms[j].get_W() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    cout << "Conv Out b: " << j << endl;
                    for(int i_=0; i_<this->convs[j].get_n_kernels(); i_++)
                    {
                        for(int j_=0; j_<this->convs[j].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->convs[j].get_W_out(); k_++)
                                cout << img_conv_out[i_*this->convs[j].get_H_out()*this->convs[j].get_W_out() + j_*this->convs[j].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    //cin >> k1;
                    */

                    // Capa convolucional 
                    img_plms_out = plms_outs + img*tam_out_pools + i_plm_out[j-1];
                    img_conv_a = conv_a + img*tam_out_convs + i_conv_out[j];
                    img_grad_w_conv = conv_grads_w + i_w[j];
                    img_grad_b_conv = conv_grads_bias + i_b[j];
                    this->convs[j].backPropagationGEMM(img_plms_out, img_conv_out, img_conv_a, img_grad_w_conv, img_grad_b_conv);
                    
                    /*
                    cout << "Conv A: " << j << ", " << this->convs[j].get_n_kernels() << "x" << this->convs[j].get_H_out() << "x" << this->convs[j].get_W_out() << endl;
                    for(int i_=0; i_<this->convs[j].get_n_kernels(); i_++)
                    {
                        for(int j_=0; j_<this->convs[j].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->convs[j].get_W_out(); k_++)
                                cout << img_conv_a[i_*this->convs[j].get_H_out()*this->convs[j].get_W_out() + j_*this->convs[j].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    cout << "Conv Out b: " << j << endl;
                    for(int i_=0; i_<this->convs[j].get_n_kernels(); i_++)
                    {
                        for(int j_=0; j_<this->convs[j].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->convs[j].get_W_out(); k_++)
                                cout << img_conv_out[i_*this->convs[j].get_H_out()*this->convs[j].get_W_out() + j_*this->convs[j].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;


                    cout << "Output MaxPool " << j-1 << endl;
                    for(int i_=0; i_<this->plms[j-1].get_C(); i_++)
                    {
                        for(int j_=0; j_<this->plms[j-1].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->plms[j-1].get_W_out(); k_++)
                                cout << img_plms_out[i_*this->plms[j-1].get_H_out()*this->plms[j-1].get_W_out() + j_*this->plms[j-1].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;


                    cout << "Grad Pesos Conv b: " << j << endl;
                    n_kernels = this->convs[j].get_n_kernels(), K = this->convs[j].get_kernel_fils();
                    C = this->convs[j].get_C();
                    for(int i=0; i<n_kernels; ++i)
                    {
                        for(int j=0; j<C; ++j)
                        {
                            for(int k=0; k<K; ++k)
                            {
                                for(int p=0; p<K; ++p)
                                    cout << img_grad_w_conv[i*C*K*K + j*K*K + k*K + p] << " ";
                                cout << endl;
                            }
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    cout << "Grad Bias : " << j << endl;
                    for(int i=0; i<this->convs[j].get_n_kernels(); ++i)
                    {
                        cout << img_grad_b_conv[i] << " ";
                    }
                    cout << endl;

                   //cin >> k1;
                   */

                }
                
                
                if(this->n_capas_conv >1)
                {
                    img_conv_out = convs_outs + img*tam_out_convs + i_conv_out[0];
                    img_plms_out = plms_outs + img*tam_out_pools + i_plm_out[0];
                    img_plms_in_copy = plms_in_copys + img*tam_in_pools + i_plm_in[0];
                    this->plms[0].backPropagationGPU(img_conv_out, img_plms_out, img_plms_in_copy);

                    /*
                    cout << "Output MaxPool " << 0 << endl;
                    for(int i_=0; i_<this->plms[0].get_C(); i_++)
                    {
                        for(int j_=0; j_<this->plms[0].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->plms[0].get_W_out(); k_++)
                                cout << img_plms_out[i_*this->plms[0].get_H_out()*this->plms[0].get_W_out() + j_*this->plms[0].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    cout << "PLM_in_copys b: " << 0 << endl;
                    for(int i_=0; i_<this->plms[0].get_C(); i_++)
                    {
                        for(int j_=0; j_<this->plms[0].get_H(); j_++)
                        {
                            for(int k_=0; k_<this->plms[0].get_W(); k_++)
                                cout << img_plms_in_copy[i_*this->plms[0].get_H()*this->plms[0].get_W() + j_*this->plms[0].get_W() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    cout << "Conv Out b: " << 0 << endl;
                    for(int i_=0; i_<this->convs[0].get_n_kernels(); i_++)
                    {
                        for(int j_=0; j_<this->convs[0].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->convs[0].get_W_out(); k_++)
                                cout << img_conv_out[i_*this->convs[0].get_H_out()*this->convs[0].get_W_out() + j_*this->convs[0].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    //cin >> k1;
                    */

                    img_conv_a = conv_a + img*tam_out_convs + i_conv_out[0];
                    img_grad_w_conv = conv_grads_w + i_w[0];
                    img_grad_b_conv = conv_grads_bias + i_b[0];
                    this->convs[0].backPropagationGEMM(img_in, img_conv_out, img_conv_a, img_grad_w_conv, img_grad_b_conv);
                    


                    /*
                    cout << "Conv A: " << 0 << ", " << this->convs[0].get_n_kernels() << "x" << this->convs[0].get_H_out() << "x" << this->convs[0].get_W_out() << endl;
                    for(int i_=0; i_<this->convs[0].get_n_kernels(); i_++)
                    {
                        for(int j_=0; j_<this->convs[0].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->convs[0].get_W_out(); k_++)
                                cout << img_conv_a[i_*this->convs[0].get_H_out()*this->convs[0].get_W_out() + j_*this->convs[0].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    cout << "Conv Out b: " << 0 << endl;
                    for(int i_=0; i_<this->convs[0].get_n_kernels(); i_++)
                    {
                        for(int j_=0; j_<this->convs[0].get_H_out(); j_++)
                        {
                            for(int k_=0; k_<this->convs[0].get_W_out(); k_++)
                                cout << img_conv_out[i_*this->convs[0].get_H_out()*this->convs[0].get_W_out() + j_*this->convs[0].get_W_out() + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;


                    cout << "Output MaxPool " << 0 << endl;
                    for(int i_=0; i_<C_ini; i_++)
                    {
                        for(int j_=0; j_<H_ini; j_++)
                        {
                            for(int k_=0; k_<W_ini; k_++)
                                cout << this->img_in[i_*H_ini*W_ini + j_*W_ini + k_] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;
                    */


                    //cout << "Grad Pesos Conv b: " << 0 << endl;
                    n_kernels = this->convs[0].get_n_kernels(), K = this->convs[0].get_kernel_fils();
                    C = this->convs[0].get_C();
                    /*
                    for(int i=0; i<n_kernels; ++i)
                    {
                        for(int j=0; j<C; ++j)
                        {
                            for(int k=0; k<K; ++k)
                            {
                                for(int p=0; p<K; ++p)
                                    cout << img_grad_w_conv[i*C*K*K + j*K*K + k*K + p] << " ";
                                cout << endl;
                            }
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;

                    cout << "Grad Bias : " << 0 << endl;
                    for(int i=0; i<this->convs[0].get_n_kernels(); ++i)
                    {
                        cout << img_grad_b_conv[i] << " ";
                    }
                    cout << endl;

                   //cin >> k1;
                    */
                }
                
            }
            
            
            // ----------------------------------------------
            // Pesos de las capas convolucionales
            // ----------------------------------------------
            // ----------------------------------------------
            // Realizar la media de los gradientes respecto a cada parámetro
            for(int i_=0; i_<tam_kernels_conv; i_++)
                conv_grads_w[i_] /= tam_batches[i];
            
            // ----------------------------------------------
            // Bias o Sesgos de las capas convolucionales
            // ----------------------------------------------
            // ----------------------------------------------
            // Realizar la media
            for(int i_=0; i_<n_bias_conv; i_++)
                conv_grads_bias[i_] /= tam_batches[i];
            
            /*
            cout << "Grad Pesos Conv b: " << 0 << endl;
            for(int r=0; r<n_capas_conv; r++)
            {
                int n_kernels = this->convs[r].get_n_kernels(), K = this->convs[r].get_kernel_fils();
                C = this->convs[r].get_C();
                for(int i=0; i<n_kernels; ++i)
                {
                    for(int j=0; j<C; ++j)
                    {
                        for(int k=0; k<K; ++k)
                        {
                            for(int p=0; p<K; ++p)
                                cout << conv_grads_w[i_w[r] + i*C*K*K + j*K*K + k*K + p] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;
            }
            cout << endl;
            
            /*
            cout << "Grad Bias : " << endl;
            for(int r=0; r<n_capas_conv; r++)
            {
                for(int i=0; i<this->convs[r].get_n_kernels(); ++i)
                {
                    cout << conv_grads_bias[i_b[r] + i] << " ";
                }
                cout << endl;
            }
            cout << endl;
            */

           //cin >> k1;
            


            // Actualizar parámetros --------------------------------------------------------------------
            /*
            cout << "Pesos Conv b: " << endl;
            for(int r=0; r<n_capas_conv; r++)
            {
                float *w_ptr = this->convs[r].get_pesos_ptr();
                int n_kernels = this->convs[r].get_n_kernels(), K = this->convs[r].get_kernel_fils();
                C = this->convs[r].get_C();
                for(int i=0; i<n_kernels; ++i)
                {
                    for(int j=0; j<C; ++j)
                    {
                        for(int k=0; k<K; ++k)
                        {
                            for(int p=0; p<K; ++p)
                                cout << w_ptr[i*C*K*K + j*K*K + k*K + p] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;
            }
            cout << endl;
            */

            
            // Actualizar parámetros de capas convolucionales 
            for(int j=0; j<this->n_capas_conv; ++j)
            {
                img_grad_w_conv = conv_grads_w + i_w[j];
                img_grad_b_conv = conv_grads_bias + i_b[j];
                this->convs[j].actualizar_grads_ptr(img_grad_w_conv, img_grad_b_conv);
            }
            
            // Actualizar parámetros de capas convolucionales 
            for(int j=0; j<this->n_capas_conv; ++j)
                this->convs[j].escalar_pesos_ptr(2);
            
            /*
            cout << "Pesos Conv b: " << endl;
            for(int r=0; r<n_capas_conv; r++)
            {
                float *w_ptr = this->convs[r].get_pesos_ptr();
                int n_kernels = this->convs[r].get_n_kernels(), K = this->convs[r].get_kernel_fils();
                C = this->convs[r].get_C();
                for(int i=0; i<n_kernels; ++i)
                {
                    for(int j=0; j<C; ++j)
                    {
                        for(int k=0; k<K; ++k)
                        {
                            for(int p=0; p<K; ++p)
                                cout << w_ptr[i*C*K*K + j*K*K + k*K + p] << " ";
                            cout << endl;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;
            }
            cout << endl;
            
            cin >> k1;
            /*
            cout << "Bias : " << endl;
            for(int r=0; r<n_capas_conv; r++)
            {
                float *bias_ptr = this->convs[r].get_bias_ptr();
                for(int i=0; i<this->convs[r].get_n_kernels(); ++i)
                {
                    cout << bias_ptr[i] << " ";
                }
                cout << endl;
            }
            cout << endl;

           // cin >> k1;
            */
            

        }
        
        fin = high_resolution_clock::now();
        duration = duration_cast<seconds>(fin - ini);
        
        cout << "Época: " << ep << ",                                           " << duration.count() << " (s)" << endl;
        
        checkCudaErrors(cudaGetLastError());
        evaluar_modelo();
        
        
    }
    //evaluar_modelo_en_test();
   

    // Liberar memoria
    free(grad_x_fully); free(flat_outs_batch); free(conv_grads_bias); free(grads_bias_fully); free(grads_pesos_fully); free(convs_outs); free(plms_outs); free(conv_grads_w);
    free(plms_in_copys); free(conv_a); free(indices); free(batch); free(tam_batches); free(y_batch);
}

void CNN::mostrar_ptr(float *x, int C, int H, int W)
{
    cout << "\nX\n";
    for(int j=0; j<C; j++)
    {
        for(int k=0; k<H; k++)
        {
            for(int p=0; p<W; p++)
                cout << x[j*H*W + k*W + p] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
}


void CNN::prueba()
{
    int C_in = this->convs[0].get_C(), H_in = this->convs[0].get_H(), W_in = this->convs[0].get_W(),
        C_out = this->convs[0].get_n_kernels(), H_out = this->convs[0].get_H_out(), W_out = this->convs[0].get_W_out();
    float * input = (float *)malloc(C_in*H_in*W_in * sizeof(float)),
          * output = (float *)malloc(C_out*H_out*W_out * sizeof(float)),
          * output_a = (float *)malloc(C_out*H_out*W_out * sizeof(float));
    
    Convolutional conv(this->convs[0].get_n_kernels(), this->convs[0].get_kernel_fils(), this->convs[0].get_kernel_cols(), C_in, H_in, W_in, 0.1);
    
    checkCudaErrors(cudaGetLastError());
    cout << "Entro\n";
    this->convs[0].forwardPropagationGEMM(input, output, output_a);
    //conv.forwardPropagationGEMM(input, output, output_a);
    checkCudaErrors(cudaGetLastError());
    cout << "Salgo\n";

    free(input); free(output); free(output_a);
}

/*
    @brief  Evalúa el modelo sobre los datos de entrenamiento. Las medidas de evaluación son Accuracy y Entropía Cruzada
*/
void CNN::evaluar_modelo()
{
    float acc=0.0,entr=0.0;
    int C, H, W, C_ini, H_ini, W_ini, H_out, W_out;

    int tam_flat_out = this->plms[this->n_capas_conv-1].get_C() * this->plms[this->n_capas_conv-1].get_H_out() * this->plms[this->n_capas_conv-1].get_W_out();
    float *img_flat_out = nullptr;

    
    // Realizar la propagación hacia delante
    for(int img=0; img<this->n_imagenes; ++img)
    {
        // Copiar imagen de entrenamiento en img_in
        C_ini = this->convs[0].get_C();
        H_ini = this->convs[0].get_H();
        W_ini = this->convs[0].get_W();
        for(int i=0; i<C_ini; i++)
            for(int j=0; j<H_ini; j++)
                for(int k=0; k<W_ini; k++)
                    img_in[i*H_ini*W_ini + j*W_ini + k] = train_imgs[i*H_ini*W_ini + j*W_ini + k + img*C_ini*H_ini*W_ini];
        
        
        // Capas convolucionales y maxpool ----------------------------
        for(int i=0; i<this->n_capas_conv; ++i)
        {
            // Capa convolucional 
            this->convs[i].forwardPropagationGEMM(this->img_in, this->img_out, this->conv_a);
            
            // Copiar img_out en img_in
            C = this->convs[i].get_n_kernels();
            H = this->convs[i].get_H_out();
            W = this->convs[i].get_W_out();

            for(int i=0; i<C; i++)
                for(int j=0; j<H; j++)
                    for(int k=0; k<W; k++)
                        this->img_in[i*H*W + j*W + k] = this->img_out[i*H*W + j*W + k];
            
            // Capa MaxPool 
            this->plms[i].forwardPropagationGPU(this->img_in, this->img_out, this->img_in_copy);
            
            // Copiar img_out en img_in
            H = this->plms[i].get_H_out();
            W = this->plms[i].get_W_out();
            for(int i=0; i<C; i++)
                for(int j=0; j<H; j++)
                    for(int k=0; k<W; k++)
                        this->img_in[i*H*W + j*W + k] = this->img_out[i*H*W + j*W + k];
            
        }

        // Capa flatten
        img_flat_out = flat_outs + img*tam_flat_out; 

        C = this->plms[this->n_capas_conv-1].get_C();
        H_out = this->plms[this->n_capas_conv-1].get_H_out();
        W_out = this->plms[this->n_capas_conv-1].get_W_out();
        //img_plms_out = plms_outs + img*tam_out_pools + i_plm_out[this->n_capas_conv-1];

        for(int i_=0; i_<C; i_++)    
            for(int j_=0; j_<H_out; j_++)    
                for(int k_=0; k_<W_out; k_++)
                    img_flat_out[i_*H_out*W_out + j_*W_out + k_] = img_out[i_*H_out*W_out + j_*W_out + k_];
        
    }
    
    /*
    cout << "FLAT_OUT : " << endl;
    for(int i=0; i<this->n_imagenes; i++)
    {
        for(int j=0; j<tam_flat_out; j++)
            cout << flat_outs[i*tam_flat_out + j] << " ";
        cout << endl;
    }
    cout << endl;
    int k1;
    //cin >> k1;
    */
    

    // Cada hebra obtiene el accuracy y la entropía cruzada sobre una porción de imágenes
    acc = (*this->fully).accuracy_ptr(flat_outs, this->train_labels, n_imagenes, a_ptr, z_ptr);
    entr = (*this->fully).cross_entropy_ptr(flat_outs, this->train_labels, n_imagenes, a_ptr, z_ptr);

    // Realizar media y obtener valores finales
    //acc = acc / n_imagenes * 100;
    //entr = -entr / n_imagenes;

    cout << "Accuracy: " << acc << " %,  ";

    cout << "Entropía cruzada: " << entr << ",         " << endl << endl;    
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