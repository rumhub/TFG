#include "CNN.h"

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
CNN::CNN(int *capas_conv, int n_capas_conv, int *tams_pool, int *padding, int *capas_fully, int n_capas_fully, int C, int H, int W, const float &lr)
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

    for(int i=0; i<n_capas_conv; i++)
        this->padding[i] = padding[i];

    vector<float> v_1D;
    //vector<float> v_1D(W_out);
    vector<vector<float>> v_2D;

    cout << "Input: " << C << "x" << H << "x" << W << endl;
    // Padding de la primera capa
    H += 2*padding[0];
    W += 2*padding[0];
    Convolutional conv1(capas_conv[0], capas_conv[1], capas_conv[2], C, H, W, lr);
    
    cout << "Capa 0, padding inicial: " << C << "x" << H << "x" << W << endl;

    // Inicializar capas convolucionales y maxpool --------------------------------------------
    for(int i=0; i<n_capas_conv; ++i)
    {   
        i_capas_conv = capas_conv + 3*i;
        i_capas_pool = tams_pool + 2*i;
        // Capas convolucionales ------------------------------------------------
        //                  nºkernels          filas_kernel      cols_kernel
        Convolutional conv(i_capas_conv[0], i_capas_conv[1], i_capas_conv[2], C, H, W, lr);
        this->convs[i] = conv;

        // H_out = H - K + 1
        C = i_capas_conv[0];
        H = H - i_capas_conv[1] + 1;
        W = W - i_capas_conv[2] + 1;

        cout << "Capa " << i << " tras conv: " << C << "x" << H << "x" << W << endl;

        // Capas MaxPool -----------------------------------------------------------
        int pad_sig = 0;    // Padding de la siguiente capa convolucional
        if(this->n_capas_conv > i+1)
            pad_sig = this->padding[i+1];
        //           filas_kernel_pool  cols_kernel_pool
        PoolingMax plm(i_capas_pool[0], i_capas_pool[1], C, H, W, pad_sig);
        this->plms[i] = plm;

        // H_out = H / K + 2*pad
        H = H / i_capas_pool[0] + 2*pad_sig;
        W = W / i_capas_pool[0] + 2*pad_sig;

        //this->outputs.push_back(img_out);
        cout << "Capa " << i << " tras pool: " << C << "x" << H << "x" << W << endl;
    }

    
    // Inicializar capa fullyconnected -----------------------------------------
    int *capas_fully_ptr = (int *)malloc((n_capas_fully+1) * sizeof(int));

    capas_fully_ptr[0] = C*H*W;

    for(int i=1; i<n_capas_fully+1; i++)
        capas_fully_ptr[i] = capas_fully[i-1];

    this->fully = new FullyConnected(capas_fully_ptr, n_capas_fully+1, lr);

    // Mostrar
    cout << "Capa fully: " << endl;
    for(int i=0; i<n_capas_fully+1; i++)
        cout << capas_fully_ptr[i] << endl;

    free(capas_fully_ptr);
}

/*
    @brief  Muestra la arquitectura de la red
*/
void CNN::mostrar_arquitectura()
{
    cout << "\n-----------Arquitectura de la red-----------\n";
    cout << "Padding por capa: ";
    for(int i=0; i<this->n_capas_conv; i++)
        cout << this->padding[i] << " ";
    cout << endl;
    
    for(int i=0; i<this->n_capas_conv; i++)
    {
        cout << "Dimensiones de entrada a " << this->convs[i].get_n_kernels() << " kernels " << this->convs[i].get_kernel_fils() << "x" << this->convs[i].get_kernel_cols() << " convolucionales: " << this->convs[i].get_C() << "x" << this->convs[i].get_H() << "x" << this->convs[i].get_W() << endl;
        cout << "Dimensiones de entrada a un kernel " << this->plms[i].get_kernel_fils() << "x" << this->plms[i].get_kernel_cols() << " MaxPool: " << this->plms[i].get_C() << "x" << this->plms[i].get_H() << "x" << this->plms[i].get_W() << endl;
    }

    // Volúmen de salida de la última capa MaxPool
    cout << "Dimensiones de salida de un kernel " << this->plms[this->n_capas_conv-1].get_kernel_fils() << "x" << this->plms[this->n_capas_conv-1].get_kernel_cols() << " MaxPool: " << this->plms[this->n_capas_conv-1].get_C() << "x" << this->plms[this->n_capas_conv-1].get_H_out() << "x" << this->plms[this->n_capas_conv-1].get_W_out() << endl;
}



void CNN::set_train(const vector<vector<vector<vector<float>>>> &x, const vector<vector<float>> &y)
{
    this->h_train_imgs = x; 
    this->h_train_labels = y;
};


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


void shuffle(vector<int> vec, mt19937& rng) {
    for (int i = vec.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng);
        std::swap(vec[i], vec[j]);
    }
}


void CNN::train(int epocas, int mini_batch)
{
    /*
    auto ini = high_resolution_clock::now();
    auto fin = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(fin - ini);

    int n=this->h_train_imgs.size();
    vector<vector<vector<vector<vector<float>>>>> convs_outs(mini_batch), plms_outs(mini_batch), conv_grads_w(this->n_capas_conv), plms_in_copys(mini_batch), conv_a(mini_batch);       // Input y output de cada capa (por cada imagen de training)
    vector<vector<vector<vector<float>>>> convs_out(this->n_capas_conv), pools_out(this->n_capas_conv);
    vector<vector<vector<float>>> grads_pesos_fully = (*this->fully).get_pesos(), img_aux;
    vector<vector<float>> grad_x_fully, flat_outs(mini_batch), grads_bias_fully = (*this->fully).get_bias(), fully_a = (*this->fully).get_a(), fully_z = fully_a, fully_grad_a = fully_a, conv_grads_bias(this->n_capas_conv), prueba(this->n_capas_conv), max_conv(this->n_capas_conv), min_conv(this->n_capas_conv); 
    vector<int> indices(n), batch(mini_batch), tam_batches;  
    const int M = n / mini_batch;
    int pad_sig;

    std::random_device rd;
    std::mt19937 g(rd());

    /*
    float *d_train_imgs = (float *)malloc(this->h_train_imgs.size() * this->h_train_imgs[0].size() * this->h_train_imgs[0][0].size() * this->h_train_imgs[0][0][0].size() * sizeof(float));

    
    // Copiar a GPU imágenes de entrenamiento
    for(int i=0; i<this->h_train_imgs.size(); ++i)
        for(int j=0; j<this->h_train_imgs[i].size(); ++j)
            for(int k=0; k<this->h_train_imgs[i][j].size(); ++k)
                for(int p=0; p<this->h_train_imgs[i][j][k].size(); ++p)

    */
   /*
    //-------------------------------------------------
    // Inicializar índices
    //-------------------------------------------------
    // Inicializar vector de índices
    for(int i=0; i<n; ++i)
        indices[i] = i;

    // Inicializar tamaño de mini-batches
    for(int i=0; i<M; ++i)
        tam_batches.push_back(mini_batch);
    
    // Último batch puede tener distinto tamaño al resto
    if(n % mini_batch != 0)
        tam_batches.push_back(n % mini_batch);   

    //-------------------------------------------------
    // Reservar espacio 
    //-------------------------------------------------

    // Capas convolucionales ---------------------------
    for(int i=0; i<this->n_capas_conv; ++i)
    {
        // Gradientes
        conv_grads_w[i] = this->convs[i].get_pesos();
        conv_grads_bias[i] = this->convs[i].get_bias();

        convs_out[i] = this->outputs[i*2];
        pools_out[i] = this->outputs[i*2+1];
    }

    for(int i=0; i<mini_batch; ++i)
    {
        // Capas convolucionales
        convs_outs[i] = convs_out;

        // Capas de agrupamiento
        plms_outs[i] = pools_out;
    }
    conv_a = convs_outs;
    plms_in_copys = convs_outs;


        for(int ep=0; ep<epocas; ++ep)
        {
            ini = high_resolution_clock::now();

            // Desordenar vector de índices
            shuffle(indices, g);

            
            // ForwardPropagation de cada batch -----------------------------------------------------------------------
            for(int i=0; i<tam_batches.size(); ++i)
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

                        padding_interno(plms_outs[img][j], pad_sig);                        
                    }


                // ---------------------------------------------------------------------------------------
                for(int img=0; img<tam_batches[i]; ++img)
                {
                    // Primera capa convolucional y maxpool -----------------------
                    // Realizar los cálculos
                    this->convs[0].forwardPropagation(this->h_train_imgs[batch[img]], convs_outs[img][0], conv_a[img][0]);

                    this->plms[0].forwardPropagation(convs_outs[img][0], plms_outs[img][0], plms_in_copys[img][0], pad_sig);

                    
                    // Resto de capas convolucionales y maxpool ----------------------------
                    for(int j=1; j<this->n_capas_conv; ++j)
                    {
                        // Capa convolucional 
                        this->convs[j].forwardPropagation(plms_outs[img][j-1], convs_outs[img][j], conv_a[img][j]);

                        // Capa MaxPool 
                        this->plms[j].forwardPropagation(convs_outs[img][j], plms_outs[img][j], plms_in_copys[img][j], pad_sig);
                    }
                    
                    (*this->flat).forwardPropagation(plms_outs[img][plms_outs[img].size()-1], flat_outs[img]);        
                         
                }
                
                
                
                // ---------------------------------------------------------------------------------------------------------------------------
                // Capa totalmente conectada
                // ---------------------------------------------------------------------------------------------------------------------------

                // Inicializar gradientes de pesos
                for(int j=0; j<grads_pesos_fully.size(); ++j)
                    for(int k=0; k<grads_pesos_fully[j].size(); ++k)
                        for(int p=0; p<grads_pesos_fully[j][k].size(); ++p)
                            grads_pesos_fully[j][k][p] = 0.0;

                // Inicializar gradientes de sesgos
                for(int j=0; j<grads_bias_fully.size(); ++j)
                    for(int k=0; k<grads_bias_fully[j].size(); ++k)
                        grads_bias_fully[j][k] = 0.0;

                // Relaizar propagación hacia delante y hacia detrás en la capa totalmente conectada
                (*this->fully).train(flat_outs, this->h_train_labels, batch, tam_batches[i], grads_pesos_fully, grads_bias_fully, grad_x_fully, fully_a, fully_z, fully_grad_a);

                // ---------------------------------------------------------------------------------------------------------------------------
                // Capas convolucionales, de agrupación y aplanado
                // ---------------------------------------------------------------------------------------------------------------------------

                // ----------------------------------------------
                // ----------------------------------------------
                // BackPropagation ------------------------------
                // ----------------------------------------------
                // ----------------------------------------------

                // Inicializar gradientes a 0
                for(int j=0; j<this->n_capas_conv; ++j)
                    this->convs[j].reset_gradients(conv_grads_w[j], conv_grads_bias[j]);

                // Cálculo de gradientes respecto a cada parámetro 
                for(int img=0; img<tam_batches[i]; ++img)
                {
                    img_aux = this->h_train_imgs[batch[img]];

                    // Última capa, su output no tiene padding
                    int i_c=this->n_capas_conv-1;
                    (*this->flat).backPropagation(plms_outs[img][i_c], grad_x_fully[img]);

                    // Capa MaxPool 
                    this->plms[i_c].backPropagation(convs_outs[img][i_c], plms_outs[img][i_c], plms_in_copys[img][i_c], 0);

                    // Capa convolucional 
                    if(this->n_capas_conv > 1)
                        this->convs[i_c].backPropagation(plms_outs[img][i_c-1], convs_outs[img][i_c], conv_a[img][i_c], conv_grads_w[i_c], conv_grads_bias[i_c], this->padding[i_c]);
                    else
                        this->convs[i_c].backPropagation(img_aux, convs_outs[img][i_c], conv_a[img][i_c], conv_grads_w[i_c], conv_grads_bias[i_c], this->padding[i_c]);

                    for(int j=this->n_capas_conv-2; j>=1; j--)
                    {
                        // Capa MaxPool 
                        this->plms[j].backPropagation(convs_outs[img][j], plms_outs[img][j], plms_in_copys[img][j], this->padding[j+1]);

                        // Capa convolucional 
                        this->convs[j].backPropagation(plms_outs[img][j-1], convs_outs[img][j], conv_a[img][j], conv_grads_w[j], conv_grads_bias[j], this->padding[j]);
                    }
                    
                    if(this->n_capas_conv >1)
                    {
                        this->plms[0].backPropagation(convs_outs[img][0], plms_outs[img][0], plms_in_copys[img][0], this->padding[1]);
                        this->convs[0].backPropagation(img_aux, convs_outs[img][0], conv_a[img][0], conv_grads_w[0], conv_grads_bias[0], this->padding[0]);
                    }
                    
                }
                
                // ----------------------------------------------
                // Pesos de la capa totalmente conectada
                // ----------------------------------------------
                // ----------------------------------------------
                // Realizar la media de los gradientes respecto a cada peso
                for(int j=0; j<grads_pesos_fully.size(); ++j)
                    for(int k=0; k<grads_pesos_fully[j].size(); ++k)
                        for(int p=0; p<grads_pesos_fully[j][k].size(); ++p)
                            grads_pesos_fully[j][k][p] /= tam_batches[i];

                // ----------------------------------------------
                // Bias o Sesgos de la capa totalmente conectada
                // ----------------------------------------------
                // ----------------------------------------------
                // Realizar la media de los gradientes respecto a cada sesgo
                for(int j=0; j<grads_bias_fully.size(); ++j)
                    for(int k=0; k<grads_bias_fully[j].size(); ++k)
                        grads_bias_fully[j][k] /= tam_batches[i];

                // ----------------------------------------------
                // Pesos de las capas convolucionales
                // ----------------------------------------------
                // ----------------------------------------------
                // Realizar la media de los gradientes respecto a cada parámetro
                for(int i_1=1; i_1<conv_grads_w.size(); ++i_1)
                    for(int i_2=0; i_2<conv_grads_w[i_1].size(); ++i_2)
                        for(int i_3=0; i_3<conv_grads_w[i_1][i_2].size(); ++i_3)
                            for(int i_4=0; i_4<conv_grads_w[i_1][i_2][i_3].size(); ++i_4)
                                for(int i_5=0; i_5<conv_grads_w[i_1][i_2][i_3][i_4].size(); ++i_5)
                                        conv_grads_w[i_1][i_2][i_3][i_4][i_5]  /= tam_batches[i];
                

                // ----------------------------------------------
                // Bias o Sesgos de las capas convolucionales
                // ----------------------------------------------
                // ----------------------------------------------
                // Realizar la media
                for(int j=0; j<conv_grads_bias.size(); j++)
                    for(int k=0; k<conv_grads_bias[j].size(); k++)
                        conv_grads_bias[j][k] /= tam_batches[i];


                // Actualizar parámetros --------------------------------------------------------------------

                // Actualizar parámetros de capas convolucionales 
                for(int j=0; j<this->n_capas_conv; ++j)
                    this->convs[j].actualizar_grads(conv_grads_w[j], conv_grads_bias[j]);


                // Actualizar parámetros de capas totalmente conectadas 
                (*this->fully).actualizar_parametros(grads_pesos_fully, grads_bias_fully);

                (*this->fully).escalar_pesos(2);
                
                // Actualizar parámetros de capas convolucionales 
                for(int j=0; j<this->n_capas_conv; ++j)
                    this->convs[j].escalar_pesos(2);
                

            
            }
            
            fin = high_resolution_clock::now();
            duration = duration_cast<seconds>(fin - ini);

            cout << "Época: " << ep << ",                                           " << duration.count() << " (s)" << endl;
            //cout << "Época: " << ep << ",                                           " << t2-t1 << " (s) " << endl;

            evaluar_modelo();
            
            
        }
        evaluar_modelo_en_test();
   
    */
}


/*
    @brief  Evalúa el modelo sobre los datos de entrenamiento. Las medidas de evaluación son Accuracy y Entropía Cruzada
*/
void CNN::evaluar_modelo()
{
    /*
    int n=this->h_train_imgs.size();
    double t1, t2;
    vector<vector<vector<float>>> img_in, img_out, img_in_copy, conv_a;
    
    vector<float> flat_out; 
    float acc ,entr;


    vector<vector<float>> flat_outs(n);

    // Inicialización de parámetros
    //t1 = omp_get_wtime();
    acc = 0.0;
    entr = 0.0;


    // Realizar la propagación hacia delante
    for(int img=0; img<n; ++img)
    {
        img_in = this->h_train_imgs[img];

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
    acc = (*this->fully).accuracy(flat_outs,this->h_train_labels);
    entr = (*this->fully).cross_entropy(flat_outs, this->h_train_labels);


    // Realizar media y obtener valores finales
    acc = acc / n * 100;
    entr = -entr / n;

    //t2 = omp_get_wtime();

    cout << "Accuracy: " << acc << " %,  ";


    cout << "Entropía cruzada: " << entr << ",         " << endl << endl;
    //cout << "Entropía cruzada: " << entr << ",         " << t2 - t1 << " (s) " << endl << endl;
    */
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

int main()
{
    //vector<vector<int>> capas_conv = {{3, 3, 3}, {3, 5, 5}}, tams_pool = {{2, 2}, {2, 2}};
    int C=3, H=32, W=32, n_capas_fully = 2, n_capas_conv = 2;
    int *capas_fully = (int *)malloc(n_capas_fully * sizeof(int)),
        *capas_conv = (int *)malloc(n_capas_conv*3 * sizeof(int)),
        *capas_pool = (int *)malloc(n_capas_conv*2 * sizeof(int)),
        *padding = (int *)malloc(n_capas_conv * sizeof(int));
    float lr = 0.1;
    int i=0;
    capas_fully[0] = 2;
    capas_fully[1] = 3;

    // Primera capa convolucional
    capas_conv[i*3 +0] = 4;      // 4 kernels
    capas_conv[i*3 +1] = 3;      // kernels de 3 filas
    capas_conv[i*3 +2] = 3;      // kernels de 2 columnas

    i = 1;
    // Segunda capa convolucional
    capas_conv[i*3 +0] = 7;      // 7 kernels
    capas_conv[i*3 +1] = 5;      // kernels de 5 filas
    capas_conv[i*3 +2] = 5;      // kernels de 5 columnas

    i=0;
    // Primera capa MaxPool
    capas_pool[i*2 +0] = 2;      // kernels de 2 filas
    capas_pool[i*2 +1] = 2;      // kernels de 2 columnas

    i = 1;
    // Segunda capa MaxPool
    capas_pool[i*2 +0] = 2;      // kernels de 2 filas
    capas_pool[i*2 +1] = 2;      // kernels de 2 columnas
    
    // Padding
    padding[0] = 1;
    padding[1] = 2;

    CNN cnn(capas_conv, n_capas_conv, capas_pool, padding, capas_fully, n_capas_fully, C, H, W, lr);
    cnn.mostrar_arquitectura();

    free(capas_fully); free(capas_conv); free(capas_pool); free(padding);
    return 0;
}