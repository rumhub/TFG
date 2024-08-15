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
CNN::CNN(const vector<vector<int>> &capas_conv, const vector<vector<int>> &tams_pool, const vector<int> &padding,  vector<int> &capas_fully, const vector<vector<vector<float>>> &input, const float &lr)
{
    
    vector<vector<vector<float>>> img_in, img_out, img_in_copy;
    vector<float> flat_out;
    int H_out, W_out;

    if(capas_conv.size() != padding.size())
    {
        cout << "ERROR Dimensiones de array capa convolucional y array de padding no coincidem.\n";
        exit(-1);
    }

    if(capas_conv[0].size() != 3)
    {
        cout << "ERROR Dimensiones de array capa convolucional incorrectas. Ejemplo de uso, capas_conv[0] = {16, 3, 3}\n";
        exit(-1);
    }

    if(input.size() <= 0)
    {
        cout << "ERROR. Hay que proporcionar un input a la red. \n";
        exit(-1);
    }

    this->n_capas_conv = capas_conv.size();
    this->lr = lr;
    this->convs = new Convolutional[this->n_capas_conv];
    this->plms = new PoolingMax[this->n_capas_conv];
    img_in = input;
    this->padding = padding;

    vector<float> v_1D;
    vector<vector<float>> v_2D;

    Convolutional conv1(capas_conv[0][0], capas_conv[0][1], capas_conv[0][2], img_in, lr);
    
    // Inicializar capas convolucionales y maxpool --------------------------------------------
    for(int i=0; i<n_capas_conv; ++i)
    {   
        if(i == 0) 
            conv1.aplicar_padding(img_in, padding[i]);

        // Crear imagen output ----------------------------------------
        // H_out = fils_img   -       K          + 1;
        H_out = img_in[0].size() - capas_conv[i][1] + 1;
        W_out = img_in[0][0].size() - capas_conv[i][2] + 1;

        v_1D.clear();
        for(int j=0; j<W_out; ++j)
            v_1D.push_back(0.0);
        
        v_2D.clear();
        for(int j=0; j<H_out; ++j)
            v_2D.push_back(v_1D);
        

        img_out.clear();
        for(int j=0; j<capas_conv[i][0]; ++j)
            img_out.push_back(v_2D);
        
        
        this->outputs.push_back(img_out);

        // Capas convolucionales ------------------------------------------------
        //                  nºkernels          filas_kernel      cols_kernel
        Convolutional conv(capas_conv[i][0], capas_conv[i][1], capas_conv[i][2], img_in, lr); 
        this->convs[i] = conv;
        vector<vector<vector<float>>> conv_a = img_out;
        this->convs[i].forwardPropagation(img_in, img_out, conv_a);
        img_in = img_out;
        
        // Crear imagen output ----------------------------------------
        // H_out = fils_img   -       K          + 1;
        H_out = img_in[0].size() / tams_pool[i][0];
        W_out = img_in[0][0].size() / tams_pool[i][1];


        v_1D.clear();
        v_2D.clear();
        img_out.clear();

        for(int j=0; j<W_out; ++j)
            v_1D.push_back(0.0);

        for(int j=0; j<H_out; ++j)
            v_2D.push_back(v_1D);

        for(int j=0; j<capas_conv[i][0]; ++j)
            img_out.push_back(v_2D);
        


        // Capas MaxPool -----------------------------------------------------------
        //           filas_kernel_pool  cols_kernel_pool
        PoolingMax plm(tams_pool[i][0], tams_pool[i][1], img_in);
        img_in_copy = img_in;
        this->plms[i] = plm;

        int pad_sig = 0;    // Padding de la siguiente capa convolucional
        if(this->n_capas_conv > i+1)
            pad_sig = this->padding[i+1];

        conv1.aplicar_padding(img_out, pad_sig);
        this->outputs.push_back(img_out);

        this->plms[i].forwardPropagation(img_in, img_out, img_in_copy, pad_sig);
        img_in = img_out;
    }

    // Inicializar capa flatten -----------------------------------------------------------------
    this->flat = new Flatten(img_out);    
    (*this->flat).forwardPropagation(img_in, flat_out);

    // Conectamos capa flatten con capa totalmente conectada
    capas_fully.insert(capas_fully.begin(),(int) flat_out.size());

    // Inicializar capa fullyconnected -----------------------------------------
    if(capas_fully[0] != (int) flat_out.size())
    {
        cout << "ERROR. Dimensión capa input de la fullyconnected (" << capas_fully[0]  <<") layer no coincide con la dimensión que produce la capa flatten (" << (int) flat_out.size() <<"). \n";
        exit(-1);
    }

    this->fully = new FullyConnected(capas_fully, lr);
}

/*
    @brief  Muestra la arquitectura de la red
*/
void CNN::mostrar_arquitectura()
{
    vector<vector<vector<float>>> img_in, img_out, img_in_copy, conv_a;
    vector<float> flat_out;

    if(this->train_imgs.size() == 0)
    {
        cout << "No se puede mostrar la arquitectura. No hay imágenes de entrenamiento. \n";
        exit(-1);
    }

    img_in = this->train_imgs[0];
    cout << "Dimensiones tras realizar la propagación hacia delante de una imagen" << endl;
    cout << "Imagen inicial, dimensiones: " << img_in.size() << "x" <<  img_in[0].size() << "x" << img_in[0][0].size() << endl;

    // Inicializar capas convolucionales y maxpool --------------------------------------------
    for(int i=0; i<n_capas_conv; ++i)
    {
        // Capas convolucionales ------------------------------------------------
        conv_a = this->outputs[i*2];
        this->convs[i].forwardPropagation(img_in, this->outputs[i*2], conv_a);
        img_in = this->outputs[i*2];
        cout << "Dimensiones tras " << this->convs[i].get_n_kernels() << " capas convolucionales de " << this->convs[i].get_kernel_fils() << "x" << this->convs[i].get_kernel_cols() << ": " << this->outputs[i*2].size() << "x" << this->outputs[i*2][0].size() << "x" << this->outputs[i*2][0][0].size() << endl;

        // Capas MaxPool -----------------------------------------------------------
        img_in_copy = img_in;
        int pad_sig = 0;    // Padding de la siguiente capa convolucional
        if(this->n_capas_conv > i+1)
            pad_sig = this->padding[i+1];

        this->plms[i].forwardPropagation(img_in, this->outputs[i*2+1], img_in_copy, pad_sig);
        cout << "Dimensiones tras una capa MaxPool de " << this->plms[i].get_kernel_fils() << "x" << this->plms[i].get_kernel_cols() << ": " << this->outputs[i*2+1].size() << "x" << this->outputs[i*2+1][0].size() << "x" << this->outputs[i*2+1][0][0].size() << endl;
        img_in = this->outputs[i*2+1];
    }

    (*this->flat).forwardPropagation(img_in, flat_out);
    cout << "Dimensiones después de una capa de flatten: " << flat_out.size() << endl;

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

/*
    @brief: Desordena un vector de datos 1D
    @vec: Vector a desordenar
    @rng: random device
*/
void shuffle(vector<int> vec, mt19937& rng) {
    for (int i = vec.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng);
        std::swap(vec[i], vec[j]);
    }
}

/*
    @brief: Entrenar el modelo
    @epocas: Número de épocas a entrenar
    @mini_batch: Tamaño de mini_batch a emplear
*/
void CNN::train(int epocas, int mini_batch)
{
    auto ini = high_resolution_clock::now();
    auto fin = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(fin - ini);

    int n=this->train_imgs.size();
    vector<vector<vector<vector<vector<float>>>>> convs_outs(mini_batch), plms_outs(mini_batch), conv_grads_w(this->n_capas_conv), plms_in_copys(mini_batch), conv_a(mini_batch);       // Input y output de cada capa (por cada imagen de training)
    vector<vector<vector<vector<float>>>> convs_out(this->n_capas_conv), pools_out(this->n_capas_conv);
    vector<vector<vector<float>>> grads_pesos_fully = (*this->fully).get_pesos(), img_aux;
    vector<vector<float>> grad_x_fully, flat_outs(mini_batch), grads_bias_fully = (*this->fully).get_bias(), fully_a = (*this->fully).get_a(), fully_z = fully_a, fully_grad_a = fully_a, conv_grads_bias(this->n_capas_conv), prueba(this->n_capas_conv), max_conv(this->n_capas_conv), min_conv(this->n_capas_conv); 
    vector<int> indices(n), batch(mini_batch), tam_batches;  
    const int M = n / mini_batch;


    std::random_device rd;
    std::mt19937 g(rd());

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

                
                // ---------------------------------------------------------------------------------------
                for(int img=0; img<tam_batches[i]; ++img)
                {
                    // Primera capa convolucional y maxpool -----------------------
                    // Realizar los cálculos
                    this->convs[0].forwardPropagation(this->train_imgs[batch[img]], convs_outs[img][0], conv_a[img][0]);
                    
                    int pad_sig = 0;    // Padding de la siguiente capa convolucional
                    if(this->n_capas_conv > 1)
                        pad_sig = this->padding[1];

                    padding_interno(plms_outs[img][0], pad_sig);

                    this->plms[0].forwardPropagation(convs_outs[img][0], plms_outs[img][0], plms_in_copys[img][0], pad_sig);

                    
                    // Resto de capas convolucionales y maxpool ----------------------------
                    for(int i=1; i<this->n_capas_conv; ++i)
                    {
                        // Capa convolucional 
                        this->convs[i].forwardPropagation(plms_outs[img][i-1], convs_outs[img][i], conv_a[img][i]);

                        // Capa MaxPool 
                        pad_sig = 0;    // Padding de la siguiente capa convolucional
                        if(this->n_capas_conv > i+1)
                            pad_sig = this->padding[i+1];

                        padding_interno(plms_outs[img][i], pad_sig);
                        this->plms[i].forwardPropagation(convs_outs[img][i], plms_outs[img][i], plms_in_copys[img][i], pad_sig);
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
                (*this->fully).train(flat_outs, this->train_labels, batch, tam_batches[i], grads_pesos_fully, grads_bias_fully, grad_x_fully, fully_a, fully_z, fully_grad_a);

                // ---------------------------------------------------------------------------------------------------------------------------
                // Capas convolucionales, de agrupación y aplanado
                // ---------------------------------------------------------------------------------------------------------------------------

                // ----------------------------------------------
                // ----------------------------------------------
                // BackPropagation ------------------------------
                // ----------------------------------------------
                // ----------------------------------------------

                // Inicializar gradientes a 0
                for(int i=0; i<this->n_capas_conv; ++i)
                    this->convs[i].reset_gradients(conv_grads_w[i], conv_grads_bias[i]);

                // Cálculo de gradientes respecto a cada parámetro 
                for(int img=0; img<tam_batches[i]; ++img)
                {
                    img_aux = this->train_imgs[batch[img]];

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

                    for(int i=this->n_capas_conv-2; i>=1; i--)
                    {
                        // Capa MaxPool 
                        this->plms[i].backPropagation(convs_outs[img][i], plms_outs[img][i], plms_in_copys[img][i], this->padding[i+1]);

                        // Capa convolucional 
                        this->convs[i].backPropagation(plms_outs[img][i-1], convs_outs[img][i], conv_a[img][i], conv_grads_w[i], conv_grads_bias[i], this->padding[i]);
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
                for(int i=0; i<this->n_capas_conv; ++i)
                    this->convs[i].actualizar_grads(conv_grads_w[i], conv_grads_bias[i]);


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
   
    
}


/*
    @brief  Evalúa el modelo sobre los datos de entrenamiento. Las medidas de evaluación son Accuracy y Entropía Cruzada
*/
void CNN::evaluar_modelo()
{
    int n=this->train_imgs.size();
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
        img_in = this->train_imgs[img];

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
    acc = (*this->fully).accuracy(flat_outs,this->train_labels);
    entr = (*this->fully).cross_entropy(flat_outs, this->train_labels);


    // Realizar media y obtener valores finales
    acc = acc / n * 100;
    entr = -entr / n;

    //t2 = omp_get_wtime();

    cout << "Accuracy: " << acc << " %,  ";


    cout << "Entropía cruzada: " << entr << ",         " << endl << endl;
    //cout << "Entropía cruzada: " << entr << ",         " << t2 - t1 << " (s) " << endl << endl;
    
}

/*
    @brief  Evalúa el modelo sobre los datos de test. Las medidas de evaluación son Accuracy y Entropía Cruzada
*/
void CNN::evaluar_modelo_en_test()
{
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
}