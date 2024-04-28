#include "CNN.h"
#include <vector>


using namespace std;

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

    vector<float> v_1D(W_out);
    vector<vector<float>> v_2D;

    Convolutional conv1(capas_conv[0][0], capas_conv[0][1], capas_conv[0][2], img_in, lr);

    // Inicializar capas convolucionales y maxpool --------------------------------------------
    for(int i=0; i<n_capas_conv; i++)
    {   
        if(i == 0) 
            conv1.aplicar_padding(img_in, padding[i]);

        // Crear imagen output ----------------------------------------
        // H_out = fils_img   -       K          + 1;
        H_out = img_in[0].size() - capas_conv[i][1] + 1;
        W_out = img_in[0][0].size() - capas_conv[i][2] + 1;

        v_1D.clear();
        for(int j=0; j<W_out; j++)
        {
            v_1D.push_back(0.0);
        }
        v_2D.clear();
        for(int j=0; j<H_out; j++)
        {
            v_2D.push_back(v_1D);
        }

        img_out.clear();
        for(int j=0; j<capas_conv[i][0]; j++)
        {
            img_out.push_back(v_2D);
        }
        
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

        for(int j=0; j<W_out; j++)
            v_1D.push_back(0.0);

        for(int j=0; j<H_out; j++)
            v_2D.push_back(v_1D);

        for(int j=0; j<capas_conv[i][0]; j++)
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
    @brief      Leer imágenes de la base de datos binaria Gatos vs Perros
    @return     Se modifican this->train_imgs y this->train_labels
*/
void CNN::leer_imagenes()
{
    vector<vector<vector<float>>> imagen_k1;

    //n_imagenes = 4000;
    //int n_imagenes = 1000;
    int n_imagenes = 200;

    this->train_imgs.clear();
    this->train_labels.clear();

    // Leer imágenes
    for(int p=1; p<n_imagenes; p++)
    {
        // Leemos perros
        string ruta_ini = "../fotos/gatos_perros/training_set/dogs/dog.";
        string ruta = ruta_ini + to_string(p) + ".jpg";

        Mat image2 = imread(ruta), image;
        
        resize(image2, image, Size(28, 28));

        // Cargamos la imagen en un vector 3D
        cargar_imagen_en_vector(image, imagen_k1);

        // Normalizar imagen
        for(int i=0; i<imagen_k1.size(); i++)
            for(int j=0; j<imagen_k1[0].size(); j++)
                for(int k=0; k<imagen_k1[0][0].size(); k++)
                    imagen_k1[i][j][k] = imagen_k1[i][j][k] / 255;
                
            
        

        // Aplicamos padding a la imagen de entrada
        this->convs[0].aplicar_padding(imagen_k1, this->padding[0]);

        // Almacenamos las imágenes de entrada de la CNN
        this->train_imgs.push_back(imagen_k1);

        // Establecemos que la imagen tiene una etiqueta, 1 = perro, 0 = gato
        this->train_labels.push_back({0.0, 1.0});         

        
        // Leemos gatos
        ruta_ini = "../fotos/gatos_perros/training_set/cats/cat.";
        ruta = ruta_ini + to_string(p) + ".jpg";

        image2 = imread(ruta), image;
        
        resize(image2, image, Size(28, 28));

        // Cargamos la imagen en un vector 3D
        cargar_imagen_en_vector(image, imagen_k1);

        // Normalizar imagen
        for(int i=0; i<imagen_k1.size(); i++)
            for(int j=0; j<imagen_k1[0].size(); j++)
                for(int k=0; k<imagen_k1[0][0].size(); k++)
                    imagen_k1[i][j][k] = imagen_k1[i][j][k] / 255;
                
            
        

        // Aplicamos padding a la imagen de entrada
        this->convs[0].aplicar_padding(imagen_k1, this->padding[0]);

        // Almacenamos las imágenes de entrada de la CNN
        this->train_imgs.push_back(imagen_k1);

        // Establecemos que la imagen tiene una etiqueta, 1 = perro, 0 = gato
        this->train_labels.push_back({1.0, 0.0});

    }  
}


/*
    @brief      Leer imágenes de la base de datos MNIST
    @return     Se modifican this->train_imgs y this->train_labels
*/
void CNN::leer_imagenes_mnist(const int n_imagenes, const int n_clases)
{
    vector<vector<vector<float>>> imagen_k1;
    vector<float> v1D, y_1D;
    string ruta_ini, ruta;

    // Crear el vector y
    for(int i=0; i<n_clases; i++)
        y_1D.push_back(0.0);

    // Leer n_imagenes de la clase c
    for(int c=0; c<n_clases; c++)
    {
        // Establecer etiqueta one-hot para la clase i
        y_1D[c] = 1.0;

        // Leer imágenes
        for(int p=1; p<n_imagenes; p++)
        {
            ruta_ini = "../fotos/mnist/training/";
            ruta = ruta_ini + to_string(c) + "/" + to_string(p) + ".jpg";

            Mat image2 = imread(ruta), image;

            image = image2;

            // Cargamos la imagen en un vector 3D
            cargar_imagen_en_vector(image, imagen_k1);

            // Normalizar imagen
            for(int i=0; i<imagen_k1.size(); i++)
                for(int j=0; j<imagen_k1[0].size(); j++)
                    for(int k=0; k<imagen_k1[0][0].size(); k++)
                        imagen_k1[i][j][k] = imagen_k1[i][j][k] / 255.0;

            // Aplicamos padding a la imagen de entrada
            this->convs[0].aplicar_padding(imagen_k1, this->padding[0]);

            // Almacenamos las imágenes de entrada de la CNN
            this->train_imgs.push_back(imagen_k1);

            // Establecemos que la imagen tiene una etiqueta, 1 = perro, 0 = gato
            this->train_labels.push_back(y_1D);
        }

        // Reset todo "y_1D" a 0
        y_1D[c] = 0.0;
    }  

    
}


/*
    @brief      Leer imágenes de la base de datos CIFAR10
    @return     Se modifican this->train_imgs y this->train_labels
*/
void CNN::leer_imagenes_cifar10(const int n_imagenes, const int n_clases)
{
    vector<vector<vector<float>>> imagen_k1;
    vector<float> v1D, y_1D;
    string ruta_ini, ruta;

    // Crear el vector y
    for(int i=0; i<n_clases; i++)
        y_1D.push_back(0.0);

    // Leer n_imagenes de la clase c
    for(int c=0; c<n_clases; c++)
    {
        // Establecer etiqueta one-hot para la clase i
        y_1D[c] = 1.0;

        // Leer imágenes
        for(int p=1; p<n_imagenes; p++)
        {
            ruta_ini = "../fotos/cifar10/train/";
            ruta = ruta_ini + to_string(c) + "/" + to_string(p) + ".png";

            Mat image2 = imread(ruta), image;

            image = image2;

            // Cargamos la imagen en un vector 3D
            cargar_imagen_en_vector(image, imagen_k1);

            // Normalizar imagen
            for(int i=0; i<imagen_k1.size(); i++)
                for(int j=0; j<imagen_k1[0].size(); j++)
                    for(int k=0; k<imagen_k1[0][0].size(); k++)
                        imagen_k1[i][j][k] = imagen_k1[i][j][k] / 255.0;

            // Aplicamos padding a la imagen de entrada
            this->convs[0].aplicar_padding(imagen_k1, this->padding[0]);

            // Almacenamos las imágenes de entrada de la CNN
            this->train_imgs.push_back(imagen_k1);

            // Establecemos que la imagen tiene una etiqueta, 1 = perro, 0 = gato
            this->train_labels.push_back(y_1D);
        }

        // Reset todo "y_1D" a 0
        y_1D[c] = 0.0;
    }  

    
}

/*
    @brief  Muestra la arquitectura de la red
*/
void CNN::mostrar_arquitectura()
{
    vector<vector<vector<float>>> img_in, img_out, img_in_copy, conv_a;
    vector<float> flat_out;

    img_in = this->train_imgs[0];
    cout << "Dimensiones tras realizar la propagación hacia delante de una imagen" << endl;
    cout << "Imagen inicial, dimensiones: " << img_in.size() << "x" <<  img_in[0].size() << "x" << img_in[0][0].size() << endl;

    // Inicializar capas convolucionales y maxpool --------------------------------------------
    for(int i=0; i<n_capas_conv; i++)
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
    for(int i=0; i<input.size(); i++)
    {
        // Primeras "pad" filas se igualan a 0.0
        for(int j=0; j<pad; j++)
            for(int k=0; k<input[i].size(); k++)
            input[i][j][k] = 0.0; 

        // Últimas "pad" filas se igualan a 0.0
        for(int j=input[i].size()-1; j>=input[i].size() - pad; j--)
            for(int k=0; k<input[i].size(); k++)
            input[i][j][k] = 0.0; 
        
        // Por cada fila
        for(int k=0; k<input[i].size(); k++)
        {
            // Primeras "pad" casillas se igualan a 0.0
            for(int j=0; j<pad; j++)
                input[i][k][j] = 0.0;

            // Últimas "pad" casillas se igualan a 0.0
            for(int j=input[i][k].size()-1; j>=input[i][k].size() - pad; j--)
                input[i][k][j] = 0.0;
        }
    }    
}


void CNN::train(int epocas, int mini_batch)
{
    double t1, t2;
    int n=this->train_imgs.size();
    int n_thrs = 8;
    vector<vector<vector<vector<vector<vector<float>>>>>> convs_outs_thr(n_thrs), plms_outs_thr(n_thrs), plms_in_copys_thr(n_thrs), conv_a_thr(n_thrs), convs_grads_w(n_thrs);
    vector<vector<vector<vector<vector<float>>>>> convs_outs(mini_batch), plms_outs(mini_batch), conv_grads_w(this->n_capas_conv);       // Input y output de cada capa (por cada imagen de training)
    vector<vector<vector<vector<float>>>> convs_out(this->n_capas_conv), pools_out(this->n_capas_conv), img_aux(n_thrs), grads_pesos_fully(n_thrs);
    vector<vector<vector<float>>> grad_x, flat_outs_thr(n_thrs), grad_x_fully(n_thrs), grad_w = (*this->fully).get_pesos(), grads_bias_fully(n_thrs), fully_a(n_thrs), fully_z(n_thrs), fully_grad_a(n_thrs), convs_grads_bias(n_thrs);
    vector<vector<float>> flat_outs(mini_batch), grad_bias = (*this->fully).get_bias(), conv_grads_bias(this->n_capas_conv), prueba(this->n_capas_conv), max_conv(this->n_capas_conv), min_conv(this->n_capas_conv); 
    vector<vector<int>> batch_thr(n_thrs);
    vector<int> indices(n), batch(mini_batch), tam_batches, n_imgs_batch(n_thrs), n_imgs_batch_ant(n_thrs);  
    vector<float> max_fully(n_thrs), min_fully(n_thrs);
    const int M = n / mini_batch;

    //-------------------------------------------------
    // Inicializar índices
    //-------------------------------------------------
    // Inicializar vector de índices
    for(int i=0; i<n; i++)
        indices[i] = i;

    // Inicializar tamaño de mini-batches
    for(int i=0; i<M; i++)
        tam_batches.push_back(mini_batch);
    
    // Último batch puede tener distinto tamaño al resto
    if(n % mini_batch != 0)
        tam_batches.push_back(n % mini_batch);   

    // Inicializar batch de cada hebra
    for(int i=0; i<n_thrs; i++)
        batch_thr[i] = batch;

    //-------------------------------------------------
    // Reservar espacio 
    //-------------------------------------------------

    // Capas convolucionales ---------------------------
    for(int i=0; i<this->n_capas_conv; i++)
    {
        // Gradientes
        conv_grads_w[i] = this->convs[i].get_pesos();
        conv_grads_bias[i] = this->convs[i].get_bias();

        // Máximos y Mínimos
        max_conv[i] = max_fully;
        min_conv[i] = min_fully;

        convs_out[i] = this->outputs[i*2];
        pools_out[i] = this->outputs[i*2+1];
    }

    for(int i=0; i<mini_batch; i++)
    {
        // Capas convolucionales
        convs_outs[i] = convs_out;

        // Capas de agrupamiento
        plms_outs[i] = pools_out;
    }



    for(int i=0; i<n_thrs; i++)
    {
        // Capas convolucionales
        convs_outs_thr[i] = convs_outs;
        convs_grads_w[i] = conv_grads_w;
        convs_grads_bias[i] = conv_grads_bias;

        // Capas de agrupamiento
        plms_outs_thr[i] = plms_outs;

        // Capa totalmente conectada
        grads_pesos_fully[i] = grad_w;
        grads_bias_fully[i] = grad_bias;
        fully_a[i] = (*this->fully).get_a();
        flat_outs_thr[i] = flat_outs;
    }
    conv_a_thr = convs_outs_thr;
    plms_in_copys_thr = convs_outs_thr;

    // Capa totalmente conectada  
    fully_z = fully_a;
    fully_grad_a = fully_a;

    #pragma omp parallel num_threads(n_thrs)
    for(int ep=0; ep<epocas; ep++)
    {
        int thr_id = omp_get_thread_num();
        int n_imgs, n_imgs_ant;

        #pragma omp master
        {
            t1 = omp_get_wtime();
 
            // Desordenar vector de índices
            random_shuffle(indices.begin(), indices.end());
        }

        #pragma omp barrier

        // ForwardPropagation de cada batch -----------------------------------------------------------------------
        for(int i=0; i<tam_batches.size(); i++)
        {
            // Crear el batch para cada hebra ----------------------
            n_imgs_batch[thr_id] = tam_batches[i] / n_thrs, n_imgs_batch_ant[thr_id] = n_imgs_batch[thr_id]; 
            
            if(n_imgs_batch[thr_id] * n_thrs < tam_batches[i] && thr_id == n_thrs-1)
                n_imgs_batch[thr_id] = n_imgs_batch[thr_id] + (tam_batches[i] % n_thrs);
                            
            for(int j=0; j<n_imgs_batch[thr_id]; j++)
                batch_thr[thr_id][j] = indices[mini_batch*i + n_imgs_batch_ant[thr_id]*thr_id + j];   

            // ---------------------------------------------------------------------------------------
            for(int img=0; img<n_imgs_batch[thr_id]; img++)
            {
                // Primera capa convolucional y maxpool -----------------------
                // Realizar los cálculos
                this->convs[0].forwardPropagation(this->train_imgs[batch_thr[thr_id][img]], convs_outs_thr[thr_id][img][0], conv_a_thr[thr_id][img][0]);

                int pad_sig = 0;    // Padding de la siguiente capa convolucional
                if(this->n_capas_conv > 1)
                    pad_sig = this->padding[1];

                padding_interno(plms_outs_thr[thr_id][img][0], pad_sig);

                this->plms[0].forwardPropagation(convs_outs_thr[thr_id][img][0], plms_outs_thr[thr_id][img][0], plms_in_copys_thr[thr_id][img][0], pad_sig);


                // Resto de capas convolucionales y maxpool ----------------------------
                for(int i=1; i<this->n_capas_conv; i++)
                {
                    // Capa convolucional 
                    this->convs[i].forwardPropagation(plms_outs_thr[thr_id][img][i-1], convs_outs_thr[thr_id][img][i], conv_a_thr[thr_id][img][i]);

                    // Capa MaxPool 
                    pad_sig = 0;    // Padding de la siguiente capa convolucional
                    if(this->n_capas_conv > i+1)
                        pad_sig = this->padding[i+1];

                    padding_interno(plms_outs_thr[thr_id][img][i], pad_sig);
                    this->plms[i].forwardPropagation(convs_outs_thr[thr_id][img][i], plms_outs_thr[thr_id][img][i], plms_in_copys_thr[thr_id][img][i], pad_sig);
                }
                
                (*this->flat).forwardPropagation(plms_outs_thr[thr_id][img][plms_outs_thr[thr_id][img].size()-1], flat_outs_thr[thr_id][img]);                
            }
            

            // ---------------------------------------------------------------------------------------------------------------------------
            // Capa totalmente conectada
            // ---------------------------------------------------------------------------------------------------------------------------

            // Inicializar gradientes de pesos
            for(int j=0; j<grads_pesos_fully[thr_id].size(); j++)
                for(int k=0; k<grads_pesos_fully[thr_id][j].size(); k++)
                    for(int p=0; p<grads_pesos_fully[thr_id][j][k].size(); p++)
                        grads_pesos_fully[thr_id][j][k][p] = 0.0;

            // Inicializar gradientes de sesgos
            for(int j=0; j<grads_bias_fully[thr_id].size(); j++)
                for(int k=0; k<grads_bias_fully[thr_id][j].size(); k++)
                    grads_bias_fully[thr_id][j][k] = 0.0;

            // Relaizar propagación hacia delante y hacia detrás en la capa totalmente conectada
            (*this->fully).train(flat_outs_thr[thr_id], this->train_labels, batch_thr[thr_id], n_imgs_batch[thr_id], grads_pesos_fully[thr_id], grads_bias_fully[thr_id], grad_x_fully[thr_id], fully_a[thr_id], fully_z[thr_id], fully_grad_a[thr_id], n_thrs);

            // ---------------------------------------------------------------------------------------------------------------------------
            // Capas convolucionales, de agrupación y aplanado
            // ---------------------------------------------------------------------------------------------------------------------------

            // ----------------------------------------------
            // ----------------------------------------------
            // BackPropagation ------------------------------
            // ----------------------------------------------
            // ----------------------------------------------

            // Inicializar gradientes a 0
            for(int i=0; i<this->n_capas_conv; i++)
                this->convs[i].reset_gradients(convs_grads_w[thr_id][i], convs_grads_bias[thr_id][i]);

            // Cálculo de gradientes respecto a cada parámetro 
            for(int img=0; img<n_imgs_batch[thr_id]; img++)
            {
                img_aux[thr_id] = this->train_imgs[batch_thr[thr_id][img]];

                // Última capa, su output no tiene padding
                int i_c=this->n_capas_conv-1;
                (*this->flat).backPropagation(plms_outs_thr[thr_id][img][i_c], grad_x_fully[thr_id][img]);

                // Capa MaxPool 
                this->plms[i_c].backPropagation(convs_outs_thr[thr_id][img][i_c], plms_outs_thr[thr_id][img][i_c], plms_in_copys_thr[thr_id][img][i_c], 0);

                // Capa convolucional 
                if(this->n_capas_conv > 1)
                    this->convs[i_c].backPropagation(plms_outs_thr[thr_id][img][i_c-1], convs_outs_thr[thr_id][img][i_c], conv_a_thr[thr_id][img][i_c], convs_grads_w[thr_id][i_c], convs_grads_bias[thr_id][i_c], this->padding[i_c]);
                else
                    this->convs[i_c].backPropagation(img_aux[thr_id], convs_outs_thr[thr_id][img][i_c], conv_a_thr[thr_id][img][i_c], convs_grads_w[thr_id][i_c], convs_grads_bias[thr_id][i_c], this->padding[i_c]);

                for(int i=this->n_capas_conv-2; i>=1; i--)
                {
                    // Capa MaxPool 
                    this->plms[i].backPropagation(convs_outs_thr[thr_id][img][i], plms_outs_thr[thr_id][img][i], plms_in_copys_thr[thr_id][img][i], this->padding[i+1]);

                    // Capa convolucional 
                    this->convs[i].backPropagation(plms_outs_thr[thr_id][img][i-1], convs_outs_thr[thr_id][img][i], conv_a_thr[thr_id][img][i], convs_grads_w[thr_id][i], convs_grads_bias[thr_id][i], this->padding[i]);
                }
                
                if(this->n_capas_conv >1)
                {
                    this->plms[0].backPropagation(convs_outs_thr[thr_id][img][0], plms_outs_thr[thr_id][img][0], plms_in_copys_thr[thr_id][img][0], this->padding[1]);
                    this->convs[0].backPropagation(img_aux[thr_id], convs_outs_thr[thr_id][img][0], conv_a_thr[thr_id][img][0], convs_grads_w[thr_id][0], convs_grads_bias[thr_id][0], this->padding[0]);
                }
                
            }
            
            #pragma omp barrier

            // ----------------------------------------------
            // Pesos de la capa totalmente conectada
            // ----------------------------------------------
            // ----------------------------------------------
            // Sumar gradientes de cada batch
            for(int c=1; c<grads_pesos_fully.size(); c++)
            {
                n_imgs = grads_pesos_fully[c].size() / n_thrs, n_imgs_ant = grads_pesos_fully[c].size() / n_thrs;

                if(thr_id == n_thrs - 1)
                    n_imgs = grads_pesos_fully[c].size() - n_imgs * thr_id;

                for(int j=n_imgs_ant*thr_id; j<n_imgs_ant*thr_id + n_imgs; j++)
                    for(int k=0; k<grads_pesos_fully[c][j].size(); k++)
                        for(int p=0; p<grads_pesos_fully[c][j][k].size(); p++)
                            grads_pesos_fully[0][j][k][p] += grads_pesos_fully[c][j][k][p];       
            }

            // Realizar la media de los gradientes respecto a cada peso
            n_imgs = grads_pesos_fully[0].size() / n_thrs, n_imgs_ant = grads_pesos_fully[0].size() / n_thrs;

            if(thr_id == n_thrs - 1)
                n_imgs = grads_pesos_fully[0].size() - n_imgs * thr_id;

            for(int j=n_imgs_ant*thr_id; j<n_imgs_ant*thr_id + n_imgs; j++)
                for(int k=0; k<grads_pesos_fully[0][j].size(); k++)
                    for(int p=0; p<grads_pesos_fully[0][j][k].size(); p++)
                        grads_pesos_fully[0][j][k][p] /= tam_batches[i];

            // ----------------------------------------------
            // Bias o Sesgos de la capa totalmente conectada
            // ----------------------------------------------
            // ----------------------------------------------
            // Sumar gradientes
            for(int c=1; c<grads_bias_fully.size(); c++)
            {
                n_imgs = grads_bias_fully[c].size() / n_thrs, n_imgs_ant = grads_bias_fully[c].size() / n_thrs;

                if(thr_id == n_thrs - 1)
                    n_imgs = grads_bias_fully[c].size() - n_imgs * thr_id;

                for(int j=n_imgs_ant*thr_id; j<n_imgs_ant*thr_id + n_imgs; j++)
                    for(int k=0; k<grads_bias_fully[c][j].size(); k++)
                        grads_bias_fully[0][j][k] += grads_bias_fully[c][j][k];
            }

            // Realizar la media de los gradientes respecto a cada sesgo
            n_imgs = grads_bias_fully[0].size() / n_thrs, n_imgs_ant = grads_bias_fully[0].size() / n_thrs;

            if(thr_id == n_thrs - 1)
                n_imgs = grads_bias_fully[0].size() - n_imgs * thr_id;

            for(int j=n_imgs_ant*thr_id; j<n_imgs_ant*thr_id + n_imgs; j++)
                for(int k=0; k<grads_bias_fully[0][j].size(); k++)
                        grads_bias_fully[0][j][k] /= tam_batches[i];


            // ----------------------------------------------
            // Pesos de las capas convolucionales
            // ----------------------------------------------
            // ----------------------------------------------
            // Sumar gradientes
            for(int i_1=1; i_1<convs_grads_w.size(); i_1++)
            {
                n_imgs = convs_grads_w[i_1].size() / n_thrs, n_imgs_ant = convs_grads_w[i_1].size() / n_thrs;

                if(thr_id == n_thrs - 1)
                    n_imgs = convs_grads_w[i_1].size() - n_imgs * thr_id;

                for(int i_2=n_imgs_ant*thr_id; i_2<n_imgs_ant*thr_id + n_imgs; i_2++)
                    for(int i_3=0; i_3<convs_grads_w[i_1][i_2].size(); i_3++)
                        for(int i_4=0; i_4<convs_grads_w[i_1][i_2][i_3].size(); i_4++)
                            for(int i_5=0; i_5<convs_grads_w[i_1][i_2][i_3][i_4].size(); i_5++)
                                for(int i_6=0; i_6<convs_grads_w[i_1][i_2][i_3][i_4][i_5].size(); i_6++)
                                    convs_grads_w[0][i_2][i_3][i_4][i_5][i_6] += convs_grads_w[i_1][i_2][i_3][i_4][i_5][i_6];
            }

            // Realizar la media de los gradientes respecto a cada parámetro
            n_imgs = convs_grads_w[0].size() / n_thrs, n_imgs_ant = convs_grads_w[0].size() / n_thrs;

            if(thr_id == n_thrs - 1)
                n_imgs = convs_grads_w[0].size() - n_imgs * thr_id;

            for(int i_2=n_imgs_ant*thr_id; i_2<n_imgs_ant*thr_id + n_imgs; i_2++)
                for(int i_3=0; i_3<convs_grads_w[0][i_2].size(); i_3++)
                    for(int i_4=0; i_4<convs_grads_w[0][i_2][i_3].size(); i_4++)
                        for(int i_5=0; i_5<convs_grads_w[0][i_2][i_3][i_4].size(); i_5++)
                            for(int i_6=0; i_6<convs_grads_w[0][i_2][i_3][i_4][i_5].size(); i_6++)
                                convs_grads_w[0][i_2][i_3][i_4][i_5][i_6] /= tam_batches[i];


            // ----------------------------------------------
            // Bias o Sesgos de las capas convolucionales
            // ----------------------------------------------
            // ----------------------------------------------
            // Sumar gradientes
            for(int c=1; c<convs_grads_bias.size(); c++)
            {
                n_imgs = convs_grads_bias[c].size() / n_thrs, n_imgs_ant = convs_grads_bias[c].size() / n_thrs;

                if(thr_id == n_thrs - 1)
                    n_imgs = convs_grads_bias[c].size() - n_imgs * thr_id;

                for(int j=n_imgs_ant*thr_id; j<n_imgs_ant*thr_id + n_imgs; j++)
                    for(int k=0; k<convs_grads_bias[c][j].size(); k++)
                        convs_grads_bias[0][j][k] += convs_grads_bias[c][j][k];
            }

            // Realizar la media
            n_imgs = convs_grads_bias[0].size() / n_thrs, n_imgs_ant = convs_grads_bias[0].size() / n_thrs;

            if(thr_id == n_thrs - 1)
                n_imgs = convs_grads_bias[0].size() - n_imgs * thr_id;

            for(int j=n_imgs_ant*thr_id; j<n_imgs_ant*thr_id + n_imgs; j++)
                for(int k=0; k<convs_grads_bias[0][j].size(); k++)
                        convs_grads_bias[0][j][k] /= tam_batches[i];

            #pragma omp barrier

            // Actualizar parámetros --------------------------------------------------------------------

            // Actualizar parámetros de capas convolucionales 
            for(int i=0; i<this->n_capas_conv; i++)
                this->convs[i].actualizar_grads(convs_grads_w[0][i], convs_grads_bias[0][i]);


            // Actualizar parámetros de capas totalmente conectadas 
            (*this->fully).actualizar_parametros(grads_pesos_fully, grads_bias_fully);

            #pragma omp barrier
            (*this->fully).escalar_pesos(2, max_fully,min_fully);
            
            // Actualizar parámetros de capas convolucionales 
            for(int j=0; j<this->n_capas_conv; j++)
                this->convs[j].escalar_pesos(2, max_conv[j], min_conv[j]);
            


            #pragma omp barrier
        }

        #pragma omp master
        {
            t2 = omp_get_wtime();
            cout << "Época: " << ep << ",                                           " << t2-t1 << " (s) " << endl;
        }

        evaluar_modelo();  
        
    }
    
}


/*
    @brief  Evalúa el modelo sobre los datos de entrenamiento. Las medidas de evaluación son Accuracy y Entropía Cruzada
*/
void CNN::evaluar_modelo()
{
    int n=this->train_imgs.size(), n_thrs = omp_get_num_threads(), n_imgs = n / n_thrs, n_imgs_ant = n / n_thrs, thr_id = omp_get_thread_num();
    double t1, t2;
    vector<vector<vector<float>>> img_in, img_out, img_in_copy, conv_a;
    
    vector<float> flat_out; 
    float acc ,entr;

    if(thr_id == n_thrs - 1)
        n_imgs = n - n_imgs * thr_id;

    vector<vector<float>> flat_outs(n_imgs);

    // Inicialización de parámetros
    #pragma omp master
    {
        t1 = omp_get_wtime();
        this->sum_acc = 0.0;
        this->sum_entr = 0.0;
    }

    #pragma omp barrier

    // Cada hebra realiza la propagación hacia delante de una porción de imágenes
    for(int img=n_imgs_ant*thr_id, k=0; img<n_imgs_ant*thr_id + n_imgs; img++, k++)
    {
        img_in = this->train_imgs[img];

        // Capas convolucionales y maxpool ----------------------------
        for(int i=0; i<this->n_capas_conv; i++)
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

        flat_outs[k] = flat_out;
    }
    
    // Cada hebra obtiene el accuracy y la entropía cruzada sobre una porción de imágenes
    acc = (*this->fully).accuracy(flat_outs,this->train_labels, n_imgs_ant*thr_id);
    entr = (*this->fully).cross_entropy(flat_outs, this->train_labels, n_imgs_ant*thr_id);

    // Sumar valores de cada hebra
    #pragma omp critical
    {
        this->sum_acc += acc;
        this->sum_entr += entr;
    }

    #pragma omp barrier

    // Realizar media y obtener valores totales
    #pragma omp master
    {
        this->sum_acc = this->sum_acc / n * 100;
        this->sum_entr = -this->sum_entr / n;

        t2 = omp_get_wtime();
    
        cout << "Accuracy: " << this->sum_acc << " %,  ";


        cout << "Entropía cruzada: " << this->sum_entr << ",         " << t2 - t1 << " (s) " << endl << endl;
    }    
}

