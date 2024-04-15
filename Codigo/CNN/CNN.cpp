#include "CNN.h"
#include <vector>


using namespace std;

/*
    Formato:
    capas_conv={{3,2,2}, {2,2,2}} --> {3,2,2} significa 3 kernels 2x2
    tams_pool={{2,2}, {3,3}} --> Una capa MaxPool de 2x2, luego otra de 3x3
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

    Convolutional conv1(capas_conv[0][0], capas_conv[0][1], capas_conv[0][2], img_in, lr);  // CAMBIAR ---------------------------

    // Inicializar capas convolucionales y maxpool --------------------------------------------
    for(int i=0; i<n_capas_conv; i++)
    {    
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
        vector<vector<vector<float>>> conv_a;
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
        
        this->outputs.push_back(img_out);

        // Capas MaxPool -----------------------------------------------------------
        //           filas_kernel_pool  cols_kernel_pool
        PoolingMax plm(tams_pool[i][0], tams_pool[i][1], img_in);
        img_in_copy = img_in;
        this->plms[i] = plm;
        this->plms[i].forwardPropagation(img_in, img_out, img_in_copy);
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

void CNN::leer_imagenes()
{
    vector<vector<vector<float>>> imagen_k1;

    //n_imagenes = 4000;
    int n_imagenes = 1000;

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
        this->convs[i].forwardPropagation(img_in, this->outputs[i*2], conv_a);
        img_in = this->outputs[i*2];
        cout << "Dimensiones tras " << this->convs[i].get_n_kernels() << " capas convolucionales de " << this->convs[i].get_kernel_fils() << "x" << this->convs[i].get_kernel_cols() << ": " << this->outputs[i*2].size() << "x" << this->outputs[i*2][0].size() << "x" << this->outputs[i*2][0][0].size() << endl;

        // Capas MaxPool -----------------------------------------------------------
        img_in_copy = img_in;
        this->plms[i].forwardPropagation(img_in, this->outputs[i*2+1], img_in_copy);
        cout << "Dimensiones tras una capa MaxPool de " << this->plms[i].get_kernel_fils() << "x" << this->plms[i].get_kernel_cols() << ": " << this->outputs[i*2+1].size() << "x" << this->outputs[i*2+1][0].size() << "x" << this->outputs[i*2+1][0][0].size() << endl;
        img_in = this->outputs[i*2+1];
    }

    (*this->flat).forwardPropagation(img_in, flat_out);
    cout << "Dimensiones después de una capa de flatten: " << flat_out.size() << endl;

}

void CNN::train(int epocas, int mini_batch)
{
    double t1, t2;
    int n=this->train_imgs.size(), n_imgs_batch;
    int ini, fin;
    vector<int> indices(n);
    vector<int> batch;
    vector<vector<float>> batch_labels;
    

    // Almacenar inputs y outputs de cada capa
    vector<vector<vector<vector<vector<float>>>>> convs_outs(mini_batch), plms_outs(mini_batch), plms_in_copys(mini_batch), conv_a(mini_batch);       // Input y output de cada capa (por cada imagen de training)
    vector<vector<vector<vector<float>>>> convs_out(this->n_capas_conv);
    vector<float> flat_out; 

    // Paso a la siguiente capa
    vector<vector<vector<float>>> img_aux, grad_x;
    vector<vector<float>> flat_outs(mini_batch), grad_x_fully; 

    // Reservar espacio
    for(int i=0; i<mini_batch; i++)
    {
        convs_outs[i] = convs_out;
        plms_outs[i] = convs_out;
        plms_in_copys[i] = convs_out;
        conv_a[i] = convs_out;
    }

    // Inicializar vector de índices
    for(int i=0; i<n; i++)
        indices[i] = i;

    Aux *aux = new Aux();

    for(int i=0; i<mini_batch; i++)
        conv_a[i] = convs_out;

    // Capa totalmente conectada  ----------------
    vector<vector<vector<float>>> grad_w = (*this->fully).get_pesos();
    vector<vector<float>> grad_bias = (*this->fully).get_bias();
    vector<vector<float>> fully_a, fully_z, fully_grad_a;

    fully_a = (*this->fully).get_a();
    fully_z = (*this->fully).get_a();
    fully_grad_a = (*this->fully).get_a();

    // Capas convolucionales ---------------------------
    vector<vector<vector<vector<vector<float>>>>> conv_grads_w(this->n_capas_conv); 
    vector<vector<float>> conv_grads_bias(this->n_capas_conv);

    for(int i=0; i<this->n_capas_conv; i++)
    {
        conv_grads_w[i] = this->convs[i].get_pesos();
        conv_grads_bias[i] = this->convs[i].get_bias();
    }
    // --------------

    for(int ep=0; ep<epocas; ep++)
    {
        t1 = omp_get_wtime();
        // ForwardPropagation -----------------------------------------------------------------------
        ini = 0;
        fin = mini_batch;

        // Desordenar vector de índices
        random_shuffle(indices.begin(), indices.end());

        while(fin <=n)
        {
            //cout << fin << " de " << n << endl;
            // Crear el batch ----------------------
            batch.clear();
            if(fin <= n)
                for(int j=ini; j<fin; j++)
                    batch.push_back(indices[j]);   
            else
                if(ini < n)
                    for(int j=ini; j<n; j++)
                        batch.push_back(indices[j]);
            

            batch_labels.clear();
            n_imgs_batch = batch.size();
            
            // Crear batch de labels
            for(int j=0; j<n_imgs_batch; j++)
                batch_labels.push_back(this->train_labels[batch[j]]);
            
            
            ini += mini_batch;
            fin += mini_batch;

            // ---------------------------------------------------------------------------------------
            #pragma omp parallel for 
            for(int img=0; img<n_imgs_batch; img++)
            {
                // Primera capa convolucional y maxpool -----------------------
                // Establecer dimensiones de salida
                convs_outs[img][0] = this->outputs[0];
                plms_outs[img][0] = this->outputs[1];

                // Realizar los cálculos
                this->convs[0].forwardPropagation(this->train_imgs[batch[img]], convs_outs[img][0], conv_a[img][0]);
                
                plms_in_copys[img][0] = convs_outs[img][0];
                this->plms[0].forwardPropagation(convs_outs[img][0], plms_outs[img][0], plms_in_copys[img][0]);
                
                // Resto de capas convolucionales y maxpool ----------------------------
                for(int i=1; i<this->n_capas_conv; i++)
                {
                    // Establecer dimensiones de salida
                    convs_outs[img][i] = this->outputs[i*2];
                    plms_outs[img][i] = this->outputs[i*2+1];

                    // Capa convolucional 
                    this->convs[i].aplicar_padding(plms_outs[img][i-1], this->padding[i]);
                    this->convs[i].forwardPropagation(plms_outs[img][i-1], convs_outs[img][i], conv_a[img][i]);

                    // Capa MaxPool 
                    plms_in_copys[img][i] = convs_outs[img][i];
                    this->plms[i].forwardPropagation(convs_outs[img][i], plms_outs[img][i], plms_in_copys[img][i]);
                }
                
                (*this->flat).forwardPropagation(plms_outs[img][plms_outs[img].size()-1], flat_outs[img]);                
            }

            // ---------------------------------------------------------------------------------------

            (*this->fully).train(flat_outs, batch_labels, n_imgs_batch, grad_w, grad_bias, grad_x_fully, fully_a, fully_z, fully_grad_a);

            (*this->fully).actualizar_parametros(grad_w, grad_bias, n_imgs_batch);

            (*this->fully).escalar_pesos(2);

            // BackPropagation -----------------------------------------------------------------------
            for(int i=0; i<this->n_capas_conv; i++)
                this->convs[i].reset_gradients(conv_grads_w[i], conv_grads_bias[i]);

            //#pragma omp parallel for
            for(int img=0; img<n_imgs_batch; img++)
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

            
            // Actualizar pesos de capas convolucionales 
            for(int i=0; i<this->n_capas_conv; i++)
            {
                this->convs[i].actualizar_grads(conv_grads_w[i], conv_grads_bias[i], n_imgs_batch);
                this->convs[i].escalar_pesos(2);
            }
        }

        t2 = omp_get_wtime();
        cout << "Época: " << ep << "                                       , " << t2-t1 << " (s) " << endl;
        accuracy();  
        
        
    }
    
}


// Accuracy sobre training 
void CNN::accuracy()
{
    
    int n=this->train_imgs.size();
    double t_ini, t_fin, t1, t2;
    vector<vector<vector<float>>> img_in, img_out, img_in_copy, conv_a;
    vector<vector<float>> flat_outs(n);
    vector<float> flat_out; 
    float acc ,entr;

    t_ini = omp_get_wtime();

    #pragma omp parallel for private(img_in, img_out, img_in_copy, flat_out, conv_a)
    for(int img=0; img<n; img++)
    {
        img_in = this->train_imgs[img];

        // Capas convolucionales y maxpool ----------------------------
        for(int i=0; i<this->n_capas_conv; i++)
        {
            // Capa convolucional 
            img_out = this->outputs[i*2];
            this->convs[i].forwardPropagation(img_in, img_out, conv_a);
            img_in = img_out;

            // Capa MaxPool 
            img_out = this->outputs[i*2+1];
            img_in_copy = img_in;
            this->plms[i].forwardPropagation(img_in, img_out, img_in_copy);
            img_in = img_out;
        }
        
        (*this->flat).forwardPropagation(img_out, flat_out);

        flat_outs[img] = flat_out;

    }
    t_fin = omp_get_wtime();


    t1 = omp_get_wtime();
    acc = (*this->fully).accuracy(flat_outs,this->train_labels);
    t2 = omp_get_wtime();
    cout << "Accuracy: " << acc << " %,                                         " << t_fin-t_ini + t2-t1 << " (s) " << endl;


    t1 = omp_get_wtime();
    entr = (*this->fully).cross_entropy(flat_outs, this->train_labels);
    t2 = omp_get_wtime();
    cout << "Entropía cruzada: " << entr << ",                                     " << t_fin-t_ini + t2-t1 << " (s) " << endl << endl;
    
}

void CNN::prueba()
{
    vector<vector<vector<float>>> imagen_k1;
    //n_imagenes = 4000;
    int n_imagenes = 100;

    this->train_imgs.clear();
    this->train_labels.clear();
    
    // Crear img de entrada 8x8
    vector<float> v1D;
    vector<vector<float>> v2D;
    for(int k=0; k<4; k++)
        v1D.push_back(1.0);

    for(int j=0; j<4; j++)
        v2D.push_back(v1D);

    imagen_k1.push_back(v2D);


    Aux *aux = new Aux();

    cout << "Img entrada red: " << endl;
    aux->mostrar_imagen(imagen_k1);

    // Aplicamos padding a la imagen de entrada
    this->convs[0].aplicar_padding(imagen_k1, this->padding[0]);

    // Almacenamos las imágenes de entrada de la CNN
    this->train_imgs.push_back(imagen_k1);

    // Establecemos que la imagen tiene una etiqueta, 1 = perro, 0 = gato
    this->train_labels.push_back({1});         
    

}