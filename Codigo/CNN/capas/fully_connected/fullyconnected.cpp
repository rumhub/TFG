#include "fullyconnected.h"
#include <vector>
#include <iostream>
#include "math.h"
#include "random"
#include <stdio.h>
#include <omp.h>

using namespace std;

FullyConnected::FullyConnected(const vector<int> &capas, const float &lr)
{
    vector<float> neuronas_capa, w_1D;
    vector<vector<float>> w_2D;
    
    // Neuronas ------------------------------------------------------------------
    // Por cada capa 
    for(int i=0; i<capas.size(); i++)
    {
        // Por cada neurona de cada capa
        for(int j=0; j<capas[i]; j++)
            neuronas_capa.push_back(0.0);

        this->a.push_back(neuronas_capa);
        neuronas_capa.clear();
    }

    // Pesos -------------------------------------------------------------------
    // Por cada capa
    for(int i=0; i<a.size()-1; i++)
    {
        // Por cada neurona de cada capa
        for(int j=0; j<a[i].size(); j++)
        {
            
            // Por cada neurona de la capa siguiente
            for(int k=0; k<a[i+1].size(); k++)
            {
                // Añadimos un peso. 
                // w[i][j][k] indica el peso que conecta la neurona j de la capa i con la neurona k de la capa i+1
                w_1D.push_back(0.0);
            }

            w_2D.push_back(w_1D);  
            w_1D.clear();          
        }
        this->w.push_back(w_2D);
        w_2D.clear();
    }    

    // Learning Rate
    this->lr = lr;

    // Bias mismas dimensiones que neuronas, 1 bias por neurona
    this->bias = this->a;

    // Inicializar pesos mediante inicialización He
    for(int i=0; i<a.size()-1; i++)
        this->generar_pesos(i);

    // Inicializamos bias con un valor random entre -0.5 y 0.5
    for(int i=0; i<this->bias.size(); i++)
        for(int j=0; j<this->bias[i].size(); j++)
            this->bias[i][j] = 0.0;
};


void FullyConnected::generar_pesos(const int &capa)
{
    // Inicialización He
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0, sqrt(2.0 / this->a[capa].size())); 

    for(int i=0; i<this->a[capa].size(); i++)
        for(int j=0; j<this->a[capa+1].size(); j++)
            this->w[capa][i][j] = distribution(gen);
}

void FullyConnected::mostrarpesos()
{
    
    cout << "Pesos: " << endl;
    for(int i=0; i<this->w.size(); i++)
    {
        cout << "Capa " << i << "------------" << endl;
        for(int j=0; j<this->w[i].size(); j++)
        {
            //cout << "Pesos de neurona " << j << " respecto a neuronas de la capa siguiente: " << endl;
            for(int k=0; k<this->w[i][j].size(); k++)
            {
                cout << this->w[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    
    /*
    cout << "Pesos: " << endl;

    //cout << "Pesos de neurona " << j << " respecto a neuronas de la capa siguiente: " << endl;
    for(int k=0; k<this->w[0][0].size(); k++)
    {
        cout << this->w[0][0][k] << " ";
    }
    cout << endl;
    */

}

void FullyConnected::mostrarbias()
{
    cout << "Bias: " << endl;
    
    for(int i=0; i<this->bias.size(); i++)
        for(int j=0; j<this->bias[i].size(); j++)
            cout << this->bias[i][j] << " ";

    cout << endl;
}

float FullyConnected::relu(const float &x)
{
    float result = 0.0;

    if(x > 0)
        result = x;
    
    return result;
}

float FullyConnected::deriv_relu(const float &x)
{
    float result = 0.0;

    if(x > 0)
        result = 1;
    
    return result;
}


float FullyConnected::sigmoid(const float &x)
{
    return 1/(1+exp(-x));
}

// x --> Input de la red
void FullyConnected::forwardPropagation(const vector<float> &x, vector<vector<float>> &a, vector<vector<float>> &z)
{
    float max, sum = 0.0, epsilon = 0.000000001;

    // Introducimos input -------------------------------------------------------
    for(int i=0; i<x.size(); i++)
    {
        z[0][i] = x[i];
        a[0][i] = x[i];
    }
        
    
    // Forward Propagation ------------------------------------------------------------
    // Por cada capa
    for(int i=0; i<this->a.size()-1; i++)
    {

        // Por cada neurona de la capa siguiente
        for(int k=0; k<this->a[i+1].size(); k++)
        {
            // Reset siguiente capa
            a[i+1][k] = 0.0;

            // w[i][j][k] indica el peso que conecta la neurona j de la capa i con la neurona k de la capa i+1
            for(int j=0; j<this->a[i].size(); j++)
                a[i+1][k] += z[i][j] * this->w[i][j][k];
            
            // Aplicar bias o sesgo
            a[i+1][k] += this->bias[i+1][k];
        }

        // Aplicamos función de activación asociada a la capa actual -------------------------------

        // En capas ocultas se emplea ReLU como función de activación
        if(i < this->a.size() - 2)
        {
            for(int k=0; k<this->a[i+1].size(); k++)
                z[i+1][k] = relu(a[i+1][k]);
            
        }else
        {
            // En la capa output se emplea softmax como función de activación
            sum = 0.0;
            max = a[i+1][0];

            // Normalizar -----------------------------------------------------------------
            // Encontrar el máximo
            for(int k=0; k<this->a[i+1].size(); k++)
                if(max < this->a[i+1][k])
                    max = a[i+1][k];
            
            // Normalizar
            for(int k=0; k<this->a[i+1].size(); k++)
                a[i+1][k] = a[i+1][k] - max;
            
            // Calculamos la suma exponencial de todas las neuronas de la capa output ---------------------
            for(int k=0; k<this->a[i+1].size(); k++)
                sum += exp(a[i+1][k]);

            for(int k=0; k<this->a[i+1].size(); k++)
                z[i+1][k] = exp(a[i+1][k]) / sum;
            
        } 
    }
}




// Ahora x es el conjunto de datos de training
// x[0] el primer dato training
// x[0][0] el primer elemento del primer dato training
float FullyConnected::cross_entropy(vector<vector<float>> x, vector<vector<float>> y)
{
    float sum = 0.0, prediccion = 0.0, epsilon = 0.000000001;
    int n=this->a.size()-1;

    vector<vector<float>> a, z;
    a = this->a;
    z = this->a;

    sum = 0.0;

    #pragma omp parallel for reduction(+:sum) firstprivate(a, z)
    for(int i=0; i<x.size(); i++)
    {
        forwardPropagation(x[i], a, z);

        for(int c=0; c<this->a[n].size(); c++)
            if(y[i][c] == 1)
                prediccion = z[n][c];
            
        sum += log(prediccion+epsilon);
    }

    sum = -sum / x.size();

    return sum;
}

float FullyConnected::accuracy(vector<vector<float>> x, vector<vector<float>> y)
{
    float sum =0.0, max;
    int prediccion, n=this->a.size()-1;

    vector<vector<float>> a, z;
    a = this->a;
    z = this->a;

    #pragma omp parallel for reduction(+:sum) firstprivate(a, z)
    for(int i=0; i<x.size(); i++)
    {
        forwardPropagation(x[i], a, z);

        // Inicialización
        max = z[n][0];
        prediccion = 0;

        // Obtener valor más alto de la capa output
        for(int c=1; c<this->a[n].size(); c++)
        {
            if(max < z[n][c])
            {
                max = z[n][c];
                prediccion = c;
            }
        }

        // Ver si etiqueta real y predicción coindicen
        sum += y[i][prediccion];
    }

    sum = sum / x.size() * 100;

    return sum;
}

void FullyConnected::train(const vector<vector<float>> &x, const vector<vector<float>> &y, const int &n_datos, vector<vector<vector<float>>> &grad_pesos, vector<vector<float>> &grad_b, vector<vector<float>> &grad_x, vector<vector<float>> &a, vector<vector<float>> &z, vector<vector<float>> &grad_a)
{
    float epsilon = 0.000000001;

    int i_output = this->a.size()-1; // índice de la capa output
    float sum, o_in, grad_x_output, sig_o_in;
    int i_last_h = i_output-1;  // Índice de la capa h1
    int i_act, i_ant;


    // Inicializar gradiente respecto a entrada 
    grad_x.clear();

    // Inicializar gradiente de pesos a 0 --------------------------
    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                grad_pesos[i][j][k] = 0.0;

    // Inicializar gradiente bias a 0 ------------------------------
    for(int i=0; i<this->bias.size(); i++)
        for(int j=0; j<this->bias[i].size(); j++)
            grad_b[i][j] = 0.0;

    // Backpropagation ----------------------------------------------
    // Hay 2 o más capas ocultas
    for(int i=0; i<n_datos; i++)
    {
        forwardPropagation(x[i], a, z);

        // Inicializar a 0 gradiente respecto a input
        for(int _i = 0; _i < grad_a.size(); _i++)
            for(int j = 0; j < grad_a[_i].size(); j++)
                grad_a[_i][j] = 0.0;


        // Capa SoftMax -----------------------------------------------
        // Se calcula gradiente del error respecto a cada Z_k
        // grad_Zk = O_k - y_k
        for(int k=0; k<this->a[i_output].size(); k++)
            grad_a[i_output][k] = z[i_output][k] - y[i][k];

        // Pesos h_last - Softmax
        for(int p=0; p<this->a[i_last_h].size(); p++)
            for(int k=0; k<this->a[i_output].size(); k++)
                grad_pesos[i_last_h][p][k] += grad_a[i_output][k] * z[i_last_h][p];
                //                                 grad_Zk                  *  z^i_last_h_p

        // Sesgos capa softmax
        for(int k=0; k<this->a[i_output].size(); k++)
            grad_b[i_output][k] += grad_a[i_output][k];
            // bk = grad_Zk

        // Última capa oculta -----------------------------------------------
        for(int p=0; p<this->a[i_last_h].size(); p++)      
            for(int k=0; k<this->a[i_output].size(); k++)
                grad_a[i_last_h][p] += grad_a[i_output][k] * this->w[i_last_h][p][k] * deriv_relu(a[i_last_h][p]);
                //                              grad_Zk           *  w^i_last_h_pk          * ...
                
        // Capas ocultas intermedias
        for(int capa= i_last_h; capa > 1; capa--)
        {
            // Pesos
            for(int i_act = 0; i_act < this->a[capa].size(); i_act++)       // Por cada neurona de la capa actual
                for(int i_ant = 0; i_ant < this->a[capa-1].size(); i_ant++)     // Por cada neurona de la capa anterior
                    grad_pesos[capa-1][i_ant][i_act] += grad_a[capa][i_act] * z[capa-1][i_ant];

            // Bias
            for(int i_act = 0; i_act < this->a[capa].size(); i_act++)
                grad_b[capa][i_act] += grad_a[capa][i_act];
            
            // Grad input
            for(int i_ant = 0; i_ant < this->a[capa-1].size(); i_ant++)     // Por cada neurona de la capa anterior
                for(int i_act = 0; i_act < this->a[capa].size(); i_act++)       // Por cada neurona de la capa actual
                    grad_a[capa-1][i_ant] += grad_a[capa][i_act] * this->w[capa-1][i_ant][i_act] * deriv_relu(a[capa-1][i_ant]);
        }

        // Capa input
        // Pesos
        int capa=1;
        for(int i_act = 0; i_act < this->a[capa].size(); i_act++)       // Por cada neurona de la capa actual
            for(int i_ant = 0; i_ant < this->a[capa-1].size(); i_ant++)     // Por cada neurona de la capa anterior
                grad_pesos[capa-1][i_ant][i_act] += grad_a[capa][i_act] * z[capa-1][i_ant];
        
        // Grad input
        for(int i_ant = 0; i_ant < this->a[capa-1].size(); i_ant++)     // Por cada neurona de la capa anterior
            for(int i_act = 0; i_act < this->a[capa].size(); i_act++)       // Por cada neurona de la capa actual
                grad_a[capa-1][i_ant] += grad_a[capa][i_act] * this->w[capa-1][i_ant][i_act];


        grad_x.push_back(grad_a[0]);
    } 
}

void FullyConnected::escalar_pesos(float clip_value)
{
    // Calcular el máximo y el mínimo de los pesos
    float max = this->w[0][0][0], min = this->w[0][0][0];

    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
            {
                if(max < this->w[i][j][k])
                    max = this->w[i][j][k];
                
                if(min > this->w[i][j][k])
                    min = this->w[i][j][k];
            }
    
    // Realizar gradient clipping
    float scaling_factor = clip_value / std::max(std::abs(max), std::abs(min));
    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                this->w[i][j][k] = std::max(std::min(this->w[i][j][k], clip_value), -clip_value);
}

void FullyConnected::actualizar_parametros(vector<vector<vector<float>>> &grad_pesos, vector<vector<float>> &grad_b, const int &n_imgs_batch)
{

    // Realizar la media
    for(int j=0; j<this->w.size(); j++)
        for(int k=0; k<this->w[j].size(); k++)
            for(int p=0; p<this->w[j][k].size(); p++)
                grad_pesos[j][k][p] = grad_pesos[j][k][p] / n_imgs_batch;

    for(int i=0; i<this->bias.size(); i++)
        for(int j=0; j<this->bias[i].size(); j++)
            grad_b[i][j] = grad_b[i][j] / n_imgs_batch;

    // Actualizar pesos
    for(int j=0; j<this->w.size(); j++)
        for(int k=0; k<this->w[j].size(); k++)
            for(int p=0; p<this->w[j][k].size(); p++)
                this->w[j][k][p] -= this->lr * grad_pesos[j][k][p];

    // Actualizar bias
    for(int i=0; i<this->bias.size(); i++)
        for(int j=0; j<this->bias[i].size(); j++)
            this->bias[i][j] -= this->lr * grad_b[i][j];
}

void FullyConnected::generarDatos(vector<vector<float>> &x, vector<float> &y)
{
    int n_datos = 1000, cont0 = 0, cont1 = 0, n_datos_por_clase = n_datos/2; 
    float x1, x2, y_, aux, sum;
    vector<float> dato_x;
    x.clear();
    y.clear();

    while(cont0 < n_datos_por_clase)
    {
        dato_x.clear();
        sum = 0.0;

        // Generamos datos input
        for(int j=0; j<this->a[0].size(); j++)
        {
            aux = rand() / float(RAND_MAX);
            aux = aux / this->a[0].size();  // Para que esté en clase 0
            dato_x.push_back(aux);

            // Generamos dato output tal que: if x+y >0, result=1, else result=0
            sum += aux;
        }

        x.push_back(dato_x);

        y.push_back(0);
        cont0++;
             
    }


    while(cont1 < n_datos_por_clase)
    {
        dato_x.clear();
        sum = 0.0;

        // Generamos datos input
        for(int j=0; j<this->a[0].size(); j++)
        {
            aux = rand() / float(RAND_MAX);
            aux = aux + 1/this->a[0].size();  // Para que esté en clase 0
            dato_x.push_back(aux);

            // Generamos dato output tal que: if x+y >0, result=1, else result=0
            sum += aux;
        }

        x.push_back(dato_x);

        y.push_back(1);
        cont1++;
             
    }

}

void FullyConnected::setLR(float lr)
{
    this->lr = lr;
}

/*
void FullyConnected::leer_imagenes_mnist(vector<vector<float>> &x, vector<vector<float>> &y, const int n_imagenes, const int n_clases)
{
    vector<vector<vector<float>>> imagen_k1;
    vector<float> v1D, y_1D;
    string ruta_ini, ruta;

    //n_imagenes = 4000;

    x.clear();
    y.clear();

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
            ruta_ini = "../../../fotos/mnist/training/";
            ruta = ruta_ini + to_string(c) + "/" + to_string(p) + ".jpg";

            Mat image2 = imread(ruta), image;

            image = image2;

            // Cargamos la imagen en un vector 3D
            cargar_imagen_en_vector(image, imagen_k1);

            // Normalizar imagen y pasar a 1D (solo queremos 1 canal porque son en blanco y negro)
            v1D.clear();
            for(int j=0; j<imagen_k1[0].size(); j++)
                for(int k=0; k<imagen_k1[0][0].size(); k++)
                {
                    //imagen_k1[0][j][k] = imagen_k1[0][j][k] / 255;
                    v1D.push_back((float) imagen_k1[0][j][k] / 255.0);
                }
            x.push_back(v1D);
            y.push_back(y_1D);
        }

        // Reset todo "y_1D" a 0
        y_1D[c] = 0.0;
    }  
}
*/

void FullyConnected::particion_k_fold(const vector<vector<float>> &x, const vector<vector<float>> &y, const int &k)
{
    int tam_particion = x.size() / k, n = x.size();
    vector<vector<vector<float>>> particiones_train, particiones_test;
    vector<vector<float>> datos_particion;
    vector<float> v_1D(x[0].size());

    for(int i=0; i<tam_particion; i++)
        datos_particion.push_back(v_1D);

    // Por cada partición
    for(int i=0; i<k; i++)
    {
        for(int j=0; j<k; j++)
        {
            for(int k=0; k<tam_particion; k++)
                datos_particion[k] = x[j*tam_particion + k];

            if(i == j)
                particiones_test.push_back(datos_particion);
            else
                particiones_train.push_back(datos_particion);
        }
    }
}


void FullyConnected::leer_atributos(vector<vector<float>> &x, vector<vector<float>> &y, string fichero)
{
    ifstream inFile;
    string dato, s, delimiter = ","; 
    bool data = false;
    int n=0;                // Num de características que tenemos sin contar la clase
    vector<float> nums, y_1D;

    x.clear();
    y.clear();
    inFile.open(fichero);

    if(!inFile)
    {
        cerr << "No se ha podido abrir el fichero " << fichero;
        exit(1);
    }

    // Inicializar y_1D
    for(int i=0; i<4; i++)
        y_1D.push_back(0.0);
    
    for(int i=0; i<3; i++)  // Hay 3 clases
    {
        y_1D[i] = 1.0;
        for(int j=0; j<50; j++) // Hay 50 ejemplos de cada clase
        {
            getline(inFile, s);
            for(int i=0; i<4; i++)  // Hay 4 números por ejemplo
            {
                dato = s.substr(0, s.find(delimiter));
                s.erase(0, s.find(delimiter)+ delimiter.length());
                nums.push_back(stof(dato));
                //cout << dato << " ";
            }
            //cout << endl;
            x.push_back(nums);
            y.push_back(y_1D);
            nums.clear();
            s.clear();
        }
        y_1D[i] = 0.0;
    }


    inFile.close();
}

void FullyConnected::copiar_parametros(FullyConnected &fc)
{
    this->w = fc.w;
    this->bias = fc.bias;
}

/*
int main()
{ 
    // Solo se meten capa input y capas ocultas, la capa output siempre tiene 1 neurona
    vector<vector<float>> x, y; 
    vector<int> capas1{784, 10, 10};        // MNIST
    FullyConnected n1(capas1, 0.1);

    //n1.leer_imagenes_mnist(x, y, 5, 10);
    n1.leer_imagenes_mnist(x, y, 3000, 10);

    //x[0] = {1, 0, 0, 1};
    //n1.forwardPropagation(x[0]);
    //n1.mostrarNeuronas();

    //vector<int> capas{(int) x[0].size(), 256, 256, 10}; // MNIST
    //vector<int> capas{4, 4, 10, 3};                  // IRIS
    //vector<int> capas{4, 4, 3};                  // IRIS
    FullyConnected n(capas1, 0.01);
    //n.forwardPropagation(x[0]);
    
    //n.leer_atributos(x, y, "../../../fotos/iris/iris.data");
    
    // SGD
    vector<int> indices(x.size()), tam_batches;
    int n_imgs=x.size(), n_epocas = 100, mini_batch = 32;
    const int M = n_imgs / mini_batch;

    // Inicializar vector de índices
    for(int i=0; i<n_imgs; i++)
        indices[i] = i;

    // Indices --------------------------------------------
    // Inicializar tamaño de mini-batches (int)
    for(int i=0; i<M; i++)
        tam_batches.push_back(mini_batch);
    
    // Último batch puede tener distinto tamaño al resto
    if(n_imgs % mini_batch != 0)
        tam_batches.push_back(n_imgs % mini_batch);    

    vector<vector<vector<vector<float>>>> grad_pesos(THREAD_NUM);
    vector<vector<vector<float>>> grad_b(THREAD_NUM), grad_x(THREAD_NUM);

    FullyConnected *fullys = new FullyConnected[THREAD_NUM];
    for(int i=0; i<THREAD_NUM; i++)
        fullys[i] = n;
    
    // ---------------------------------------------------
    // Por cada trabajador p 
    #pragma omp parallel num_threads(THREAD_NUM)
    {
        int thread_id = omp_get_thread_num();

        vector<int> batch_p(mini_batch);
        vector<vector<float>> batch_xp(mini_batch);
        vector<vector<float>> batch_yp(mini_batch);


        for(int ep=0; ep<n_epocas; ep++)
        {
            // Desordenar vector de índices
            #pragma omp single
            {
                random_shuffle(indices.begin(), indices.end());
            }
            #pragma omp barrier

            // Por cada mini-batch
            for(int i=0; i<tam_batches.size(); i++)
            {
                fullys[thread_id].copiar_parametros(n);
                
                // Cada trabajador obtiene N/P imágenes, N = Nº imágenes por mini-batch 
                int n_imgs_batch = tam_batches[i] / THREAD_NUM, n_imgs_batch_ant = n_imgs_batch; 

                if(n_imgs_batch * THREAD_NUM < tam_batches[i] && thread_id == THREAD_NUM-1)
                    n_imgs_batch = n_imgs_batch + (tam_batches[i] % THREAD_NUM);
                                
                for(int j=0; j<n_imgs_batch; j++)
                    batch_p[j] = indices[mini_batch*i + n_imgs_batch_ant*thread_id + j];   

                // X ---------------------------------------------------
                for(int j=0; j<n_imgs_batch; j++)
                    batch_xp[j] = x[batch_p[j]];

                // Y ---------------------------------------------------
                for(int j=0; j<n_imgs_batch; j++)
                    batch_yp[j] = y[batch_p[j]];              
                
                // Realizar backpropagation y acumular gradientes
                fullys[thread_id].train(batch_xp, batch_yp, n_imgs_batch, grad_pesos[thread_id], grad_b[thread_id], grad_x[thread_id]);

                #pragma omp barrier
                #pragma omp critical
                {
                    // Sumar gradientes
                    if(thread_id != 0)
                    {
                        for(int c=0; c<grad_pesos[0].size(); c++)
                            for(int j=0; j<grad_pesos[0][c].size(); j++)
                                for(int k=0; k<grad_pesos[0][c][j].size(); k++)
                                    grad_pesos[0][c][j][k] += grad_pesos[thread_id][c][j][k];

                        for(int c=0; c<grad_b[0].size(); c++)
                            for(int j=0; j<grad_b[0][c].size(); j++)
                                    grad_b[0][c][j] += grad_b[thread_id][c][j];
                    }
                }

                #pragma omp barrier  
                #pragma omp single
                {
                    // Realizar la media
                    for(int c=0; c<grad_pesos[0].size(); c++)
                        for(int j=0; j<grad_pesos[0][c].size(); j++)
                            for(int k=0; k<grad_pesos[0][c][j].size(); k++)
                                grad_pesos[0][c][j][k] = grad_pesos[0][c][j][k] / tam_batches[i];

                    for(int c=0; c<grad_b[0].size(); c++)
                        for(int j=0; j<grad_b[0][c].size(); j++)
                                grad_b[0][c][j] = grad_b[0][c][j] / tam_batches[i];
                    
                    // Actualizar parámetros
                    n.actualizar_parametros(grad_pesos[0], grad_b[0]);
                }

                #pragma omp barrier  
                
            }

            #pragma omp single
            {
                cout << "Época: " << ep << endl;
                cout << "Entropía cruzada: " << n.cross_entropy(x, y) << endl;
                cout << "Accuracy: " << n.accuracy(x,y) << " %" << endl;
            }
            #pragma omp barrier 
        }
    }

    
    
    
    
    return 0;
}
*/