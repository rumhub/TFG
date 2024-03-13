#include "fullyconnected.h"
#include <vector>
#include <iostream>
#include "math.h"
#include "random"

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
        {
            neuronas_capa.push_back(0.0);
            //neuronas_capa.push_back((float) rand() / float(RAND_MAX) - 0.5);
        }

        this->a.push_back(neuronas_capa);
        neuronas_capa.clear();
    }

    this->z = this->a;
    this->grad_a = this->a;

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
                w_1D.push_back(this->generar_peso(a[i].size()));
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

    // Inicializamos bias con un valor random entre -0.5 y 0.5
    for(int i=0; i<this->bias.size(); i++)
        for(int j=0; j<this->bias[i].size(); j++)
            this->bias[i][j] = this->generar_peso(a[i].size());

    // Inicializar gradiente de pesos a 0
    this->grad_w = w;
    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                this->grad_w[i][j][k] = 0;

    // Inicializar gradiente de bias a 0
    this->grad_bias = this->bias;
    for(int i=0; i<this->grad_bias.size(); i++)
        for(int j=0; j<this->grad_bias[i].size(); j++)
            this->grad_bias[i][j] = 0.0;
};


float FullyConnected::generar_peso(int neuronas_in)
{
    // Inicialización He
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0, sqrt(2.0 / neuronas_in)); 

    return distribution(gen);
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

void FullyConnected::mostrarNeuronas()
{
    cout << "NEURONAS: " << endl;
    for(int i=0; i<this->z.size(); i++)
    {
        for(int j=0; j<this->z[i].size(); j++)
        {
            cout << this->z[i][j] << " ";
        }
        cout << endl;
    }
}

void FullyConnected::mostrarbias()
{
    cout << "Bias: " << endl;
    
    for(int i=0; i<this->bias.size(); i++)
        for(int j=0; j<this->bias[i].size(); j++)
            cout << this->bias[i][j] << " ";

    cout << endl;
}

float FullyConnected::relu(float x)
{
    float result = 0.0;

    if(x > 0)
        result = x;
    
    return result;
}

float FullyConnected::deriv_relu(float x)
{
    float result = 0.0;

    if(x > 0)
        result = 1;
    
    return result;
}


float FullyConnected::sigmoid(float x)
{
    return 1/(1+exp(-x));
}

// x --> Input de la red
void FullyConnected::forwardPropagation(const vector<float> &x)
{
    float max, sum = 0.0, epsilon = 0.000000001;

    // Introducimos input -------------------------------------------------------
    for(int i=0; i<x.size(); i++)
        this->a[0][i] = x[i];
    
    // Forward Propagation ------------------------------------------------------------
    // Por cada capa
    for(int i=0; i<this->a.size()-1; i++)
    {

        // Por cada neurona de la capa siguiente
        for(int k=0; k<this->a[i+1].size(); k++)
        {
            // Reset siguiente capa
            this->a[i+1][k] = 0.0;

            // w[i][j][k] indica el peso que conecta la neurona j de la capa i con la neurona k de la capa i+1
            for(int j=0; j<this->a[i].size(); j++)
                this->a[i+1][k] += this->a[i][j] * this->w[i][j][k];
            
            // Aplicar bias o sesgo
            this->a[i+1][k] += this->bias[i+1][k];
        }

        // Aplicamos función de activación asociada a la capa actual -------------------------------

        // En capas ocultas (y capa input) se emplea ReLU como función de activación
        if(i < this->a.size() - 2)
        {
            for(int k=0; k<this->a[i+1].size(); k++)
                this->z[i+1][k] = relu(this->a[i+1][k]);
            
        }else
        {
            // En la capa output se emplea softmax como función de activación
            sum = 0.0;
            max = this->a[i+1][0];

            // Normalizar -----------------------------------------------------------------
            // Encontrar el máximo
            for(int k=0; k<this->a[i+1].size(); k++)
                if(max < this->a[i+1][k])
                    max = this->a[i+1][k];
            
            // Normalizar
            for(int k=0; k<this->a[i+1].size(); k++)
                this->a[i+1][k] = this->a[i+1][k] - max;
            
            // Calculamos la suma exponencial de todas las neuronas de la capa output ---------------------
            for(int k=0; k<this->a[i+1].size(); k++)
                sum += exp(this->a[i+1][k]);

            for(int k=0; k<this->a[i+1].size(); k++)
                this->z[i+1][k] = exp(this->a[i+1][k]) / sum;
            
        } 
    }
}


void FullyConnected::mostrar_prediccion(vector<float> x, float y)
{
    forwardPropagation(x);
    cout << "Input: ";
    int n=this->a.size()-1;

    for(int i=0; i<x.size(); i++)
    {
        cout << x[i] << " ";
    }
    cout << "y: " << y << ", predicción: " << this->z[n][0] << endl;
}

void FullyConnected::mostrar_prediccion_vs_verdad(vector<float> x, float y)
{
    int n=this->a.size()-1;
    forwardPropagation(x);

    cout << "y: " << y << ", predicción: " << this->z[n][0] << endl;
}

// Ahora x es el conjunto de datos de training
// x[0] el primer dato training
// x[0][0] el primer elemento del primer dato training
float FullyConnected::cross_entropy(vector<vector<float>> x, vector<vector<float>> y)
{
    float sum = 0.0, prediccion = 0.0, epsilon = 0.000000001;
    int n=this->a.size()-1;

    for(int i=0; i<x.size(); i++)
    {
        forwardPropagation(x[i]);

        for(int c=0; c<this->a[n].size(); c++)
            if(y[i][c] == 1)
                prediccion = this->z[n][c];
            
        sum += log(prediccion+epsilon);
    }

    sum = -sum / x.size();


    return sum;
}

float FullyConnected::accuracy(vector<vector<float>> x, vector<vector<float>> y)
{
    float sum =0.0, max;
    int prediccion, n=this->z.size()-1;

    for(int i=0; i<x.size(); i++)
    {
        forwardPropagation(x[i]);

        // Inicialización
        max = this->z[n][0];
        prediccion = 0;

        // Obtener valor más alto de la capa output
        for(int c=1; c<this->z[n].size(); c++)
        {
            if(max < this->z[n][c])
            {
                max = this->z[n][c];
                prediccion = c;
            }
        }

        // Ver si etiqueta real y predicción coindicen
        if(y[i][prediccion] == 1)
            sum++;
    }

    sum = sum / x.size() * 100;


    return sum;
}

void FullyConnected::train(const vector<vector<float>> &x, const vector<vector<float>> &y, vector<vector<float>> &grad_x)
{
    int n_datos = x.size();
    float epsilon = 0.000000001;

    int i_output = this->a.size()-1; // índice de la capa output
    float sum, o_in, grad_x_output, sig_o_in;
    int i_last_h = i_output-1;  // Índice de la capa h1
    int i_act, i_ant;

    // Ver máximo número de neuronas por capa
    int max=0;

    for(int i=0; i<this->a.size(); i++)
        if(a[i].size() > max)
            max = a[i].size();

    // Inicializar gradiente respecto a entrada a 0 --------------------------
    grad_x.clear();

    // Inicializar gradiente de pesos a 0 --------------------------
    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                this->grad_w[i][j][k] = 0.0;

    // Inicializar gradiente bias a 0 ------------------------------
    for(int i=0; i<this->grad_bias.size(); i++)
        for(int j=0; j<this->grad_bias[i].size(); j++)
            this->grad_bias[i][j] = 0.0;

    // Backpropagation ----------------------------------------------
    // Hay 2 o más capas ocultas
    for(int i=0; i<n_datos; i++)
    {
        forwardPropagation(x[i]);

        // Inicializar a 0 gradiente respecto a input
        for(int _i = 0; _i < this->grad_a.size(); _i++)
            for(int j = 0; j < this->grad_a[_i].size(); j++)
                this->grad_a[_i][j] = 0.0;


        // Capa SoftMax -----------------------------------------------
        // Se calcula gradiente del error respecto a cada Z_k
        // grad_Zk = O_k - y_k
        for(int k=0; k<this->a[i_output].size(); k++)
            this->grad_a[i_output][k] = this->z[i_output][k] - y[i][k];

        // Pesos h_last - Softmax
        for(int p=0; p<this->a[i_last_h].size(); p++)
            for(int k=0; k<this->a[i_output].size(); k++)
                this->grad_w[i_last_h][p][k] += this->grad_a[i_output][k] * this->z[i_last_h][p];
                //                                 grad_Zk                  *  z^i_last_h_p

        // Sesgos
        for(int k=0; k<this->a[i_output].size(); k++)
            this->grad_bias[i_output][k] += this->grad_a[i_output][k];
            // bk = sum(grad_Zk)

        // Última capa oculta -----------------------------------------------
        for(int p=0; p<this->a[i_last_h].size(); p++)      
            for(int k=0; k<this->a[i_output].size(); k++)
                this->grad_a[i_last_h][p] += this->grad_a[i_output][k] * this->w[i_last_h][p][k] * deriv_relu(this->a[i_last_h][p]);
                //this->grad_a[i_last_h][p] += this->grad_a[i_output][k] * this->w[i_last_h][p][k] * sigmoid(this->a[i_last_h][p]) * (1- sigmoid(this->a[i_last_h][p]));
                //                              grad_Zk           *  w^i_last_h_pk          * ...

        // Capas ocultas intermedias
        for(int capa= i_last_h; capa >= 1; capa--)
        {
            // Pesos
            for(int i_act = 0; i_act < this->a[capa].size(); i_act++)       // Por cada neurona de la capa actual
                for(int i_ant = 0; i_ant < this->a[capa-1].size(); i_ant++)     // Por cada neurona de la capa anterior
                    this->grad_w[capa-1][i_ant][i_act] += this->grad_a[capa][i_act] * this->z[capa-1][i_ant];

            // Bias
            for(int i_act = 0; i_act < this->a[capa].size(); i_act++)
                this->grad_bias[capa][i_act] += this->grad_a[capa][i_act];
            
            // Grad input
            for(int i_ant = 0; i_ant < this->a[capa-1].size(); i_ant++)     // Por cada neurona de la capa anterior
                for(int i_act = 0; i_act < this->a[capa].size(); i_act++)       // Por cada neurona de la capa actual
                    this->grad_a[capa-1][i_ant] += this->grad_a[capa][i_act] * this->w[capa-1][i_ant][i_act] * deriv_relu(this->a[capa-1][i_ant]);
        }

        grad_x.push_back(this->grad_a[0]);
    }

    // Realizar medias -----------------------------------------------------------------
    // Realizar la media de los gradientes de pesos
    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                this->grad_w[i][j][k] = this->grad_w[i][j][k] / n_datos;
            
    // Realizar la media de los gradientes de bias
    for(int i=0; i<this->grad_bias.size(); i++)
        for(int j=0; j<this->grad_bias[i].size(); j++)
            this->grad_bias[i][j] = this->grad_bias[i][j] / n_datos;

    // Gradient clipping --------------------------------------------------------------------
    float max_grad = -2, min_grad = 2;

    // Normalizar pesos a rango [-1,1]
    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
            {
                if(max_grad < this->grad_w[i][j][k])
                    max_grad = this->grad_w[i][j][k];

                if(min_grad > this->grad_w[i][j][k])
                    min_grad = this->grad_w[i][j][k];
            }

    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                2 * ((this->grad_w[i][j][k] - min_grad) / (max_grad - min_grad + epsilon)) -1;
    
    // Normalizar bias a rango [-1,1]
    max_grad = -2;
    min_grad = 2;
    for(int i=0; i<this->grad_bias.size(); i++)
        for(int j=0; j<this->grad_bias[i].size(); j++)
        {
            if(max_grad < this->grad_bias[i][j])
                max_grad = this->grad_bias[i][j];
            
            if(min_grad > this->grad_bias[i][j])
                min_grad = this->grad_bias[i][j];
        }

    for(int i=0; i<this->grad_bias.size(); i++)
        for(int j=0; j<this->grad_bias[i].size(); j++)
            if(max_grad < this->grad_bias[i][j])
                2 * ((this->grad_bias[i][j] - min_grad) / (max_grad - min_grad + epsilon)) -1;
                
    // Actualizar parámetros ----------------------------------------------------------
    // Actualizar pesos
    for(int j=0; j<this->w.size(); j++)
        for(int k=0; k<this->w[j].size(); k++)
            for(int p=0; p<this->w[j][k].size(); p++)
                this->w[j][k][p] -= this->lr * this->grad_w[j][k][p];

    // Actualizar bias
    for(int i=0; i<this->grad_bias.size(); i++)
        for(int j=0; j<this->grad_bias[i].size(); j++)
            this->bias[i][j] -= this->lr * this->grad_bias[i][j];
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



int main()
{
    // Solo se meten capa input y capas ocultas, la capa output siempre tiene 1 neurona
    
    //vector<int> capas{4, 4, 2, 2};
    vector<vector<float>> x, grad_x; 
    vector<vector<float>> y;
    //vector<int> capas1{4, 3, 4};
    vector<int> capas1{784, 10, 10};
    FullyConnected n1(capas1, 0.1);

    n1.leer_imagenes_mnist(x, y, 3000, 10);

    //x[0] = {1, 0, 0, 1};
    //n1.forwardPropagation(x[0]);
    //n1.mostrarNeuronas();

    vector<int> capas{(int) x[0].size(), 256, 256, 10};
    FullyConnected n(capas, 0.01);
    n.forwardPropagation(x[0]);
    
    
    int n_epocas = 100000;

    /*
    for(int i=0; i<n_epocas; i++)
    {
        if(i % 10 == 0)
        {
            cout << "Después de entrenar " << i << " épocas -----------------------------------" << endl;
            cout << "Entropía cruzada: " << n.cross_entropy(x, y) << endl;
            cout << "Accuracy: " << n.accuracy(x,y) << " %" << endl;
        }
        
        n.train(x, y, grad_x);
    }
    cout << "Después de entrenar " << n_epocas << " épocas -----------------------------------" << endl;
    cout << "Entropía cruzada: " << n.cross_entropy(x, y) << endl;
    cout << "Accuracy: " << n.accuracy(x,y) << " %" << endl;
    */

    // SGD
    vector<int> indices(x.size());
    vector<int> batch;
    vector<vector<float>> batch_labels, grad_x_fully, x_batch;
    int n_imgs_batch, ini, fin, n_imgs=x.size();

    // Inicializar vector de índices
    for(int i=0; i<n_imgs; i++)
        indices[i] = i;

    int mini_batch = 32;
    for(int ep=0; ep<n_epocas; ep++)
    {
        ini = 0;
        fin = mini_batch;

        // Desordenar vector de índices
        random_shuffle(indices.begin(), indices.end());

        while(fin <=n_imgs)
        {
            //cout << fin << " de " << n_imgs << endl;
            // Crear el batch ----------------------
            batch.clear();
            n_imgs_batch = 0;
            if(fin <= n_imgs)
                for(int j=ini; j<fin; j++)
                    batch.push_back(indices[j]);   
            else
                if(ini < n_imgs)
                    for(int j=ini; j<n_imgs; j++)
                        batch.push_back(indices[j]);
            

            batch_labels.clear();
            n_imgs_batch = batch.size();
            
            // Crear batch de labels
            for(int j=0; j<n_imgs_batch; j++)
                batch_labels.push_back(y[batch[j]]);
            
            // Crear el conjunto de entrenamiento del batch
            x_batch.clear();
            for(int j=0; j<n_imgs_batch; j++)
                x_batch.push_back(x[batch[j]]);

            ini += mini_batch;
            fin += mini_batch;

            n.train(x_batch, batch_labels, grad_x_fully);
        }


        cout << "Época: " << ep << endl;
        cout << "Entropía cruzada: " << n.cross_entropy(x, y) << endl;
        cout << "Accuracy: " << n.accuracy(x,y) << " %" << endl;

    }

    return 0;
}
