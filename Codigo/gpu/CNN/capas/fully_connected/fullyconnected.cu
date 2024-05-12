#include "fullyconnected.h"
#include <vector>
#include <iostream>
#include "math.h"
#include "random"
#include <stdio.h>
#include <omp.h>

using namespace std;


/*
    CONSTRUCTOR de la clase FullyConnected
    --------------------------------------
  
    @capas  Vector de enteros que indica el número de neuronas por capa. Habrá capas.size() capas y cada una contendrá capas[i] neuronas
    @lr     Learning Rate o Tasa de Aprendizaje
*/
FullyConnected::FullyConnected(const vector<int> &capas, const float &lr)
{
    vector<float> neuronas_capa, w_1D;
    vector<vector<float>> w_2D;
    
    // Neuronas ------------------------------------------------------------------
    // Por cada capa 
    for(int i=0; i<capas.size(); ++i)
    {
        // Por cada neurona de cada capa
        for(int j=0; j<capas[i]; ++j)
            neuronas_capa.push_back(0.0);

        this->a.push_back(neuronas_capa);
        neuronas_capa.clear();
    }

    // Pesos -------------------------------------------------------------------
    // Por cada capa
    for(int i=0; i<a.size()-1; ++i)
    {
        // Por cada neurona de cada capa
        for(int j=0; j<a[i].size(); ++j)
        {
            
            // Por cada neurona de la capa siguiente
            for(int k=0; k<a[i+1].size(); ++k)
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
    for(int i=0; i<a.size()-1; ++i)
        this->generar_pesos(i);

    // Inicializar bias con un valor de 0.0
    for(int i=0; i<this->bias.size(); ++i)
        for(int j=0; j<this->bias[i].size(); ++j)
            this->bias[i][j] = 0.0;
};

/*
    @brief  Genera los pesos entre 2 capas de neuronas
    @capa   Capa a generar los pesos con la siguiente
    @return Se actualizan los valores de w (pesos de la red)
*/
void FullyConnected::generar_pesos(const int &capa)
{
    // Inicialización He
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0, sqrt(2.0 / this->a[capa].size())); 

    for(int i=0; i<this->a[capa].size(); ++i)
        for(int j=0; j<this->a[capa+1].size(); ++j)
            this->w[capa][i][j] = distribution(gen);
}

/*
    @brief      Función de activación ReLU
    @x          Valor sobre el cual aplicar ReLU
    @return     @x tras aplicar ReLU sobre él
*/
float FullyConnected::relu(const float &x)
{
    float result = 0.0;

    if(x > 0)
        result = x;
    
    return result;
}

/*
    @brief      Derivada de la función de activación ReLU
    @x          Valor sobre el cual aplicar la derivada de ReLU
    @return     @x tras aplicar la derivada de ReLU sobre él
*/
float FullyConnected::deriv_relu(const float &x)
{
    float result = 0.0;

    if(x > 0)
        result = 1;
    
    return result;
}

/*
    @brief      Función de activación Sigmoid
    @x          Valor sobre el cual aplicar Sigmoid
    @return     @x tras aplicar Sigmoid sobre él
*/
float FullyConnected::sigmoid(const float &x)
{
    return 1/(1+exp(-x));
}

/*
    @brief  Propagación hacia delante por toda la red totalmente conectada
    @x      Valores de entrada a la red
    @a      Neuronas de la red antes de aplicar la función de activación
    @z      Neuronas de la red después de aplicar la función de activación

    @return Se actualizan los valores de @a y @z
*/
void FullyConnected::forwardPropagation(const vector<float> &x, vector<vector<float>> &a, vector<vector<float>> &z)
{
    float max, sum = 0.0, epsilon = 0.000000001;

    // Introducimos input -------------------------------------------------------
    for(int i=0; i<x.size(); ++i)
    {
        z[0][i] = x[i];
        a[0][i] = x[i];
    }
        
    
    // Forward Propagation ------------------------------------------------------------
    // Por cada capa
    for(int i=0; i<this->a.size()-1; ++i)
    {

        // Por cada neurona de la capa siguiente
        for(int k=0; k<this->a[i+1].size(); ++k)
        {
            // Reset siguiente capa
            a[i+1][k] = 0.0;

            // w[i][j][k] indica el peso que conecta la neurona j de la capa i con la neurona k de la capa i+1
            for(int j=0; j<this->a[i].size(); ++j)
                a[i+1][k] += z[i][j] * this->w[i][j][k];
            
            // Aplicar bias o sesgo
            a[i+1][k] += this->bias[i+1][k];
        }

        // Aplicamos función de activación asociada a la capa actual -------------------------------

        // En capas ocultas se emplea ReLU como función de activación
        if(i < this->a.size() - 2)
        {
            for(int k=0; k<this->a[i+1].size(); ++k)
                z[i+1][k] = relu(a[i+1][k]);
            
        }else
        {
            // En la capa output se emplea softmax como función de activación
            sum = 0.0;
            max = a[i+1][0];

            // Normalizar -----------------------------------------------------------------
            // Encontrar el máximo
            for(int k=0; k<this->a[i+1].size(); ++k)
                if(max < this->a[i+1][k])
                    max = a[i+1][k];
            
            // Normalizar
            for(int k=0; k<this->a[i+1].size(); ++k)
                a[i+1][k] = a[i+1][k] - max;
            
            // Calculamos la suma exponencial de todas las neuronas de la capa output ---------------------
            for(int k=0; k<this->a[i+1].size(); ++k)
                sum += exp(a[i+1][k]);

            for(int k=0; k<this->a[i+1].size(); ++k)
                z[i+1][k] = exp(a[i+1][k]) / sum;
            
        } 
    }
}

/*
    @brief  Realiza la medida de Entropía Cruzada sobre un conjunto de datos
    @x      Conjunto de datos de entrada
    @y      Etiquetas de los datos de entrada
    @ini    Primera posición i en cumplir {y[i] corresponde a x[i], y[i+1] corresponde a x[i+1], ...} 
    @return Valor de entropía cruzada sobre el conjunto de datos de entrada x
*/
float FullyConnected::cross_entropy(vector<vector<float>> x, vector<vector<float>> y, const int &ini)
{
    float sum = 0.0, prediccion = 0.0, epsilon = 0.000000001;
    int n=this->a.size()-1;

    vector<vector<float>> a, z;
    a = this->a;
    z = this->a;

    for(int i=0; i<x.size(); ++i)
    {
        forwardPropagation(x[i], a, z);

        for(int c=0; c<this->a[n].size(); ++c)
            if(y[ini + i][c] == 1)
                prediccion = z[n][c];
            
        sum += log(prediccion+epsilon);
    }

    //sum = -sum / x.size();

    return sum;
}

/*
    @brief  Realiza la medida de Accuracy sobre un conjunto de datos
    @x      Conjunto de datos de entrada
    @y      Etiquetas de los datos de entrada
    @ini    Primera posición i en cumplir {y[i] corresponde a x[i], y[i+1] corresponde a x[i+1], ...} 
    @return Valor de accuracy sobre el conjunto de datos de entrada x
*/
float FullyConnected::accuracy(vector<vector<float>> x, vector<vector<float>> y, const int &ini)
{
    float sum =0.0, max;
    int prediccion, n=this->a.size()-1;

    vector<vector<float>> a, z;
    a = this->a;
    z = this->a;

    for(int i=0; i<x.size(); ++i)
    {
        // Propagación hacia delante de la entrada x[i] a lo largo de la red
        forwardPropagation(x[i], a, z);

        // Inicialización
        max = z[n][0];
        prediccion = 0;

        // Obtener valor más alto de la capa output
        for(int c=1; c<this->a[n].size(); ++c)
        {
            if(max < z[n][c])
            {
                max = z[n][c];
                prediccion = c;
            }
        }

        // Ver si etiqueta real y predicción coindicen
        sum += y[ini + i][prediccion];
    }

    //sum = sum / x.size() * 100;

    return sum;
}

/*
    @brief      Entrenamiento de la red
    @x          Conjunto de datos de entrada
    @y          Etiquetas asociadas a cada dato de entrada
    @batch      Conjunto de índices desordenados tal que para cada dato de entrada x[i], este está asociado con y[batch[i]]
    @n_datos    Número de datos con los que entrenar
    @grad_pesos Gradiente de cada peso de la red
    @grad_b     Gradiente de cada bias de la red
    @grad_x     Gradiente respecto a la entrada x
    @a          Neuronas de la red antes de aplicar la función de activación
    @z          Neuronas de la red después de aplicar la función de activación
    @grad_a     Gradiente respecto a cada neurona de la red antes de aplicar la función de activación
    @n_thrs     Número de hebras a emplear
    @return     Se actualizan los valores de @grad_pesos, @grad_b, @grad_x, @a, @z, y @grad_a
*/
void FullyConnected::train(const vector<vector<float>> &x, const vector<vector<float>> &y, const vector<int> &batch, const int &n_datos, vector<vector<vector<float>>> &grad_pesos, vector<vector<float>> &grad_b, vector<vector<float>> &grad_x, vector<vector<float>> &a, vector<vector<float>> &z, vector<vector<float>> &grad_a, const int &n_thrs)
{
    float epsilon = 0.000000001;
    int i_output = this->a.size()-1, i_last_h = i_output-1; // índice de la capa output, Índice de la capa h1 respectivamente
    vector<vector<float>> vector_2d(n_datos);
    grad_x = vector_2d;


    for(int i=0; i<n_datos; ++i)
    {
        // Propagación hacia delante
        forwardPropagation(x[i], a, z);
        
        // Propagación hacia detrás
        // Inicializar a 0 gradiente respecto a input
        for(int _i = 0; _i < grad_a.size(); ++_i)
            for(int j = 0; j < grad_a[_i].size(); ++j)
                grad_a[_i][j] = 0.0;


        // Capa SoftMax -----------------------------------------------
        // Se calcula gradiente del error respecto a cada Z_k
        for(int k=0; k<this->a[i_output].size(); ++k)
            grad_a[i_output][k] = z[i_output][k] - y[batch[i]][k];
            // grad_Zk = O_k - y_k
        
        // Pesos h_last - Softmax
        for(int p=0; p<this->a[i_last_h].size(); ++p)
            for(int k=0; k<this->a[i_output].size(); ++k)
                grad_pesos[i_last_h][p][k] += grad_a[i_output][k] * z[i_last_h][p];
                //                                 grad_Zk                  *  z^i_last_h_p
        
        // Sesgos capa softmax
        for(int k=0; k<this->a[i_output].size(); ++k)
            grad_b[i_output][k] += grad_a[i_output][k];
            // bk = grad_Zk

        // Última capa oculta -----------------------------------------------
        for(int p=0; p<this->a[i_last_h].size(); ++p)      
            for(int k=0; k<this->a[i_output].size(); ++k)
                grad_a[i_last_h][p] += grad_a[i_output][k] * this->w[i_last_h][p][k] * deriv_relu(a[i_last_h][p]);
                //                              grad_Zk           *  w^i_last_h_pk          * ...
                
        // Capas ocultas intermedias
        for(int capa= i_last_h; capa > 1; capa--)
        {
            // Pesos
            for(int i_act = 0; i_act < this->a[capa].size(); ++i_act)       // Por cada neurona de la capa actual
                for(int i_ant = 0; i_ant < this->a[capa-1].size(); ++i_ant)     // Por cada neurona de la capa anterior
                    grad_pesos[capa-1][i_ant][i_act] += grad_a[capa][i_act] * z[capa-1][i_ant];

            // Bias
            for(int i_act = 0; i_act < this->a[capa].size(); ++i_act)
                grad_b[capa][i_act] += grad_a[capa][i_act];
            
            // Grad input
            for(int i_ant = 0; i_ant < this->a[capa-1].size(); ++i_ant)     // Por cada neurona de la capa anterior
                for(int i_act = 0; i_act < this->a[capa].size(); ++i_act)       // Por cada neurona de la capa actual
                    grad_a[capa-1][i_ant] += grad_a[capa][i_act] * this->w[capa-1][i_ant][i_act] * deriv_relu(a[capa-1][i_ant]);
        }
        
        
        // Capa input
        // Pesos
        int capa=1;
        for(int i_act = 0; i_act < this->a[capa].size(); ++i_act)       // Por cada neurona de la capa actual
            for(int i_ant = 0; i_ant < this->a[capa-1].size(); ++i_ant)     // Por cada neurona de la capa anterior
                grad_pesos[capa-1][i_ant][i_act] += grad_a[capa][i_act] * z[capa-1][i_ant];
        
        // Grad input
        for(int i_ant = 0; i_ant < this->a[capa-1].size(); ++i_ant)     // Por cada neurona de la capa anterior
            for(int i_act = 0; i_act < this->a[capa].size(); ++i_act)       // Por cada neurona de la capa actual
                grad_a[capa-1][i_ant] += grad_a[capa][i_act] * this->w[capa-1][i_ant][i_act];


        grad_x[i] = grad_a[0];
        
    } 
}

/*
    @brief          Escalar los pesos para evitar que los gradientes "exploten"
    @clip_value     Valor a emplear para realizar el "clip" o escalado
    @return         Se actualizan los valores de w (pesos de la red)
*/
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

/*
    @brief          Actualizar los pesos y sesgos de la red
    @grad_pesos     Gradiente de cada peso de la red
    @grad_b         Gradiente de cada sesgo de la red
    @return         Se actualizar los valores de w y bias (pesos y sesgos de la red)
*/
void FullyConnected::actualizar_parametros(vector<vector<vector<float>>> &grad_pesos, vector<vector<float>> &grad_b)
{
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


int main()
{
    vector<int> capas = {3, 2};
    vector<float> X = {1, 1, 1};
    vector<vector<float>> a, z, grad_b;
    vector<vector<vector<float>>> w, grad_pesos;
    FullyConnected fully(capas, 0.1);
    w = fully.get_pesos();
    a = fully.get_a();
    z = a;
    grad_pesos = w;
    grad_b = a;

    for(int i=0; i<w.size(); i++)
        for(int j=0; j<w[i].size(); j++)
            for(int k=0; k<w[i][j].size(); k++)
                w[i][j][k] = 0.7;

    for(int i=0; i<w.size(); i++)
        for(int j=0; j<w[i].size(); j++)
            for(int k=0; k<w[i][j].size(); k++)
                grad_pesos[i][j][k] = 1;
            
    for(int i=0; i<z.size(); i++)
        for(int j=0; j<z[i].size(); j++)
            grad_b[i][j] = 0;




    //fully.set_pesos(w);

    fully.forwardPropagation(X, a, z);
    /*
    cout << "z: " << endl;
    for(int i=0; i<z.size(); i++)
    {
        for(int j=0; j<z[i].size(); j++)
            cout << z[i][j] << " " << endl;
        cout << endl;
    }
    */
    w = fully.get_pesos();
    cout << "pesos: " << endl;
    for(int i=0; i<w.size(); i++)
    {
        for(int j=0; j<w[i].size(); j++)
        {
            for(int k=0; k<w[i][j].size(); k++)
                cout << w[i][j][k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    fully.actualizar_parametros(grad_pesos, grad_b);
    fully.escalar_pesos(2);
    w = fully.get_pesos();

    cout << "pesos: " << endl;
    for(int i=0; i<w.size(); i++)
    {
        for(int j=0; j<w[i].size(); j++)
        {
            for(int k=0; k<w[i][j].size(); k++)
                cout << w[i][j][k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;

    return 0;
}