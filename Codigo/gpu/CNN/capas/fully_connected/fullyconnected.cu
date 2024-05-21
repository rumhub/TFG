#include "fullyconnected.h"

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
    CONSTRUCTOR de la clase FullyConnected
    --------------------------------------
  
    @capas  Vector de enteros que indica el número de neuronas por capa. Habrá capas.size() capas y cada una contendrá capas[i] neuronas
    @lr     Learning Rate o Tasa de Aprendizaje
*/
FullyConnected::FullyConnected(int *capas, int n_capas, float lr)
{
    
    vector<float> neuronas_capa, w_1D;
    vector<vector<float>> w_2D;
    int n_neuronas = 0, n_pesos = 0;
    this->n_capas = n_capas;
    this->capas = (int *)malloc(n_capas * sizeof(int));
    this->i_capa = (int *)malloc(n_capas * sizeof(int));
    this->i_w_ptr = (int *)malloc(n_capas * sizeof(int));

    // Neuronas ------------------------------------------------------------------
    for(int i=0; i<n_capas; i++)
    {
        this->i_capa[i] = n_neuronas;
        n_neuronas += capas[i];
        this->capas[i] = capas[i];
    }
    this->n_neuronas = n_neuronas;

    this->bias_ptr = (float *)malloc(n_neuronas * sizeof(float));       // Cada neurona tiene asociado un bias

    // Mostrar neuronas
    //mostrar_neuronas();
    

    // Pesos ------------------------------------------------------------------
    for(int i=0; i<n_capas-1; i++)
    {
        this->i_w_ptr[i] = n_pesos;
        n_pesos += capas[i] * capas[i+1];   // Cada neurona de la capa actual se conecta a cada neurona de la capa siguiente
    }
    
    this->n_pesos = n_pesos;
    this->w_ptr = (float *)malloc(n_pesos * sizeof(float));

    // Learning Rate
    this->lr = lr;
    
    // Inicializar pesos mediante inicialización He
    for(int i=0; i<n_capas-1; ++i)
        this->generar_pesos_ptr(i);

    /*
    // Mostrar pesos
    cout << "Pesos" << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        cout << "Capa " << i << endl;
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
            {
                cout << "W(" << j << "," << k << "): ";
                cout << this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    */

    // Inicializar bias con un valor de 0.0
    for(int i=0; i<n_neuronas; i++)
        this->bias_ptr[i] = 0.0;
};

void FullyConnected::mostrar_pesos_ptr()
{
    // Mostrar pesos
    cout << "Pesos" << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        cout << "Capa " << i << endl;
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
            {
                cout << "W(" << j << "," << k << "): ";
                cout << this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
}


void FullyConnected::mostrar_pesos()
{
    cout << "Pesos" << endl;
    for(int i=0; i<w.size(); i++)
    {
        cout << "Capa " << i << endl;
        for(int j=0; j<w[i].size(); j++)
        {
            for(int k=0; k<w[i][j].size(); k++)
            {
                cout << "W(" << j << "," << k << "): ";
                cout << w[i][j][k] << " ";

            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
}

void FullyConnected::mostrar_neuronas_ptr(float *z_ptr)
{
    // Mostrar neuronas
    cout << "Neuronas" << endl;
    for(int i=0; i<n_capas; i++)
    {
        cout << "Capa " << i << endl;
        for(int j=0; j<capas[i]; j++)
            cout << z_ptr[i_capa[i] + j] << " ";
        cout << endl;
    }
    cout << endl;
}


void FullyConnected::mostrar_neuronas(const vector<vector<float>> &z)
{
    // Mostrar neuronas
    cout << "Neuronas" << endl;
    for(int i=0; i<a.size(); i++)
    {
        cout << "Capa " << i << endl;
        for(int j=0; j<a[i].size(); j++)
            cout << z[i][j] << " ";
        cout << endl;
    }
    cout << endl;
}

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

    int cont = 0;
    for(int i=0; i<this->a[capa].size(); ++i)
        for(int j=0; j<this->a[capa+1].size(); ++j)
            this->w[capa][i][j] = distribution(gen);
}

void FullyConnected::copiar_w_de_vector_a_ptr(vector<vector<vector<float>>> w_)
{
    for(int i=0; i<w_.size(); i++)
        for(int j=0; j<w_[i].size(); j++)
            for(int k=0; k<w_[i][j].size(); k++)
                w_ptr[i_w_ptr[i] + j*capas[i+1] + k] = w_[i][j][k];
}


/*
    @brief  Genera los pesos entre 2 capas de neuronas
    @capa   Capa a generar los pesos con la siguiente
    @return Se actualizan los valores de w (pesos de la red)
*/
void FullyConnected::generar_pesos_ptr(const int &capa)
{
    // Inicialización He
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0, sqrt(2.0 / this->capas[capa])); 

    int cont = 0;
    for(int i=0; i<this->capas[capa]; ++i)
        for(int j=0; j<this->capas[capa+1]; ++j)
            this->w_ptr[i_w_ptr[capa] + i*capas[capa+1] + j] = 1.0;
            //this->w_ptr[i_w_ptr[capa] + i*capas[capa+1] + j] = distribution(gen);
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
    @brief  Propagación hacia delante por toda la red totalmente conectada
    @x      Valores de entrada a la red
    @a      Neuronas de la red antes de aplicar la función de activación
    @z      Neuronas de la red después de aplicar la función de activación

    @return Se actualizan los valores de @a y @z
*/
void FullyConnected::forwardPropagation_ptr(float *x, float *a_ptr, float *z_ptr)
{
    float max, sum = 0.0, epsilon = 0.000000001;

    // Introducimos input -------------------------------------------------------
    for(int i=0; i<capas[0]; ++i)
    {
        z_ptr[i] = x[i];
        a_ptr[i] = x[i];
    }

    
    // Forward Propagation ------------------------------------------------------------
    // Por cada capa
    for(int i=0; i<n_capas -1; ++i)
    {

        // Por cada neurona de la capa siguiente
        for(int k=0; k<capas[i+1]; ++k)
        {
            // Reset siguiente capa
            a_ptr[i_capa[i+1] + k] = 0.0;

            for(int j=0; j<capas[i]; ++j)
                a_ptr[i_capa[i+1] + k] += z_ptr[i_capa[i] + j] * w_ptr[i_w_ptr[i] + j*capas[i+1] + k];

            // Aplicar bias o sesgo
            a_ptr[i_capa[i+1] + k] += this->bias_ptr[i_capa[i+1] + k];
        }

        // Aplicamos función de activación asociada a la capa actual -------------------------------

        // En capas ocultas se emplea ReLU como función de activación
        if(i < n_capas - 2)
        {
            for(int k=0; k<capas[i+1]; ++k)
                z_ptr[i_capa[i+1] + k] = relu(a_ptr[i_capa[i+1] + k]);
        }else
        {
            // En la capa output se emplea softmax como función de activación
            sum = 0.0;
            max = a_ptr[i_capa[i+1]];

            // Normalizar -----------------------------------------------------------------
            // Encontrar el máximo
            for(int k=0; k<capas[i+1]; ++k)
                if(max < a_ptr[i_capa[i+1] + k])
                    max = a_ptr[i_capa[i+1] + k];
            
            // Normalizar
            for(int k=0; k<capas[i+1]; ++k)
                a_ptr[i_capa[i+1] + k] = a_ptr[i_capa[i+1] + k] - max;
            
            // Calculamos la suma exponencial de todas las neuronas de la capa output ---------------------
            for(int k=0; k<capas[i+1]; ++k)
                sum += exp(a_ptr[i_capa[i+1] + k]);

            for(int k=0; k<capas[i+1]; ++k)
                z_ptr[i_capa[i+1] + k] = exp(a_ptr[i_capa[i+1] + k]) / sum;
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
float FullyConnected::cross_entropy(vector<vector<float>> x, vector<vector<float>> y)
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
            if(y[i][c] == 1)
                prediccion = z[n][c];
            
        sum += log(prediccion+epsilon);
    }

    //sum = -sum / x.size();

    return sum;
}

/*
    @brief  Realiza la medida de Entropía Cruzada sobre un conjunto de datos
    @x      Conjunto de datos de entrada
    @y      Etiquetas de los datos de entrada
    @ini    Primera posición i en cumplir {y[i] corresponde a x[i], y[i+1] corresponde a x[i+1], ...} 
    @return Valor de entropía cruzada sobre el conjunto de datos de entrada x
*/
float FullyConnected::cross_entropy_ptr(float *x, float *y, int n_datos, float *a_ptr, float *z_ptr)
{
    float sum = 0.0, prediccion = 0.0, epsilon = 0.000000001;
    int n=n_capas-1;

    for(int i=0; i<n_datos; ++i)
    {
        float *x_i = x + i*capas[0];                                // Cada x[i] tiene tantos valores como neuronas hay en la primera capa
        forwardPropagation_ptr(x_i, a_ptr, z_ptr);

        for(int c=0; c<capas[n]; ++c)
            if(y[i*capas[n] + c] == 1)                      // Cada y[i] tiene valores como neuronas hay en la última capa. 1 neurona y 1 valor por clase. One-hot.
                prediccion = z_ptr[i_capa[n] + c];
            
        sum += log(prediccion+epsilon);
    }

    //sum = -sum / x.size();

    return sum;
}


/*
    @brief  Realiza la medida de Accuracy sobre un conjunto de datos
    @x      Conjunto de datos de entrada
    @y      Etiquetas de los datos de entrada
    @return Valor de accuracy sobre el conjunto de datos de entrada x
*/
float FullyConnected::accuracy_ptr(float *x, float *y, int n_datos, float *a_ptr, float *z_ptr)
{
    float sum =0.0, max;
    int prediccion, n=n_capas-1;

    for(int i=0; i<n_datos; ++i)
    {
        // Propagación hacia delante de la entrada x[i] a lo largo de la red
        float *x_i = x + i*capas[0];                                // Cada x[i] tiene tantos valores como neuronas hay en la primera capa
        forwardPropagation_ptr(x_i, a_ptr, z_ptr);

        // Inicialización
        max = z_ptr[i_capa[n]];
        prediccion = 0;

        // Obtener valor más alto de la capa output
        for(int c=1; c<capas[n]; ++c)
        {
            if(max < z_ptr[i_capa[n] + c])
            {
                max = z_ptr[i_capa[n] + c];
                prediccion = c;
            }
        }

        // Ver si etiqueta real y predicción coindicen
        sum += y[i*capas[n] + prediccion];
    }

    //sum = sum / x.size() * 100;

    return sum;
}


/*
    @brief  Realiza la medida de Accuracy sobre un conjunto de datos
    @x      Conjunto de datos de entrada
    @y      Etiquetas de los datos de entrada
    @return Valor de accuracy sobre el conjunto de datos de entrada x
*/
float FullyConnected::accuracy(vector<vector<float>> x, vector<vector<float>> y)
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
        sum += y[i][prediccion];
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
    @return     Se actualizan los valores de @grad_pesos, @grad_b, @grad_x, @a, @z, y @grad_a
*/
//void FullyConnected::train(const vector<vector<float>> &x, const vector<vector<float>> &y, const vector<int> &batch, const int &n_datos, vector<vector<vector<float>>> &grad_pesos, vector<vector<float>> &grad_b, vector<vector<float>> &grad_x, vector<vector<float>> &a, vector<vector<float>> &z, vector<vector<float>> &grad_a)
void FullyConnected::train_ptr(float *x, float *y, int *batch, const int &n_datos, float * grad_w_ptr, float * grad_bias_ptr, float *grad_x, float *a_ptr, float *z_ptr, float *grad_a_ptr)
{
    float epsilon = 0.000000001;
    int i_output = n_capas -1, i_last_h = i_output-1; // índice de la capa output, Índice de la capa h1 respectivamente

    for(int i=0; i<n_datos; ++i)
    {
        // Propagación hacia delante
        float *x_i = x + i*capas[0];                                // Cada x[i] tiene tantos valores como neuronas hay en la primera capa
        forwardPropagation_ptr(x_i, a_ptr, z_ptr);
        
        // Propagación hacia detrás
        // Inicializar a 0 gradiente respecto a input
        for(int _i = 0; _i < n_capas; ++_i)
            for(int j = 0; j < capas[_i]; ++j)
                grad_a_ptr[i_capa[_i] + j] = 0.0;

        // Capa SoftMax -----------------------------------------------
        // Se calcula gradiente del error respecto a cada Z_k
        for(int k=0; k<capas[i_output]; ++k)
            grad_a_ptr[i_capa[i_output] + k] = z_ptr[i_capa[i_output] + k] - y[batch[i]*capas[i_output] + k];
            // grad_Zk = O_k - y_k


        // Pesos h_last - Softmax
        for(int p=0; p<capas[i_last_h]; ++p)
            for(int k=0; k<capas[i_output]; ++k)
                grad_w_ptr[i_w_ptr[i_last_h] + p*capas[i_last_h+1] + k] += grad_a_ptr[i_capa[i_output] + k] * z_ptr[i_capa[i_last_h] + p];
                //                                 grad_Zk                  *  z^i_last_h_p
        
        // Sesgos capa softmax
        for(int k=0; k<capas[i_output]; ++k)
            grad_bias_ptr[i_capa[i_output] + k] += grad_a_ptr[i_capa[i_output] + k];
            // bk = grad_Zk

        // Última capa oculta -----------------------------------------------
        for(int p=0; p<capas[i_last_h]; ++p)      
            for(int k=0; k<capas[i_output]; ++k)
                grad_a_ptr[i_capa[i_last_h] + p] += grad_a_ptr[i_capa[i_output] + k] * w_ptr[i_w_ptr[i_last_h] + p*capas[i_last_h+1] + k] * deriv_relu(a_ptr[i_capa[i_last_h] + p]);
                //                              grad_Zk           *  w^i_last_h_pk          * ...
                
        // Capas ocultas intermedias
        for(int capa= i_last_h; capa > 1; capa--)
        {
            // Pesos
            for(int i_act = 0; i_act < capas[capa]; ++i_act)       // Por cada neurona de la capa actual
                for(int i_ant = 0; i_ant < capas[capa-1]; ++i_ant)     // Por cada neurona de la capa anterior
                    grad_w_ptr[i_w_ptr[capa-1] + i_ant*capas[capa] + i_act] += grad_a_ptr[i_capa[capa] + i_act] * z_ptr[i_capa[capa-1] + i_ant];

            // Bias
            for(int i_act = 0; i_act < capas[capa]; ++i_act)
                grad_bias_ptr[i_capa[capa] + i_act] += grad_a_ptr[i_capa[capa] + i_act];
            
            // Grad input
            for(int i_ant = 0; i_ant < capas[capa-1]; ++i_ant)     // Por cada neurona de la capa anterior
                for(int i_act = 0; i_act < capas[capa]; ++i_act)       // Por cada neurona de la capa actual
                    grad_a_ptr[i_capa[capa-1] + i_ant] += grad_a_ptr[i_capa[capa] + i_act] * w_ptr[i_w_ptr[capa-1] + i_ant*capas[capa] + i_act] * deriv_relu(a_ptr[i_capa[capa-1] + i_ant]);
        }
        
        
        // Capa input
        // Pesos
        int capa=1;
        for(int i_act = 0; i_act < capas[capa]; ++i_act)       // Por cada neurona de la capa actual
            for(int i_ant = 0; i_ant < capas[capa-1]; ++i_ant)     // Por cada neurona de la capa anterior
                grad_w_ptr[i_w_ptr[capa-1] + i_ant*capas[capa] + i_act] += grad_a_ptr[i_capa[capa] + i_act] * z_ptr[i_capa[capa-1] + i_ant];
        
        // Grad input
        for(int i_ant = 0; i_ant < capas[capa-1]; ++i_ant)     // Por cada neurona de la capa anterior
            for(int i_act = 0; i_act < capas[capa]; ++i_act)       // Por cada neurona de la capa actual
                grad_a_ptr[i_capa[capa-1] + i_ant] += grad_a_ptr[i_capa[capa] + i_act] * w_ptr[i_w_ptr[capa-1] + i_ant*capas[capa] + i_act];

        // Copiar gradiente respecto a input de primera capa en grad_x[i]
        for(int j=0; j<capas[0]; j++)
            grad_x[i*capas[0] + j] = grad_a_ptr[j];
    } 
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
    @return     Se actualizan los valores de @grad_pesos, @grad_b, @grad_x, @a, @z, y @grad_a
*/
void FullyConnected::train(const vector<vector<float>> &x, const vector<vector<float>> &y, const vector<int> &batch, const int &n_datos, vector<vector<vector<float>>> &grad_pesos, vector<vector<float>> &grad_b, vector<vector<float>> &grad_x, vector<vector<float>> &a, vector<vector<float>> &z, vector<vector<float>> &grad_a)
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
void FullyConnected::escalar_pesos_ptr(float clip_value)
{
    // Calcular el máximo y el mínimo de los pesos
    float max = this->w[0][0][0], min = this->w[0][0][0];

    for(int i=0; i<n_capas-1; i++)
        for(int j=0; j<capas[i]; j++)
            for(int k=0; k<capas[i+1]; k++)
            {
                if(max < this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k])
                    max = this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k];
                
                if(min > this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k])
                    min = this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k];
            }

    // Realizar gradient clipping
    float scaling_factor = clip_value / std::max(std::abs(max), std::abs(min));
    for(int i=0; i<n_capas-1; i++)
        for(int j=0; j<capas[i]; j++)
            for(int k=0; k<capas[i+1]; k++)
                this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k] = std::max(std::min(this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k], clip_value), -clip_value);
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
void FullyConnected::actualizar_parametros_ptr(float *grad_pesos, float *grad_b)
{
    // Actualizar pesos
    for(int i=0; i<n_capas-1; i++)
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
                this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k] -= this->lr * grad_pesos[i_w_ptr[i] + j*capas[i+1] + k];

    for(int i=0; i<n_capas; i++)
        for(int j=0; j<capas[i]; j++)
            this->bias_ptr[i_capa[i] + j] -= this->lr * grad_b[i_capa[i] + j];
}

/*
    @brief          Actualizar los pesos y sesgos de la red
    @grad_pesos     Gradiente de cada peso de la red
    @grad_b         Gradiente de cada sesgo de la red
    @return         Se actualizar los valores de w y bias (pesos y sesgos de la red)
*/
void FullyConnected::actualizar_parametros(const vector<vector<vector<float>>> &grad_pesos, const vector<vector<float>> &grad_b)
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

/*
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
*/

void mostrar_ptr_2D(float *x, int H, int W)
{
    for(int i=0; i<H; i++)
    {
        for(int j=0; j<W; j++)
            cout << x[i*W + j] << " ";
        cout << endl;
    }
}

void mostrar_vector_2D(vector<vector<float>> x)
{
    for(int i=0; i<x.size(); i++)
    {
        for(int j=0; j<x[i].size(); j++)
            cout << x[i][j] << " ";
        cout << endl;
    }
}

/*
int main()
{
    // CPU --------------
    cout << " ---------- CPU ---------- " << endl; 
    int tam_x = 3, n_datos = 2, n_clases = 2;
    vector<int> capas = {tam_x, 2, 3, n_clases};
    vector<vector<float>> a_cpu, z_cpu, grad_a_cpu, y_cpu = {{0.0, 1.0}, {0.0, 1.0}};
    vector<vector<vector<float>>> grad_w_cpu;
    vector<float> x_cpu;
    vector<vector<float>> X_cpu, grad_b_cpu, grad_x_cpu;
    vector<int> batch_cpu(n_datos);
    FullyConnected fully_cpu(capas, 0.1);
    a_cpu = fully_cpu.get_a();
    z_cpu = fully_cpu.get_a();
    grad_a_cpu = fully_cpu.get_a();
    grad_w_cpu = fully_cpu.get_pesos();
    grad_b_cpu = fully_cpu.get_bias();

    for(int i=0; i<n_datos; i++)
        batch_cpu[i] = i;

    for(int i=0; i<tam_x; i++)
        x_cpu.push_back(i);

    for(int i=0; i<n_datos; i++)
        X_cpu.push_back(x_cpu);

    for(int i=0; i<grad_w_cpu.size(); i++)
        for(int j=0; j<grad_w_cpu[i].size(); j++)
            for(int k=0; k<grad_w_cpu[i][j].size(); k++)
                grad_w_cpu[i][j][k] = 0.0;

    for(int i=0; i<grad_b_cpu.size(); i++)
        for(int j=0; j<grad_b_cpu[i].size(); j++)
            grad_b_cpu[i][j] = 0.0;

    //fully_cpu.forwardPropagation(x_cpu, a_cpu, z_cpu);
    //fully_cpu.mostrar_neuronas(z_cpu);
    mostrar_vector_2D(X_cpu);
    mostrar_vector_2D(y_cpu);
    cout << "Entr: " << fully_cpu.cross_entropy(X_cpu, y_cpu) << endl;
    cout << "Acc: " << fully_cpu.accuracy(X_cpu, y_cpu) << endl;

    vector<vector<vector<float>>> w_cpu_copy = fully_cpu.get_pesos();
    
    fully_cpu.train(X_cpu, y_cpu, batch_cpu, n_datos, grad_w_cpu, grad_b_cpu, grad_x_cpu, a_cpu, z_cpu, grad_a_cpu);
    fully_cpu.actualizar_parametros(grad_w_cpu, grad_b_cpu);
    fully_cpu.escalar_pesos(2);
    fully_cpu.mostrar_pesos();
    



    // GPU --------------
    cout << " ---------- GPU ---------- " << endl; 
    int n_capas = 4;
    int *capas_ptr = (int *)malloc(n_capas * sizeof(int));
    int *i_capa = (int *)malloc(n_capas * sizeof(int));

    for(int i=0; i<n_capas; i++)
        capas_ptr[i] = capas[i];
    FullyConnected fully_gpu(capas_ptr, n_capas, 0.1);

    int n_neuronas = 0, n_pesos = 0;

    // Pesos ------------------------------------------------------------------
    int *i_w_ptr = (int *)malloc(n_capas * sizeof(int));

    // Neuronas ------------------------------------------------------------------
    for(int i=0; i<n_capas; i++)
    {
        i_capa[i] = n_neuronas;
        n_neuronas += capas_ptr[i];
        capas_ptr[i] = capas[i];
    }

    for(int i=0; i<n_capas-1; i++)
    {
        i_w_ptr[i] = n_pesos;
        n_pesos += capas_ptr[i] * capas_ptr[i+1];   // Cada neurona de la capa actual se conecta a cada neurona de la capa siguiente
    }
    
    float *grad_w_ptr = (float *)malloc(n_pesos * sizeof(float));

    // Inicializar gradientes de pesos a 0.0
    for(int i=0; i<n_pesos; i++)
        grad_w_ptr[i] = 0.0;

    // Entrada y salida ------------------------------------------------------------------
    float *X_gpu = (float *)malloc(n_datos * tam_x * sizeof(float));
    float *y_gpu = (float *)malloc(n_datos * tam_x * sizeof(float));
    int *batch_gpu = (int *)malloc(n_datos * sizeof(int));
    
    for(int i=0; i<n_datos; i++)
        batch_gpu[i] = i;

    for(int i=0; i<n_datos; i++)
        for(int j=0; j<tam_x; j++)
            X_gpu[i*tam_x + j] = X_cpu[i][j];

    for(int i=0; i<n_datos; i++)
        for(int j=0; j<n_clases; j++)
            y_gpu[i*n_clases + j] = y_cpu[i][j];


    float *a_ptr = (float *)malloc(n_neuronas * sizeof(float));
    float *z_ptr = (float *)malloc(n_neuronas * sizeof(float));
    float *grad_bias_ptr = (float *)malloc(n_neuronas * sizeof(float));       
    float *grad_a_ptr = (float *)malloc(n_neuronas * sizeof(float));

    // Inicializar gradientes de sesgos a 0.0
    for(int i=0; i<n_neuronas; i++)
        grad_bias_ptr[i] = 0.0;

    fully_gpu.copiar_w_de_vector_a_ptr(w_cpu_copy);
    //fully_gpu.mostrar_neuronas_ptr();
    mostrar_ptr_2D(X_gpu, n_datos, tam_x);
    mostrar_ptr_2D(y_gpu, n_datos, n_clases);
    cout << "Entr: " << fully_gpu.cross_entropy_ptr(X_gpu, y_gpu, n_datos, a_ptr, z_ptr) << endl;
    cout << "Acc: " << fully_gpu.accuracy_ptr(X_gpu, y_gpu, n_datos, a_ptr, z_ptr) << endl;

    
    float *grad_x_gpu = (float *)malloc(tam_x * n_datos * sizeof(float));
    fully_gpu.train_ptr(X_gpu, y_gpu, batch_gpu, n_datos, grad_w_ptr, grad_bias_ptr, grad_x_gpu, a_ptr, z_ptr, grad_a_ptr);
    fully_gpu.actualizar_parametros_ptr(grad_w_ptr, grad_bias_ptr);
    fully_gpu.escalar_pesos_ptr(2);
    fully_gpu.mostrar_pesos_ptr();
    

    free(capas_ptr); free(i_w_ptr); free(grad_w_ptr); free(X_gpu); free(y_gpu); free(batch_gpu); free(a_ptr); free(z_ptr); free(grad_x_gpu); free(grad_bias_ptr); free(grad_a_ptr);

    return 0;
}
*/