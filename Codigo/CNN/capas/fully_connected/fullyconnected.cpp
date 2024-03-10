#include "fullyconnected.h"
#include <vector>
#include <iostream>
#include "math.h"

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

        this->neuronas.push_back(neuronas_capa);
        neuronas_capa.clear();
    }

    // Pesos -------------------------------------------------------------------
    // Por cada capa
    for(int i=0; i<neuronas.size()-1; i++)
    {
        // Por cada neurona de cada capa
        for(int j=0; j<neuronas[i].size(); j++)
        {
            
            // Por cada neurona de la capa siguiente
            for(int k=0; k<neuronas[i+1].size(); k++)
            {
                // Añadimos un peso. 
                // w[i][j][k] indica el peso que conecta la neurona j de la capa i con la neurona k de la capa i+1
                w_1D.push_back((float) rand() / float(RAND_MAX) -0.5);
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
    this->bias = this->neuronas;

    // Inicializamos bias con un valor random entre -0.5 y 0.5
    for(int i=0; i<this->bias.size(); i++)
        for(int j=0; j<this->bias[i].size(); j++)
            this->bias[i][j] = (float) rand() / float(RAND_MAX) -0.5;

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
    for(int i=0; i<this->neuronas.size(); i++)
    {
        for(int j=0; j<this->neuronas[i].size(); j++)
        {
            cout << this->neuronas[i][j] << " ";
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
    float sum = 0.0;

    // Introducimos input -------------------------------------------------------
    for(int i=0; i<x.size(); i++)
        this->neuronas[0][i] = x[i];
    
    // Forward Propagation ------------------------------------------------------------
    // Por cada capa
    for(int i=0; i<this->neuronas.size()-1; i++)
    {

        // Por cada neurona de la capa siguiente
        for(int k=0; k<this->neuronas[i+1].size(); k++)
        {
            // Reset siguiente capa
            this->neuronas[i+1][k] = 0.0;

            // w[i][j][k] indica el peso que conecta la neurona j de la capa i con la neurona k de la capa i+1
            for(int j=0; j<this->neuronas[i].size(); j++)
                this->neuronas[i+1][k] += this->neuronas[i][j] * this->w[i][j][k];
            
            // Aplicar bias o sesgo
            this->neuronas[i+1][k] += this->bias[i+1][k];
        }

        // Aplicamos función de activación asociada a la capa actual -------------------------------

        // En capas intermedias (y capa input) se emplea ReLU como función de activación
        if(i < this->neuronas.size() - 3)
        {
            for(int k=0; k<this->neuronas[i+1].size(); k++)
                this->neuronas[i+1][k] = relu(this->neuronas[i+1][k]);
        }else
        {
            // En la última capa oculta se emplea sigmoide como función de activación
            if(i == this->neuronas.size() - 3)
            {
                for(int k=0; k<this->neuronas[i+1].size(); k++)
                    this->neuronas[i+1][k] = sigmoid(this->neuronas[i+1][k]);
            }else
            {
                // En la capa output se emplea softmax como función de activación
                sum = 0.0;
                // Calculamos la suma exponencial de todas las neuronas de la capa output
                for(int k=0; k<this->neuronas[i+1].size(); k++)
                    sum += exp(this->neuronas[i+1][k]);

                for(int k=0; k<this->neuronas[i+1].size(); k++)
                    this->neuronas[i+1][k] = exp(this->neuronas[i+1][k]) / sum;
            }
        } 
    }
}


void FullyConnected::mostrar_prediccion(vector<float> x, float y)
{
    forwardPropagation(x);
    cout << "Input: ";
    int n=this->neuronas.size()-1;

    for(int i=0; i<x.size(); i++)
    {
        cout << x[i] << " ";
    }
    cout << "y: " << y << ", predicción: " << this->neuronas[n][0] << endl;
}

void FullyConnected::mostrar_prediccion_vs_verdad(vector<float> x, float y)
{
    int n=this->neuronas.size()-1;
    forwardPropagation(x);

    cout << "y: " << y << ", predicción: " << this->neuronas[n][0] << endl;
}

// Ahora x es el conjunto de datos de training
// x[0] el primer dato training
// x[0][0] el primer elemento del primer dato training
float FullyConnected::cross_entropy(vector<vector<float>> x, vector<vector<float>> y)
{
    float sum = 0.0, prediccion = 0.0, epsilon = 0.000000001;
    int n=this->neuronas.size()-1;

    for(int i=0; i<x.size(); i++)
    {
        forwardPropagation(x[i]);

        for(int c=0; c<this->neuronas[n].size(); c++)
            if(y[i][c] == 1)
                prediccion = this->neuronas[n][c];
            
        sum += log(prediccion+epsilon);
    }

    sum = -sum / x.size();


    return sum;
}

float FullyConnected::accuracy(vector<vector<float>> x, vector<float> y)
{
    float sum =0.0, prediccion;
    int n=this->neuronas.size()-1;

    for(int i=0; i<x.size(); i++)
    {
        forwardPropagation(x[i]);
        prediccion = this->neuronas[n][0];

        if((prediccion < 0.5 && y[i] == 0) || (prediccion >= 0.5 && y[i] == 1))
            sum ++;

    }

    sum = sum / x.size() * 100;


    return sum;
}


void FullyConnected::train(const vector<vector<float>> &x, const vector<float> &y, vector<vector<float>> &grad_x)
{
    /*
    int n_datos = x.size();
    float sum_b = 0.0, sum_w1 = 0.0, sum_w2 = 0.0, prediccion;
    float grad_w_aux, aux, epsilon = 0.000000001;

    int i_output = this->neuronas.size()-1; // índice de la capa output
    float sum, o_in, grad_x_output, sig_o_in;
    int i_last_h = i_output-1;  // Índice de la capa h1
    int i_h1 = 1, i_h2 = 2;
    vector<float> grad_x_i;

    grad_x.clear();

    for(int i=0; i<this->neuronas[0].size(); i++)
        grad_x_i.push_back(0);

    // Inicializar gradiente de pesos a 0 --------------------------
    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                this->grad_w[i][j][k] = 0;

    // Inicializar gradiente bias a 0 ------------------------------
    for(int i=0; i<this->grad_bias.size(); i++)
        for(int j=0; j<this->grad_bias[i].size(); j++)
            this->grad_bias[i][j] = 0;
    

    // Backpropagation ----------------------------------------------

    // Backpropagation ----------------------------------------------
    // Si hay 2 o más capas ocultas
    if(this->neuronas.size() > 3)
    { 
        for(int i=0; i<n_datos; i++)
        {
            forwardPropagation(x[i]);
            prediccion = this->neuronas[i_output][0];
            
            o_in = 0;

            // Por cada neurona j de i_last_h, sumamos i_last_h(out_j) * el peso que lo conecta con la capa output
            // Es decir, queremos obtener O_in (solo hay una neurona en la capa output)
            for(int j=0; j<this->neuronas[i_last_h].size(); j++)
                o_in += this->neuronas[i_last_h][j] * this->w[i_last_h][j][0];
            
            o_in += this->bias[i_output][0];

            grad_x_output = (y[i] / prediccion) - (1-y[i])/(1-prediccion);

            sig_o_in = sigmoid(o_in);

            // Grad_x --> Hasta Oout / Oin
            grad_x_output = grad_x_output * (sig_o_in * (1-sig_o_in));
            

            // Pesos h_last - Oin
            
            // Por cada neurona de la capa output
            for(int i_o=0; i_o<this->neuronas[i_output].size(); i_o++)
            {
                // Por cada neurona de la capa h_last
                for(int j=0; j<this->neuronas[i_last_h].size(); j++)
                {
                    // Calcular gradiente de peso
                    grad_w_aux = this->neuronas[i_last_h][j];    //hlast_out_j
                    this->grad_w[i_last_h][j][i_o] += grad_w_aux * grad_x_output;

                    // Actualizar grad_bias capa h_last
                    this->grad_bias[i_last_h][j] += grad_x_output;
                }


            }
            
            
            // Pesos entre capas ocultas intermedias
            
            vector<float> grads_x_n;
            int n_hidden_intermedias = this->neuronas.size() - 3;   // n_hidden_intermedias = nº capas - (capa input, última capa oculta, capa output)
            int n_capa_ant; // capa anterior a n_capa
            float h_in;

            // Por cada capa oculta intermedia
            // Si hay 3 capas intermedias, la primera de ellas es la que tiene índice 1, pues la capa input es la 0
            for(int n_capa= i_last_h; n_capa > i_last_h - n_hidden_intermedias; n_capa--)
            {
                n_capa_ant = n_capa -1;

                // Calcular gradiente de x hasta hn_in (grads_x_n) -----------------------------------------------
                // Por cada neurona de n_capa
                for(int k=0; k<this->neuronas[n_capa].size(); k++)
                {
                    h_in = 0;

                    // Por cada neurona j de la capa anterior
                    for(int j=0; j<this->neuronas[n_capa_ant].size(); j++)
                    {
                        h_in += this->neuronas[n_capa_ant][j] * this->w[n_capa_ant][j][k];
                    }
                    
                    h_in += this->bias[n_capa_ant-1];    // sumarle el bias de h_in-1 a h_in

                    grads_x_n.push_back(grad_x_output * this->w[n_capa][k][0] * deriv_relu(h_in)); // 0 porque solo hay una neurona en la capa output0;
                }

                // Usando grads_x_n, calcular el gradiente de cada peso que conecta las capas h1 y h2
                // Por cada neurona de la capa h1
                for(int j=0; j<this->neuronas[n_capa_ant].size(); j++)
                {
                    // Por cada neurona de h2
                    for(int k=0; k<this->neuronas[n_capa].size(); k++)
                    {
                        // Calcular gradiente de peso
                        grad_w_aux = this->neuronas[n_capa_ant][j];    // x_(j)
                        this->grad_w[n_capa_ant][j][k] += grad_w_aux * grads_x_n[k];
                    }
                }       

                // Por cada neurona de h2
                for(int k=0; k<this->neuronas[n_capa].size(); k++)
                {
                    // Actualizar grad_bias capa h1
                    this->grad_bias[n_capa_ant] += grads_x_n[k];
                }
                // this->grad_bias[n_capa_ant] += grad_x_output; 

                grads_x_n.clear();
            }


            
            // Pesos input - h1

            vector<float> grads_x_h1;
            float sum = 0;

            // Calcular gradiente de x hasta h_in (grads_x_h1) -----------------------------------------------
            // Por cada neurona de h1
            for(int j=0; j<this->neuronas[i_h1].size(); j++)
            {  
                h_in = 0;

                // Por cada neurona de la capa input
                for(int i_input=0; i_input<this->neuronas[i_h1].size(); i_input++)
                {
                    h_in += this->neuronas[i_h1][i_input] * this->w[i_h1][i_input][j];
                }

                // Calcular suma de gradientes que vienen de h2 -------------------
                sum = 0;
                for(int k=0; k<this->neuronas[i_h2].size(); k++)
                {
                    sum += grads_x_n[k] * this->w[i_h1][j][k];
                }

                grads_x_h1.push_back(sum * deriv_relu(h_in));

            }

            // Calcular el gradiente de cada peso que conecta las capas input y h1
            for(int j=0; j<this->neuronas[i_h1].size(); j++)
            {
                for(int i_input=0; i_input<this->neuronas[0].size(); i_input++)
                {
                    // Calcular gradiente de peso
                    grad_w_aux = this->neuronas[0][i_input];
                    this->grad_w[0][i_input][j] += grad_w_aux * grads_x_h1[j];
                }
            }

            // Actualizar bias capa input
            for(int j=0; j<this->neuronas[i_h1].size(); j++)
            {
                this->grad_bias[0] += grads_x_h1[j];
            }
            
            // ---------------------------------------------------------------------------

            // Reset gradiente respecto a input
            for(int i=0; i<this->neuronas[0].size(); i++)
            {
                grad_x_i[i] = 0;
            }   

            // Actualizar gradiente respecto a input               
            for(int i=0; i<this->neuronas[0].size(); i++)
            {
                for(int j=0; j<this->neuronas[1].size(); j++)
                {
                    // w[i][j][k] indica el peso que conecta la neurona j de la capa i con la neurona k de la capa i+1
                    grad_x_i[i] += grad_w_aux * this->w[0][i][j]; 
                }                        
            }

            grad_x.push_back(grad_x_i); 

            // --------------------------------------------------------------------------
        }
    }else
    {
        // Si solo hay una capa oculta
        if(this->neuronas.size() == 3)
        {

            for(int d=0; d<n_datos; d++)
            {
                forwardPropagation(x[d]);
                prediccion = this->neuronas[i_output][0];

                o_in = 0;

                // Por cada neurona j de h1, sumamos h1_out_j * el peso que lo conecta con la capa output
                // Es decir, queremos obtener O_in (solo hay una neurona en la capa output)
                for(int j=0; j<this->neuronas[i_h1].size(); j++)
                {
                    o_in += this->neuronas[i_h1][j] * this->w[i_h1][j][0];
                }
                
                o_in += this->bias[i_h1];
                grad_x_output = ((y[d]+epsilon) / (prediccion + epsilon)) - (1-y[d]+epsilon)/(1-prediccion+epsilon);
                sig_o_in = sigmoid(o_in);

                // Grad_x --> Hasta Oout / Oin
                grad_x_output = grad_x_output * ((sig_o_in+epsilon) * (1-sig_o_in+epsilon));

                // Pesos h1 - Oin
                
                // Por cada neurona de la capa output
                for(int i_o=0; i_o<this->neuronas[i_output].size(); i_o++)
                {
                    // Por cada neurona de la capa h1
                    for(int j=0; j<this->neuronas[i_h1].size(); j++)
                    {
                        // Calcular gradiente de peso
                        grad_w_aux = this->neuronas[i_h1][j];    //h1_out_j
                        this->grad_w[i_h1][j][i_o] += grad_x_output * grad_w_aux;
                    }
                }
                
                // Actualizar grad_bias capa h1
                this->grad_bias[i_h1] += grad_x_output;

                // Pesos input - h1

                vector<float> grads_x_h1;
                float h1_in;

                // Por cada neurona de h1
                for(int j=0; j<this->neuronas[i_h1].size(); j++)
                {
                    h1_in = 0;

                    // Por cada neurona de la capa input
                    for(int i=0; i<this->neuronas[0].size(); i++)
                    {
                        h1_in += this->neuronas[0][i] * this->w[0][i][j];
                    }

                    grads_x_h1.push_back(grad_x_output * this->w[i_h1][j][0] * deriv_relu(h1_in)); // 0 porque solo hay una neurona en la capa output0;
                }

                // Por cada neurona de la capa input
                for(int i=0; i<this->neuronas[0].size(); i++)
                {
                    // Por cada neurona de h1
                    for(int j=0; j<this->neuronas[i_h1].size(); j++)
                    {
                        // Calcular gradiente de peso
                        grad_w_aux = this->neuronas[0][i];    // x_(i_input)
                        this->grad_w[0][i][j] += grads_x_h1[j] * grad_w_aux;
                    }

                    
                }       

                // Por cada neurona de h1
                float sum_grad_x = 0.0;
                for(int j=0; j<this->neuronas[i_h1].size(); j++)
                    sum_grad_x += grads_x_h1[j];
                
                // Actualizar grad_bias capa input
                this->grad_bias[0] += sum_grad_x;
                

                // Gradiente de output respecto a input ---------------------------  

                // w[i][j][k] indica el peso que conecta la neurona j de la capa i con la neurona k de la capa i+1
                for(int i=0; i<this->neuronas[0].size(); i++) 
                    for(int j=0; j<this->neuronas[i_h1].size(); j++)
                        grad_x_i[i] = sum_grad_x * this->w[0][i][j];
                    
                grad_x.push_back(grad_x_i); 


            }

        }else
        {
            // Si solo hay capa input y output. Es decir, no hay capas ocultas
            if(this->neuronas.size() == 2)
            {
                for(int d=0; d<n_datos; d++)
                {
                    
                    forwardPropagation(x[d]);
                    prediccion = this->neuronas[i_output][0];

                    //cout << "p: " << prediccion << ", r: " << y[i] << endl;

                    o_in = 0;
                    
                    // Para obtener o_in, multiplicamos cada neurona de entrada j por el peso que la conecta con la capa output
                    for(int i=0; i<this->neuronas[0].size(); i++)
                    {
                        o_in += this->neuronas[0][i] * this->w[0][i][0];
                    }
                    o_in += this->bias[0]; 

                    grad_x_output = ((y[d]+epsilon) / (prediccion + epsilon)) - (1-y[d]+epsilon)/(1-prediccion+epsilon);
                    sig_o_in = sigmoid(o_in);
                    grad_x_output = grad_x_output * (sig_o_in * (1-sig_o_in+epsilon));
                    
                    //cout << "o_in: " << o_in << endl;
                    // Actualizar gradiente de pesos
                    for(int i=0; i<this->neuronas[0].size(); i++)
                    {
                        this->grad_w[0][i][0] += grad_x_output * x[d][i];
                    }
                    
                    // Actualizar gradiente de bias
                    this->grad_bias[0] += grad_x_output;    

                    
                    // Reset gradiente respecto a input
                    for(int i=0; i<this->neuronas[0].size(); i++)
                    {
                        grad_x_i[i] = 0;
                    }   

                    // Actualizar gradiente respecto a input               
                    for(int i=0; i<this->neuronas[0].size(); i++)
                    {
                        for(int j=0; j<this->neuronas[i_h1].size(); j++)
                        {
                            // w[i][j][k] indica el peso que conecta la neurona j de la capa i con la neurona k de la capa i+1
                            grad_x_i[i] += grad_x_output * this->w[0][i][j]; 
                            //cout << this->w[0][k][j] << " "; 
                        }  
                        //cout << grad_x_output << " ";  
                                            
                    }
                    //cout << "fin "<< endl;

                        
                    
                    grad_x.push_back(grad_x_i); 
                }

            }
        }
    }

    //cout << n_datos_reales << endl;

    // Realizar la media de los gradientes de pesos
    for(int i=0; i<this->w.size(); i++)
    {
        for(int j=0; j<this->w[i].size(); j++)
        {
            for(int k=0; k<this->w[i][j].size(); k++)
            {
                this->grad_w[i][j][k] = -this->grad_w[i][j][k] / n_datos;
            }
        }
    }

    // Realizar la media de los gradientes de bias
    for(int i=0; i<this->grad_bias.size(); i++)
    {
        this->grad_bias[i] = -this->grad_bias[i] / n_datos;
    }

    // Actualizar pesos
    for(int j=0; j<this->w.size(); j++)
    {
        for(int k=0; k<this->w[j].size(); k++)
        {
            for(int p=0; p<this->w[j][k].size(); p++)
            {
                this->w[j][k][p] -= this->lr * this->grad_w[j][k][p];
            }
        }
    }

    // ARREGLAR -----------------------------------------------------------------------------------
    //cout << endl << this->grad_w[0][0][0] << endl;
    // ARREGLAR -----------------------------------------------------------------------------------


    // Actualizar bias
    for(int i=0; i<this->bias.size(); i++)
    {
        this->bias[i] -= this->lr * this->grad_bias[i];
    }
        
    
    */

    


    







    /*
    // Normalizar pesos --------------------------------------
    for(int j=0; j<this->w.size(); j++)
    {
        for(int k=0; k<this->w[j].size(); k++)
        {
            for(int p=0; p<this->w[j][k].size(); p++)
            {
                if(this->w[j][k][p] < -1)
                    this->w[j][k][p] = -1;
                
                if(this->w[j][k][p] > 1)
                    this->w[j][k][p] = 1;
            }
        }
    }

    // Normalizar bias --------------------------------------
    for(int i=0; i<this->bias.size(); i++)
    {
        if(this->bias[i] < -1)
            this->bias[i] = -1;
        
        if(this->bias[i] > 1)
            this->bias[i] = 1;
    }
    */
    
}




/*
void FullyConnected::generarDatos(vector<vector<float>> &x, vector<float> &y)
{
    int n_datos = 1000, cont = 0; 
    float x1, x2, y_, aux, sum;
    vector<float> dato_x;
    x.clear();
    y.clear();

    
    for(int i=0; i<n_datos; i++)
    {
        dato_x.clear();
        sum = 0.0;

        // Generamos datos input
        for(int j=0; j<this->neuronas[0].size(); j++)
        {
            aux = rand() / float(RAND_MAX);
            dato_x.push_back(aux);

            // Generamos dato output tal que: if x+y >0, result=1, else result=0
            sum += aux;
        }

        x.push_back(dato_x);

        if(sum >=2)
            y.push_back(1);
        else
            y.push_back(0);
    }
}
*/

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
        for(int j=0; j<this->neuronas[0].size(); j++)
        {
            aux = rand() / float(RAND_MAX);
            aux = aux / this->neuronas[0].size();  // Para que esté en clase 0
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
        for(int j=0; j<this->neuronas[0].size(); j++)
        {
            aux = rand() / float(RAND_MAX);
            aux = aux + 1/this->neuronas[0].size();  // Para que esté en clase 0
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
                    v1D.push_back(imagen_k1[0][j][k]);
                }
            x.push_back(v1D);
            y.push_back(y_1D);
        }

        // Reset todo "y_1D" a 0
        y_1D[c] = 0.0;
    }  
}

/*
void FullyConnected::leer_imagenes_mnist(vector<vector<float>> &x, vector<float> &y)
{
    vector<vector<vector<float>>> imagen_k1;
    vector<float> v1D;

    //n_imagenes = 4000;
    int n_imagenes = 2000;

    x.clear();
    y.clear();

    // Leer imágenes
    for(int p=1; p<n_imagenes; p++)
    {

        // Leemos 0s
        string ruta_ini = "../../../fotos/mnist/training/0/";
        string ruta = ruta_ini + to_string(p) + ".jpg";

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
                v1D.push_back(imagen_k1[0][j][k]);
            }
        x.push_back(v1D);
        y.push_back(0);


        // Leemos 1s
        ruta_ini = "../../../fotos/mnist/training/0/";
        ruta = ruta_ini + to_string(p) + ".jpg";

        image2 = imread(ruta), image;

        image = image2;

        // Cargamos la imagen en un vector 3D
        cargar_imagen_en_vector(image, imagen_k1);

        // Normalizar imagen y pasar a 1D (solo queremos 1 canal porque son en blanco y negro)

        v1D.clear();
        for(int j=0; j<imagen_k1[0].size(); j++)
            for(int k=0; k<imagen_k1[0][0].size(); k++)
            {
                imagen_k1[0][j][k] = imagen_k1[0][j][k] / 255;
                v1D.push_back(imagen_k1[0][j][k]);
            }
        x.push_back(v1D);
        y.push_back(1);
    }  
}
*/

/*
int main()
{
    // Solo se meten capa input y capas ocultas, la capa output siempre tiene 1 neurona
    
    //vector<int> capas{4, 4, 2, 2};
    vector<int> capas{4, 2};
    vector<vector<float>> x, grad_x; 
    vector<float> y;

    FullyConnected n(capas, 0.1);

    n.generarDatos(x, y);
    //n.mostrarNeuronas();


    //int n_epocas = 28000;
    int n_epocas = 100000;
    //vector<float> do_back;

    for(int i=0; i<n_epocas; i++)
    {
        if(i % 1000 == 0)
        {
            cout << "Después de entrenar " << i << " épocas -----------------------------------" << endl;
            cout << "binary_loss: " << n.binary_loss(x, y) << endl;
            cout << "Accuracy: " << n.accuracy(x,y) << " %" << endl;
        }
        //n.train(x, y, grad_x, do_back);
        n.train(x, y, grad_x);
        
    }
    cout << "Después de entrenar " << n_epocas << " épocas -----------------------------------" << endl;
    cout << "binary_loss: " << n.binary_loss(x, y) << endl;
    cout << "Accuracy: " << n.accuracy(x,y) << " %" << endl;


    n.generarDatos(x, y);
    cout << endl << "Accuracy en TEST: " << n.accuracy(x,y) << " %" << endl;

    float sum;
    for(int k=496; k<506; k++)
    {
        sum = 0;
        cout << "¿";
        for(int l=0; l<x[0].size(); l++)
        {
            cout << x[k][l] << " + ";
            sum += x[k][l];
        }

        cout << " = " << sum << ", sum>= 1 --> y=1, sum < 1 --> y=0, " << endl;
        n.mostrar_prediccion_vs_verdad(x[k], y[k]);
        cout << "----------------------------" << endl;
    }
        

    //n.mostrarNeuronas();
    //n.mostrarpesos();

    //cout << "Grad_x.size() = " << grad_x.size() << endl;

    return 0;
}
*/