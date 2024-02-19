#include "fullyconnected_no_hidden.h"
#include <iostream>

using namespace std;

void FullyConnected_NoH::train(const vector<vector<float>> &x, const vector<float> &y, vector<vector<float>> &grad_x)
{
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
    {
        grad_x_i.push_back(0);
    }


    // Inicializar gradiente de pesos a 0 --------------------------
    for(int i=0; i<this->w.size(); i++)
    {
        for(int j=0; j<this->w[i].size(); j++)
        {
            for(int k=0; k<this->w[i][j].size(); k++)
            {
                this->grad_w[i][j][k] = 0;
            }
        }
    }

    // Inicializar gradiente bias a 0 ------------------------------
    for(int i=0; i<this->grad_bias.size(); i++)
    {
        this->grad_bias[i] = 0;
    }

    // Backpropagation ----------------------------------------------
    // Solo hay capa input y output. Es decir, no hay capas ocultas
    for(int d=0; d<n_datos; d++)
    {
        
        forwardPropagation(x[d]);
        prediccion = this->neuronas[i_output][0];

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
            }  
                                
        }   
        
        grad_x.push_back(grad_x_i); 
    }

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

    // Actualizar bias
    for(int i=0; i<this->bias.size(); i++)
    {
        this->bias[i] -= this->lr * this->grad_bias[i];
    }
}


int main()
{
    // Solo se meten capa input y capas ocultas, la capa output siempre tiene 1 neurona
    
    //vector<int> capas{4, 4, 2, 2};
    vector<int> capas{4};
    vector<vector<float>> x, grad_x; 
    vector<float> y;

    FullyConnected_NoH n(capas, 0.1);

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