#include "fullyconnected_hidden.h"
#include <iostream>

using namespace std;

void FullyConnected_H::train(const vector<vector<float>> &x, const vector<float> &y, vector<vector<float>> &grad_x)
{
    int n_datos = x.size();
    float sum_b = 0.0, sum_w1 = 0.0, sum_w2 = 0.0, prediccion;
    float grad_w_aux, aux, epsilon = 0.000000001;

    int i_output = this->neuronas.size()-1; // índice de la capa output
    float sum, o_in, grad_x_output, sig_o_in;
    int i_last_h = i_output-1;  // Índice de la capa h1
    int i_h1 = 1, i_h2 = 2;
    int i_act, i_ant;

    // Ver máximo número de neuronas por capa
    int max=0;

    for(int i=0; i<this->neuronas.size(); i++)
        if(neuronas[i].size() > max)
            max = neuronas[i].size();

    vector<float> grads_output_h(max, 0.0);
    vector<float> grad_x_i(this->neuronas[0].size(), 0.0);

    grad_x.clear();

    // Inicializar gradiente de pesos a 0 --------------------------
    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                this->grad_w[i][j][k] = 0;

    // Inicializar gradiente bias a 0 ------------------------------
    for(int i=0; i<this->grad_bias.size(); i++)
        this->grad_bias[i] = 0;

    // Backpropagation ----------------------------------------------
    // Hay 2 o más capas ocultas
    for(int i=0; i<n_datos; i++)
    {
        forwardPropagation(x[i]);
        prediccion = this->neuronas[i_output][0];
        
        o_in = 0;

        // Por cada neurona j de i_last_h, sumamos i_last_h(out_j) * el peso que lo conecta con la capa output
        // Es decir, queremos obtener O_in (solo hay una neurona en la capa output)
        for(int j=0; j<this->neuronas[i_last_h].size(); j++)
            o_in += this->neuronas[i_last_h][j] * this->w[i_last_h][j][0];
        
        o_in += this->bias[i_last_h];

        grad_x_output = ((y[i]+epsilon) / (prediccion + epsilon)) - (1-y[i]+epsilon)/(1-prediccion+epsilon);

        sig_o_in = sigmoid(o_in);

        // Grad_x --> Hasta Oout / Oin
        grad_x_output = grad_x_output * (sig_o_in * (1-sig_o_in+epsilon));

        // Pesos h_last - Oin
        
        // Por cada neurona de la capa output
        for(int i_o=0; i_o<this->neuronas[i_output].size(); i_o++)
            for(int j=0; j<this->neuronas[i_last_h].size(); j++)    // Por cada neurona de la capa h_last                                
                this->grad_w[i_last_h][j][i_o] += grad_x_output* this->neuronas[i_last_h][j];   // grad_output * hlast_out_j
        
        // Actualizar grad_bias capa h_last
        this->grad_bias[i_last_h] += grad_x_output;

        // Calcular gradiente hasta última capa oculta
        float h_in;
        i_act = i_last_h;
        i_ant = i_last_h-1;
        
        for(int p=0; p<this->neuronas[i_act].size(); p++)
        {
            h_in = 0.0;
            
            for(int j=0; j<this->neuronas[i_ant].size(); j++)
                h_in += this->neuronas[i_ant][j] * this->w[i_ant][j][p];
            
            h_in += this->bias[i_ant];
            grads_output_h[p] = grad_x_output * this->w[i_last_h][p][0] * deriv_relu(h_in);
        }

        // Actualizar pesos penúltima capa oculta - última capa oculta
        for(int p=0; p<this->neuronas[i_act].size(); p++)
        {
            // Por cada neurona de la capa h_last
            for(int k=0; k<this->neuronas[i_ant].size(); k++)
                this->grad_w[i_ant][k][p] += grads_output_h[p]* this->neuronas[i_ant][k];   // grad_output * hlast_out_j
            

            this->grad_bias[i_act] += grads_output_h[p];
        }

        // Capas ocultas repetitivas --------------------------------
        for(int l=i_output-2; l>0; l--)
        {
            sum = 0.0;
            i_act = l;
            i_ant = l-1;
            for(int k=0; k<this->neuronas[i_act].size(); k++)
            {
                h_in = 0.0;
                
                for(int j=0; j<this->neuronas[i_ant].size(); j++)
                    h_in += this->neuronas[i_ant][j] * this->w[i_ant][j][k];
                
                h_in += this->bias[i_ant];
                
                // Suma del gradiente de la capa siguiente
                for(int p=0; p<grads_output_h.size(); p++)
                    sum += grads_output_h[p] * this->w[i_act][k][p];
                
                grads_output_h[k] = sum * deriv_relu(h_in);
            }
        }
        
        // Capa input
        for(int i_=0; i_<this->neuronas[0].size(); i_++)
        {
            sum = 0.0;
            for(int j=0; j<this->neuronas[1].size(); j++)
            {
                sum += grads_output_h[j] * this->w[0][i_][j];
                this->grad_bias[0] += grads_output_h[j];
            }

            grad_x_i[i_] = sum;
        }

        grad_x.push_back(grad_x_i); 
    }

    // Realizar la media de los gradientes de pesos
    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                this->grad_w[i][j][k] = -this->grad_w[i][j][k] / n_datos;
            
    // Realizar la media de los gradientes de bias
    for(int i=0; i<this->grad_bias.size(); i++)
        this->grad_bias[i] = -this->grad_bias[i] / n_datos;
    
    // Actualizar pesos
    for(int j=0; j<this->w.size(); j++)
        for(int k=0; k<this->w[j].size(); k++)
            for(int p=0; p<this->w[j][k].size(); p++)
                this->w[j][k][p] -= this->lr * this->grad_w[j][k][p];

    // Actualizar bias
    for(int i=0; i<this->bias.size(); i++)
        this->bias[i] -= this->lr * this->grad_bias[i];
    
}

/*
int main()
{
    // Solo se meten capa input y capas ocultas, la capa output siempre tiene 1 neurona
    
    //vector<int> capas{4, 4, 2, 2};
    vector<int> capas{4, 8, 2};
    vector<vector<float>> x, grad_x; 
    vector<float> y;

    FullyConnected_H n(capas, 0.001);

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