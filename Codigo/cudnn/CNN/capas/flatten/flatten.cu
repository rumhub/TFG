#include "flatten.h"

Flatten::Flatten(const vector<vector<vector<float>>> &input)
{              
    this->canales = input.size();
    this->filas = input[0].size();
    this->cols = input[0][0].size();
    this->max = numeric_limits<float>::max();
};

// input --> volumen 3D
void Flatten::forwardPropagation(const vector<vector<vector<float>>> &input, vector<float> &output)
{                 
    int n1 = input.size(), n2 = input[0].size(), n3 = input[0][0].size(), n_out = n1*n2*n3, cont=0;
    vector<float> out(n_out);

    for(int i=0; i<n1; ++i)
        for(int j=0; j<n2; ++j)
            for(int k=0; k<n3; ++k, ++cont)
                out[cont] = input[i][j][k];

    output = out;
};



void Flatten::backPropagation(vector<vector<vector<float>>> &errores_matriz, const vector<float> &errores_primera_capa)
{
    vector<float> v_1D(this->cols);
    vector<vector<float>> v_2D(this->filas);
    vector<vector<vector<float>>> v_3D(this->canales);

    // Para cada canal de cada imagen
    for(int i=0; i<this->canales; ++i)
    {
        for(int j=0; j<this->filas; ++j)
        {
            for(int k=0; k<this->cols; ++k)
            {
                v_1D[k] = errores_primera_capa[i*this->filas*this->cols + j*this->cols + k];
            }
            v_2D[j] = v_1D;
        }
        v_3D[i] = v_2D;
    }

    errores_matriz = v_3D;
    
};

int Flatten::get_canales()
{
    return this->canales;
}
