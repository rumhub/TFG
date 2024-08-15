#include <vector>
#include "flatten.h"
#include <iostream>
#include <chrono>
#include <limits>

using namespace std::chrono;
using namespace std;

/*
    @brief: Constructor
    @input: Volumen 3D de entrada
*/
Flatten::Flatten(const vector<vector<vector<float>>> &input)
{              
    this->canales = input.size();
    this->filas = input[0].size();
    this->cols = input[0][0].size();
};


/*
    @brief: Propagación hacia delante en una capa de aplanado
    @input: Volumen 3D de entrada
    @output: Vector 1D de salida
*/
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


/*
    @brief: Retropropagación en una capa de aplanado
    @errores_primera_capa: Vector 1D de entrada
    @errores_matriz: Volumen 3D de salida
*/
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
