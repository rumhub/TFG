#include "poolingMax.h"
#include <iostream>
#include <limits>

using namespace std;

/*
    @brief: Constructor
    @kernel_fils: Número de filas del kernel o ventana
    @kernel_cols: Nümero de columnas del kernel o ventana
    @input:       Entrada de la capa
*/
PoolingMax::PoolingMax(int kernel_fils, int kernel_cols, vector<vector<vector<float>>> &input)
{
    this->kernel_fils = kernel_fils;
    this->kernel_cols = kernel_cols;
    this->image_fils = input[0].size();
    this->image_cols = input[0][0].size();
    this->image_canales = input.size();
    this->n_filas_eliminadas = 0;

    if(this->image_fils % kernel_fils != 0 || this->image_cols % kernel_cols != 0)
        cout << "Warning. Las dimensiones del volumen de entrada(" << this->image_fils << ") no son múltiplos del kernel max_pool(" << kernel_fils << "). \n";
    

};

/*
    @brief: Propagación hacia delante en una capa de agupación máxima
    @input: Entrada de la capa
    @output: Salida de la capa
    @input_copy: Copia de la entrada (solo tiene que tener las mismas dimensiones, no los mismos valores)
    @pad: Padding o relleno de la capa convolucional conectada
*/
void PoolingMax::forwardPropagation(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad)
{
    int M = input.size(), K=kernel_fils, n_veces_fils = input[0].size() / K , n_veces_cols = input[0][0].size() / K;
    float max;
    
    if(input_copy.size() != input.size() || input_copy[0].size() != input[0].size() || input_copy[0][0].size() != input[0][0].size())
    {
        cout << "Error. En la capa Max_Pool no coinciden las dimensiones de input e input_copy. " << endl;
        exit(-1);
    }

    // Inicializamos input_copy a 0
    for(int i=0; i<input_copy.size(); ++i)
    for(int j=0; j<input_copy[0].size(); ++j)
        for(int k=0; k<input_copy[0][0].size(); ++k)
            input_copy[i][j][k] = 0.0;

    int i_m, j_m;
    bool encontrado;

    for(int m=0; m<M; ++m) // Para cada canal
        for(int h=0; h<n_veces_fils; ++h) // Se calcula el nº de desplazamientos del kernel a lo largo del volumen de entrada 3D
            for(int w=0; w<n_veces_cols; ++w)  
            {
                max = numeric_limits<float>::min();
                encontrado = false;
                // Para cada subregión, realizar el pooling
                for(int p=0; p<K; ++p)
                    for(int q=0; q<K; ++q)
                    {
                        if(input[m][h*K + p][w*K + q] > max)
                        {
                            max = input[m][h*K + p][w*K + q];
                            i_m = h*K + p;
                            j_m = w*K + q;
                            encontrado = true;
                        }
                            
                    }
                
                output[m][pad+h][pad+w] = max;

                if(encontrado)
                    input_copy[m][i_m][j_m] = 1.0;
            }
};

/*
    @brief: Retropropagación en una capa de agupación máxima
    @input: Entrada de la capa
    @output: Salida de la capa
    @input_copy: Copia de la entrada (solo tiene que tener las mismas dimensiones, no los mismos valores)
    @pad_output: Padding o relleno de la capa convolucional conectada
*/
void PoolingMax::backPropagation(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad_output)
{
    int n_canales = this->image_canales, n_veces_fils = this->image_fils / kernel_fils, n_veces_cols = this->image_cols / kernel_cols;
    int fila, columna;
    float max = 0.0;
    int output_fil = 0, output_col = 0;

    // Inicializar imagen de entrada a 0
    for(int i=0; i<input.size(); ++i)
        for(int j=0; j<input[0].size(); ++j)
            for(int k=0; k<input[0][0].size(); ++k)
                input[i][j][k] = 0.0;

    // Para cada imagen 2D
    for(int t=0; t<n_canales; ++t)
    {
        output_fil = 0;
        for(int i=0; i<n_veces_fils*kernel_fils; i+=kernel_fils, ++output_fil)
        {
            output_col = 0;
            // En este momento imagenes_2D[t][i][j] contiene la casilla inicial del kernel en la imagen t para realizar el pooling
            // Casilla inicial = esquina superior izquierda
            for(int j=0; j<n_veces_cols*kernel_cols; j+=kernel_cols, ++output_col)  
            {
                max = output[t][pad_output + output_fil][pad_output + output_col];

                // Para cada subregión, realizar el pooling
                for(int k=i; k<(i+kernel_fils)  && k<this->image_fils; ++k)
                {
                    for(int h=j; h<(j+kernel_cols) && h<this->image_cols; ++h)
                    {
                        // Si es el valor máximo, dejarlo como estaba
                        if(input_copy[t][k][h] != 0)
                            input[t][k][h] = max;
                        else
                            input[t][k][h] = 0.0;
                    }
                }
            }
            
        }
    }
    
};
