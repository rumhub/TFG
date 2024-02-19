#include "poolingMax.h"
#include <iostream>

using namespace std;

PoolingMax::PoolingMax(int kernel_fils, int kernel_cols, const vector<vector<vector<float>>> &input)
{
    this->kernel_fils = kernel_fils;
    this->kernel_cols = kernel_cols;
    this->image_fils = input[0].size();
    this->image_cols = input[0][0].size();
    this->image_canales = input.size();
};


// Suponemos que H y W son múltiplos de K
// Es decir, suponemos que tanto el ancho como el alto de la imagen de entrada "input" son múltiplos del tamaño del kernel a aplicar
void PoolingMax::forwardPropagation(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output)
{
    int M = input.size(), K=kernel_fils, n_veces_fils = input[0].size() / K , n_veces_cols = input[0][0].size() / K;
    float max;

    for(int m=0; m<M; m++) // Para cada canal
        for(int h=0; h<n_veces_fils; h++) // Se calcula el nº de desplazamientos del kernel a lo largo del volumen de entrada 3D
            for(int w=0; w<n_veces_cols; w++)  
            {
                max = 0.0;

                // Para cada subregión, realizar el pooling
                for(int p=0; p<K; p++)
                    for(int q=0; q<K; q++)
                    {
                        if(input[m][h*K + p][w*K + q] > max)
                            max = input[m][h*K + p][w*K + q];
                            
                    }
                
                output[m][h][w] = max;
            }
};


void PoolingMax::backPropagation(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output, const vector<vector<vector<float>>> &grad_output)
{
    int n_canales = this->image_canales, n_veces_fils = this->image_fils / kernel_fils, n_veces_cols = this->image_cols / kernel_cols;
    int fila, columna;
    float max = 0.0, grad;
    int output_fil = 0, output_col = 0;

    // Para cada imagen 2D
    for(int t=0; t<n_canales; t++)
    {
        output_fil = 0;
        for(int i=0; i<n_veces_fils*kernel_fils; i+=kernel_fils, output_fil++)
        {
            output_col = 0;
            // En este momento imagenes_2D[t][i][j] contiene la casilla inicial del kernel en la imagen t para realizar el pooling
            // Casilla inicial = esquina superior izquierda
            for(int j=0; j<n_veces_cols*kernel_cols; j+=kernel_cols, output_col++)  
            {
                max = output[t][output_fil][output_col];
                grad = grad_output[t][output_fil][output_col];

                // Para cada subregión, realizar el pooling
                for(int k=i; k<(i+kernel_fils)  && k<this->image_fils; k++)
                {
                    for(int h=j; h<(j+kernel_cols) && h<this->image_cols; h++)
                    {
                        // Si no es el valor máximo, asignar un 0.0
                        // Si es el valor máximo, dejarlo como estaba
                        if(input[t][k][h] < max)
                            input[t][k][h] = 0.0;
                        else
                            input[t][k][h] = grad;
                    }
                }
            }
            
        }
    }
};

void PoolingMax::mostrar_tam_kernel()
{
    cout << "Estructura kernel "<< this->kernel_fils << "x" << this->kernel_cols << "x" << this->image_canales << endl; 
}

/*

// https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/pooling_layer
int main() 
{
    
    // Ejemplo de uso
    vector<vector<float>> imagen = {
        {1.0, 1.0, 2.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {3.0, 2.0, 1.0, 0.0},
        {1.0, 2.0, 3.0, 4.0}
    };
    
    vector<vector<vector<float>>> imagenes_2D, output, grad_output, v_3D;
    vector<vector<float>> v_2D;
    vector<float> v_1D;

    imagenes_2D.push_back(imagen);
    imagenes_2D.push_back(imagen);


    // H_out = fils_img   -       K          + 1;
    int H_out = imagenes_2D[0].size() / 2;
    int W_out = imagenes_2D[0][0].size() / 2;


    output.clear();
    for(int j=0; j<W_out; j++)
    {
        v_1D.push_back(0.0);
    }

    for(int j=0; j<H_out; j++)
    {
        v_2D.push_back(v_1D);
    }


    for(int j=0; j<imagenes_2D.size(); j++)
    {
        output.push_back(v_2D);
    }
    

    cout << "------------ Imagen inicial: ------------" << endl;
    mostrar_imagenes_2D(imagenes_2D);

    PoolingMax plm(2, 2, imagenes_2D);
    
    plm.forwardPropagation(imagenes_2D, output);
    
    cout << "------------ Pooling Max, Forward Propagation: ------------" << endl;
    mostrar_imagenes_2D(output);

    grad_output = output;
    int p=0;
    // Inicializar grad_output todo a 10
    for(int i=0; i<grad_output.size(); i++)
    {
        for(int j=0; j<grad_output[i].size(); j++)
        {
            for(int k=0; k<grad_output[i][j].size(); k++)
            {
                grad_output[i][j][k] = p++;
            }
        }
    }




    cout << "------------ Pooling Max, Back Propagation: ------------" << endl;
    plm.backPropagation(imagenes_2D, output, grad_output);
    mostrar_imagenes_2D(imagenes_2D);
    
    cout << "Grad_output\n";
    mostrar_imagenes_2D(grad_output);


    return 0;
}
*/
