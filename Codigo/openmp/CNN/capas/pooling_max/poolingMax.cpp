#include "poolingMax.h"
#include <iostream>
#include <limits>
//#include "../../auxiliar/auxiliar.cpp"

using namespace std;

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

// Idea de input_copy --> Inicializar a 0. Cada kernel se quedará solo con 1 valor, pues lo pones a 1 en input_copy para luego saber cuál era al hacer backpropagation
// Suponemos que H y W son múltiplos de K
// Es decir, suponemos que tanto el ancho como el alto de la imagen de entrada "input" son múltiplos del tamaño del kernel a aplicar
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

void PoolingMax::mostrar_tam_kernel()
{
    cout << "Estructura kernel "<< this->kernel_fils << "x" << this->kernel_cols << "x" << this->image_canales << endl; 
}
/*
void aplicar_padding(vector<vector<vector<float>>> &imagen_3D, int pad)
{
    vector<vector<vector<float>>> imagen_3D_aux;
    vector<vector<float>> imagen_aux;
    vector<float> fila_aux;

    // Por cada imagen
    for(int i=0; i<imagen_3D.size();i++)
    {
        // Añadimos padding superior
        for(int j=0; j<imagen_3D[i].size() + pad*2; j++) // pad*2 porque hay padding tanto a la derecha como a la izquierda
            fila_aux.push_back(0.0);
        
        for(int k=0; k<pad; k++)
            imagen_aux.push_back(fila_aux);
        
        fila_aux.clear();

        // Padding lateral (izquierda y derecha)
        // Por cada fila de cada imagen
        for(int j=0; j<imagen_3D[i].size(); j++)
        {
            // Añadimos padding lateral izquierdo
            for(int t=0; t<pad; t++)
                fila_aux.push_back(0.0);

            // Dejamos casillas centrales igual que en la imagen original
            for(int k=0; k<imagen_3D[i][j].size(); k++)
                fila_aux.push_back(imagen_3D[i][j][k]);
            
            // Añadimos padding lateral derecho
            for(int t=0; t<pad; t++)
                fila_aux.push_back(0.0);
            
            // Añadimos fila construida a la imagen
            imagen_aux.push_back(fila_aux);
            fila_aux.clear();
        }
        
        // Añadimos padding inferior
        fila_aux.clear();

        for(int j=0; j<imagen_3D[i].size() + pad*2; j++) // pad*2 porque hay padding tanto a la derecha como a la izquierda
            fila_aux.push_back(0.0);
        
        for(int k=0; k<pad; k++)
            imagen_aux.push_back(fila_aux);
        
        fila_aux.clear();
        
        // Añadimos imagen creada al conjunto de imágenes
        imagen_3D_aux.push_back(imagen_aux);
        imagen_aux.clear();
    }

    imagen_3D = imagen_3D_aux;
};


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
    
    vector<vector<vector<float>>> imagenes_2D, output, v_3D, input_copy;
    vector<vector<float>> v_2D;
    vector<float> v_1D;
    int pad = 1;    // Padding se mete en capas convolucionales. Por tanto, si metemos padding de pad, antes de la capa pooling_max (input) hay que quitarlo al hacer backprop
    Aux *aux = new Aux();

    cout << "--------------- SIMULACIÓN GRADIENTE ------------------ " << endl;
    // Simulación. Viene gradiente de una capa convolucional con padding = pad

    // Imágenes input con dimensiones sin padding
    imagenes_2D.push_back(imagen);
    //imagenes_2D.push_back(imagen);

    // H_out = fils_img   -       K          + 1;
    int H_out = imagenes_2D[0].size() / 2;

    output.clear();
    v_1D.clear();
    v_2D.clear();
    for(int j=0; j<H_out; j++)
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

    // Aplicamos padding al output
    // Output viene ya con las dimensiones tras aplicar padding
    aplicar_padding(output, pad);

    cout << "------------ Imagen inicial: ------------" << endl;
    aux->mostrar_imagen(imagenes_2D);
    input_copy = imagenes_2D;

    PoolingMax plm1(2, 2, imagenes_2D);

    plm1.forwardPropagation(imagenes_2D, output, input_copy, pad);

    cout << "Output \n";
    aux->mostrar_imagen(output);

    cout << "------------ Pooling Max, Back Propagation: ------------" << endl;

    // Cambiamos el output porque contendrá un gradiente desconocido
    for(int i=0; i<output.size(); i++)
        for(int j=0; j<output[0].size(); j++)
            for(int k=0; k<output[0][0].size(); k++)
                output[i][j][k] = 9;

    //cout << "--- Output modificado ---- \n";
    //aux->mostrar_imagen(output);

    plm1.backPropagation(imagenes_2D, output, input_copy, pad);

    cout << "Input\n";
    aux->mostrar_imagen(imagenes_2D);

    return 0;
}
*/
