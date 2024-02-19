#include "convolutional.h"
#include <vector>
#include <math.h>
#include <iostream>
#include <chrono>

using namespace std;

void mostrar_imagen(vector<vector<float>> imagen)
{
    int n = imagen.size();

    for(int i=0; i<n; i++)
    {
        for(int j=0; j<imagen[i].size(); j++)
        {
            cout << imagen[i][j] << " ";
        }
        cout << endl;
    }
};

void mostrar_imagen(vector<vector<vector<float>>> imagenes_2D)
{
    int n = imagenes_2D.size();

    for(int k=0; k<n; k++)
    {
        cout << "IMAGEN " << k << endl;

        mostrar_imagen(imagenes_2D[k]);
        cout << endl;
    }

};

void mostrar_imagen(vector<vector<vector<vector<float>>>> imagenes)
{
    int n = imagenes.size();

    for(int k=0; k<n; k++)
    {
        cout << "Capa " << k << endl;

        mostrar_imagen(imagenes[k]);
        cout << endl;
    }

};


// Constructor
Convolutional::Convolutional(int n_kernels, int kernel_fils, int kernel_cols, const vector<vector<vector<float>>> &input, float lr)
{
    this->n_kernels = n_kernels;
    this->kernel_fils = kernel_fils;
    this->kernel_cols = kernel_cols;
    this->kernel_depth = input.size();
    this->image_fils = input[0].size();
    this->image_cols = input[0][0].size();
    this->lr = lr;


    vector<vector<vector<vector<float>>>> pesos_por_kernel;
    vector<vector<vector<float>>> pesos_totales;    // Representa los pesos de un kernel (los pesos de todas sus dimensiones)
    vector<vector<float>> pesos_kernel_2D;      // Representa los pesos de un kernel 2D
    vector<float> pesos_fila;               // Representa los pesos de una fila de un kernel 2D. Podría verse como un kernel 1D (un array)
    float random;
    
    // Inicialización pesos con valores random entre 0 y 1
    
    // Por cada imagen
    double varianza = 2.0 / (kernel_fils*kernel_cols*n_kernels);
    // Por cada kernel
    for(int f=0; f< n_kernels; f++)
    {
        for(int k=0; k< kernel_depth; k++)
        {
            for(int i=0; i<kernel_fils; i++)
            {
                for(int j=0; j<kernel_cols;j++)
                {
                    // Inicialización He
                    random = (rand() / double(RAND_MAX)) * sqrt(varianza);
                    pesos_fila.push_back(random);
                }
                pesos_kernel_2D.push_back(pesos_fila);
                pesos_fila.clear();
            }
            pesos_totales.push_back(pesos_kernel_2D);
            pesos_kernel_2D.clear();
        }
        pesos_por_kernel.push_back(pesos_totales);
        pesos_totales.clear();
    }     
    
    this->w = pesos_por_kernel;
    this->grad_w = this->w;


    // Bias    
    // Un bias por filtro, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks

    for(int i=0; i<n_kernels; i++)
        this->bias.push_back((float) (rand()) / (RAND_MAX));
    

    // Grad Bias
    this->grad_bias = this->bias;

};

float Convolutional::activationFunction(float x)
{
	// RELU
	return (x > 0.0) ? x : 0;
};

void Convolutional::aplicar_padding(vector<vector<vector<float>>> &imagen_3D, int pad)
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

void Convolutional::mostrar_pesos()
{
    
    for(int n=0; n<this->n_kernels; n++)
        for(int d=0; d<this->kernel_depth; d++)
        {
            for(int i=0; i<this->kernel_fils; i++)
            {
                for(int j=0; j<this->kernel_cols; j++)
                    cout << this->w[n][d][i][j] << " ";
                
                cout << endl;
            }
            cout << "Bias kernel " << n << ": " << this->bias[n] << endl;
            cout << endl << endl;
        }

    cout << endl; 
}

void Convolutional::mostrar_grad()
{
    
    for(int n=0; n<this->n_kernels; n++)
        for(int d=0; d<this->kernel_depth; d++)
            for(int i=0; i<this->kernel_fils; i++)
            {
                for(int j=0; j<this->kernel_cols; j++)
                    cout << this->grad_w[n][d][i][j] << " ";
                
                cout << endl;
            }

    cout << endl; 
}

// La salida es una imagen 2D, un canal de profundidad
void Convolutional::forwardPropagation(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output)
{
    // https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
    vector<vector<vector<float>>> input_copy;
    vector<vector<float>> conv_imagen;
    vector<float> conv_fila;
    
    // nº de kernels, nº de "capas 2D" por kernel, nº de filas del kernel, nº de columnas del kernel a aplicar
    int M = this->w.size(), depth_k = pesos[0].size(), K = pesos[0][0].size();

    // nº de "capas 2D",   nº de filas del volumen de entrada
    int C = input.size(), fils_img = input[0].size();

    // nº de veces que se deslizará el kernel sobre el volumen de entrada input
    // Suponemos nº filas = nº columnas
    int n_veces = fils_img - K + 1;

    if(C != depth_k)
    {
        cout << "Error. La profundidad del volumen del entrada no coincide con la profundidad de los kernels proporcionados. " << endl;
        exit(-1);
    }
        

    // Por cada kernel M 
    for(int img_out=0; img_out<M; img_out++)
        for(int i=0; i<n_veces; i++)    
            for(int j=0; j<n_veces; j++)  
            {
                output[img_out][i][j] = 0.0;
                
                // Realizar convolución 3D
                for(int c=0; c<C; c++)
                    for(int i_k=0; i_k<K; i_k++)
                        for(int j_k=0; j_k<K; j_k++)
                            output[img_out][i][j] += input[c][i+i_k][j+j_k] * this->w[img_out][c][i_k][j_k];                            

                // Sumamos bias a la suma ponderada obtenida
                output[img_out][i][j] = activationFunction(output[img_out][i][j] + this->bias[img_out]);
            }
};

void Convolutional::flip_w(vector<vector<vector<vector<float>>>> &w_flipped)
{
    w_flipped = w;

    // Por cada kernel
    for(int k=0; k<w.size(); k++)
        for(int d=0; d<w[k].size(); d++)    // Por cada canal del kernel
            for(int j=w[k][d].size()-1, j_w=0; j>=0; j--, j_w++)    // Por cada fila
                for(int c=w[k][d][j].size()-1, c_w=0; c>=0; c--, c_w++)
                    w_flipped[k][d][j_w][c_w] = w[k][d][j][c];
    
};

void Convolutional::reset_gradients()
{
    // Reset gradientes -------------------------------------------------------------------
    
    // Reset gradiende del bias
    for(int k=0; k<this->n_kernels; k++)
        this->grad_bias[k] = 0;

    // Reset del gradiente de cada peso
    for(int i=0; i<this->grad_w.size(); i++)
        for(int j=0; j<this->grad_w[0].size(); j++)
            for(int k=0; k<this->grad_w[0][0].size(); k++)
                for(int l=0; l<this->grad_w[0][0][0].size(); l++)
                    this->grad_w[i][j][k][l] = 0;
                
}


void Convolutional::backPropagation_libro(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output)
{
    // Cálculo de los gradientes ---------------------------------------------------------------------------------------------
   
    // Cálculo del gradiente de bias --------------------------------------

    // Cada kernel tiene un bias --> nº bias = nº kernels = nº de dimensiones del volumen output
    // Por cada kernel
    for(int k=0; k<this->n_kernels; k++)
    {
        // Tras cada correlación se obtiene una imagen 2D
        // En cada pixel de dicha imagen 2D influye el bias del kernel empleado
        // Es decir, el bias b del kernel k influye en cada píxel del output producido en la correlación del volumen de entrada
        // con el kernel k. De esta forma, b influye en cada píxel de la imagen 2D número k del volumen de salida

        // https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
        for(int j=0; j<output[k].size(); j++)
        {
            for(int p=0; p<output[k][j].size(); p++)
            {
                this->grad_bias[k] += output[k][j][p];
            }
        } 
    }

    
    // Cálculo del gradiente de pesos --------------------------------------
    int n_imgs_sep, tam_fil=this->w[0][0].size(), tam_col=this->w[0][0][0].size(), fil, col;
    int n_veces_fils = input[0].size() - tam_fil + 1;
    int n_veces_cols = input[0][0].size() - tam_col + 1;
    float sum;

    
    // Libro
    // ------------------------------------------------------------------------------------------
    
    int M = this->grad_w.size(), C = input.size(), K = this->kernel_fils;
    for(int m=0; m<M; m++)
    {
        for(int h=0; h<n_veces_fils; h++)
        {
            for(int w=0; w<n_veces_cols; w++)
            {
                for(int c=0; c<C; c++)
                {
                    for(int p=0; p<K; p++)
                    {
                        for(int q=0; q<K; q++)
                        {
                            this->grad_w[m][c][p][q] += input[c][h+p][w+q] * output[m][h][w];
                        }

                    }
                }
            }
        }
    }
    
    // ----------------------------------------------------------------------------------

        
    // Gradiente respecto a input -----------------------------
    
    // Reset gradiente respecto a input
    for(int i=0; i<input.size(); i++)
    {
        for(int j=0; j<input[0].size(); j++)
        {
            for(int k=0; k<input[0][0].size(); k++)
            {
                input[i][j][k] = 0;
            }
        }
    } 

    // Libro
    // ------------------------------------------------------------------------------------------
    int H = input[0].size();
    int W = input[0][0].size();

    for(int m=0; m<M; m++)
    {
        for(int h=0; h<H-1; h++)
        {
            for(int w=0; w<W-1; w++)
            {
                for(int c=0; c<C; c++)
                {
                    for(int p=0; p<K; p++)
                    {
                        for(int q=0; q<K; q++)
                        {
                            if(h-p >= 0 && w-p >= 0 && h-p < n_veces_fils && w-p < n_veces_cols)
                                input[c][h][w] += output[m][h-p][w-p] * this->w[m][c][K-1-p][K-1-q];
                        }
                    }
                }
            }
        }
    }

}

void Convolutional::backPropagation_sin_padding(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output)
{
    // Cálculo de los gradientes ---------------------------------------------------------------------------------------------
   
    // Cálculo del gradiente de bias --------------------------------------

    // Cada kernel tiene un bias --> nº bias = nº kernels = nº de dimensiones del volumen output
    // Por cada kernel
    for(int k=0; k<this->n_kernels; k++)
    {
        // Tras cada correlación se obtiene una imagen 2D
        // En cada pixel de dicha imagen 2D influye el bias del kernel empleado
        // Es decir, el bias b del kernel k influye en cada píxel del output producido en la correlación del volumen de entrada
        // con el kernel k. De esta forma, b influye en cada píxel de la imagen 2D número k del volumen de salida

        // https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
        for(int j=0; j<output[k].size(); j++)
            for(int p=0; p<output[k][j].size(); p++)
                this->grad_bias[k] += output[k][j][p]; 
    }
    
    // Cálculo del gradiente de pesos --------------------------------------
    int n_imgs_sep, tam_fil=this->w[0][0].size(), tam_col=this->w[0][0][0].size(), fil, col;

    // Suponemos filas = columnas
    int n = input[0].size() - output[0].size() + 1;

    // Número de "imágenes 2D" de la salida es igual al número de filtros que tengamos en la capa anterior, tam_output
    // tam_input se refiere al número de canales o "imágenes 2D" que tiene la imagen input
    // También suponemos que solo usamos filtros con nºcols = nº fils

    int tam_output = output.size(), tam_input = input.size(), K = output[0].size();
    for(int img_out=0; img_out<tam_output; img_out++)
        for(int c=0; c<tam_input; c++)
            for(int i=0; i<n; i++)
                for(int j=0; j<n; j++)
                    for(int i_k=0; i_k<K; i_k++)
                        for(int j_k=0; j_k<K; j_k++)
                            this->grad_w[img_out][c][i][j] += input[c][i+i_k][j+j_k] * output[img_out][i_k][j_k];
                        
        
    // Gradiente respecto a input -----------------------------
    
    // Reset gradiente respecto a input
    for(int i=0; i<input.size(); i++)
        for(int j=0; j<input[0].size(); j++)
            for(int k=0; k<input[0][0].size(); k++)
                input[i][j][k] = 0;

    // Invertimos los pesos
    vector<vector<vector<vector<float>>>> w_flipped;
    flip_w(w_flipped);
    
    // Añadimos padding al output
    vector<vector<vector<float>>> output_con_pad;
    output_con_pad = output;
    aplicar_padding(output_con_pad, K-1);

    n = input[0].size();

    for(int c=0; c<tam_input; c++)
        for(int img_out=0; img_out<tam_output; img_out++)
            for(int i=0; i<n; i++)
                for(int j=0; j<n; j++)
                    for(int i_k=0; i_k<tam_fil; i_k++)
                        for(int j_k=0; j_k<tam_fil; j_k++)
                            input[c][i][j] += this->w[img_out][c][i_k][j_k] * output[img_out][i+i_k][j+j_k];
       
}

void Convolutional::backPropagation_con_padding(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output, const int &pad)
{
    // Cálculo de los gradientes ---------------------------------------------------------------------------------------------
   
    // Cálculo del gradiente de bias --------------------------------------

    // Cada kernel tiene un bias --> nº bias = nº kernels = nº de dimensiones del volumen output
    // Por cada kernel
    for(int k=0; k<this->n_kernels; k++)
    {
        // Tras cada correlación se obtiene una imagen 2D
        // En cada pixel de dicha imagen 2D influye el bias del kernel empleado
        // Es decir, el bias b del kernel k influye en cada píxel del output producido en la correlación del volumen de entrada
        // con el kernel k. De esta forma, b influye en cada píxel de la imagen 2D número k del volumen de salida

        // https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
        for(int j=0; j<output[k].size(); j++)
            for(int p=0; p<output[k][j].size(); p++)
                this->grad_bias[k] += output[k][j][p];
    }
    
    // Cálculo del gradiente de pesos --------------------------------------
    int n_imgs_sep, tam_fil=this->w[0][0].size(), tam_col=this->w[0][0][0].size(), fil, col;

    // Suponemos filas = columnas

    // Número de "imágenes 2D" de la salida es igual al número de filtros que tengamos en la capa anterior, tam_output
    // tam_input se refiere al número de canales o "imágenes 2D" que tiene la imagen input
    // También suponemos que solo usamos filtros con nºcols = nº fils

    int tam_output = output.size(), tam_input = input.size(), K = input[0].size() - pad;
    int n = input[0].size() - output[0].size() + 1, j_esq, i_esq, i, j, i_k, j_k;

    for(int img_out=0; img_out<tam_output; img_out++)
        for(int c=0; c<tam_input; c++)
        {
            i=0;

            // Primera fila ------------------------------------------------
            for(j=0; j<n-1; j++)
                for(i_k=0; i_k<K; i_k++)
                    for(j_k=0; j_k<K; j_k++)
                        this->grad_w[img_out][c][i][j] += output[img_out][pad + i_k][pad - j + j_k] * input[c][pad +i_k][pad +j_k];         
            
            // Forzamos la posición de la esquina de arriba a la izquierda para la última columna
            j_esq = output[0].size() - pad - input[0].size();
            i_esq = pad;
            
            for(i_k=0; i_k<K; i_k++)
                for(j_k=0; j_k<K; j_k++)
                    this->grad_w[img_out][c][i_esq][j] += output[img_out][i_esq + i_k][j_esq + j_k] * input[c][pad +i_k][pad +j_k];
        
            
            // Filas intermedias ------------------------------------------------
            for(i=1; i<n-1; i++)
                for(j=0; j<n-1; j++)
                    for(i_k=0; i_k<K; i_k++)
                        for(j_k=0; j_k<K; j_k++)
                            this->grad_w[img_out][c][i][j] += output[img_out][pad - i + i_k][pad - j + j_k] * input[c][pad + i_k][pad + j_k];

            // Forzamos la posición de la esquina de arriba a la izquierda para la última columna
            j_esq = output[0].size() - pad - input[0].size();
            
            for(i_k=0; i_k<K; i_k++)
                for(j_k=0; j_k<K; j_k++)
                    this->grad_w[img_out][c][i][j] += output[img_out][pad -i + i_k][j_esq + j_k] * input[c][pad +i_k][pad +j_k];
        

            
            // Última fila ------------------------------------------------
            i_esq = output[0].size() - input[0].size();
            for(j=0; j<n-1; j++)
                for(i_k=0; i_k<K; i_k++)
                    for(j_k=0; j_k<K; j_k++)
                        this->grad_w[img_out][c][i_esq][j] += output[img_out][i_esq + i_k][pad - j + j_k] * input[c][pad + i_k][pad + j_k];
            
            // Forzamos la posición de la esquina de arriba a la izquierda para la última columna
            j_esq = output[0].size() - pad - input[0].size();
            
            for(i_k=0; i_k<K; i_k++)
                for(j_k=0; j_k<K; j_k++)
                    this->grad_w[img_out][c][i_esq][j] += output[img_out][i_esq + i_k][j_esq + j_k] * input[c][pad +i_k][pad +j_k];

        }

    // Gradiente respecto a input -----------------------------

    // Reset gradiente respecto a input
    for(int i=0; i<input.size(); i++)
        for(int j=0; j<input[0].size(); j++)
            for(int k=0; k<input[0][0].size(); k++)
                input[i][j][k] = 0;

    // Invertimos los pesos
    vector<vector<vector<vector<float>>>> w_flipped;
    flip_w(w_flipped);
    
    // Añadimos padding al output
    vector<vector<vector<float>>> output_con_pad;
    output_con_pad = output;
    aplicar_padding(output_con_pad, tam_fil-1);

    n = input[0].size();

    for(int c=0; c<tam_input; c++)
        for(int img_out=0; img_out<tam_output; img_out++)
            for(int i=0; i<n; i++)
                for(int j=0; j<n; j++)
                    for(int i_k=0; i_k<tam_fil; i_k++)
                        for(int j_k=0; j_k<tam_fil; j_k++)
                            input[c][i][j] += this->w[img_out][c][i_k][j_k] * output_con_pad[img_out][i+i_k][j+j_k];
}

void Convolutional::backPropagation(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output, const int &pad)
{
    if(pad == 0)
        backPropagation_sin_padding(input, output);
    else
    {
        if(output[0].size() - pad >= input[0].size())
        {
            cout << "ERROR. Caso 3 sin implementar. \n";
            exit(-1);
        }else
            backPropagation_con_padding(input, output, pad);
    }
    
}

void Convolutional::actualizar_grads(int n_datos)
{
    //cout << "Paso 1: " << this->grad_w[0][0][0][0] << endl;
    // Realizar la media de los gradientes de pesos
    for(int i=0; i<this->grad_w.size(); i++)
        for(int j=0; j<this->grad_w[i].size(); j++)
            for(int k=0; k<this->grad_w[i][j].size(); k++)
                for(int l=0; l<this->grad_w[i][j][k].size(); l++)
                    this->grad_w[i][j][k][l] = -this->grad_w[i][j][k][l] / n_datos;

    // Actualizar pesos
    for(int j=0; j<this->w.size(); j++)
        for(int k=0; k<this->w[j].size(); k++)
            for(int p=0; p<this->w[j][k].size(); p++)
                for(int l=0; l<this->grad_w[j][k][p].size(); l++)
                    this->w[j][k][p][l] += this->lr * this->grad_w[j][k][p][l];

    // Realizar la media de los gradientes de bias
    for(int i=0; i<this->grad_bias.size(); i++)
        this->grad_bias[i] = -this->grad_bias[i] / n_datos;
    
    // Actualizar bias
    for(int i=0; i<this->bias.size(); i++)
        this->bias[i] += this->lr * this->grad_bias[i];
    
}

// https://towardsdatascience.com/forward-and-backward-propagation-in-convolutional-neural-networks-64365925fdfa
// https://colab.research.google.com/drive/13MLFWdi3uRMZB7UpaJi4wGyGAZ9ftpPD?authuser=1#scrollTo=FEFgOKF4gGv2
