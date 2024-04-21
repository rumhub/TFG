#include "convolutional.h"
#include <vector>
#include <math.h>
#include <iostream>
#include <chrono>
#include "random"
#include "omp.h"

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

// Debug -------
void Convolutional::w_a_1()
{
    vector<vector<vector<vector<float>>>> w;    // w[n][d][i][j]   --> Matriz de pesos([d][i][j]) respecto al kernel n. (d = depth del kernel)

    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[0].size(); j++)
            for(int k=0; k<this->w[0][0].size(); k++)
                for(int m=0; m<this->w[0][0][0].size(); m++)
                    this->w[i][j][k][m] = 1;

}


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
                    pesos_fila.push_back(0.0);
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

    this->generar_pesos();

    // Bias    
    // Un bias por filtro, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks

    for(int i=0; i<n_kernels; i++)
        this->bias.push_back(0.0);
    
};

void Convolutional::generar_pesos() 
{
    // Inicialización He
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> distribution(0.0, sqrt(2.0 / (this->n_kernels * this->kernel_depth * this->kernel_fils * this->kernel_fils)));

    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[0].size(); j++)
            for(int k=0; k<this->w[0][0].size(); k++)
                for(int p=0; p<this->w[0][0][0].size(); p++)
                    this->w[i][j][k][p] = distribution(gen);
}

float Convolutional::activationFunction(float x)
{
	// RELU
	return (x > 0.0) ? x : 0;
};

float Convolutional::deriv_activationFunction(float x)
{
    float result = 0.0;

    if(x > 0)
        result = 1;
    
    return result;
}

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

// a --> // Convolución antes de aplicar la función de activación
// La salida es una imagen 2D, un canal de profundidad
void Convolutional::forwardPropagation(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &a)
{
    // https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
    vector<vector<vector<float>>> input_copy;
    vector<vector<float>> conv_imagen;
    vector<float> conv_fila;
    
    // nº de kernels, nº de "capas 2D" por kernel, nº de filas del kernel, nº de columnas del kernel a aplicar
    int M = this->w.size(), depth_k = this->w[0].size(), K = this->w[0][0].size();

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
    
    a = output;

    // Por cada kernel M 
    for(int img_out=0; img_out<M; img_out++)
        for(int i=0; i<n_veces; i++)    
            for(int j=0; j<n_veces; j++)  
            {
                a[img_out][i][j] = 0.0;
                
                // Realizar convolución 3D
                for(int c=0; c<C; c++)
                    for(int i_k=0; i_k<K; i_k++)
                        for(int j_k=0; j_k<K; j_k++)
                            a[img_out][i][j] += input[c][i+i_k][j+j_k] * this->w[img_out][c][i_k][j_k];                            

                // Sumamos bias a la suma ponderada obtenida
                a[img_out][i][j] += this->bias[img_out];


                // Aplicamos función de activación
                output[img_out][i][j] = activationFunction(a[img_out][i][j]);
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

void Convolutional::reset_gradients(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias)
{
    // Reset gradientes -------------------------------------------------------------------
    
    // Reset gradiende del bias
    for(int k=0; k<this->n_kernels; k++)
        grad_bias[k] = 0.0;

    // Reset del gradiente de cada peso
    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[0].size(); j++)
            for(int k=0; k<this->w[0][0].size(); k++)
                for(int l=0; l<this->w[0][0][0].size(); l++)
                    grad_w[i][j][k][l] = 0.0;
                
}


// a --> // Convolución antes de aplicar la función de activación
// OJO --> El input viene ya con el padding aplicado
void Convolutional::backPropagation(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> output, vector<vector<vector<float>>> &a, vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias, const int &pad)
{
    // https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
    vector<vector<vector<float>>> output_copy, grad_input;
    vector<vector<float>> conv_imagen;
    vector<float> conv_fila;
    
    // nº de kernels, nº de filas del kernel, nº de columnas del kernel a aplicar
    int F = this->w.size(), K = this->w[0][0].size(), tam_fil_y = output[0][0].size();

    // nº de "capas 2D",   nº de filas del volumen de entrada
    int C = input.size();

    grad_input = input;

    // Inicializar input a 0
    for(int i=0; i<input.size(); i++)
        for(int j=0; j<input[0].size(); j++)    
            for(int k=0; k<input[0][0].size(); k++)
                grad_input[i][j][k] = 0.0;

    // Realizar derivada Y_out/Y_in
    for(int i=0; i<output.size(); i++)
        for(int j=0; j<output[0].size(); j++)    
            for(int k=0; k<output[0][0].size(); k++)
                output[i][j][k] = output[i][j][k] * deriv_activationFunction(a[i][j][k]);
    
    // Aplicar padding
    output_copy = output;
    aplicar_padding(output_copy, pad + K-2);    // Se añade + K-2 para hacer una convolución completa
    
    // Suponemos nº filas = nº columnas
    // nº veces k sobre y, nº de veces o deslizamientos de y sobre x
    int n_veces_ky = output_copy[0].size() - K + 1, n_veces_yx = input[0].size() - output[0].size() + 1;

    // Convolución entre output y pesos    
    for(int f=0; f<F; f++)  // Por cada filtro f
    {
        // Gradiente respecto a entrada
        for(int i=0; i<n_veces_ky; i++)    
            for(int j=0; j<n_veces_ky; j++)    
                for(int c=0; c<C; c++)  // Convolución entre salida y pesos invertidos
                    for(int i_k=0; i_k<K; i_k++)
                        for(int j_k=0; j_k<K; j_k++)
                            grad_input[c][i][j] += output_copy[f][i+i_k][j+j_k] * this->w[f][c][K -1 - i_k][K -1 - j_k];                            
        
        // Gradiente respecto a pesos
        for(int i=0; i<n_veces_yx; i++)    
            for(int j=0; j<n_veces_yx; j++)  
                for(int c=0; c<C; c++)  // Convolución entre entrada y salida
                    for(int i_k=0; i_k<tam_fil_y; i_k++)
                        for(int j_k=0; j_k<tam_fil_y; j_k++)
                            grad_w[f][c][i][j] += input[c][i + i_k][j + j_k] * output[f][i_k][j_k];
    }


    // Gradiente respecto a entrada
    for(int i=0; i<input.size(); i++)
        for(int j=0; j<input[0].size(); j++)    
            for(int k=0; k<input[0][0].size(); k++)
                input[i][j][k] = grad_input[i][j][k];

    // Calcular el gradiente del bias
    for(int i=0; i<output.size(); i++)
        for(int j=0; j<output[0].size(); j++)    
            for(int k=0; k<output[0][0].size(); k++)
                grad_bias[i] += output[i][j][k];
};


void Convolutional::escalar_pesos(float clip_value)
{
    // Calculate the maximum and minimum values of weights
    float max = this->w[0][0][0][0], min = this->w[0][0][0][0];

    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                for(int l=0; l<this->w[i][j][k].size(); l++)
                {
                    if(max < this->w[i][j][k][l])
                        max = this->w[i][j][k][l];
                    
                    if(min > this->w[i][j][k][l])
                        min = this->w[i][j][k][l];
                }

    // Perform gradient clipping
    float scaling_factor = clip_value / std::max(std::abs(max), std::abs(min));
    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                for(int l=0; l<this->w[i][j][k].size(); l++)
                    this->w[i][j][k][l] = std::max(std::min(this->w[i][j][k][l] * scaling_factor, clip_value), -clip_value);
}

void Convolutional::actualizar_grads(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias)
{
    int n_thrs = 8, thr_id = omp_get_thread_num(), n_imgs, n_imgs_ant;

    // Actualizar pesos
    for(int c=0; c<this->w.size(); c++)
    {
        n_imgs = this->w[c].size() / n_thrs, n_imgs_ant = this->w[c].size() / n_thrs;

        if(thr_id == n_thrs - 1)
            n_imgs = this->w[c].size() - n_imgs * thr_id;

        for(int k=n_imgs_ant*thr_id; k<n_imgs_ant*thr_id + n_imgs; k++)
            for(int p=0; p<this->w[c][k].size(); p++)
                for(int l=0; l<this->w[c][k][p].size(); l++)
                    this->w[c][k][p][l] -= this->lr * grad_w[c][k][p][l];
    }

    // Actualizar Bias
    n_imgs = this->bias.size() / n_thrs, n_imgs_ant = this->bias.size() / n_thrs;

    if(thr_id == n_thrs - 1)
        n_imgs = this->bias.size() - n_imgs * thr_id;

    for(int j=n_imgs_ant*thr_id; j<n_imgs_ant*thr_id + n_imgs; j++)
        this->bias[j] -= this->lr * grad_bias[j];
}

// https://towardsdatascience.com/forward-and-backward-propagation-in-convolutional-neural-networks-64365925fdfa
// https://colab.research.google.com/drive/13MLFWdi3uRMZB7UpaJi4wGyGAZ9ftpPD?authuser=1#scrollTo=FEFgOKF4gGv2
/*
int main() 
{
    
    // Ejemplo de uso
    vector<vector<float>> imagen = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    
    vector<vector<vector<float>>> imagenes_2D, output, grad_output, v_3D, imagenes_2D_copy;
    vector<vector<float>> v_2D;
    vector<float> v_1D;
    int n_kernels = 1, pad=0, K=3;

    //imagenes_2D.push_back(imagen);
    //imagenes_2D.push_back(imagen);

     // Input tensor
    vector<vector<vector<float>>> x =
        {
            {
                {1, 2, 3, 4, 5},
                {6, 7, 8, 9, 10},
                {11, 12, 13, 14, 15},
                {16, 17, 18, 19, 20},
                {21, 22, 23, 24, 25}
            },
            {
                {25, 24, 23, 22, 21},
                {20, 19, 18, 17, 16},
                {15, 14, 13, 12, 11},
                {10, 9, 8, 7, 6},
                {5, 4, 3, 2, 1}
            }
        };

    imagenes_2D = x;
    //aplicar_padding(imagenes_2D, 1);

    // H_out = fils_img   -       K          + 1;
    int H_out = imagenes_2D[0].size() + 2*pad - K + 1;
    int W_out = imagenes_2D[0][0].size() + 2*pad - K + 1;

    output.clear();
    for(int j=0; j<W_out; j++)
    {
        v_1D.push_back(0.0);
    }

    for(int j=0; j<H_out; j++)
    {
        v_2D.push_back(v_1D);
    }


    for(int j=0; j<n_kernels; j++)
    {
        output.push_back(v_2D);
    }
    

    cout << "------------ Imagen inicial: ------------" << endl;
    mostrar_imagen(imagenes_2D);
    Convolutional conv_aux(1, K, K, imagenes_2D, 0.1);
    
    cout << "Aplicamos padding \n";
    conv_aux.aplicar_padding(imagenes_2D, pad);
    mostrar_imagen(imagenes_2D);

    Convolutional conv(n_kernels, K, K, imagenes_2D, 0.1);

    conv.w_a_1();
    conv.mostrar_pesos();

    imagenes_2D_copy = imagenes_2D;

    conv.forwardPropagation(imagenes_2D, output);
    
    cout << "------------ Conv, Forward Propagation --> output: ------------" << endl;
    mostrar_imagen(output);

    
    cout << "------------ Conv, Back Propagation: ------------" << endl;

    conv.backPropagation(imagenes_2D, output, pad);
    mostrar_imagen(imagenes_2D);

    conv.mostrar_grad();

    conv.reset_gradients();

    cout << "------------ Conv, Back Propagation Biblio: ------------" << endl;
    conv.backPropagation_bibliografia(imagenes_2D_copy, output, pad);
    mostrar_imagen(imagenes_2D_copy);

    conv.mostrar_grad();

    return 0;
}
*/