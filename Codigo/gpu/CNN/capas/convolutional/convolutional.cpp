#include "convolutional.h"
#include <vector>
#include <math.h>
#include <iostream>
#include <chrono>
#include "random"
#include "omp.h"

using namespace std;


/*
    CONSTRUCTOR de la clase Convolutional
    --------------------------------------
  
    @n_kernels      Número de kernels a aplicar
    @kernel_fils    Número de filas por kernel
    @kernel_cols    Número de columnas por kernel
    @input          Volumen 3D de entrada
    @lr             Learning Rate o Tasa de Aprendizaje
*/
Convolutional::Convolutional(int n_kernels, int kernel_fils, int kernel_cols, const vector<vector<vector<float>>> &input, float lr)
{
    this->n_kernels = n_kernels;
    this->kernel_fils = kernel_fils;
    this->kernel_cols = kernel_cols;
    this->kernel_depth = input.size();
    this->lr = lr;
    this->w.clear();

    vector<vector<vector<vector<float>>>> pesos_por_kernel;
    vector<vector<vector<float>>> pesos_totales;    // Representa los pesos de un kernel (los pesos de todas sus dimensiones)
    vector<vector<float>> pesos_kernel_2D;      // Representa los pesos de un kernel 2D
    vector<float> pesos_fila;               // Representa los pesos de una fila de un kernel 2D. Podría verse como un kernel 1D (un array)
    
    // Crear estructura de los pesos
    for(int f=0; f< n_kernels; ++f) // Por cada kernel
    {
        for(int k=0; k< kernel_depth; ++k)
        {
            for(int i=0; i<kernel_fils; ++i)
            {
                for(int j=0; j<kernel_cols; ++j)
                {
                    pesos_fila.push_back(0.0);
                }
                pesos_kernel_2D.push_back(pesos_fila);
                pesos_fila.clear();
            }
            pesos_totales.push_back(pesos_kernel_2D);
            pesos_kernel_2D.clear();
        }
        this->w.push_back(pesos_totales);
        pesos_totales.clear();
    }     

    // Inicializar pesos mediante Inicialización He
    this->generar_pesos();

    // Bias    
    // Un bias por filtro, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
    for(int i=0; i<n_kernels; i++)
        this->bias.push_back(0.0);
    
};

/*
    @brief      Inicializa los pesos de la capa convolucional según la inicialización He
    @return     Se modifica w (los pesos de la capa)
*/
void Convolutional::generar_pesos() 
{
    // Inicialización He
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> distribution(0.0, sqrt(2.0 / (this->n_kernels * this->kernel_depth * this->kernel_fils * this->kernel_fils)));

    for(int i=0; i<this->w.size(); ++i)
        for(int j=0; j<this->w[0].size(); ++j)
            for(int k=0; k<this->w[0][0].size(); ++k)
                for(int p=0; p<this->w[0][0][0].size(); ++p)
                    this->w[i][j][k][p] = distribution(gen);
}

/*
    @brief      Función de activación ReLU
    @x          Valor sobre el cual aplicar ReLU
    @return     @x tras aplicar ReLU sobre él
*/
float Convolutional::activationFunction(float x)
{
	// RELU
	return (x > 0.0) ? x : 0;
};

/*
    @brief      Derivada de la función de activación ReLU
    @x          Valor sobre el cual aplicar la derivada de ReLU
    @return     @x tras aplicar la derivada de ReLU sobre él
*/
float Convolutional::deriv_activationFunction(float x)
{
    float result = 0.0;

    if(x > 0)
        result = 1;
    
    return result;
}

/*
    @brief      Aplicar padding a un volumen de entrada 3D
    @imagen_3D  Volumen 3D de entrada
    @pad        Nivel de padding a aplicar
    @return     Se añaden @pad niveles de padding al volumen de entrada @imagen_3D
*/
void Convolutional::aplicar_padding(vector<vector<vector<float>>> &imagen_3D, int pad)
{
    vector<vector<vector<float>>> imagen_3D_aux;
    vector<vector<float>> imagen_aux;
    vector<float> fila_aux;

    // Por cada imagen
    for(int i=0; i<imagen_3D.size(); ++i)
    {
        // Añadimos padding superior
        for(int j=0; j<imagen_3D[i].size() + pad*2; ++j) // pad*2 porque hay padding tanto a la derecha como a la izquierda
            fila_aux.push_back(0.0);
        
        for(int k=0; k<pad; ++k)
            imagen_aux.push_back(fila_aux);
        
        fila_aux.clear();

        // Padding lateral (izquierda y derecha)
        // Por cada fila de cada imagen
        for(int j=0; j<imagen_3D[i].size(); ++j)
        {
            // Añadimos padding lateral izquierdo
            for(int t=0; t<pad; ++t)
                fila_aux.push_back(0.0);

            // Dejamos casillas centrales igual que en la imagen original
            for(int k=0; k<imagen_3D[i][j].size(); ++k)
                fila_aux.push_back(imagen_3D[i][j][k]);
            
            // Añadimos padding lateral derecho
            for(int t=0; t<pad; ++t)
                fila_aux.push_back(0.0);
            
            // Añadimos fila construida a la imagen
            imagen_aux.push_back(fila_aux);
            fila_aux.clear();
        }
        
        // Añadimos padding inferior
        fila_aux.clear();

        for(int j=0; j<imagen_3D[i].size() + pad*2; ++j) // pad*2 porque hay padding tanto a la derecha como a la izquierda
            fila_aux.push_back(0.0);
        
        for(int k=0; k<pad; ++k)
            imagen_aux.push_back(fila_aux);
        
        fila_aux.clear();
        
        // Añadimos imagen creada al conjunto de imágenes
        imagen_3D_aux.push_back(imagen_aux);
        imagen_aux.clear();
    }

    imagen_3D = imagen_3D_aux;
};


/*
    @brief      Propagación hacia delante a lo largo de toda la capa convolucional
    @input      Volumen de entrada 3D
    @output     Volumen de salida 3D
    @a          Valor de las neuronas antes de aplicar la función de activación
    @return     Se modifica @output y @a
*/
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
    
    //a = output;

    // Por cada kernel M 
    for(int img_out=0; img_out<M; ++img_out)
        for(int i=0; i<n_veces; ++i)    
            for(int j=0; j<n_veces; ++j)  
            {
                a[img_out][i][j] = 0.0;
                
                // Realizar convolución 3D
                for(int c=0; c<C; ++c)
                    for(int i_k=0; i_k<K; ++i_k)
                        for(int j_k=0; j_k<K; ++j_k)
                            a[img_out][i][j] += input[c][i+i_k][j+j_k] * this->w[img_out][c][i_k][j_k];                            

                // Sumamos bias a la suma ponderada obtenida
                a[img_out][i][j] += this->bias[img_out];


                // Aplicamos función de activación
                output[img_out][i][j] = activationFunction(a[img_out][i][j]);
            }
};


/*
    @brief      Establece el valor de todos gradientes de la capa convolucional a 0.0
    @grad_w     Gradientes respecto a pesos
    @grad_bias  Gradientes respecto a sesgos
    @return     Se modifica tantp @grad_w como @grad_bias
*/
void Convolutional::reset_gradients(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias)
{
    // Reset gradientes -------------------------------------------------------------------
    
    // Reset gradiende del bias
    for(int k=0; k<this->n_kernels; ++k)
        grad_bias[k] = 0.0;

    // Reset del gradiente de cada peso
    for(int i=0; i<this->w.size(); ++i)
        for(int j=0; j<this->w[0].size(); ++j)
            for(int k=0; k<this->w[0][0].size(); ++k)
                for(int l=0; l<this->w[0][0][0].size(); ++l)
                    grad_w[i][j][k][l] = 0.0;
                
}


/*
    @brief      Retropropagación de la capa convolucional
    @input      Volumen 3D de entrada de la capa
    @output     Volumen 3D de salida de la capa
    @a          Valor de las neuronas antes de aplicar la función de activación
    @grad_w     Gradientes respecto a pesos
    @grad_bias  Gradientes respecto a sesgos
    @pad        Nivel de padding que se aplicó anteriormente   
*/
void Convolutional::backPropagation(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> output, const vector<vector<vector<float>>> &a, vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias, const int &pad)
{
    // https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
    vector<vector<vector<float>>> grad_input = input;
    vector<vector<float>> conv_imagen;
    vector<float> conv_fila;
    
    // nº de kernels, nº de filas del kernel, nº de columnas del kernel a aplicar
    int F = this->w.size(), K = this->w[0][0].size(), C = input.size();

    // Inicializar input a 0
    for(int i=0; i<input.size(); ++i)
        for(int j=0; j<input[0].size(); ++j)    
            for(int k=0; k<input[0][0].size(); ++k)
                grad_input[i][j][k] = 0.0;

    // Realizar derivada Y_out/Y_in
    for(int i=0; i<output.size(); ++i)
        for(int j=0; j<output[0].size(); ++j)    
            for(int k=0; k<output[0][0].size(); ++k)
                output[i][j][k] = output[i][j][k] * deriv_activationFunction(a[i][j][k]);
    
    // Suponemos nº filas = nº columnas
    int H = input[0].size(), W = input[0][0].size();
    int H_out = H-K+1, W_out = W-K+1;

    // Convolución entre output y pesos    
    for(int f=0; f<F; ++f)  // Por cada filtro f
    {
        // Gradiente respecto a entrada
        for(int i=0; i<H; ++i)    
            for(int j=0; j<W; ++j)    
                for(int c=0; c<C; ++c)  // Convolución entre salida y pesos invertidos
                    for(int i_k=0; i_k<K; ++i_k)
                        for(int j_k=0; j_k<K; ++j_k)
                            if(i-i_k >= 0 && j-j_k >= 0 && i-i_k < H_out && j-j_k < W_out)
                                grad_input[c][i][j] += output[f][i-i_k][j-j_k] * this->w[f][c][K -1 - i_k][K -1 - j_k];                            
        
        // Gradiente respecto a pesos
        for(int i=0; i<H_out; ++i)    
            for(int j=0; j<W_out; ++j)  
                for(int c=0; c<C; ++c)  // Convolución entre entrada y salida
                    for(int i_k=0; i_k<this->kernel_fils; ++i_k)
                        for(int j_k=0; j_k<this->kernel_cols; ++j_k)
                            grad_w[f][c][i_k][j_k] += input[c][i + i_k][j + j_k] * output[f][i][j];
    }


    // Gradiente respecto a entrada
    for(int i=0; i<input.size(); ++i)
        for(int j=0; j<input[0].size(); ++j)    
            for(int k=0; k<input[0][0].size(); ++k)
                input[i][j][k] = grad_input[i][j][k];

    // Calcular el gradiente del bias
    for(int i=0; i<output.size(); ++i)
        for(int j=0; j<output[0].size(); ++j)    
            for(int k=0; k<output[0][0].size(); ++k)
                grad_bias[i] += output[i][j][k];
};

/*
    @brief          Escalar los pesos para evitar que los gradientes "exploten"
    @clip_value     Valor a emplear para realizar el "clip" o escalado
    @maxs           Se asigna una posición a cada hebra. Contiene el valor máximo encontrado
    @mins           Se asigna una posición a cada hebra. Contiene el valor mínimo encontrado
    @return         Se actualizan los valores de w (pesos de la capa)
*/
void Convolutional::escalar_pesos(float clip_value, vector<float> &maxs, vector<float> &mins)
{
    // Cada hebra busca el máximo y mínimo de su conjunto de datos
    int n_thrs = 8, thr_id = omp_get_thread_num(), n_imgs, n_imgs_ant;
    maxs[thr_id] = this->w[0][0][0][0];
    mins[thr_id] = this->w[0][0][0][0];

    // Buscar máximo y mínimo locales
    for(int i=0; i<this->w.size(); ++i)
    {
        // Reparto de carga
        n_imgs = this->w[i].size() / n_thrs, n_imgs_ant = this->w[i].size() / n_thrs;

        if(thr_id == n_thrs - 1)
            n_imgs = this->w[i].size() - n_imgs * thr_id;

        // Cada hebra busca en "n_imgs" pesos
        for(int j=n_imgs_ant*thr_id; j<n_imgs_ant*thr_id + n_imgs; ++j)
            for(int k=0; k<this->w[i][j].size(); ++k)
                for(int l=0; l<this->w[i][j][k].size(); ++l)
                {
                    if(maxs[thr_id] < this->w[i][j][k][l])
                        maxs[thr_id] = this->w[i][j][k][l];
                    
                    if(mins[thr_id] > this->w[i][j][k][l])
                        mins[thr_id] = this->w[i][j][k][l];
                }
    }
    #pragma omp barrier

    // Buscar valor ḿaximo y mínimo globales
    #pragma omp master
    {
        for(int i=1; i<n_thrs; ++i)
        {
            if(maxs[0] < maxs[i])
                maxs[0] = maxs[i];
            
            if(mins[0] > mins[i])
                mins[0] = mins[i];
        }
    }
    #pragma omp barrier

    // Perform gradient clipping
    float scaling_factor = clip_value / std::max(std::abs(maxs[0]), std::abs(mins[0]));
    for(int i=0; i<this->w.size(); ++i)
    {
        // Reparto de carga
        n_imgs = this->w[i].size() / n_thrs, n_imgs_ant = this->w[i].size() / n_thrs;

        if(thr_id == n_thrs - 1)
            n_imgs = this->w[i].size() - n_imgs * thr_id;

        // Cada hebra actualiza "n_imgs" pesos
        for(int j=n_imgs_ant*thr_id; j<n_imgs_ant*thr_id + n_imgs; ++j)
            for(int k=0; k<this->w[i][j].size(); ++k)
                for(int l=0; l<this->w[i][j][k].size(); ++l)
                    this->w[i][j][k][l] = std::max(std::min(this->w[i][j][k][l] * scaling_factor, clip_value), -clip_value);
    }

}

/*
    @brief          Actualizar los pesos y sesgos de la capa
    @grad_w         Gradiente de cada peso de la capa
    @grad_b         Gradiente de cada sesgo de la capa
    @return         Se actualizar los valores de w y bias (pesos y sesgos de la capa)
*/
void Convolutional::actualizar_grads(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias)
{
    int n_thrs = 8, thr_id = omp_get_thread_num(), n_imgs, n_imgs_ant;

    // Actualizar pesos
    for(int c=0; c<this->w.size(); ++c)
    {
        // Reparto de carga
        n_imgs = this->w[c].size() / n_thrs, n_imgs_ant = this->w[c].size() / n_thrs;

        if(thr_id == n_thrs - 1)
            n_imgs = this->w[c].size() - n_imgs * thr_id;

        // Cada hebra actualiza "n_imgs" pesos
        for(int k=n_imgs_ant*thr_id; k<n_imgs_ant*thr_id + n_imgs; ++k)
            for(int p=0; p<this->w[c][k].size(); ++p)
                for(int l=0; l<this->w[c][k][p].size(); ++l)
                    this->w[c][k][p][l] -= this->lr * grad_w[c][k][p][l];
    }

    // Actualizar Bias
    // Reparto de carga
    n_imgs = this->bias.size() / n_thrs, n_imgs_ant = this->bias.size() / n_thrs;

    if(thr_id == n_thrs - 1)
        n_imgs = this->bias.size() - n_imgs * thr_id;

    // Cada hebra actualiza "n_imgs" sesgos
    for(int j=n_imgs_ant*thr_id; j<n_imgs_ant*thr_id + n_imgs; ++j)
        this->bias[j] -= this->lr * grad_bias[j];
}

// https://towardsdatascience.com/forward-and-backward-propagation-in-convolutional-neural-networks-64365925fdfa
// https://colab.research.google.com/drive/13MLFWdi3uRMZB7UpaJi4wGyGAZ9ftpPD?authuser=1#scrollTo=FEFgOKF4gGv2