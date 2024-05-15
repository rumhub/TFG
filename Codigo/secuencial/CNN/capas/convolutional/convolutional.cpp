#include "convolutional.h"

using namespace std::chrono;

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

void Convolutional::unroll(int C, int n, int K, float *X, float *X_unroll){
    int H_out = n-K+1;
    int W_out = n-K+1;
    int w_base;
    int W = H_out * W_out;

    for(int c=0; c<C; c++)
    {
        w_base = c * (K*K);
        for(int p=0; p<K; p++)
            for(int q=0; q<K; q++)
                for(int h=0; h<H_out; h++){
                    int h_unroll = w_base + p*K + q;
                    for(int w=0; w < W_out; w++){
                        int w_unroll = h * W_out + w;
                        X_unroll[h_unroll*W + w_unroll] = X[c*n*n + (h+p)*n + (w+q)];
                        //cout << X[c*n*n + (h+p)*n + (w+q)] << " ";
                    }
                }

        
    } 
    //cout << " ---------------------232 ------------------ " << endl;
}

void Convolutional::unroll_1dim(int C, int H, int W, int K, float *X, float *X_unroll){

    int H_out = H - K+1, W_out = W -K +1;
    int cont = 0;

    for(int i=0; i<C; i++)
    {
        for(int k=0; k<W_out; k++)
        {
            for(int j=0; j<H_out; j++)
            {
                // Guardar K*K elementos de "convolución"
                for(int ky=0; ky < K; ky++)
                    for(int kx=0; kx<K; kx++)
                    {

                        
                        X_unroll[cont++] = X[i*H*W + (j+ky)*W + (k+kx)];
                        //X_unroll[cont++] = X[i*H*W + (j+ky)*W + (k+kx)];
                        //cout << X[i*H*W + (j+ky)*W + (k+kx)] << " ";
                    }
                //cout << endl;
            }
        }
    }        
    //cout << " ---------- " << endl;
}

/*
void Convolutional::unroll_3dim(int C, int H, int W, int K, float *X, float *X_unroll){

    int H_out = H - K+1, W_out = W -K +1;
    int cont = 0;

    cout << " ---------------------232 ------------------ " << endl;
    for(int j=0; j<H_out; j++)
    {
        for(int k=0; k<W_out; k++)
        {
            for(int i=0; i<C; i++)
            {
                // Guardar K*K elementos de "convolución"
                for(int ky=0; ky < K; ky++)
                    for(int kx=0; kx<K; kx++)
                    {
                        
                        X_unroll[cont++] = X[i*H*W + (j+ky)*W + (k+kx)];
                        //X_unroll[cont++] = X[i*H*W + (j+ky)*W + (k+kx)];
                        cout << X[i*H*W + (j+ky)*W + (k+kx)] << " ";
                    }
                cout << endl;
            }
        }
    }        
    cout << " ---------------------232 ------------------ " << endl;

}
*/


void Convolutional::unroll_3dim(int C, int H, int W, int K, float *X, float *X_unroll){

        // Calculate the size of output
        int H_out = H - K + 1;
        int W_out = W - K + 1;

        int outputIndex = 0;
        for (int i = 0; i < H_out; ++i) {
            for (int j = 0; j < W_out; ++j) {
                int inputIndex = 0;
                for (int d = 0; d < C; ++d) {
                    for (int ki = 0; ki < K; ++ki) {
                        for (int kj = 0; kj < K; ++kj) {
                            X_unroll[outputIndex * (K * K * C) + inputIndex++] = X[d * (H * W) + (i + ki) * W + (j + kj)];
                        }
                    }
                }
                ++outputIndex;
            }
        }

}

/*
// Function to perform convolution with loop unrolling
std::vector<std::vector<int>> convolutionUnrolled(const std::vector<std::vector<std::vector<int>>>& input,
                                                  const std::vector<std::vector<int>>& kernel) {
    // Assuming input dimensions are [depth][height][width]
    int C = input.size();
    int H = input[0].size();
    int W = input[0][0].size();
    int K = kernel.size();
    int K = kernel[0].size();

    // Calculate the size of output
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    std::vector<std::vector<int>> output(H_out * W_out,
                                          std::vector<int>(K * K * C));

    int outputIndex = 0;
    for (int i = 0; i < H_out; ++i) {
        for (int j = 0; j < W_out; ++j) {
            int inputIndex = 0;
            for (int d = 0; d < C; ++d) {
                for (int ki = 0; ki < K; ++ki) {
                    for (int kj = 0; kj < K; ++kj) {
                        output[outputIndex][inputIndex++] = input[d][i + ki][j + kj];
                    }
                }
            }
            ++outputIndex;
        }
    }

    return output;
}
*/

void Convolutional::printMatrix_3D(float* matrix, int C, int n) {
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) 
                cout << matrix[i*n*n +j*n +k] << " ";
            cout << endl;
        }
        cout << endl;
    }
}

void Convolutional::printMatrix(float* matrix, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) 
            cout << matrix[i * w + j] << " ";
        cout << endl;
    }
}

void Convolutional::matrizTranspuesta(float* matrix, int rows, int cols)
{
    // Allocate a new matrix to hold the transposed data
    float* transposedMatrix = (float*)malloc(cols * rows * sizeof(float));
    
    if (transposedMatrix == nullptr) {
        cerr << "Memory allocation for transposed matrix failed" << endl;
        exit(1);
    }

    // Transpose the matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Copy the element from original matrix at position (i, j) to
            // transposed matrix at position (j, i)
            transposedMatrix[j * rows + i] = matrix[i * cols + j];
        }
    }

    // Copy the transposed data back to the original matrix
    // This requires the original matrix to be able to hold the new dimensions
    // i.e., it must have space for 'cols * rows' elements
    for (int i = 0; i < cols * rows; i++) {
        matrix[i] = transposedMatrix[i];
    }

    // Free the allocated memory for the transposed matrix
    free(transposedMatrix);
}

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


void Convolutional::printMatrix_4D(float* matrix, int F, int C, int n) {
    for (int f = 0; f < F; f++) {
        for (int i = 0; i < C; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) 
                    cout << matrix[f*C*n*n + i*n*n +j*n +k] << " ";
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }
}

void Convolutional::multiplicarMatrices(float* m1, int rows1, int cols1, float* m2, int cols2, float* result) {
    for (int i = 0; i < rows1; i++) 
        for (int j = 0; j < cols2; j++) {
            result[i * cols2 + j] = 0.0f;

            for (int k = 0; k < cols1; k++) 
                result[i * cols2 + j] += m1[i * cols1 + k] * m2[k * cols2 + j];
            
        }
}

void Convolutional::printMatrix_vector(const vector<vector<vector<float>>> &X) {
    for (int i = 0; i < X.size(); i++) {
        for (int j = 0; j < X[i].size(); j++) {
            for (int k = 0; k < X[i][j].size(); k++) 
                cout << X[i][j][k] << " ";
            cout << endl;
        }
        cout << endl;
    }
}


/*
    @brief          Escalar los pesos para evitar que los gradientes "exploten"
    @clip_value     Valor a emplear para realizar el "clip" o escalado
    @return         Se actualizan los valores de w (pesos de la capa)
*/
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

/*
    @brief          Actualizar los pesos y sesgos de la capa
    @grad_w         Gradiente de cada peso de la capa
    @grad_b         Gradiente de cada sesgo de la capa
    @return         Se actualizar los valores de w y bias (pesos y sesgos de la capa)
*/
void Convolutional::actualizar_grads(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias)
{
    // Actualizar pesos
    for(int j=0; j<this->w.size(); j++)
        for(int k=0; k<this->w[j].size(); k++)
            for(int p=0; p<this->w[j][k].size(); p++)
                for(int l=0; l<this->w[j][k][p].size(); l++)
                    this->w[j][k][p][l] -= this->lr * grad_w[j][k][p][l];
    
    // Actualizar bias
    for(int i=0; i<this->bias.size(); i++)
        this->bias[i] -= this->lr * grad_bias[i];
    
}



// https://towardsdatascience.com/forward-and-backward-propagation-in-convolutional-neural-networks-64365925fdfa
// https://colab.research.google.com/drive/13MLFWdi3uRMZB7UpaJi4wGyGAZ9ftpPD?authuser=1#scrollTo=FEFgOKF4gGv2


// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void printMatrix_3D(float* matrix, int C, int n) {
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) 
                cout << matrix[i*n*n +j*n +k] << " ";
            cout << endl;
        }
        cout << endl;
    }
}

void printMatrix_vector(const vector<vector<vector<float>>> &X) {
    for (int i = 0; i < X.size(); i++) {
        for (int j = 0; j < X[i].size(); j++) {
            for (int k = 0; k < X[i][j].size(); k++) 
                cout << X[i][j][k] << " ";
            cout << endl;
        }
        cout << endl;
    }
}

/*
int main()
{
    // -----------------------------------------------------------------------------------------------------
    // Método estándar
    // -----------------------------------------------------------------------------------------------------
    auto ini = high_resolution_clock::now();
    auto fin = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(fin - ini);
    int n_kernels = 26, K=5, H=50, W=50, H_out = H-K+1, W_out = W-K+1, pad = 0, C=100;   // C=9
    //int n_kernels = 2, K=2, H=3, W=3, H_out = H-K+1, W_out = W-K+1, pad = 0, C=2;   // C=9
    vector<vector<vector<float>>> a, input_gpu, a_gpu;
    vector<vector<vector<vector<float>>>> grad_w, grad_w2;
    vector<float> grad_bias;

    vector<vector<vector<float>>> input(C, vector<vector<float>>(H, vector<float>(W, 0)));
    vector<vector<vector<float>>> output(n_kernels, vector<vector<float>>(H_out, vector<float>(W_out, 0))), output_gpu = output;

    for(int i=0; i<C; i++)
        for(int j=0; j<H; j++)
            for(int k=0; k<W; k++)
                input[i][j][k] = (float) (j+k) / (H*W);
                //h_A[i*K + j] = (rand() % 100) + 1;
                //input[i][j][k] = 3.0;
    
    a = output, input_gpu = input, a_gpu = a;

    // Kernel de pesos
    vector<vector<vector<vector<float>>>> w(n_kernels, vector<vector<vector<float>>>(C, vector<vector<float>>(K, vector<float>(K, 0.7))));
    
    // Vector de sesgos
    vector<float> bias(n_kernels, 0.0);
    /*
    for(int i = 0; i < n_kernels; i++) 
        for(int j = 0; j < C; j++)
            for(int kx = 0; kx < K; kx++)
                for(int ky = 0; ky < K; ky++)
                    w[i][j][kx][ky] = (float) i+j / (n_kernels*C*2);
    
    // Inicializar gradientes a 0.0
    grad_w = w;
    grad_bias = bias;

    for(int i = 0; i < n_kernels; i++) 
        for(int j = 0; j < C; j++)
            for(int kx = 0; kx < K; kx++)
                for(int ky = 0; ky < K; ky++)
                    grad_w[i][j][kx][ky] = 0.0;

    grad_w2 = grad_w;


    // Crear capa convolucional
    Convolutional conv(n_kernels, K, K, input, 0.1);
    //conv.set_w(w);
    conv.set_b(bias);

    // Establecer device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cout << "Estableciendo dispotisivo para GPU...\n";
    printf("Usando el dispositivo %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    cout << " -------------------------- Método Estándar -------------------------- " << endl;
    //cout << "Input" << endl;
    //printMatrix_vector(input);
    ini = high_resolution_clock::now();
    conv.forwardPropagation(input, output, a);
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);
    
    // Mostrar resultado
    cout << "Tiempo CPU: " << duration.count() << " (us)" << endl;
    
    //cout << "Ouput" << endl;
    //printMatrix_vector(output);
    
    //cout << "-- Backprop --" << endl;
    //cout << "Input" << endl;
    conv.backPropagation(input, output, a, grad_w, grad_bias, pad);
    //printMatrix_vector(input);

    /*
    cout << "Gradientes de pesos" << endl;
    // Mostrar gradientes de pesos
    for(int i = 0; i < n_kernels; i++) 
    {
        for(int j = 0; j < C; j++)
        {
            for(int kx = 0; kx < K; kx++)
            {
                for(int ky = 0; ky < K; ky++)
                    cout << grad_w[i][j][kx][ky] << " ";
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }
    
    
    cout << " -------------------------- Método GEMM -------------------------- " << endl;
    //cout << "Input" << endl;
    //printMatrix_vector(input_gpu);
    ini = high_resolution_clock::now();
    conv.forwardPropagationGEMM(input_gpu, output_gpu, a_gpu);
    cudaDeviceSynchronize();
    fin = high_resolution_clock::now();
    duration = duration_cast<microseconds>(fin - ini);
    
    // Mostrar resultado
    cout << "Tiempo GPU: " << duration.count() << " (us)" << endl;
    //cout << "Ouput" << endl;
    //printMatrix_vector(output_gpu);
    
    //cout << "-- Backprop --" << endl;
    conv.backPropagationGEMM(input_gpu, output_gpu, a_gpu, grad_w2, grad_bias, pad);
    /*
    cout << "Input" << endl;
    printMatrix_vector(input_gpu);

    */
    /*
    cout << "Gradientes de pesos" << endl;
    // Mostrar gradientes de pesos
    for(int i = 0; i < n_kernels; i++) 
    {
        for(int j = 0; j < C; j++)
        {
            for(int kx = 0; kx < K; kx++)
            {
                for(int ky = 0; ky < K; ky++)
                    cout << grad_w2[i][j][kx][ky] << " ";
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }
    */
   /*
    int i_=0;
    cout << "Input\n";
    for(int j=0; j<H; j++)
    {
        for(int k=0; k<W; k++)
            cout << input[i_][j][k] << " ";
        cout << endl;
    }
    cout << endl;

    cout << "Input GEMM\n";
    for(int j=0; j<H; j++)
    {
        for(int k=0; k<W; k++)
            cout << input_gpu[i_][j][k] << " ";
        cout << endl;
    }
    cout << endl;
    


   // Comprobar resultados
    bool correcto = true;
    float epsilon = 0000000.1;
    int n_errores = 0;
    float err_medio_input = 0.0, err_medio_w = 0.0;

    for(int i=0; i<n_kernels; i++)
        for(int j=0; j<H_out; j++)
            for(int k=0; k<W_out; k++)
                if(abs(output[i][j][k] - output_gpu[i][j][k]) > epsilon)
                {
                    correcto = false;
                    cout << abs(output[i][j][k] - output_gpu[i][j][k]) << "output" << endl;
                    n_errores++;
                }

    for(int i=0; i<C; i++)
        for(int j=0; j<H; j++)
            for(int k=0; k<W; k++)
                if(abs(input[i][j][k] - input_gpu[i][j][k]) > epsilon)
                {
                    correcto = false;
                    //cout << abs(input[i][j][k] - input_gpu[i][j][k]) << " input. " << input[i][j][k] << " vs " << input_gpu[i][j][k] << endl;
                    n_errores++;
                    err_medio_input += abs(input[i][j][k] - input_gpu[i][j][k]);
                }


    for(int i = 0; i < n_kernels; i++) 
        for(int j = 0; j < C; j++)
            for(int kx = 0; kx < K; kx++)
                for(int ky = 0; ky < K; ky++)
                if(abs(grad_w[i][j][kx][ky] - grad_w2[i][j][kx][ky]) > epsilon)
                {
                    correcto = false;
                    //cout << abs(grad_w[i][j][kx][ky] - grad_w2[i][j][kx][ky]) << " pesos " << endl;
                    n_errores++;
                    err_medio_w += abs(grad_w[i][j][kx][ky] - grad_w2[i][j][kx][ky]);
                }



    if(correcto)
        cout << "Todo correcto" << endl;
    else
    {
        cout << "Incorrecto (" << n_errores << " errores) " << endl;
        cout << "Error medio input: " << err_medio_input / C*H*W << endl;
        cout << "Error medio w: " << err_medio_w / n_kernels*C*K*K << endl;
    }

    return 0;
}
*/