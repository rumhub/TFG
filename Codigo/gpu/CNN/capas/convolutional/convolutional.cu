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


/*
    @brief      Propagación hacia delante a lo largo de toda la capa convolucional
    @input      Volumen de entrada 3D
    @output     Volumen de salida 3D
    @a          Valor de las neuronas antes de aplicar la función de activación
    @return     Se modifica @output y @a
*/
void Convolutional::forwardPropagationGEMM(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &a)
{
    // Tamaños
    int K = kernel_fils, C = input.size(), H = input[0].size(), W = input[0][0].size(), H_out = H -K +1, W_out = W -K +1, 
        fils_input_unroll = K*K*C, cols_input_unroll = H_out * W_out,                       // Tamaños de la entrada 'desplegada'
        fils_w = this->n_kernels, cols_w = K*K*C,                                           // Tamaños de los pesos
        bytes_input = input.size() * input[0].size() * input[0][0].size() * sizeof(float),      // Espacio para la entrada
        bytes_input_unroll = fils_input_unroll * cols_input_unroll *sizeof(float),          // Espacio para input 'desplegado'
        bytes_w = fils_w * cols_w * sizeof(float),              // Espacio para pesos
        bytes_bias = this->n_kernels * sizeof(float),           // Espacio para sesgos
        bytes_output = cols_input_unroll * fils_w *sizeof(float);              // Espacio para la salida

    // Crear bloque y grid
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((cols_input_unroll  + BLOCK_SIZE -1) / BLOCK_SIZE, (fils_w + BLOCK_SIZE -1) / BLOCK_SIZE);

    // Punteros host
    float *input_ = (float*)malloc(bytes_input), 
    *h_input_unroll = (float*)malloc(bytes_input_unroll),
    *output_ = (float*)malloc(bytes_output),
    *h_a = (float*)malloc(bytes_output),
    *h_w = (float*)malloc(bytes_w),
    *b = (float*)malloc(bytes_bias); 


    // Punteros device
    float *d_input_unroll, *d_a, *d_w; 
    cudaMalloc((void **) &d_input_unroll, bytes_input_unroll);
    cudaMalloc((void **) &d_a, bytes_output);
    cudaMalloc((void **) &d_w, bytes_w);

    // Copiar valores a host -----------------------------------
    // Input
    for(int i = 0; i < C; i++) 
        for(int j = 0; j < H; j++)
            for(int k = 0; k < W; k++)
                input_[i*H*W +j*W +k] = input[i][j][k];
    
    // Pesos
    for(int i = 0; i < this->n_kernels; i++) 
        for(int j = 0; j < C; j++)
            for(int kx = 0; kx < K; kx++)
                for(int ky = 0; ky < K; ky++)
                    h_w[i*C*K*K + j*K*K + kx*K + ky] = this->w[i][j][kx][ky];

    // Sesgos
    for(int i = 0; i < this->n_kernels; i++) 
        b[i] = this->bias[i];

    this->unroll(C, H, K, input_, h_input_unroll);

    /*
    cout << "input" << endl;
    for(int i=0; i<C; i++)
    {
        for(int j=0; j<H; j++)
        {
            for(int k=0; k<W; k++)
                cout << input_[i*H*W + j*W + k] << " ";
            cout << endl;
        }
        cout << endl;
    }

    cout << "input unroll\n";

    for(int j=0; j<K*K*C; j++)
    {
        for(int k=0; k<H_out*W_out; k++)
            cout << input_[j*H_out*W_out + k] << " ";
        cout << endl;
    }
    cout << endl;
    */
    


    // Copiar de CPU a GPU
    cudaMemcpy(d_input_unroll, h_input_unroll, bytes_input_unroll, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, bytes_w, cudaMemcpyHostToDevice);

    // Multiplicación de matrices
    size_t smem = (2*block.x * block.y) *sizeof(float);
    multiplicarMatricesGPU<<<grid, block, smem>>>(fils_w, cols_input_unroll, cols_w, d_w, d_input_unroll, d_a);

    // Paso de GPU a CPU
    cudaMemcpy(h_a, d_a, bytes_output, cudaMemcpyDeviceToHost);
    
    // Sumar bias
    for(int i = 0; i < this->n_kernels; i++) 
        for(int j = 0; j < H_out; j++)
            for(int k = 0; k < W_out; k++)
                h_a[i*H_out*W_out +j*W_out +k] += b[i];

    // Aplicar función de activación
    for(int i = 0; i < this->n_kernels; i++) 
        for(int j = 0; j < H_out; j++)
            for(int k = 0; k < W_out; k++)
                output_[i*H_out*W_out +j*W_out +k] = activationFunction(h_a[i*H_out*W_out +j*W_out +k]);

    // Copiar valores de salida
    for(int i = 0; i < this->n_kernels; i++) 
        for(int j = 0; j < H_out; j++)
            for(int k = 0; k < W_out; k++)
                output[i][j][k] = output_[i*H_out*W_out +j*W_out +k];
    
    // Copiar "a" en CPU secuencial
    for(int i = 0; i < this->n_kernels; i++) 
        for(int j = 0; j < H_out; j++)
            for(int k = 0; k < W_out; k++)
                a[i][j][k] = h_a[i*H_out*W_out +j*W_out +k];

    // Liberar memoria
    free(input_); free(h_input_unroll); free(output_); free(h_a); free(h_w); free(b);
    cudaFree(d_input_unroll); cudaFree(d_a); cudaFree(d_w);
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

/*
    @brief      Retropropagación de la capa convolucional
    @input      Volumen 3D de entrada de la capa
    @output     Volumen 3D de salida de la capa
    @a          Valor de las neuronas antes de aplicar la función de activación
    @grad_w     Gradientes respecto a pesos
    @grad_bias  Gradientes respecto a sesgos
    @pad        Nivel de padding que se aplicó anteriormente   
*/
void Convolutional::backPropagationGEMM(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> output, const vector<vector<vector<float>>> &a, vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias, const int &pad)
{
    // https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939
    vector<vector<vector<float>>> grad_input = input, output_pad;
    vector<vector<float>> conv_imagen;
    vector<float> conv_fila;    
    int K = this->kernel_fils, C = input.size(), H = input[0].size(), W = input[0][0].size(), H_out = H-K+1, W_out = W-K+1, M=this->n_kernels;

    
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

    output_pad = output;
    aplicar_padding(output_pad, K-1);
    int H_out_pad = output_pad[0].size(), W_out_pad = output_pad[0][0].size();

    // Reserva de memoria en host
    int fils_output_unroll = K*K*M, cols_output_unroll = H*W, fils_matriz_pesos = C, cols_matriz_pesos = K*K*M,
        fils_input_unroll = H_out * W_out, cols_input_unroll = K*K*C,
        bytes_output_unroll = fils_output_unroll * cols_output_unroll * sizeof(float),
        bytes_matriz_pesos = fils_matriz_pesos * cols_matriz_pesos * sizeof(float),
        bytes_input_unroll = fils_input_unroll * cols_input_unroll * sizeof(float);
    float *output_pad_ptr = (float *)malloc(M*H_out_pad*W_out_pad * sizeof(float)),
        *h_output_unroll = (float *)malloc(bytes_output_unroll),
        *h_matriz_pesos = (float *)malloc(bytes_matriz_pesos),
        *h_input = (float *)malloc(C*H*W * sizeof(float)),
        *h_output = (float*)malloc(M*H_out*W_out*sizeof(float)),
        *h_input_unroll = (float *)malloc(bytes_input_unroll),
        *h_grad_w = (float *)malloc(this->n_kernels*C*K*K * sizeof(float));

    // Crear bloque y grid
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_grad_w((cols_input_unroll  + BLOCK_SIZE -1) / BLOCK_SIZE, (this->n_kernels + BLOCK_SIZE -1) / BLOCK_SIZE);
    dim3 grid_grad_input((cols_output_unroll  + BLOCK_SIZE -1) / BLOCK_SIZE, (C + BLOCK_SIZE -1) / BLOCK_SIZE);
    size_t smem = (2*block.x * block.y) *sizeof(float);

    // Reserva de memoria en device
    float * d_output_unroll, *d_matriz_pesos, *d_input, *d_input_unroll, *d_output, *d_grad_w;
    cudaMalloc((void **) &d_output_unroll, bytes_output_unroll);
    cudaMalloc((void **) &d_matriz_pesos, bytes_matriz_pesos);
    cudaMalloc((void **) &d_input, C*H*W * sizeof(float));
    cudaMalloc((void **) &d_output, M*H_out*W_out * sizeof(float));
    cudaMalloc((void **) &d_input_unroll, bytes_input_unroll);
    cudaMalloc((void **) &d_grad_w, this->n_kernels*C*K*K * sizeof(float));

    // Output con padding
    for(int i = 0; i < M; i++) 
        for(int j = 0; j < H_out_pad; j++)
            for(int k = 0; k < W_out_pad; k++)
                output_pad_ptr[i*H_out_pad*W_out_pad +j*W_out_pad +k] = output_pad[i][j][k];
    
    // Output sin padding
    for(int i = 0; i < M; i++) 
        for(int j = 0; j < H_out; j++)
            for(int k = 0; k < W_out; k++)
                h_output[i*H_out*W_out +j*W_out +k] = output[i][j][k];
    
    // Input
    for(int i = 0; i < C; i++) 
        for(int j = 0; j < H; j++)
            for(int k = 0; k < W; k++)
                h_input[i*H*W +j*W +k] = input[i][j][k];

    unroll_3dim(M, H_out_pad, W_out_pad, K, output_pad_ptr, h_output_unroll);
    //unroll(this->n_kernels, H_out_pad, K, output_pad_ptr, h_output_unroll);
    unroll_1dim(C, H, W, H_out, h_input, h_input_unroll);
    matrizTranspuesta(h_input_unroll, K*K*C, H_out*W_out);
    matrizTranspuesta(h_output_unroll, cols_output_unroll, fils_output_unroll);

    // Concatenar los pesos de todos los kernels para una misma capa de profundidad C
    for(int j = 0; j < C; j++)
        for(int i = 0; i < this->n_kernels; i++) 
            for(int kx = K-1; kx >= 0; kx--)
                for(int ky = K-1; ky >=0; ky--)
                    h_matriz_pesos[j*this->n_kernels*K*K + i*K*K + kx*K + ky] = this->w[i][j][kx][ky];

    // Copiar datos de CPU a GPU
    cudaMemcpy(d_matriz_pesos, h_matriz_pesos, bytes_matriz_pesos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_unroll, h_output_unroll, bytes_output_unroll, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_unroll, h_input_unroll, bytes_input_unroll, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, M*H_out*W_out * sizeof(float), cudaMemcpyHostToDevice);

    // Multiplicación de matrices
    multiplicarMatricesGPU<<<grid_grad_w, block, smem>>>(this->n_kernels, cols_input_unroll, H_out * W_out, d_output, d_input_unroll, d_grad_w);    // Gradiente respecto a pesos
    multiplicarMatricesGPU<<<grid_grad_input, block, smem>>>(fils_matriz_pesos, cols_output_unroll, cols_matriz_pesos, d_matriz_pesos, d_output_unroll, d_input);  // Gradiente respecto a entrada

    // Paso de GPU a CPU
    cudaMemcpy(h_grad_w, d_grad_w, this->n_kernels*C*K*K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_input, d_input, C*H*W * sizeof(float), cudaMemcpyDeviceToHost);

    // Input
    for(int i = 0; i < C; i++) 
        for(int j = 0; j < H; j++)
            for(int k = 0; k < W; k++)
                input[i][j][k] = h_input[i*H*W +j*W +k];
    
    // Sumar el gradiente respecto a pesos
    for(int i = 0; i < this->n_kernels; i++) 
        for(int j = 0; j < C; j++)
            for(int kx = 0; kx < K; kx++)
                for(int ky = 0; ky < K; ky++)
                    grad_w[i][j][kx][ky] += h_grad_w[i*C*K*K + j*K*K + kx*K + ky];

    // Calcular el gradiente del bias
    for(int i=0; i<output.size(); ++i)
        for(int j=0; j<output[0].size(); ++j)    
            for(int k=0; k<output[0][0].size(); ++k)
                grad_bias[i] += output[i][j][k];
    

    // Liberar espacio
    free(output_pad_ptr); free(h_output_unroll); free(h_matriz_pesos); free(h_input);
    cudaFree(d_output_unroll); cudaFree(d_matriz_pesos); cudaFree(d_input);
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
    @maxs           Se asigna una posición a cada hebra. Contiene el valor máximo encontrado
    @mins           Se asigna una posición a cada hebra. Contiene el valor mínimo encontrado
    @return         Se actualizan los valores de w (pesos de la capa)
*/
void Convolutional::escalar_pesos(float clip_value, vector<float> &maxs, vector<float> &mins)
{
    /*
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
    */
}

/*
    @brief          Actualizar los pesos y sesgos de la capa
    @grad_w         Gradiente de cada peso de la capa
    @grad_b         Gradiente de cada sesgo de la capa
    @return         Se actualizar los valores de w y bias (pesos y sesgos de la capa)
*/
void Convolutional::actualizar_grads(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias)
{
    /*
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
    */
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

int main()
{
    // -----------------------------------------------------------------------------------------------------
    // Método estándar
    // -----------------------------------------------------------------------------------------------------
    auto ini = high_resolution_clock::now();
    auto fin = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(fin - ini);
    int n_kernels = 26, K=5, H=50, W=50, H_out = H-K+1, W_out = W-K+1, pad = 0, C=100;   // C=9
    vector<vector<vector<float>>> a, input_gpu, a_gpu;
    vector<vector<vector<vector<float>>>> grad_w, grad_w2;
    vector<float> grad_bias;

    vector<vector<vector<float>>> input(C, vector<vector<float>>(H, vector<float>(W, 0)));
    vector<vector<vector<float>>> output(n_kernels, vector<vector<float>>(H_out, vector<float>(W_out, 0))), output_gpu = output;

    for(int i=0; i<C; i++)
        for(int j=0; j<H; j++)
            for(int k=0; k<W; k++)
                input[i][j][k] = (float) (i+j+k) / (C*H*W);
                //h_A[i*K + j] = (rand() % 100) + 1;
                //input[i][j][k] = 3.0;
    
    a = output, input_gpu = input, a_gpu = a;

    // Kernel de pesos
    vector<vector<vector<vector<float>>>> w(n_kernels, vector<vector<vector<float>>>(C, vector<vector<float>>(K, vector<float>(K, 0.7))));
    
    // Vector de sesgos
    vector<float> bias(n_kernels, 0.0);

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
    Convolutional conv(n_kernels, K, K, input, 0.01);
    conv.set_w(w);
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
    */
    
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
    */

   // Comprobar resultados
    bool correcto = true;
    float epsilon = 0000000.1;
    int n_errores = 0;
    float err_medio_input = 0.0;

    for(int i=0; i<n_kernels; i++)
        for(int j=0; j<H_out; j++)
            for(int k=0; k<W_out; k++)
                if(abs(output[i][j][k] - output_gpu[i][j][k]) > epsilon)
                {
                    correcto = false;
                    cout << abs(output[i][j][k] - output_gpu[i][j][k]) << "output" << endl;
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
                }



    if(correcto)
        cout << "Todo correcto" << endl;
    else
    {
        cout << "Incorrecto (" << n_errores << " errores) " << endl;
        cout << "Error medio input: " << err_medio_input / C*H*W << endl;
    }

    return 0;
}