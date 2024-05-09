#include "poolingMax.h"
#include <iostream>
#include <limits>
#include "../../auxiliar/auxiliar.cpp"

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


void PoolingMax::forwardPropagationGPU(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad)
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

    int C = input.size(), H = input[0].size(), W =input[0][0].size(), H_out = H/ this->kernel_fils +2*pad, W_out = W / this->kernel_cols +2*pad;
    int bytes_input = C*H*W * sizeof(float), bytes_output = C*H_out*W_out * sizeof(float);

    // Crear tamaño de bloque y grid
    dim3 block(8, 8);
    dim3 grid(ceil( (float)(W + block.x -1) / block.x), ceil((float)(H + block.y -1) / block.y));

    // Reserva memoria en host
    float *h_input = (float *)malloc(bytes_input), 
          *h_output = (float *)malloc(bytes_output);

    // Reserva memoria en device
    float *d_input, *d_output;
    cudaMalloc((void **) &d_input, bytes_input);
    cudaMalloc((void **) &d_output, bytes_output);

    // Copiar entrada en CPU
    for(int i=0; i<C; i++)
        for(int j=0; j<H; j++)
            for(int k=0; k<W; k++)
                h_input[i*H*W + j*W + k] = input[i][j][k];

    // Copiar datos de CPU a GPU
    cudaMemcpy(d_input, h_input, bytes_input, cudaMemcpyHostToDevice);

    // Realizar MaxPool
    maxpool<<<grid, block>>>(C, H, W, K, d_input, d_output, pad);

    cudaMemcpy(h_output, d_output, bytes_output, cudaMemcpyDeviceToHost);

    // Copiar datos de GPU a CPU
    for(int i=0; i<C; i++)
        for(int j=0; j<H_out; j++)
            for(int k=0; k<W_out; k++)
                output[i][j][k] = h_output[i*H_out*W_out + j*W_out + k];
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
    int C=2, H=15, W=15, K=5, H_out = H/K, W_out = W/K;
    vector<vector<vector<float>>> input_cpu(C, vector<vector<float>>(H, vector<float>(W, 0))), input_copy_cpu = input_cpu, input_gpu, input_copy_gpu;
    vector<vector<vector<float>>> output_cpu(C, vector<vector<float>>(H_out, vector<float>(W_out, 0))), output_gpu;

    for(int i=0; i<C; i++)
        for(int j=0; j<H; j++)
            for(int k=0; k<W; k++)
                input_cpu[i][j][k] = i+j+k;


    int pad = 1;    // Padding se mete en capas convolucionales. Por tanto, si metemos padding de pad, antes de la capa pooling_max (input) hay que quitarlo al hacer backprop
    Aux *aux = new Aux();

    cout << "--------------- SIMULACIÓN GRADIENTE ------------------ " << endl;

    // Aplicamos padding al output
    // Output viene ya con las dimensiones tras aplicar padding
    aplicar_padding(output_cpu, pad);
    output_gpu = output_cpu;

    cout << "------------ Imagen inicial: ------------" << endl;
    aux->mostrar_imagen(input_cpu);
    input_copy_cpu = input_cpu;
    input_gpu = input_cpu;
    input_copy_gpu = input_copy_cpu;

    PoolingMax plm1(K, K, input_cpu);

    plm1.forwardPropagation(input_cpu, output_cpu, input_copy_cpu, pad);

    cout << "Output \n";
    aux->mostrar_imagen(output_cpu);


    // ---------------- GPU --------------------------
    cout << " ------------------ GPU ---------------------" << endl;
    plm1.forwardPropagationGPU(input_gpu, output_gpu, input_copy_gpu, pad);
    aux->mostrar_imagen(output_gpu);

    
    /*
    cout << "------------ Pooling Max, Back Propagation: ------------" << endl;

    // Cambiamos el output porque contendrá un gradiente desconocido
    for(int i=0; i<output_cpu.size(); i++)
        for(int j=0; j<output_cpu[0].size(); j++)
            for(int k=0; k<output_cpu[0][0].size(); k++)
                output_cpu[i][j][k] = 9;

    //cout << "--- Output modificado ---- \n";
    //aux->mostrar_imagen(output);

    plm1.backPropagation(input, output_cpu, input_copy, pad);

    cout << "Input\n";
    aux->mostrar_imagen(input);
    */

    return 0;
}

