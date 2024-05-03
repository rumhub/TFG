#include "convolutional.cpp"

// Function to print a matrix
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

void printMatrix_4D(float* matrix, int F, int C, int n) {
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

void unroll(int C, int n, int K, float *X, float *X_unroll){
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
                    }
                }

        
    } 
}

void printMatrix(float* matrix, int h, int w) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) 
            cout << matrix[i * w + j] << " ";
        cout << endl;
    }
}

void multiplyMatrices(float* m1, int rows1, int cols1, float* m2, int cols2, float* result) {
    for (int i = 0; i < rows1; i++) 
        for (int j = 0; j < cols2; j++) {
            result[i * cols2 + j] = 0.0f;

            for (int k = 0; k < cols1; k++) 
                result[i * cols2 + j] += m1[i * cols1 + k] * m2[k * cols2 + j];
            
        }
}

int main()
{
    // -----------------------------------------------------------------------------------------------------
    // Método estándar
    // -----------------------------------------------------------------------------------------------------
    int n_kernels = 2, K=2, H=3, W=H, H_out = H-K+1, W_out = W-K+1;
    vector<vector<vector<float>>> input = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}}}, a = input;
    vector<vector<vector<float>>> output(n_kernels);

    // Crear volumen de salida
    vector<vector<float>> aux_2D(H_out);
    vector<float> aux_1D(W_out);

    for(int i=0; i<aux_2D.size(); i++)
        aux_2D[i] = aux_1D;
    
    for(int i=0; i<output.size(); i++)
        output[i] = aux_2D;

    // Kernel de pesos
    vector<vector<vector<vector<float>>>> w = {{{{1, 1}, {1,1}}, {{1, 1}, {1,1}}}, {{{2, 2}, {2,2}}, {{2, 2}, {2,2}}}};
    // Vector de sesgos
    vector<float> bias(n_kernels, 0.0);

    Convolutional conv(n_kernels, K, K, input, 0.01);
    conv.set_w(w);
    conv.set_b(bias);

    cout << " -------------------------- Método Estándar -------------------------- " << endl;
    cout << "Input" << endl;
    printMatrix_vector(input);

    conv.forwardPropagation(input, output, a);

    cout << "Ouput" << endl;
    printMatrix_vector(output);


    // -----------------------------------------------------------------------------------------------------
    // Método GEMM
    // -----------------------------------------------------------------------------------------------------
    int n = 3; // Size of the matrix (n x n)

    int C = 2;

    float* X = (float*)malloc(n * n * C * sizeof(float));
    float* X_unroll = (float*)malloc(K*K*C *H_out*W_out *sizeof(float));
    float *kernel_W = (float *) malloc(K*K*C*n_kernels * sizeof(float));
    float *kernel_bias = (float *) malloc(n_kernels * sizeof(float));
    float *result = (float*)malloc(H_out * W_out * n_kernels * sizeof(float));

    // Inicializar matriz de entrada X
    int cont = 1;
    for (int i = 0; i < C; i++) 
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                X[i*n*n +j*n +k] = cont++;
            }
        }

    // Inicializar matriz de pesos 
    for (int f = 0; f < n_kernels; f++) 
        for (int i = 0; i < C; i++) 
            for (int j = 0; j < K; j++) {
                for (int k = 0; k < K; k++) {
                    kernel_W[f*C*K*K + i*K*K + j*K + k] = f+1;
                }
            }
    
    // Inicializar vector de sesgos
    for (int f = 0; f < n_kernels; f++) 
        kernel_bias[f] = 0.0;

    cout << " -------------------------- Método GEMM -------------------------- " << endl;

    cout << "Input" << endl;
    printMatrix_3D(X, C, n);

    //cout << "Kernel de pesos" << endl;
    //printMatrix_4D(kernel_W, n_kernels, C, K);
    //printMatrix(kernel_W, n_kernels, K*K*C);

    
    unroll(C, n, K, X, X_unroll);
    //cout << "-------- Unroll ----------:" << endl;
    //printMatrix(X_unroll, K*K*C, H_out * W_out);

    multiplyMatrices(kernel_W, n_kernels, K*K*C, X_unroll, H_out*W_out, result); 
    cout << "Output" << endl;
    printMatrix_3D(result, n_kernels, H_out);
    

    return 0;
}