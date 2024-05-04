#include <vector>

using namespace std;

class Convolutional
{
    private:
        vector<vector<vector<vector<float>>>> w;    // Pesos de la capa convolucional
        vector<vector<vector<float>>> a;            // Valor de cada neurona antes de aplicar la función de activación

        int n_kernels;                          // Número de kernels 3D
        int kernel_fils;                        // Número de filas por kernel
        int kernel_cols;                        // Número ded columnas por kernel
        int kernel_depth;                       // Número de canales de profundidad por kernel

        vector<float> bias;                     // Un bias por filtro, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
        float lr;                               // Learning Rate o Tasa de Aprendizaje

    public:

        // Constructores
        Convolutional(int n_kernels, int kernel_fils, int kernel_cols, const vector<vector<vector<float>>> &input, float lr);
        Convolutional(){};

        // Funciones de activación
		float activationFunction(float x);
        float deriv_activationFunction(float x);

        // Propagación hacia delante
        void forwardPropagation(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &a);

        void forwardPropagationGEMM(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &a);

        // Retropropagación
        void backPropagation(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> output, const vector<vector<vector<float>>> &a, vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias, const int &pad);
        
        void backPropagationGEMM(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> output, const vector<vector<vector<float>>> &a, vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias, const int &pad);

        // Modificación de parámetros
        void generar_pesos();
        void reset_gradients(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias);
        void actualizar_grads(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias);
        void escalar_pesos(float clip_value, vector<float> &maxs, vector<float> &mins);
        void matrizTranspuesta(float* matrix, int rows, int cols);
        void unroll(int C, int n, int K, float *X, float *X_unroll);

        // Aplicar padding
        void aplicar_padding(vector<vector<vector<float>>> &imagen_3D, int pad);

        // Gets
        int get_kernel_fils(){return this->kernel_fils;};
        int get_kernel_cols(){return this->kernel_cols;};
        int get_kernel_depth(){return this->kernel_depth;};
        int get_n_kernels(){return this->n_kernels;};
        vector<vector<vector<vector<float>>>> get_pesos(){return this->w;};
        vector<float> get_bias(){return this->bias;};
        // https://calvinfeng.gitbook.io/machine-learning-notebook/supervised-learning/convolutional-neural-network/convolution_operation

        // Debug
        void set_w(const vector<vector<vector<vector<float>>>> &w_){this->w = w_;};
        void set_b(const vector<float> &b){this->bias = b;};
        void printMatrix_3D(float* matrix, int C, int n);
        void printMatrix(float* matrix, int h, int w);
        void printMatrix_vector(const vector<vector<vector<float>>> &X);
        void multiplicarMatrices(float* m1, int rows1, int cols1, float* m2, int cols2, float* result);
        void printMatrix_4D(float* matrix, int F, int C, int n);
};