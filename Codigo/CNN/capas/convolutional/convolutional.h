#include <vector>

using namespace std;

class Convolutional
{
    private:
        vector<vector<vector<vector<float>>>> w;    // w[n][d][i][j]   --> Matriz de pesos([d][i][j]) respecto al kernel n. (d = depth del kernel)
        vector<vector<vector<vector<float>>>> grad_w;   // Gradiente de los pesos
        int n_kernels;
        int kernel_fils;
        int kernel_cols;
        int kernel_depth;
        int image_fils;
        int image_cols;

        // Un bias por filtro, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
        vector<float> bias; 
        vector<float> grad_bias;
        float lr;
    
    public:

        // Constructor
        Convolutional(int n_kernels, int kernel_fils, int kernel_cols, const vector<vector<vector<float>>> &input, float lr);
        Convolutional(){};

		float activationFunction(float x);

        float deriv_activationFunction(float x);

        void aplicar_padding(vector<vector<vector<float>>> &imagen_3D, int pad);

        void forwardPropagation(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output);

        // https://calvinfeng.gitbook.io/machine-learning-notebook/supervised-learning/convolutional-neural-network/convolution_operation

        void backPropagation_libro(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output);

        void backPropagation_bibliografia(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> output, const int &pad);

        void backPropagation(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> output, const int &pad);

        void mostrar_pesos();

        void mostrar_grad();

        void flip_w(vector<vector<vector<vector<float>>>> &w_flipped);

        // Capa de convolución o capa de correlación
        void conv_corr(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<vector<float>>>> pesos, bool valid, bool conv);

        void reset_gradients();

        void actualizar_grads(int n_datos);

        void mostrar_tam_kernel();

        int get_kernel_fils(){return this->kernel_fils;};
        int get_kernel_cols(){return this->kernel_cols;};
        int get_kernel_depth(){return this->kernel_depth;};
        int get_n_kernels(){return this->n_kernels;};

        // Debug ---------
        void w_a_1();
};