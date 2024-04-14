#include <vector>

using namespace std;

class Convolutional
{
    private:
        vector<vector<vector<vector<float>>>> w;    // w[n][d][i][j]   --> Matriz de pesos([d][i][j]) respecto al kernel n. (d = depth del kernel)
        //vector<vector<vector<vector<float>>>> grad_w;   // Gradiente de los pesos
        vector<vector<vector<float>>> a;

        int n_kernels;
        int kernel_fils;
        int kernel_cols;
        int kernel_depth;
        int image_fils;
        int image_cols;

        // Un bias por filtro, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
        vector<float> bias; 
        //vector<float> grad_bias;
        float lr;

    public:

        // Constructor
        Convolutional(int n_kernels, int kernel_fils, int kernel_cols, const vector<vector<vector<float>>> &input, float lr);
        Convolutional(){};

        void generar_pesos();
         
		float activationFunction(float x);

        float deriv_activationFunction(float x);

        void aplicar_padding(vector<vector<vector<float>>> &imagen_3D, int pad);

        void forwardPropagation(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &a);

        // https://calvinfeng.gitbook.io/machine-learning-notebook/supervised-learning/convolutional-neural-network/convolution_operation

        void backPropagation(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> output, vector<vector<vector<float>>> &a, vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias, const int &pad);


        void flip_w(vector<vector<vector<vector<float>>>> &w_flipped);

        void reset_gradients(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias);

        void actualizar_grads(vector<vector<vector<vector<float>>>> &grad_w, vector<float> &grad_bias, int n_datos);

        int get_kernel_fils(){return this->kernel_fils;};
        int get_kernel_cols(){return this->kernel_cols;};
        int get_kernel_depth(){return this->kernel_depth;};
        int get_n_kernels(){return this->n_kernels;};

        void escalar_pesos(float clip_value);

        vector<vector<vector<vector<float>>>> get_pesos(){return this->w;};
        vector<float> get_bias(){return this->bias;};

        // Debug ---------
        void w_a_1();
};