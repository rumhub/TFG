#include <vector>

using namespace std;

class PoolingMax
{
    private:
        int kernel_fils;
        int kernel_cols;
        int image_fils;
        int image_cols;
        int image_canales;
        
    public:
        PoolingMax(int kernel_fils, int kernel_cols, const vector<vector<vector<float>>> &input);
        PoolingMax(){};

        // Aplica padding a un conjunto de imágenes 2D
        void forwardPropagation(const vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output);

        void backPropagation(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output, const vector<vector<vector<float>>> &grad_output);

        void mostrar_tam_kernel();

        int get_kernel_fils(){return this->kernel_fils;};
        int get_kernel_cols(){return this->kernel_cols;};
        int get_image_canales(){return this->image_canales;};
};

// https://www.linkedin.com/pulse/implementation-from-scratch-forward-back-propagation-layer-coy-ulloa