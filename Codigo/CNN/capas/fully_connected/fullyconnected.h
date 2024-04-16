
#ifndef FULLYCONNECTED_INCLUDED
#define FULLYCONNECTED_INCLUDED

//#include "../../auxiliar/leer_imagenes.cpp"
#include <vector>
#include <fstream>

#define THREAD_NUM 3

using namespace std;

// Prueba, una red con una sola capa oculta. Siempre capa output solo 1 neurona. Clasificación binaria
class FullyConnected
{
    protected:
        vector<vector<vector<float>>> w;        // Pesos

        // Neuronas
        vector<vector<float>> a;    // X*W + B, neurona antes de aplicar función de activación   

        // Bias
        vector<vector<float>> bias; // Un sesgo o bias por neurona

        // Learning Rate o Tasa de Aprendizaje
        float lr;

    public:
        FullyConnected(const vector<int> &capas, const float & lr=0.1);
        FullyConnected(){};

        void generar_pesos(const int &capa);

        void particion_k_fold(const vector<vector<float>> &x, const vector<vector<float>> &y, const int &k);

        void leer_atributos(vector<vector<float>> &x, vector<vector<float>> &y, string fichero);

        void mostrarpesos();

        void mostrarbias();

        float deriv_relu(const float &x);

        float relu(const float &x);

        float sigmoid(const float &x);

        // z -->  f(X*W + B), neurona después de aplicar función de activación
        void forwardPropagation(const vector<float> &x, vector<vector<float>> &a, vector<vector<float>> &z);

        float accuracy(vector<vector<float>> x, vector<vector<float>> y);

        float cross_entropy(vector<vector<float>> x, vector<vector<float>> y);

        // grad_a --> Gradiente respecto a la entrada de la neurona
        void train(const vector<vector<float>> &x, const vector<vector<float>> &y, const int &n_datos, vector<vector<vector<vector<float>>>> &grad_pesos, vector<vector<vector<float>>> &grad_b, vector<vector<float>> &grad_x, vector<vector<vector<float>>> &a, vector<vector<vector<float>>> &z, vector<vector<vector<float>>> &grad_a, const int &n_thrs);

        void actualizar_parametros(vector<vector<vector<vector<float>>>> &grad_pesos, vector<vector<vector<float>>> &grad_b);

        void escalar_pesos(float clip_value);

        void inicializar_parametros();

        void generarDatos(vector<vector<float>> &x, vector<float> &y);

        void setLR(float lr);

        //void leer_imagenes_mnist(vector<vector<float>> &x, vector<vector<float>> &y, const int n_imagenes, const int n_clases);

        void copiar_parametros(FullyConnected &fc);

        vector<vector<vector<float>>> get_pesos(){return this->w;};
        vector<vector<float>> get_bias(){return this->bias;};
        vector<vector<float>> get_a(){return this->a;};
};

#endif
