
#ifndef FULLYCONNECTED_INCLUDED
#define FULLYCONNECTED_INCLUDED

#include "../../auxiliar/leer_imagenes.cpp"
#include <vector>

using namespace std;

// Prueba, una red con una sola capa oculta. Siempre capa output solo 1 neurona. Clasificación binaria
class FullyConnected
{
    protected:
        vector<vector<vector<float>>> w;        // Pesos
        vector<vector<vector<float>>> grad_w;   // Gradiente de los pesos

        // Neuronas
        vector<vector<float>> a;    // X*W + B, neurona antes de aplicar función de activación   
        vector<vector<float>> z;    // f(X*W + B), neurona después de aplicar función de activación
        vector<vector<float>> grad_a;   // Gradiente respecto a la entrada de la neurona
        vector<vector<float>> bias; // Un sesgo o bias por neurona
        vector<vector<float>> grad_bias;
        float lr;

    public:
        FullyConnected(const vector<int> &capas, const float & lr=0.1);
        FullyConnected(){};

        float generar_peso(int neuronas_in);

        void mostrarpesos();

        void mostrarNeuronas();

        void mostrarbias();

        float deriv_relu(float x);

        float relu(float x);

        float sigmoid(float x);

        void forwardPropagation(const vector<float> &x);

        void mostrar_prediccion(vector<float> x, float y);

        void mostrar_prediccion_vs_verdad(vector<float> x, float y);

        float accuracy(vector<vector<float>> x, vector<vector<float>> y);

        float cross_entropy(vector<vector<float>> x, vector<vector<float>> y);

        void train(const vector<vector<float>> &x, const vector<vector<float>> &y, vector<vector<float>> &grad_x);

        void generarDatos(vector<vector<float>> &x, vector<float> &y);

        void setLR(float lr);

        void leer_imagenes_mnist(vector<vector<float>> &x, vector<vector<float>> &y, const int n_imagenes, const int n_clases);

};

#endif
