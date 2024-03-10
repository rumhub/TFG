
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
        vector<vector<float>> neuronas;   // Nº de neuronas por capa. (neuronas.size() es el nº de capas que tiene la red)
        vector<vector<float>> bias; // Un bias por neurona
        vector<vector<float>> grad_bias;
        float lr;

    public:
        FullyConnected(const vector<int> &capas, const float & lr=0.1);
        FullyConnected(){};

        void mostrarpesos();

        void mostrarNeuronas();

        void mostrarbias();

        float deriv_relu(float x);

        float relu(float x);

        float sigmoid(float x);

        void forwardPropagation(const vector<float> &x);

        void mostrar_prediccion(vector<float> x, float y);

        void mostrar_prediccion_vs_verdad(vector<float> x, float y);

        float accuracy(vector<vector<float>> x, vector<float> y);

        float binary_loss(vector<vector<float>> x, vector<float> y);

        virtual void train(const vector<vector<float>> &x, const vector<float> &y, vector<vector<float>> &grad_x);

        void generarDatos(vector<vector<float>> &x, vector<float> &y);

        void setLR(float lr);

        void leer_imagenes_mnist(vector<vector<float>> &x, vector<vector<float>> &y, const int n_imagenes, const int n_clases);

};

#endif
