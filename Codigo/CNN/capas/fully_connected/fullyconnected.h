
#ifndef FULLYCONNECTED_INCLUDED
#define FULLYCONNECTED_INCLUDED

//#include "../../auxiliar/leer_imagenes.cpp"
#include <vector>
#include <fstream>

using namespace std;

// Prueba, una red con una sola capa oculta. Siempre capa output solo 1 neurona. Clasificación binaria
class FullyConnected
{
    protected:
        vector<vector<vector<float>>> w;        // Pesos

        vector<vector<float>> a;
        vector<vector<float>> bias; // Un sesgo o bias por neurona
        float lr;

    public:
        FullyConnected(const vector<int> &capas, const float & lr);
        FullyConnected(){};

        void generar_pesos(const int &capa, vector<vector<float>> &a);

        void particion_k_fold(const vector<vector<float>> &x, const vector<vector<float>> &y, const int &k);

        void leer_atributos(vector<vector<float>> &x, vector<vector<float>> &y, string fichero);

        void mostrarpesos();

        void mostrarbias();

        float deriv_relu(const float &x);

        float relu(const float &x);

        float sigmoid(const float &x);

        void forwardPropagation(const vector<float> &x, vector<vector<float>> &a, vector<vector<float>> &z);

        float accuracy(vector<vector<float>> x, vector<vector<float>> y, vector<vector<float>> &a, vector<vector<float>> &z);

        float cross_entropy(vector<vector<float>> x, vector<vector<float>> y, vector<vector<float>> &a, vector<vector<float>> &z);

        void train(const vector<vector<float>> &x, const vector<vector<float>> &y, const vector<int> &indices, const int &n_datos, vector<vector<vector<float>>> &grad_w, vector<vector<float>> &grad_bias, vector<vector<float>> &grad_x, vector<vector<float>> &a, vector<vector<float>> &z, vector<vector<float>> &grad_a);

        void actualizar_parametros(vector<vector<vector<float>>> &grad_pesos, vector<vector<float>> &grad_b);

        void escalar_pesos(float clip_value);

        void inicializar_parametros();

        void setLR(float lr);

        //void leer_imagenes_mnist(vector<vector<float>> &x, vector<vector<float>> &y, const int n_imagenes, const int n_clases);

        void copiar_parametros(FullyConnected &fc);

        vector<vector<vector<float>>> get_pesos(){return this->w;};
        vector<vector<float>> get_bias(){return this->bias;};

        void reservar_espacio(vector<vector<float>> &a, vector<vector<float>> &z, vector<vector<float>> &grad_a){a = this->a; z=this->a; grad_a = this->a;};

        void reset_gradients(vector<vector<vector<float>>> &grad_w, vector<vector<float>> &grad_bias);
};

#endif
