
#ifndef FULLYCONNECTED_INCLUDED
#define FULLYCONNECTED_INCLUDED

#include <vector>
#include <fstream>

using namespace std;

class FullyConnected
{
    protected:
        vector<vector<vector<float>>> w;    // Pesos
        vector<vector<float>> a;        // Neuronas
        vector<vector<float>> bias;     // Bias o Sesgos
        float lr;   // Learning Rate o Tasa de Aprendizaje

    public:
        // Constructores
        FullyConnected(const vector<int> &capas, const float & lr=0.1);
        FullyConnected(){};

        // Funciones de activación
        float deriv_relu(const float &x);
        float relu(const float &x);
        float sigmoid(const float &x);

        // Propagación hacia delante
        void forwardPropagation(const vector<float> &x, vector<vector<float>> &a, vector<vector<float>> &z);

        // Cálculo de gradientes
        void train(const vector<vector<float>> &x, const vector<vector<float>> &y, const vector<int> &batch, const int &n_datos, vector<vector<vector<float>>> &grad_pesos, vector<vector<float>> &grad_b, vector<vector<float>> &grad_x, vector<vector<float>> &a, vector<vector<float>> &z, vector<vector<float>> &grad_a, const int &n_thrs);

        // Medidas de evaluación
        float accuracy(vector<vector<float>> x, vector<vector<float>> y, const int &ini);
        float cross_entropy(vector<vector<float>> x, vector<vector<float>> y, const int &ini);

        // Modificación de parámetros
        void actualizar_parametros(vector<vector<vector<vector<float>>>> &grad_pesos, vector<vector<vector<float>>> &grad_b);
        void escalar_pesos(float clip_value, vector<float> &maxs, vector<float> &mins);
        void generar_pesos(const int &capa);

        // Gets
        vector<vector<vector<float>>> get_pesos(){return this->w;};
        vector<vector<float>> get_bias(){return this->bias;};
        vector<vector<float>> get_a(){return this->a;};
};

#endif
