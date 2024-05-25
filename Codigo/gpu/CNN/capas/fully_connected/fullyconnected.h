
#ifndef FULLYCONNECTED_INCLUDED
#define FULLYCONNECTED_INCLUDED

#include <vector>
#include <fstream>
#include <iostream>
#include "math.h"
#include "random"
#include <stdio.h>
#include <omp.h>

using namespace std;

class FullyConnected
{
    protected:
        // CPU -----------------------------
        vector<vector<vector<float>>> w;    // Pesos
        vector<vector<float>> a;        // Neuronas
        vector<vector<float>> bias;     // Bias o Sesgos
        float lr;   // Learning Rate o Tasa de Aprendizaje

        // GPU -----------------------------
        float * w_ptr = nullptr;
        float * bias_ptr = nullptr;
        int n_capas;
        int *capas = nullptr;
        int *i_w_ptr = nullptr;     // Índice de cada capa de pesos
        int *i_capa = nullptr;      // Índice de cada capa de neuronas
        int n_neuronas;
        int n_pesos;

    public:
        // GPU ---------------------------------------
        FullyConnected(int *capas, int n_capas, float lr);
        ~FullyConnected(){free(w_ptr); free(bias_ptr); free(capas); free(i_w_ptr); free(i_capa);};
        void generar_pesos_ptr(const int &capa);
        void forwardPropagation_ptr(float *x, float *a, float *z);
        void mostrar_neuronas_ptr(float *z_ptr);
        void copiar_w_de_vector_a_ptr(vector<vector<vector<float>>> w_);
        void mostrar_pesos_ptr();
        float cross_entropy_ptr(float *x, float *y, int n_datos, float *a_ptr, float *z_ptr);
        float accuracy_ptr(float *x, float *y, int n_datos, float *a_ptr, float *z_ptr);
        void train_ptr(float *x, float *y, int *batch, const int &n_datos, float * grad_w_ptr, float * grad_bias_ptr, float *grad_x, float *a_ptr, float *z_ptr, float *grad_a_ptr);
        void actualizar_parametros_ptr(float *grad_pesos, float *grad_b);
        void escalar_pesos_ptr(float clip_value);

        // CPU ---------------------------------------
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
        void train(const vector<vector<float>> &x, const vector<vector<float>> &y, const vector<int> &batch, const int &n_datos, vector<vector<vector<float>>> &grad_pesos, vector<vector<float>> &grad_b, vector<vector<float>> &grad_x, vector<vector<float>> &a, vector<vector<float>> &z, vector<vector<float>> &grad_a);

        // Medidas de evaluación
        float accuracy(vector<vector<float>> x, vector<vector<float>> y);
        float cross_entropy(vector<vector<float>> x, vector<vector<float>> y);

        // Modificación de parámetros
        void actualizar_parametros(const vector<vector<vector<float>>> &grad_pesos, const vector<vector<float>> &grad_b);
        void escalar_pesos(float clip_value);
        void generar_pesos(const int &capa);

        // Gets
        float * get_pesos_ptr(){return this->w_ptr;};
        float * get_bias_ptr(){return this->bias_ptr;};
        vector<vector<vector<float>>> get_pesos(){return this->w;};
        vector<vector<float>> get_bias(){return this->bias;};
        vector<vector<float>> get_a(){return this->a;};
        int get_n_capas(){return this->n_capas;};
        int * get_capas(){return this->capas;};
        int get_n_neuronas(){return this->n_neuronas;};
        int get_n_pesos(){return this->n_pesos;};

        // Debug
        void set_pesos(const vector<vector<vector<float>>> &w_){this->w = w_;};
        void mostrar_neuronas(const vector<vector<float>> &z);
        void mostrar_pesos();
        int * get_i_w_ptr(){return this->i_w_ptr;};
        int * get_i_capa(){return this->i_capa;};
};

#endif
