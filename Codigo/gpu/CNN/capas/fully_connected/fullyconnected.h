
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
        int *capas_wT = nullptr;
        int *i_w_ptr = nullptr;     // Índice de cada capa de pesos
        int *i_capa = nullptr;      // Índice de cada capa de neuronas
        float * wT_ptr = nullptr;
        int *i_wT = nullptr;     // Índice de cada capa de pesos
        int n_neuronas;
        int n_pesos;
        bool liberar_memoria;
        float ** capasGEMM = nullptr;
        int mini_batch;
        int * i_capasGEMM = nullptr;

        // Punteros device
        float * d_wT = nullptr;
        float * d_w = nullptr;
        float * d_b = nullptr;
        float * d_z = nullptr;
        float * d_a = nullptr;
        float * d_aT = nullptr;
        float * d_grad_w = nullptr;
        float * d_grad_b = nullptr;
        float * d_y = nullptr;
        int * d_i_capasGEMM = nullptr;
        float *d_sum_acc_entr = nullptr;
        int * d_capas_wT = nullptr, *d_capas = nullptr;
        float *d_max = nullptr, *d_min = nullptr;
        float *d_grad_x = nullptr;


        // Tamaño de bloque
        dim3 block;
        dim3 block_softmax;

        // Tamaño de grid
        dim3 grid_forward;          // Grid para realizar la propagación hacia delante
        dim3 grid_softmax;          
        dim3 grid_reduce;
        dim3 bloque_reduce;

        
        int block_size;

        size_t smem;            // Memoria compartida requerida por el kernel
        size_t smem_reduce;

        float *d_softmax = nullptr;


    public:
        // GPU ---------------------------------------
        FullyConnected(int *capas, int n_capas, float lr, int mini_batch);
        
        ~FullyConnected()
        {
            if(liberar_memoria)
            {
                free(w_ptr); free(bias_ptr); free(capas); free(i_w_ptr); free(i_capa); free(wT_ptr); free(capas_wT); free(i_wT); free(i_capasGEMM);
                for(int i=0; i<n_capas; i++)
                    free(capasGEMM[i]);
                 
                free(capasGEMM); 
                cudaFree(d_wT); cudaFree(d_z); cudaFree(d_a); cudaFree(d_aT); cudaFree(d_w); cudaFree(d_b); cudaFree(d_grad_w); cudaFree(d_grad_b);
                cudaFree(d_y); cudaFree(d_i_capasGEMM); cudaFree(d_sum_acc_entr); cudaFree(d_capas_wT); cudaFree(d_capas); cudaFree(d_capas_wT);
                cudaFree(d_max); cudaFree(d_min); cudaFree(d_grad_x);
            }
        };
        
        void set_train(float *x, float *y);
        void generar_pesos_ptr(const int &capa);
        void mostrar_neuronas_ptr(float *z_ptr);
        void copiar_w_de_vector_a_ptr(vector<vector<vector<float>>> w_);
        void mostrar_pesos_ptr();
        void mostrar_pesos_Traspuestos_ptr();
        float accuracy_ptr(float *x, float *y, int n_datos, float *a_ptr, float *z_ptr);
        void actualizar_parametros_ptr(float *grad_pesos, float *grad_b);
        void actualizar_parametros_gpu();
        void escalar_pesos_ptr(float clip_value);
        void escalar_pesos_GEMM(float clip_value);
        void matrizTranspuesta(float* matrix, int rows, int cols);
        void matrizTranspuesta(float* X, float *Y, int rows, int cols);

        // CPU ---------------------------------------
        // Constructores
        FullyConnected(){};

        // Funciones de activación
        float deriv_relu(const float &x);
        float relu(const float &x);
        float sigmoid(const float &x);

        // Propagación hacia delante
        void forwardPropagation_ptr(float *x, float *a, float *z);
        void forwardPropagationGEMM();

        // Cálculo de gradientes
        void train_ptr(float *x, float *y, int *batch, const int &n_datos, float * grad_w_ptr, float * grad_bias_ptr, float *grad_x, float *a_ptr, float *z_ptr, float *grad_a_ptr);
        void trainGEMM(float *grad_x);

        // Medidas de evaluación
        void evaluar_modelo_GEMM();
        float cross_entropy_ptr(float *x, float *y, int n_datos, float *a_ptr, float *z_ptr);

        // Modificación de parámetros
        void escalar_pesos(float clip_value);
        void generar_pesos(const int &capa);

        // Gets
        float * get_pesos_ptr(){return this->w_ptr;};
        float * get_bias_ptr(){return this->bias_ptr;};
        int get_n_capas(){return this->n_capas;};
        int * get_capas(){return this->capas;};
        int get_n_neuronas(){return this->n_neuronas;};
        int get_n_pesos(){return this->n_pesos;};

        void set_biasGEMM(float *bias);
        void set_wGEMM(float *w);

        // Debug
        void mostrar_pesos();
        int * get_i_w_ptr(){return this->i_w_ptr;};
        int * get_i_capa(){return this->i_capa;};
};

#endif
