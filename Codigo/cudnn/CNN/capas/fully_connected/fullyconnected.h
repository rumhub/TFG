
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
        float * d_wT = nullptr;     // Matriz de pesos transpuesta
        float * d_w = nullptr;      // Pesos
        float * d_b = nullptr;      // Bias o Sesgos
        float * d_z = nullptr;      // Neuronas después de aplicar la función de activación
        float * d_a = nullptr;      // Neuronas antes de aplicar la función de activación
        float * d_aT = nullptr;
        float * d_grad_w = nullptr;
        float * d_grad_b = nullptr;
        float * d_y = nullptr;
        int * d_i_capasGEMM = nullptr;
        float *d_sum_acc_entr = nullptr;
        int * d_capas_wT, *d_capas;
        float *d_max = nullptr;
        float *d_min = nullptr;
        float *d_softmax = nullptr;


        // Tamaño de bloque
        dim3 block_2D;
        dim3 block_1D;

        // Tamaño de grid
        dim3 grid_2D;
        dim3 grid_1D;

        int block_size;

        size_t smem_2D;            // Memoria compartida requerida por el kernel
        size_t smem_1D;


    public:
        // Constructores
        FullyConnected(int *capas, int n_capas, float lr, int mini_batch);
        FullyConnected(){};

        // Destructor
        ~FullyConnected()
        {
            if(liberar_memoria)
            {
                free(w_ptr); free(bias_ptr); free(capas); free(i_w_ptr); free(i_capa); free(wT_ptr); free(capas_wT);
                for(int i=0; i<n_capas; i++)
                {
                    free(capasGEMM[i]);
                }
                free(capasGEMM);
                cudaFree(d_wT);
                cudaFree(d_z);
                cudaFree(d_a);
            }
        };

        // Funciones de activación
        float deriv_relu(const float &x);
        float relu(const float &x);
        float sigmoid(const float &x);

        // Propagación hacia delante
        void forwardPropagationGEMM();

        // Cálculo de gradientes
        void train_vectores_externos(float *grad_x);

        // Medidas de evaluación
        float accuracy_ptr(float *x, float *y, int n_datos, float *a_ptr, float *z_ptr);
        float cross_entropy_ptr(float *x, float *y, int n_datos, float *a_ptr, float *z_ptr);
        void evaluar_modelo_GEMM();

        // Modificación de parámetros
        void generar_pesos_ptr(const int &capa);
        void escalar_pesos_GEMM(float clip_value);
        void actualizar_parametros_gpu();

        // Gets
        float * get_pesos_ptr(){return this->w_ptr;};
        float * get_bias_ptr(){return this->bias_ptr;};
        int get_n_capas(){return this->n_capas;};
        int * get_capas(){return this->capas;};
        int get_n_neuronas(){return this->n_neuronas;};
        int get_n_pesos(){return this->n_pesos;};

        // Sets
        void set_biasGEMM(float *bias);
        void set_wGEMM(float *w);
        void set_train(float *x, float *y, int mini_batch);
        void set_train_gpu(float *x, float *y, int mini_batch);

        // Debug
        void mostrar_neuronas_ptr(float *z_ptr);
        void mostrar_pesos_ptr();
        void mostrar_pesos_Traspuestos_ptr();
        int * get_i_w_ptr(){return this->i_w_ptr;};
        int * get_i_capa(){return this->i_capa;};
};

#endif
