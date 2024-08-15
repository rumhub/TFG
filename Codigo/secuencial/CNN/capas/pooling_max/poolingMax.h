#ifndef MAXPOOL_H
#define MAXPOOL_H

#include <vector>
#include <math.h>
#include <iostream>
#include <chrono>
#include "random"
#include "omp.h"
#include <stdlib.h>
#include <stdio.h>
#include <cfloat>
#include <limits>

using namespace std;

class PoolingMax
{
    private:

        // Dimensiones de la ventana o kernel
        int kernel_fils;
        int kernel_cols;

        // Dimensiones de la imagen
        int image_fils;
        int image_cols;
        int image_canales;
        
    public:

        // Constructor
        PoolingMax(int kernel_fils, int kernel_cols, vector<vector<vector<float>>> &input);
        
        // Destructor
        PoolingMax(){};

        // Propagación hacia delante
        void forwardPropagation(vector<vector<vector<float>>> &input, vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad);

        // Retropropagación
        void backPropagation(vector<vector<vector<float>>> &input, const vector<vector<vector<float>>> &output, vector<vector<vector<float>>> &input_copy, const int &pad_output);
        
        // Gets
        int get_kernel_fils(){return this->kernel_fils;};
        int get_kernel_cols(){return this->kernel_cols;};
        int get_image_canales(){return this->image_canales;};

};


// https://www.linkedin.com/pulse/implementation-from-scratch-forward-back-propagation-layer-coy-ulloa

#endif
