#ifndef FLATTEN_H
#define FLATTEN_H

#include <iostream>
#include <vector>
#include <chrono>
#include <limits>

using namespace std::chrono;
using namespace std;

class Flatten
{
    private:
        int n_imagenes;
        int canales;
        int filas;
        int cols;
        float max;

    public:
        Flatten(const vector<vector<vector<float>>> &input);

        void forwardPropagation(const vector<vector<vector<float>>> &input, vector<float> &output);

        void backPropagation(vector<vector<vector<float>>> &errores_matriz, const vector<float> &errores_primera_capa);                     

        int get_canales();
};

#endif
