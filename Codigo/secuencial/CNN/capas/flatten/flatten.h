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
        // Imagen 3D o volumen 3D de entrada
        int n_imagenes;     // Número de imágenes 2D por volumen 3D de entrada
        int canales;        // Número de canales
        int filas;          // Filas por imagen   
        int cols;           // Columnas por imagen

    public:
        // Constructor
        Flatten(const vector<vector<vector<float>>> &input);

        // Propagación hacia delante
        void forwardPropagation(const vector<vector<vector<float>>> &input, vector<float> &output);

        // Retropropagación
        void backPropagation(vector<vector<vector<float>>> &errores_matriz, const vector<float> &errores_primera_capa);                     

        // Gets
        int get_canales();
};

#endif
