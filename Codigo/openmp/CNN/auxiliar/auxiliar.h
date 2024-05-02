#ifndef AUXILIAR_H_INCLUDED
#define AUXILIAR_H_INCLUDED

#include <vector>

using namespace std;

class Aux
{
    private:
    public:
        Aux(){};

        void mostrar_imagen(vector<vector<float>> imagen);

        void mostrar_imagen(vector<vector<vector<float>>> imagenes_2D);

        void mostrar_imagen(vector<vector<vector<vector<float>>>> imagenes);

};

#endif