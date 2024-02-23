#include "auxiliar.h"
#include <vector>
#include <iostream>

using namespace std;

void Aux::mostrar_imagen(vector<vector<float>> imagen)
{
    int n = imagen.size();

    for(int i=0; i<n; i++)
    {
        for(int j=0; j<imagen[i].size(); j++)
        {
            cout << imagen[i][j] << " ";
        }
        cout << endl;
    }
};

void Aux::mostrar_imagen(vector<vector<vector<float>>> imagenes_2D)
{
    int n = imagenes_2D.size();

    for(int k=0; k<n; k++)
    {
        cout << "IMAGEN " << k << endl;

        mostrar_imagen(imagenes_2D[k]);
        cout << endl;
    }

};

void Aux::mostrar_imagen(vector<vector<vector<vector<float>>>> imagenes)
{
    int n = imagenes.size();

    for(int k=0; k<n; k++)
    {
        cout << "Capa " << k << endl;

        mostrar_imagen(imagenes[k]);
        cout << endl;
    }

};
