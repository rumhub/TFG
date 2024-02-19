#include <vector>
#include "flatten.h"
#include <iostream>
#include <chrono>

using namespace std::chrono;
using namespace std;

Flatten::Flatten(const vector<vector<vector<float>>> &input)
{              
    this->canales = input.size();
    this->filas = input[0].size();
    this->cols = input[0][0].size();
};

// input --> volumen 3D
void Flatten::forwardPropagation(const vector<vector<vector<float>>> &input, vector<float> &output)
{                 
    int n1 = input.size(), n2 = input[0].size(), n3 = input[0][0].size(), n_out = n1*n2*n3, cont=0;
    vector<float> out(n_out);

    float max = 0.0;
    for(int i=0; i<n1; i++)
        for(int j=0; j<n2; j++)
            for(int k=0; k<n3; k++)
            {
                out[cont] = input[i][j][k];
                
                //if(max < out[cont])
                //    max = out[cont]; 

                cont++;
            }

    // Normalizar entre 0 y 1 -------------------------------------------------------------
    //for(int i=0; i<cont; i++)
    //    out[i] = out[i]/max;  
    

    output = out;
};



void Flatten::backPropagation(vector<vector<vector<float>>> &errores_matriz, const vector<float> &errores_primera_capa)
{
    vector<float> v_1D(this->cols);
    vector<vector<float>> v_2D(this->filas);
    vector<vector<vector<float>>> v_3D(this->canales);
    
    // Para cada canal de cada imagen
    for(int i=0; i<this->canales; i++)
    {
        for(int j=0; j<this->filas; j++)
        {
            for(int k=0; k<this->cols; k++)
            {
                v_1D[k] = errores_primera_capa[i*this->filas*this->cols + j*this->cols + k];
            }
            v_2D[j] = v_1D;
        }
        v_3D[i] = v_2D;
    }

    errores_matriz = v_3D;
    
};

int Flatten::get_canales()
{
    return this->canales;
}

/*
void mostrar_imagen(vector<vector<float>> imagen)
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

void mostrar_imagenes_2D(vector<vector<vector<float>>> imagenes_2D)
{
    int n = imagenes_2D.size();

    for(int k=0; k<n; k++)
    {
        cout << "IMAGEN " << k << endl;

        mostrar_imagen(imagenes_2D[k]);
        cout << endl;
    }

};

void mostrar_imagenes(vector<vector<vector<vector<float>>>> imagenes)
{
    int n = imagenes.size();

    for(int k=0; k<n; k++)
    {
        cout << "Capa " << k << endl;

        mostrar_imagenes_2D(imagenes[k]);
        cout << endl;
    }

};



int main()
{
    vector<float> output;
    vector<vector<vector<float>>> imagenes_2D;
    vector<vector<float>> imagen_2D{{1.0, 2.0, 3.0, 4.0},
                                      {5.0, 6.0, 7.0, 8.0},
                                      {9.0, 10.0, 11.0, 12.0},
                                      {13.0, 14.0, 15.0, 16.0}};

    imagenes_2D.push_back(imagen_2D);

    for(int i=0; i<imagen_2D.size(); i++)
    {
        for(int j=0; j<imagen_2D[i].size(); j++)
        {
            imagen_2D[i][j] += imagen_2D.size()*imagen_2D[i].size();
        }
    }

    imagenes_2D.push_back(imagen_2D);

    cout << endl << "------------ Imagen 3D inicial -------------" << endl;
    mostrar_imagenes_2D(imagenes_2D);

    
    Flatten flt(imagenes_2D.size(), imagenes_2D[0].size(), imagenes_2D[0][0].size());
    flt.forwardPropagation(imagenes_2D, output);

    cout << "------------ Forward Propagation de la capa Flatten -------------" << endl;
    cout << endl << "Imagen 1D: " << endl;
    for(int i=0; i<output.size(); i++)
    {
        cout << output[i] << " ";
    }
    
    
    cout << "------------ Back Propagation de la capa Flatten -------------" << endl;
    flt.backPropagation(imagenes_2D, output);

    mostrar_imagenes_2D(imagenes_2D);
    

    return 0;
}
*/