#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

void modificar_imagen(Mat imagen, const int &p_ini_x, const int &p_ini_y, const int &p_fin_x, const int &p_fin_y, const int &color_r, const int &color_g, const int &color_b)
{
    for (int x = p_ini_x; x < p_fin_x; x++) 
    {
        for (int y = p_ini_y; y < p_fin_y; y++) 
        {
            imagen.at<Vec3b>(x, y) = Vec3b(color_b, color_g, color_r);  // Establecer los valores RGB
        }
    }

    imshow("Imagen", imagen);

    // Esperar hasta que se presione una tecla
    waitKey(0);
};

void cargar_imagen_en_vector(const Mat &imagen, vector<vector<vector<float>>> &v)
{
    vector<vector<float>> canal_rojo, canal_verde, canal_azul;
    vector<float> fila_rojo, fila_verde, fila_azul;
    v.clear();

    for (int x = 0; x < imagen.rows; x++)    
    {
        for (int y = 0; y < imagen.cols; y++) 
        {
            Vec3b pixel = imagen.at<Vec3b>(x, y);
            fila_azul.push_back(pixel[0]);
            fila_verde.push_back(pixel[1]);
            fila_rojo.push_back(pixel[2]);
        }
        canal_rojo.push_back(fila_rojo);
        fila_rojo.clear();

        canal_verde.push_back(fila_verde);
        fila_verde.clear();

        canal_azul.push_back(fila_azul);
        fila_azul.clear();
    }

    // Guardamos en orden RGB
    v.push_back(canal_rojo);
    v.push_back(canal_verde);
    v.push_back(canal_azul);
};


Mat de_vector_a_imagen(const vector<vector<vector<float>>> &v)
{
    Mat imagen(v[0].size(), v[0][0].size(), CV_8UC3);

    float r, g, b;
    for(int x=0; x<v[0].size(); x++)
    {
        for(int y=0; y<v[0][0].size(); y++)
        {
            // Por cada canal RGB
            r = v[0][x][y];
            g = v[1][x][y];
            b = v[2][x][y];
            imagen.at<Vec3b>(x,y) = Vec3b(b, g, r);
        }
    }

    return imagen;
};

void mostrar_vector_como_imagen(const vector<vector<vector<float>>> &v)
{
    Mat image = de_vector_a_imagen(v);
    imshow("Imagen", image);
    // Esperar hasta que se presione una tecla
    waitKey(0);
};

