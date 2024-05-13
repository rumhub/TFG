#ifndef LEER_IMAGENES_H
#define LEER_IMAGENES_H
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

void modificar_imagen(Mat imagen, const int &p_ini_x, const int &p_ini_y, const int &p_fin_x, const int &p_fin_y, const int &color_r, const int &color_g, const int &color_b);
void cargar_imagen_en_vector(const Mat &imagen, vector<vector<vector<float>>> &v);
Mat de_vector_a_imagen(const vector<vector<vector<float>>> &v);
void mostrar_vector_como_imagen(const vector<vector<vector<float>>> &v);
void leer_imagen(vector<vector<vector<vector<float>>>> &imagenes_input);
void eee();

#endif
