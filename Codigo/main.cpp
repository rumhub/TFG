#include "CNN/CNN.cpp"
#include <vector>

using namespace std;


void leer_imagen(vector<vector<vector<vector<float>>>> &imagenes_input)
{
    vector<vector<vector<float>>> imagen_k1;

    imagenes_input.clear();

    // Leer imagen
    string ruta = "fotos/training_set/dogs/dog.1.jpg";

    Mat image2 = imread(ruta), image;
    
    resize(image2, image, Size(32, 32));

    // Cargamos la imagen en un vector 3D
    cargar_imagen_en_vector(image, imagen_k1);

    // Normalizar imagen
    for(int i=0; i<imagen_k1.size(); i++)
        for(int j=0; j<imagen_k1[0].size(); j++)
            for(int k=0; k<imagen_k1[0][0].size(); k++)
                imagen_k1[i][j][k] = imagen_k1[i][j][k] / 255;
            
    // Se almacena la imagen obtenida
    imagenes_input.push_back(imagen_k1);
    
};

int main()
{
    vector<vector<vector<vector<float>>>> input;
    vector<float> output;
    int n_imagenes;
    leer_imagen(input);

    vector<vector<int>> capas_conv = {{32,5,5}, {32,3,3}}, capas_pool={{2,2}, {2,2}};
    vector<int> capas_fully = {100, 50, 25}, padding = {2, 2};
    CNN cnn(capas_conv, capas_pool, padding, capas_fully, input, 0.1);
    
    cnn.leer_imagenes();
    cnn.mostrar_arquitectura();
    
    cnn.train(10000, 64);
    

    return 0;
}

