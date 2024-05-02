#include "CNN/CNN.cpp"
#include <vector>

using namespace std;


void leer_imagen(vector<vector<vector<vector<float>>>> &imagenes_input, const int &pad)
{
    vector<vector<vector<float>>> imagen_k1;

    imagenes_input.clear();

    // Leer imagen
    string ruta = "../../fotos/gatos_perros/training_set/dogs/dog.1.jpg";

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
    omp_set_num_threads(omp_get_num_procs());
    vector<vector<vector<vector<float>>>> input;
    vector<float> output;
    int n_imagenes, pad=1;
    leer_imagen(input, pad);

    vector<vector<int>> capas_conv = {{16,3,3}, {32,3,3}}, capas_pool={{2,2}, {2,2}};
    vector<int> capas_fully = {100, 10}, padding = {pad, pad};
    CNN cnn(capas_conv, capas_pool, padding, capas_fully, input[0], 0.001);
    
    //cnn.leer_imagenes();
    //cnn.leer_imagenes_mnist(3000, 10);
    cnn.leer_imagenes_cifar10(100, 100, 10);
    cnn.mostrar_arquitectura();
    cnn.train(20, 32);

    return 0;
}

