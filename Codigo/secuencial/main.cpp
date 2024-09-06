#include "CNN/auxiliar/leer_imagenes.h"
#include "CNN/CNN.h"
#include <vector>

using namespace std;


int main()
{
    // Creación de vectores
    vector<vector<vector<vector<float>>>> input, train_imgs, test_imgs;
    vector<vector<float>> train_labels, test_labels;
    vector<float> output;
    leer_imagen(input);
    
    // Arquitectura del modelo
    int pad=1;
    vector<vector<int>> capas_conv = {{16,3,3}, {32,3,3}}, capas_pool={{2,2}, {2,2}};
    vector<int> capas_fully = {100, 10}, padding = {pad, pad};
    CNN cnn(capas_conv, capas_pool, padding, capas_fully, input[0], 0.01);
    
    // Lectura de imágenes y entrenamiento
    leer_imagenes_cifar10(train_imgs, train_labels, test_imgs, test_labels, pad, 100, 10, 10);
    cnn.set_train(train_imgs, train_labels);
    
    cnn.mostrar_arquitectura();
    cnn.train(20, 32);
    

    return 0;
}

