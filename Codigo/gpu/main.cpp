#include "CNN/auxiliar/leer_imagenes.h"
#include "CNN/CNN.h"
/*
#include "CNN/capas/convolutional/convolutional.h"
#include "CNN/capas/flatten/flatten.h"
#include "CNN/capas/pooling_max/poolingMax.h"
#include "CNN/capas/fully_connected/fullyconnected.h"
*/
#include <vector>

using namespace std;


int main()
{
    vector<vector<vector<vector<float>>>> input;
    vector<float> output;
    vector<int> capas = {3, 2}, pad = {1, 1};
    vector<vector<int>> capas_conv = {{5, 2, 2}, {6, 2, 2}}, capas_pool = {{2, 2}, {2, 2}};
    int n_imagenes;
    leer_imagen(input);

    Flatten flat(input[0]);
    PoolingMax plm(2, 2, input[0]);
    Convolutional conv(2, 2, 2, input[0], 0.1);
    FullyConnected fully(capas, 0.1);

    CNN cnn(capas_conv, capas_pool, pad, capas, input[0], 0.1);

    /*
    vector<vector<int>> capas_conv = {{16,3,3}, {32,3,3}}, capas_pool={{2,2}, {2,2}};
    vector<int> capas_fully = {100, 10}, padding = {pad, pad};
    CNN cnn(capas_conv, capas_pool, padding, capas_fully, input[0], 0.001);
    
    //cnn.leer_imagenes();
    //cnn.leer_imagenes_mnist(3000, 10);
    cnn.leer_imagenes_cifar10(100, 100, 10);
    cnn.mostrar_arquitectura();
    cnn.train(20, 32);
    */

    return 0;
}

