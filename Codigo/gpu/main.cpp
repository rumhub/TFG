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

/*
int main()
{
    vector<vector<vector<vector<float>>>> input, train_imgs, test_imgs;
    vector<vector<float>> train_labels, test_labels;
    vector<float> output;
    vector<vector<int>> capas_conv = {{6,3,3}, {8,3,3}}, capas_pool = {{2, 2}, {2, 2}};
    vector<int> capas = {50, 10}, pad = {1, 1};
    int n_imgs_train = 50, n_imgs_test = 50, n_clases = 10;
    leer_imagen(input);

    Flatten flat(input[0]);
    PoolingMax plm(2, 2, input[0]);
    Convolutional conv(2, 2, 2, input[0], 0.1);
    FullyConnected fully(capas, 0.1);

    CNN cnn(capas_conv, capas_pool, pad, capas, input[0], 0.001);

    //leer_imagenes_gatos_perros(train_imgs, train_labels, pad[0]);
    //leer_imagenes_mnist(train_imgs, train_labels, pad[0], 100, 10);
    leer_imagenes_cifar10(train_imgs, train_labels, test_imgs, test_labels, pad[0], n_imgs_train, n_imgs_test, n_clases);

    
    float *train_imgs_ptr, *train_labels_ptr, *test_imgs_ptr, *test_labels_ptr;

    const unsigned int img_size = 32 + 2*pad[0], tam_img = img_size*img_size*3; // 3 -> RGB
    train_imgs_ptr = (float *)malloc(tam_img * n_imgs_train * n_clases * sizeof(float));
    train_labels_ptr = (float *)malloc(n_imgs_train*n_clases * n_clases * sizeof(float));

    leer_imagenes_cifar10_ptr(train_imgs_ptr, train_labels_ptr, test_imgs_ptr, test_labels_ptr, pad[0], n_imgs_train, n_imgs_test, n_clases);
    
    
    int n_errores = 0;
    int n = train_imgs.size(), c = train_imgs[0].size(), h = train_imgs[0][0].size(), w = train_imgs[0][0][0].size();
    for(int i=0; i<n; i++)
        for(int j=0; j<c; j++)
            for(int k=0; k<h; k++)
                for(int p=0; p<w; p++)
                    if(train_imgs[i][j][k][p] != train_imgs_ptr[i*c*h*w + j*h*w + k*w +p])
                    {
                        //cout << train_imgs[i][j][k][p] << " vs " << train_imgs_ptr[i*c*h*w + j*h*w + k*w +p] << endl;
                        n_errores++;
                    }
    
    cout << "Errores: " << n_errores << endl;
    
    for(int i=0; i<train_labels.size(); i++)
        for(int j=0; j<train_labels[i].size(); j++)
            if(train_labels[i][j] != train_labels_ptr[i*train_labels[i].size() + j])
                cout << train_labels[i][j] << " vs " << train_labels_ptr[i*train_labels[i].size() + j] << endl;
    

    prueba();
    
    /*
    cnn.set_train(train_imgs, train_labels);
    cnn.set_test(test_imgs, test_labels);
    cnn.mostrar_arquitectura();
    cnn.train(30, 25);
    */
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
    /*
    return 0;
}
*/


int main()
{
    //vector<vector<int>> capas_conv = {{3, 3, 3}, {3, 5, 5}}, tams_pool = {{2, 2}, {2, 2}};
    //int C=3, H=5, W=5, n_capas_fully = 2, n_capas_conv = 2, n_imgs_train = 100, n_imgs_test = 100, n_clases = 10;
    //int C=3, H=4, W=4, n_capas_fully = 2, n_capas_conv = 2, n_imgs_train = 100, n_imgs_test = 100, n_clases = 10;
    //int C=3, H=32, W=32, n_capas_fully = 2, n_capas_conv = 2, n_imgs_train = 100, n_imgs_test = 100, n_clases = 10, mini_batch = 32;
    int C=3, H=32, W=32, n_capas_fully = 2, n_capas_conv = 2, n_imgs_train = 10, n_imgs_test = 10, n_clases = 10, mini_batch = 32;
    int *capas_fully = (int *)malloc(n_capas_fully * sizeof(int)),
        *capas_conv = (int *)malloc(n_capas_conv*3 * sizeof(int)),
        *capas_pool = (int *)malloc(n_capas_conv*2 * sizeof(int)),
        *padding = (int *)malloc(n_capas_conv * sizeof(int));
        
    float *train_imgs_ptr, *train_labels_ptr, *test_imgs_ptr, *test_labels_ptr;

    // Padding
    padding[0] = 1;
    padding[1] = 1;

    const unsigned int img_size = H + 2*padding[0], tam_img = img_size*img_size*3; // 3 -> RGB
    train_imgs_ptr = (float *)malloc(tam_img * n_imgs_train * n_clases * sizeof(float));
    train_labels_ptr = (float *)malloc(n_imgs_train*n_clases * n_clases * sizeof(float));

    float lr = 0.01;
    int i=0;
    capas_fully[0] = 100;
    capas_fully[1] = n_clases;

    // Primera capa convolucional
    capas_conv[i*3 +0] = 16;      // 4 kernels
    capas_conv[i*3 +1] = 3;      // kernels de 3 filas
    capas_conv[i*3 +2] = 3;      // kernels de 2 columnas

    i = 1;
    // Segunda capa convolucional
    capas_conv[i*3 +0] = 32;      // 7 kernels
    capas_conv[i*3 +1] = 3;      // kernels de 5 filas
    capas_conv[i*3 +2] = 3;      // kernels de 5 columnas

    i=0;
    // Primera capa MaxPool
    capas_pool[i*2 +0] = 2;      // kernels de 2 filas
    capas_pool[i*2 +1] = 2;      // kernels de 2 columnas

    i = 1;
    // Segunda capa MaxPool
    capas_pool[i*2 +0] = 2;      // kernels de 2 filas
    capas_pool[i*2 +1] = 2;      // kernels de 2 columnas


    CNN cnn(capas_conv, n_capas_conv, capas_pool, padding, capas_fully, n_capas_fully, C, H, W, lr, mini_batch);
    //CNN cnn(capas_conv, n_capas_conv, capas_pool, padding, capas_fully, n_capas_fully, C, H-2*padding[0], W-2*padding[0], lr);
    cnn.mostrar_arquitectura();
    leer_imagenes_cifar10_ptr(train_imgs_ptr, train_labels_ptr, test_imgs_ptr, test_labels_ptr, padding[0], n_imgs_train, n_imgs_test, n_clases);
    cnn.set_train(train_imgs_ptr, train_labels_ptr, n_imgs_train, n_clases, C, H, W);
    //cnn.set_train(X, Y, n_imagenes, n_clases, C, H, W);
    //cnn.prueba();
    cnn.train(50, mini_batch);
    //cnn.evaluar_modelo();
    
    /*
    cnn.set_train(train_imgs, train_labels);
    cnn.set_test(test_imgs, test_labels);
    cnn.mostrar_arquitectura();
    cnn.train(30, 25);
    */
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
    
    free(capas_fully); free(capas_conv); free(capas_pool); free(padding);
    return 0;
}