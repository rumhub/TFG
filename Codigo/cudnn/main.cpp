#include "CNN/auxiliar/leer_imagenes.h"
#include "CNN/CNN.h"
#include <vector>

using namespace std;


int main()
{
    int C=3, H=32, W=32, n_capas_fully = 2, n_capas_conv = 2, n_imgs_train = 100, n_imgs_test = 100, n_clases = 10, mini_batch = 32;
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

    capas_conv[i*3 +0] = 16;      // 4 kernels
    capas_conv[i*3 +1] = 3;      // kernels de 3 filas
    capas_conv[i*3 +2] = 3;      // kernels de 2 columnas

    i=1;
    capas_conv[i*3 +0] = 32;      // 4 kernels
    capas_conv[i*3 +1] = 3;      // kernels de 3 filas
    capas_conv[i*3 +2] = 3;      // kernels de 2 columnas

    i=0;
    // Primera capa MaxPool
    capas_pool[i*2 +0] = 2;      // kernels de 2 filas
    capas_pool[i*2 +1] = 2;      // kernels de 2 columnas

    i=1;
    // Primera capa MaxPool
    capas_pool[i*2 +0] = 2;      // kernels de 2 filas
    capas_pool[i*2 +1] = 2;      // kernels de 2 columnas

    CNN cnn(capas_conv, n_capas_conv, capas_pool, padding, capas_fully, n_capas_fully, C, H, W, lr, n_imgs_train, mini_batch);
    cnn.mostrar_arquitectura();
    leer_imagenes_cifar10_sin_pad(train_imgs_ptr, train_labels_ptr, test_imgs_ptr, test_labels_ptr, n_imgs_train, n_imgs_test, n_clases);
    cnn.set_train(train_imgs_ptr, train_labels_ptr, n_imgs_train, n_clases, C, H, W);
    cnn.train(50, mini_batch);

    free(capas_fully); free(capas_conv); free(capas_pool); free(padding);
    return 0;
}
