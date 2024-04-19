#include "CNN/CNN.cpp"
#include <vector>

using namespace std;


void leer_imagen(vector<vector<vector<vector<float>>>> &imagenes_input, const int &pad)
{
    vector<vector<vector<float>>> imagen_k1;

    imagenes_input.clear();

    // Leer imagen
    string ruta = "../fotos/gatos_perros/training_set/dogs/dog.1.jpg";

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
    //omp_set_num_threads(omp_get_num_procs());
    vector<vector<vector<vector<float>>>> input;
    vector<float> output;
    int n_imagenes, pad=1;
    leer_imagen(input, pad);

    vector<vector<int>> capas_conv = {{32,3,3}}, capas_pool={{2,2}};
    vector<int> capas_fully = {128, 10}, padding = {pad};
    CNN cnn(capas_conv, capas_pool, padding, capas_fully, input[0], 0.01);
    
    //cnn.leer_imagenes();
    //cnn.leer_imagenes_mnist(3000, 10);
    cnn.leer_imagenes_cifar10(100, 10);
    cnn.mostrar_arquitectura();
    cnn.train(50, 32);

    /*
    // PRUEBAS ----------------------------------------
    vector<vector<vector<float>>> input;
    vector<vector<int>> capas_conv = {{1,2,2}}, capas_pool={{2,2}};
    vector<int> capas_fully = {2}, padding = {1};

    // Crear img de entrada 8x8
    vector<float> v1D;
    vector<vector<float>> v2D;
    for(int k=0; k<4; k++)
        v1D.push_back(2.0);

    for(int j=0; j<4; j++)
        v2D.push_back(v1D);

    input.push_back(v2D);


    CNN cnn(capas_conv, capas_pool, padding, capas_fully, input, 0.1);
    cnn.prueba();
    cnn.train(1, 1);
    */

    return 0;
}


/*
void quitar_padding(vector<vector<vector<float>>> &imagen_3D, int padding)
{
    vector<vector<vector<float>>> v_3D;
    vector<vector<float>> v_2D;
    vector<float> v_1D;

    for(int i=0; i<imagen_3D.size(); i++)
    {
        for(int j=padding; j<imagen_3D[i].size() - padding; j++)
        {
            for(int k=padding; k<imagen_3D[i][j].size() - padding; k++)
            {
                v_1D.push_back(imagen_3D[i][j][k]);
            }
            v_2D.push_back(v_1D);
            v_1D.clear();
        }
        v_3D.push_back(v_2D);
        v_2D.clear();
    }

    imagen_3D = v_3D;
};

void aplicar_padding(vector<vector<vector<float>>> &imagen_3D, int pad)
{
    vector<vector<vector<float>>> imagen_3D_aux;
    vector<vector<float>> imagen_aux;
    vector<float> fila_aux;

    // Por cada imagen
    for(int i=0; i<imagen_3D.size();i++)
    {
        // Añadimos padding superior
        for(int j=0; j<imagen_3D[i].size() + pad*2; j++) // pad*2 porque hay padding tanto a la derecha como a la izquierda
            fila_aux.push_back(0.0);
        
        for(int k=0; k<pad; k++)
            imagen_aux.push_back(fila_aux);
        
        fila_aux.clear();

        // Padding lateral (izquierda y derecha)
        // Por cada fila de cada imagen
        for(int j=0; j<imagen_3D[i].size(); j++)
        {
            // Añadimos padding lateral izquierdo
            for(int t=0; t<pad; t++)
                fila_aux.push_back(0.0);

            // Dejamos casillas centrales igual que en la imagen original
            for(int k=0; k<imagen_3D[i][j].size(); k++)
                fila_aux.push_back(imagen_3D[i][j][k]);
            
            // Añadimos padding lateral derecho
            for(int t=0; t<pad; t++)
                fila_aux.push_back(0.0);
            
            // Añadimos fila construida a la imagen
            imagen_aux.push_back(fila_aux);
            fila_aux.clear();
        }
        
        // Añadimos padding inferior
        fila_aux.clear();

        for(int j=0; j<imagen_3D[i].size() + pad*2; j++) // pad*2 porque hay padding tanto a la derecha como a la izquierda
            fila_aux.push_back(0.0);
        
        for(int k=0; k<pad; k++)
            imagen_aux.push_back(fila_aux);
        
        fila_aux.clear();
        
        // Añadimos imagen creada al conjunto de imágenes
        imagen_3D_aux.push_back(imagen_aux);
        imagen_aux.clear();
    }

    imagen_3D = imagen_3D_aux;
};


int main() 
{
    
    // Ejemplo de uso
    vector<vector<float>> imagen = {
        {1.0, 1.0, 2.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {3.0, 2.0, 1.0, 0.0},
        {1.0, 2.0, 3.0, 4.0}
    };
    
    vector<vector<vector<float>>> imagenes_2D, output, v_3D, out_conv, out_conv2, input_copy;
    vector<vector<float>> v_2D;
    vector<float> v_1D;
    int pad = 1, pad2 = 2;

    cout << "--------------- SIMULACIÓN GRADIENTE ------------------ " << endl;
    // Simulación. Viene gradiente de una capa convolucional con padding = pad

    // Padding se mete en capas convolucionales. Por tanto, si metemos padding de pad, antes de la capa pooling_max (input) hay que quitarlo al hacer backprop
    // Imágenes input con dimensiones sin padding
    imagenes_2D.push_back(imagen);
    imagenes_2D.push_back(imagen);

    int K = 2, n_kernels = 1;
    // H_out = fils_img   -       K          + 1;
    int n = imagenes_2D[0].size()+2*pad - K + 1;
    //int n = imagenes_2D[0].size() / 2;

    // Cálculo dimensiones out_conv -----------------
    out_conv.clear();
    v_1D.clear();
    v_2D.clear();
    for(int j=0; j<n; j++)
        v_1D.push_back(0.0);

    for(int j=0; j<n; j++)
        v_2D.push_back(v_1D);

    for(int j=0; j<n_kernels; j++)
        out_conv.push_back(v_2D);
    
    // Cálculo dimensiones output -----------------
    n = out_conv[0].size() /2;
    output.clear();
    v_1D.clear();
    v_2D.clear();
    for(int j=0; j<n; j++)
        v_1D.push_back(0.0);
    
    for(int j=0; j<n; j++)
        v_2D.push_back(v_1D);

    for(int j=0; j<out_conv.size(); j++)
        output.push_back(v_2D);
    

    // Cálculo dimensiones out_conv2 -----------------
    n = output[0].size()+2*pad2 - K + 1;
    out_conv2.clear();
    v_1D.clear();
    v_2D.clear();
    for(int j=0; j<n; j++)
        v_1D.push_back(0.0);

    for(int j=0; j<n; j++)
        v_2D.push_back(v_1D);

    for(int j=0; j<n_kernels; j++)
        out_conv2.push_back(v_2D);

    
    cout << "------------ Imagen inicial: ------------" << endl;
    mostrar_imagen(imagenes_2D);

    // Aplicamos padding a la entrada
    aplicar_padding(imagenes_2D, pad);

    cout << "Aplicamos padding: " << endl;
    mostrar_imagen(imagenes_2D);

    Convolutional conv(1, K, K, imagenes_2D, 0.1);
    conv.w_a_1();

    cout << "---------------- Conv, forward. Output: --------------- " << endl;
    conv.forwardPropagation(imagenes_2D, out_conv);
    mostrar_imagen(out_conv);

    
    cout << "---------------- plm, forward. Output: --------------- " << endl;
    PoolingMax plm1(2, 2, out_conv);
    input_copy = out_conv;
    plm1.forwardPropagation(out_conv, output, input_copy);

    cout << "Output \n";
    mostrar_imagen(output);



    cout << " ------------------ Conv2 ---------------------- " << endl;
    // Aplicamos padding a la entrada
    aplicar_padding(output, pad2);

    cout << "Aplicamos padding \n";
    mostrar_imagen(output);

    cout << "---------------- Conv2, forward. Output: --------------- " << endl;
    Convolutional conv2(1, K, K, output, 0.1);
    conv2.w_a_1();
    conv2.forwardPropagation(output, out_conv2);
    mostrar_imagen(out_conv2);


    cout << "--------------- BACKPROP ------------------------- " << endl;
    cout << "------------ Conv2, Back Propagation, Input: ------------" << endl;
    
    conv.backPropagation(output, out_conv2, 0);

    mostrar_imagen(output);

    quitar_padding(output, pad2);
    cout << "Quitamos padding \n";
    mostrar_imagen(output);

    cout << "------------ Pooling Max, Back Propagation: ------------" << endl;
    plm1.backPropagation(out_conv, output, input_copy, 0);

    cout << "Input\n";
    mostrar_imagen(out_conv);

    cout << "------------ Conv, Back Propagation: ------------" << endl;
    conv.backPropagation(imagenes_2D, out_conv, 0);

    cout << "Input \n";
    mostrar_imagen(imagenes_2D);

    cout << "Quitamos padding. Imagen inicial tras backpropagation \n";
    quitar_padding(imagenes_2D, pad);
    mostrar_imagen(imagenes_2D);
    

    return 0;
}
*/