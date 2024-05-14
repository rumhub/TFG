#include "leer_imagenes.h"
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


void leer_imagen(vector<vector<vector<vector<float>>>> &imagenes_input)
{
    vector<vector<vector<float>>> imagen_k1;

    imagenes_input.clear();

    // Leer imagen
    string ruta = "../fotos/gatos_perros/training_set/dogs/dog.1.jpg";

    Mat image2 = imread(ruta), image;
    
    resize(image2, image, Size(TAM_IMAGE, TAM_IMAGE));

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

void aplicar_padding(vector<vector<vector<float>>> &imagen_3D, int pad)
{
    vector<vector<vector<float>>> imagen_3D_aux;
    vector<vector<float>> imagen_aux;
    vector<float> fila_aux;

    // Por cada imagen
    for(int i=0; i<imagen_3D.size(); ++i)
    {
        // Añadimos padding superior
        for(int j=0; j<imagen_3D[i].size() + pad*2; ++j) // pad*2 porque hay padding tanto a la derecha como a la izquierda
            fila_aux.push_back(0.0);
        
        for(int k=0; k<pad; ++k)
            imagen_aux.push_back(fila_aux);
        
        fila_aux.clear();

        // Padding lateral (izquierda y derecha)
        // Por cada fila de cada imagen
        for(int j=0; j<imagen_3D[i].size(); ++j)
        {
            // Añadimos padding lateral izquierdo
            for(int t=0; t<pad; ++t)
                fila_aux.push_back(0.0);

            // Dejamos casillas centrales igual que en la imagen original
            for(int k=0; k<imagen_3D[i][j].size(); ++k)
                fila_aux.push_back(imagen_3D[i][j][k]);
            
            // Añadimos padding lateral derecho
            for(int t=0; t<pad; ++t)
                fila_aux.push_back(0.0);
            
            // Añadimos fila construida a la imagen
            imagen_aux.push_back(fila_aux);
            fila_aux.clear();
        }
        
        // Añadimos padding inferior
        fila_aux.clear();

        for(int j=0; j<imagen_3D[i].size() + pad*2; ++j) // pad*2 porque hay padding tanto a la derecha como a la izquierda
            fila_aux.push_back(0.0);
        
        for(int k=0; k<pad; ++k)
            imagen_aux.push_back(fila_aux);
        
        fila_aux.clear();
        
        // Añadimos imagen creada al conjunto de imágenes
        imagen_3D_aux.push_back(imagen_aux);
        imagen_aux.clear();
    }

    imagen_3D = imagen_3D_aux;
};

/*
    @brief      Leer imágenes de la base de datos binaria Gatos vs Perros
    @return     Se modifican train_imgs y train_labels
*/
void leer_imagenes_gatos_perros(vector<vector<vector<vector<float>>>> &train_imgs, vector<vector<float>> &train_labels, const int &pad)
{
    vector<vector<vector<float>>> imagen_k1;

    //n_imagenes = 4000;
    //int n_imagenes = 1000;
    int n_imagenes = 200;

    train_imgs.clear();
    train_labels.clear();

    // Leer imágenes
    for(int p=1; p<n_imagenes; ++p)
    {
        // Leemos perros
        string ruta_ini = "../fotos/gatos_perros/training_set/dogs/dog.";
        string ruta = ruta_ini + to_string(p) + ".jpg";

        Mat image2 = imread(ruta), image;
        
        resize(image2, image, Size(TAM_IMAGE, TAM_IMAGE));

        // Cargamos la imagen en un vector 3D
        cargar_imagen_en_vector(image, imagen_k1);

        // Normalizar imagen
        for(int i=0; i<imagen_k1.size(); ++i)
            for(int j=0; j<imagen_k1[0].size(); ++j)
                for(int k=0; k<imagen_k1[0][0].size(); ++k)
                    imagen_k1[i][j][k] = imagen_k1[i][j][k] / 255;
                
            
        

        // Aplicamos padding a la imagen de entrada
        aplicar_padding(imagen_k1, pad);

        // Almacenamos las imágenes de entrada de la CNN
        train_imgs.push_back(imagen_k1);

        // Establecemos que la imagen tiene una etiqueta, 1 = perro, 0 = gato
        train_labels.push_back({0.0, 1.0});         

        
        // Leemos gatos
        ruta_ini = "../fotos/gatos_perros/training_set/cats/cat.";
        ruta = ruta_ini + to_string(p) + ".jpg";

        image2 = imread(ruta), image;
        
        resize(image2, image, Size(TAM_IMAGE, TAM_IMAGE));

        // Cargamos la imagen en un vector 3D
        cargar_imagen_en_vector(image, imagen_k1);

        // Normalizar imagen
        for(int i=0; i<imagen_k1.size(); ++i)
            for(int j=0; j<imagen_k1[0].size(); ++j)
                for(int k=0; k<imagen_k1[0][0].size(); ++k)
                    imagen_k1[i][j][k] = imagen_k1[i][j][k] / 255;
                
            
        

        // Aplicamos padding a la imagen de entrada
        aplicar_padding(imagen_k1, pad);

        // Almacenamos las imágenes de entrada de la CNN
        train_imgs.push_back(imagen_k1);

        // Establecemos que la imagen tiene una etiqueta, 1 = perro, 0 = gato
        train_labels.push_back({1.0, 0.0});

    }  
}


/*
    @brief      Leer imágenes de la base de datos MNIST
    @return     Se modifican train_imgs y train_labels
*/
void leer_imagenes_mnist(vector<vector<vector<vector<float>>>> &train_imgs, vector<vector<float>> &train_labels, const int &pad, const int n_imagenes, const int n_clases)
{
    vector<vector<vector<float>>> imagen_k1;
    vector<float> v1D, y_1D;
    string ruta_ini, ruta;

    train_imgs.clear();
    train_labels.clear();

    // Crear el vector y
    for(int i=0; i<n_clases; ++i)
        y_1D.push_back(0.0);

    // Leer n_imagenes de la clase c
    for(int c=0; c<n_clases; ++c)
    {
        // Establecer etiqueta one-hot para la clase i
        y_1D[c] = 1.0;

        // Leer imágenes
        for(int p=1; p<n_imagenes; ++p)
        {
            ruta_ini = "../fotos/mnist/training/";
            ruta = ruta_ini + to_string(c) + "/" + to_string(p) + ".jpg";

            Mat image2 = imread(ruta), image;

            image = image2;

            // Cargamos la imagen en un vector 3D
            cargar_imagen_en_vector(image, imagen_k1);

            // Normalizar imagen
            for(int i=0; i<imagen_k1.size(); ++i)
                for(int j=0; j<imagen_k1[0].size(); ++j)
                    for(int k=0; k<imagen_k1[0][0].size(); ++k)
                        imagen_k1[i][j][k] = imagen_k1[i][j][k] / 255.0;

            // Aplicamos padding a la imagen de entrada
            aplicar_padding(imagen_k1, pad);

            // Almacenamos las imágenes de entrada de la CNN
            train_imgs.push_back(imagen_k1);

            // Establecemos que la imagen tiene una etiqueta, 1 = perro, 0 = gato
            train_labels.push_back(y_1D);
        }

        // Reset todo "y_1D" a 0
        y_1D[c] = 0.0;
    }  
}



/*
    @brief      Leer imágenes de la base de datos CIFAR10
    @return     Se modifican train_imgs, train_labels, test_imgs y test_labels
*/
void leer_imagenes_cifar10(vector<vector<vector<vector<float>>>> &train_imgs, vector<vector<float>> &train_labels, vector<vector<vector<vector<float>>>> &test_imgs, vector<vector<float>> &test_labels, const int &pad, const int &n_imagenes_train, const int &n_imagenes_test, const int n_clases)
{
    vector<vector<vector<float>>> imagen_k1;
    vector<float> v1D, y_1D;
    string ruta_ini, ruta;

    train_imgs.clear();
    train_labels.clear();
    test_imgs.clear();
    test_labels.clear();

    // Crear el vector y
    for(int i=0; i<n_clases; ++i)
        y_1D.push_back(0.0);

    // TRAIN ------------------------------------------------------------------------
    // Leer n_imagenes de la clase c
    for(int c=0; c<n_clases; ++c)
    {
        // Establecer etiqueta one-hot para la clase i
        y_1D[c] = 1.0;

        // Leer imágenes
        for(int p=1; p<n_imagenes_train; ++p)
        {
            ruta_ini = "../fotos/cifar10/train/";
            ruta = ruta_ini + to_string(c) + "/" + to_string(p) + ".png";

            Mat image2 = imread(ruta), image;

            image = image2;

            // Cargamos la imagen en un vector 3D
            cargar_imagen_en_vector(image, imagen_k1);

            // Normalizar imagen
            for(int i=0; i<imagen_k1.size(); ++i)
                for(int j=0; j<imagen_k1[0].size(); ++j)
                    for(int k=0; k<imagen_k1[0][0].size(); ++k)
                        imagen_k1[i][j][k] = imagen_k1[i][j][k] / 255.0;

            // Aplicamos padding a la imagen de entrada
            aplicar_padding(imagen_k1, pad);

            // Almacenamos las imágenes de entrada de la CNN
            train_imgs.push_back(imagen_k1);

            // Establecemos que la imagen tiene una etiqueta, 1 = perro, 0 = gato
            train_labels.push_back(y_1D);
        }

        // Reset todo "y_1D" a 0
        y_1D[c] = 0.0;
    }  

    
    // TEST ------------------------------------------------------------------------
    // Leer n_imagenes de la clase c
    for(int c=0; c<n_clases; ++c)
    {
        // Establecer etiqueta one-hot para la clase i
        y_1D[c] = 1.0;

        // Leer imágenes
        for(int p=1; p<n_imagenes_test; ++p)
        {
            ruta_ini = "../fotos/cifar10/test/";
            ruta = ruta_ini + to_string(c) + "/" + to_string(p) + ".png";

            Mat image2 = imread(ruta), image;

            image = image2;

            // Cargamos la imagen en un vector 3D
            cargar_imagen_en_vector(image, imagen_k1);

            // Normalizar imagen
            for(int i=0; i<imagen_k1.size(); ++i)
                for(int j=0; j<imagen_k1[0].size(); ++j)
                    for(int k=0; k<imagen_k1[0][0].size(); ++k)
                        imagen_k1[i][j][k] = imagen_k1[i][j][k] / 255.0;

            // Aplicamos padding a la imagen de entrada
            aplicar_padding(imagen_k1, pad);

            // Almacenamos las imágenes de entrada de la CNN
            test_imgs.push_back(imagen_k1);

            // Establecemos que la imagen tiene una etiqueta, 1 = perro, 0 = gato
            test_labels.push_back(y_1D);
        }

        // Reset todo "y_1D" a 0
        y_1D[c] = 0.0;
    }  
    
}


void eee()
{
    cout << "EEEEE desde leer_img" << endl;
}

/*
int main()
{
    return 0;
}
*/