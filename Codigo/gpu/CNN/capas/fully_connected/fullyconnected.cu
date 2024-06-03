#include "fullyconnected.h"

using namespace std;

/*
    Realiza una multiplicación matricial con tiles. Un tile por bloque. Usa memoria compartida.
    Además, añade una fila al final de la matriz "B" con todos sus valores a 1.
    Por último, al establecer el valor final de cada casilla de la matriz resultado C, aplica la función de activación ReLU.
*/
__global__ void forward_capas_intermedias(int M, int N, int K, const float *A, const float *B, float *C, float *a, bool aplicar_relu)
{
    // Memoria compartida dinámica
	extern __shared__ float sdata[];

    // Convertir de índices de hebra a índices de matriz 
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x, 
        idA = iy*K + ix, idB = iy*N + ix, id_tile = threadIdx.y * blockDim.x + threadIdx.x;
    //int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);
    int n_tiles = (K + blockDim.x - 1) / blockDim.x;

    // Punteros a A y B
    float *sA = sdata,
          *sB = sdata + blockDim.x * blockDim.y;

    float sum = 0.0f;    

    int lim = blockDim.x;

    // Si tam_bloque > tam_A
    if(lim > K)
        lim = K;

    /*
        Para multiplicar A(MxK) x B(KxN) hay que multiplicar una fila de A x una columna de B
        Es decir, multiplicar KxK elementos y sumarlos
        Un tile es más pequeño que K -> Dividir K en tiles e iterar sobre ellos
    */
    for(int tile=0; tile < n_tiles; ++tile)
    {
        idA = iy*K + tile * blockDim.x + threadIdx.x;
        idB = (tile * blockDim.x + threadIdx.y)*N + ix;
       
        // Cargar submatrices de A y B en memoria compartida (tamaño tilex x tiley)
        // Cada hebra carga en memoria compartida un elemento de A y otro de B
        (iy < M && tile * blockDim.x + threadIdx.x < K) ? sA[id_tile] = A[idA] : sA[id_tile] = 0.0;
        
        if(tile * blockDim.x + threadIdx.y < K && ix < N)
            (tile * blockDim.x + threadIdx.y < K-1) ? sB[id_tile] = B[idB] : sB[id_tile] = 1.0;             // Añadir 1 fila de 1s al final
        else
            sB[id_tile] = 0.0;
        
        //(tile * blockDim.x + threadIdx.y < K && ix < N) ? sB[id_tile] = B[idB] : sB[id_tile] = 0.0;

        // Sincronizar hebras
        __syncthreads();

        // Realizar multiplicación matricial
        if(iy < M && ix < N)
        {
            // Si última iteración
            if(tile == n_tiles -1)
                lim = K - tile * blockDim.x;

            // Cada hebra calcula una posición de C (una fila de A * una columna de B)
            for (int i = 0; i < lim; i++) 
                sum += sA[threadIdx.y*blockDim.x + i] * sB[threadIdx.x + i*blockDim.x];
        }

        // Sincronizar hebras
        __syncthreads();
    }

    // Establecer valor en matriz final
    if(iy < M && ix < N)
    {
        // Aplicar ReLU si estamos en alguna capa intermedia
        if(aplicar_relu)
            (sum > 0) ? C[iy*N + ix] = sum : C[iy*N + ix] = 0.0;
        else                        
            C[iy*N + ix] = sum;     // No aplicar ReLU si estamos en SoftMax

        // Almacenar en "a" la matriz resultado antes de aplicar la función de activación
        a[iy*N + ix] = sum;     
        
    }    
    
}

/*
    Entrada: Matrix X(MxK)
    Salida: Y(K), cada Y[i] contiene el valor máximo de la columna X[i]
*/
__global__ void kernel_sofmax(int M, int K, float *X, float *a)
{
	int tid = threadIdx.x,
        i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float maximo = -9999999, sum = 0.0;

    if(i < M*K) 
    {
        for(int j=0; j<K; j++)
            maximo = max(maximo, X[i*K + j]);     // Cada hebra obtiene el máximo de una fila

        // Sincronizar hebras
    	__syncthreads();

        // Normalizar con el valor máximo de cada fila (dato del minibatch)
        for(int j=0; j<K; j++)
        {
            X[i*K + j] -= maximo;
            a[i*K + j] = X[i*K + j];
        }

        // Sincronizar hebras
    	__syncthreads();

        // Realizar la suma exponencial de cada fila
        for(int j=0; j<K; j++)
            sum += expf(X[i*K + j]);

        // Normalizar cada fila
        for(int j=0; j<K; j++)
            X[i*K + j] = expf(X[i*K + j]) / sum;   
    }    
}

__global__ void matrizTranspuesta_GPU(float* X, float *Y, int rows, int cols)
{
	int tid = threadIdx.x,
        i = blockIdx.x * blockDim.x + threadIdx.x;

    // Cada hebra se encarga de una fila
    if(i < rows)
        for (int j = 0; j < cols; j++)
            Y[j * rows + i] = X[i * cols + j];
}


__global__ void kernel_back_softmax(int M, int K, float *grad_a, float *Z, float *Y, int n_clases)
{
    int tid = threadIdx.x,
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < M*K) 
    {
        // Cada hebra se encarga de una fila
        for(int j=0; j<K; ++j)
            grad_a[i*K + j] = Z[i*K + j] - Y[i*n_clases + j];
    
    }
}

__global__ void kernel_grad_bias(int M, int K, float *grad_a, float *grad_b)
{
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x, 
        tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);

    if(iy < M && ix <K) 
    {
        // Inicializar valor del gradiente de bias a 0.0
        grad_b[iy*K + ix] = 0.0f;

        // Cada hebra se encarga de una fila
        for(int j=0; j<K; ++j)
            grad_b[iy*K + ix] += grad_a[tid*K + j];
    }
}


__global__ void kernel_actualizar_bias(int N, float *bias, float *grad_bias, float lr)
{
    int tid = threadIdx.x,
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N) 
        bias[i] -= lr * grad_bias[i];    
}

__global__ void kernel_actualizar_pesos(int M, int K, float *w, float *grad_w, float lr)
{
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x, 
        tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);

    if(iy < M && ix <K) 
    {
        // Cada hebra se encarga de un peso
        w[iy*K + ix] -= lr * grad_w[iy*K + ix];
    }
}


/*
    Emplea tiles. Un tile por bloque. Usa memoria compartida
*/
__global__ void backprop_capas_intermedias(int M, int N, int K, const float *A, const float *B, float *C, bool aplicar_deriv_relu)
{
    // Memoria compartida dinámica
	extern __shared__ float sdata[];

    // Convertir de índices de hebra a índices de matriz 
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x, 
        idA = iy*K + ix, idB = iy*N + ix, id_tile = threadIdx.y * blockDim.x + threadIdx.x;
    //int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);
    int n_tiles = (K + blockDim.x - 1) / blockDim.x;

    // Punteros a A y B
    float *sA = sdata,
          *sB = sdata + blockDim.x * blockDim.y;

    float sum = 0.0f;    

    int lim = blockDim.x;

    // Si tam_bloque > tam_A
    if(lim > K)
        lim = K;

    /*
        Para multiplicar A(MxK) x B(KxN) hay que multiplicar una fila de A x una columna de B
        Es decir, multiplicar KxK elementos y sumarlos
        Un tile es más pequeño que K -> Dividir K en tiles e iterar sobre ellos
    */
    for(int tile=0; tile < n_tiles; ++tile)
    {
        idA = iy*K + tile * blockDim.x + threadIdx.x;
        idB = (tile * blockDim.x + threadIdx.y)*N + ix;
       
        // Cargar submatrices de A y B en memoria compartida (tamaño tilex x tiley)
        // Cada hebra carga en memoria compartida un elemento de A y otro de B
        (iy < M && tile * blockDim.x + threadIdx.x < K) ? sA[id_tile] = A[idA] : sA[id_tile] = 0.0;
        (tile * blockDim.x + threadIdx.y < K && ix < N) ? sB[id_tile] = B[idB] : sB[id_tile] = 0.0;

        // Sincronizar hebras
        __syncthreads();

        // Realizar multiplicación matricial
        if(iy < M && ix < N)
        {
            // Si última iteración
            if(tile == n_tiles -1)
                lim = K - tile * blockDim.x;

            // Cada hebra calcula una posición de C (una fila de A * una columna de B)
            for (int i = 0; i < lim; i++) 
                sum += sA[threadIdx.y*blockDim.x + i] * sB[threadIdx.x + i*blockDim.x];
        }

        // Sincronizar hebras
        __syncthreads();
    }

    if(iy < M && ix < N)
    {
        if(aplicar_deriv_relu)
            (C[iy*N + ix] > 0) ? C[iy*N + ix] = sum : C[iy*N + ix] = 0;
        else 
            C[iy*N + ix] = sum;
    }
}


/*
    Emplea tiles. Un tile por bloque. Usa memoria compartida
*/
__global__ void multiplicarMatricesGPU(int M, int N, int K, const float *A, const float *B, float *C)
{
    // Memoria compartida dinámica
	extern __shared__ float sdata[];

    // Convertir de índices de hebra a índices de matriz 
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x, 
        idA = iy*K + ix, idB = iy*N + ix, id_tile = threadIdx.y * blockDim.x + threadIdx.x;
    //int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);
    int n_tiles = (K + blockDim.x - 1) / blockDim.x;

    // Punteros a A y B
    float *sA = sdata,
          *sB = sdata + blockDim.x * blockDim.y;

    float sum = 0.0f;    

    int lim = blockDim.x;

    // Si tam_bloque > tam_A
    if(lim > K)
        lim = K;

    /*
        Para multiplicar A(MxK) x B(KxN) hay que multiplicar una fila de A x una columna de B
        Es decir, multiplicar KxK elementos y sumarlos
        Un tile es más pequeño que K -> Dividir K en tiles e iterar sobre ellos
    */
    for(int tile=0; tile < n_tiles; ++tile)
    {
        idA = iy*K + tile * blockDim.x + threadIdx.x;
        idB = (tile * blockDim.x + threadIdx.y)*N + ix;
       
        // Cargar submatrices de A y B en memoria compartida (tamaño tilex x tiley)
        // Cada hebra carga en memoria compartida un elemento de A y otro de B
        (iy < M && tile * blockDim.x + threadIdx.x < K) ? sA[id_tile] = A[idA] : sA[id_tile] = 0.0;
        (tile * blockDim.x + threadIdx.y < K && ix < N) ? sB[id_tile] = B[idB] : sB[id_tile] = 0.0;

        // Sincronizar hebras
        __syncthreads();

        // Realizar multiplicación matricial
        if(iy < M && ix < N)
        {
            // Si última iteración
            if(tile == n_tiles -1)
                lim = K - tile * blockDim.x;

            // Cada hebra calcula una posición de C (una fila de A * una columna de B)
            for (int i = 0; i < lim; i++) 
                sum += sA[threadIdx.y*blockDim.x + i] * sB[threadIdx.x + i*blockDim.x];
        }

        // Sincronizar hebras
        __syncthreads();
    }

    if(iy < M && ix < N)
        C[iy*N + ix] = sum;
}



__global__ void kernel_evaluacion_modelo(int M, int K, float * X, float * Y, float *sumas) 
{
	int tid = threadIdx.x,
        i = blockIdx.x * blockDim.x + threadIdx.x, predic_acc = 0;
    
    float predic_entr = 0.0, epsilon = 0.000000001, maximo = -9999999, sum = 0.0, entr = 0.0, acc = 0.0;

    float *sum_entr = sumas, *sum_acc = sumas + M-1;

    // Calcular entropía cruzada --------------------------
    if(i < M) 
    {
        // Inicializar suma a 0.0
        sum_entr[i] = 0.0f;

        for(int j=0; j<K; j++)
            if(Y[i*K + j] == 1)                     
                predic_entr = X[i*K + j];

        sum_entr[i] += log(predic_entr+epsilon);
    }   

    // Sincronizar hebras
    __syncthreads();

    if(i == 0)
    {
        for(int j=1; j<M; j++)
            sum_entr[0] += sum_entr[j];
        sum_entr[0] = -sum_entr[0] / M;

        entr = sum_entr[0];
    }
    
    // Calcular Accuracy -----------------------------
    if(i < M) 
    {
        for(int j=0; j<K; j++)
        {
            if(maximo < X[i*K + j])
            {
                maximo = X[i*K + j];
                predic_acc = j;
            }
        }

        sum_acc[i] = Y[i*K + predic_acc];
    }   


    // Sincronizar hebras
    __syncthreads();

    if(i == 0)
    {
        for(int j=1; j<M; j++)
            sum_acc[0] += sum_acc[j];
        sum_acc[0] = sum_acc[0] / M * 100;

        acc = sum_acc[0];
        printf("Accuracy: %f, Entropía cruzada: %f\n", acc, entr);
    }
}

__global__ void reduceMax(float * X, float * Y, const int N, float *maximo) 
{
	extern __shared__ float sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;


    // Cargar los datos en memoria compartida
	sdata[tid] = ((i < N) ? X[i] : -100000000.0f);
	__syncthreads();

    // Obtener el máximo local de cada bloque
	for(int s = blockDim.x/2; s > 0; s >>= 1)
    {
	  if (tid < s)
	        sdata[tid]=max(sdata[tid],sdata[tid+s]); 	
	  __syncthreads();
	}
	if (tid == 0) 
        Y[blockIdx.x] = sdata[0];
	__syncthreads();

    // Obtener el máximo total en todos los bloques
    if(i == 0)
    {
        maximo[0] = Y[0];

        for(int j=1; j<gridDim.x; j++)
            maximo[0] = max(maximo[0], Y[j]);
    }
}

__global__ void reduceMin(float * X, float * Y, const int N, float *minimo) 
{
	extern __shared__ float sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;


    // Cargar los datos en memoria compartida
	sdata[tid] = ((i < N) ? X[i] : 100000000.0f);
	__syncthreads();

    // Obtener el máximo local de cada bloque
	for(int s = blockDim.x/2; s > 0; s >>= 1)
    {
	  if (tid < s)
	        sdata[tid]=min(sdata[tid],sdata[tid+s]); 	
	  __syncthreads();
	}
	if (tid == 0) 
        Y[blockIdx.x] = sdata[0];
	__syncthreads();

    // Obtener el máximo total en todos los bloques
    if(i == 0)
    {
        minimo[0] = Y[0];

        for(int j=1; j<gridDim.x; j++)
            minimo[0] = min(minimo[0], Y[j]);
    }
}

__global__ void kernel_escalar_pesos(float * X, const int N, float *maxi, float *mini, float valor_clip) 
{
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    float maximo = abs(maxi[0]), minimo = abs(mini[0]),
          factor_de_escala = valor_clip / max(maximo, minimo);

    // Escalar los pesos
    if(i < N)
        X[i] = max(min(X[i], valor_clip), -valor_clip);
}


__global__ void kernel_transpuesta_pesos(int *capas_wT, int *capas, float *w, float *wT, int capa, float *bias)
{
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x, 
        tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);

    // Cada hebra se encarga de un peso
    if(iy < capas[capa] && ix < capas[capa+1]) 
        wT[ix*capas_wT[capa] + iy] = w[iy*capas[capa+1] + ix];	

    // Bias
    if(iy == 0)
        wT[ix*capas_wT[capa] + capas[capa]] = bias[ix];     
}



/*
    CONSTRUCTOR de la clase FullyConnected
    --------------------------------------
  
    @capas  Vector de enteros que indica el número de neuronas por capa. Habrá capas.size() capas y cada una contendrá capas[i] neuronas
    @lr     Learning Rate o Tasa de Aprendizaje
*/
FullyConnected::FullyConnected(const vector<int> &capas, const float &lr)
{
    liberar_memoria = false;
    vector<float> neuronas_capa, w_1D;
    vector<vector<float>> w_2D;
    
    // Neuronas ------------------------------------------------------------------
    // Por cada capa 
    for(int i=0; i<capas.size(); ++i)
    {
        // Por cada neurona de cada capa
        for(int j=0; j<capas[i]; ++j)
            neuronas_capa.push_back(0.0);

        this->a.push_back(neuronas_capa);
        neuronas_capa.clear();
    }

    // Pesos -------------------------------------------------------------------
    // Por cada capa
    for(int i=0; i<a.size()-1; ++i)
    {
        // Por cada neurona de cada capa
        for(int j=0; j<a[i].size(); ++j)
        {
            
            // Por cada neurona de la capa siguiente
            for(int k=0; k<a[i+1].size(); ++k)
            {
                // Añadimos un peso. 
                // w[i][j][k] indica el peso que conecta la neurona j de la capa i con la neurona k de la capa i+1
                w_1D.push_back(0.0);
            }

            w_2D.push_back(w_1D);  
            w_1D.clear();          
        }
        this->w.push_back(w_2D);
        w_2D.clear();
    }    

    // Learning Rate
    this->lr = lr;

    // Bias mismas dimensiones que neuronas, 1 bias por neurona
    this->bias = this->a;

    // Inicializar pesos mediante inicialización He
    for(int i=0; i<a.size()-1; ++i)
        this->generar_pesos(i);

    // Inicializar bias con un valor de 0.0
    for(int i=0; i<this->bias.size(); ++i)
        for(int j=0; j<this->bias[i].size(); ++j)
            this->bias[i][j] = 0.0;
};


/*
    CONSTRUCTOR de la clase FullyConnected
    --------------------------------------
  
    @capas  Vector de enteros que indica el número de neuronas por capa. Habrá capas.size() capas y cada una contendrá capas[i] neuronas
    @lr     Learning Rate o Tasa de Aprendizaje
*/
FullyConnected::FullyConnected(int *capas, int n_capas, float lr, int mini_batch)
{
    
    liberar_memoria = true;
    vector<float> neuronas_capa, w_1D;
    vector<vector<float>> w_2D;
    int n_neuronas = 0, n_pesos = 0, n_neuronas_GEMM = 0;
    this->mini_batch = mini_batch;
    this->n_capas = n_capas;
    this->capas = (int *)malloc(n_capas * sizeof(int));
    this->capas_wT = (int *)malloc(n_capas * sizeof(int));
    this->i_capa = (int *)malloc(n_capas * sizeof(int));
    this->i_w_ptr = (int *)malloc(n_capas * sizeof(int));
    this->capasGEMM = (float **)malloc(n_capas * sizeof(float*));
    this->i_capasGEMM = (int *)malloc(n_capas * sizeof(int));

    // Neuronas ------------------------------------------------------------------
    for(int i=0; i<n_capas; i++)
    {
        this->i_capa[i] = n_neuronas;
        n_neuronas += capas[i];
        this->capas[i] = capas[i];
        capas_wT[i] = capas[i]+1;

        capasGEMM[i] = (float *)malloc((capas[i]+1) * mini_batch * sizeof(float)); 
    }
    this->n_neuronas = n_neuronas;

    this->i_capasGEMM[0] = 0;
    for(int i=1; i<n_capas; i++)
    {
        this->i_capasGEMM[i] = i_capasGEMM[i-1] + (capas[i-1] + 1) * mini_batch;
        n_neuronas_GEMM += i_capasGEMM[i];
    }


    this->bias_ptr = (float *)malloc(n_neuronas * sizeof(float));       // Cada neurona tiene asociado un bias
    this->i_wT = (int *)malloc(n_capas * n_neuronas * sizeof(int));

    // Mostrar neuronas
    //mostrar_neuronas();
    

    // Pesos ------------------------------------------------------------------
    int n_pesos_T = 0;
    for(int i=0; i<n_capas-1; i++)
    {
        this->i_w_ptr[i] = n_pesos;
        n_pesos += capas[i] * capas[i+1];   // Cada neurona de la capa actual se conecta a cada neurona de la capa siguiente
        
        
        this->i_wT[i] = n_pesos_T;
        n_pesos_T += capas[i+1] * (capas[i]+1); // Añadir bias
        
    }
    
    this->n_pesos = n_pesos;
    this->w_ptr = (float *)malloc(n_pesos * sizeof(float));
    this->wT_ptr = (float *)malloc(n_pesos * n_neuronas * sizeof(float));

    // Learning Rate
    this->lr = lr;
    
    // Inicializar pesos mediante inicialización He
    for(int i=0; i<n_capas-1; ++i)
        this->generar_pesos_ptr(i);

    // Inicializar bias con un valor de 0.0
    for(int i=0; i<n_neuronas; i++)
        this->bias_ptr[i] = 0.0;
    
    
    // Almacenar matriz Transpuesta de pesos
    int cont = 0;
    for(int i=0; i<n_capas-1; i++)
    {
        for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
        {
            for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
            {
                this->wT_ptr[i_wT[i] + k*capas_wT[i] + j] = this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k];	
                //this->wT_ptr[cont++] = this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k];	
                //cont++;
            }
            
            // Añadir bias
            this->wT_ptr[i_wT[i] + k*capas_wT[i] + capas[i]] = this->bias_ptr[i_capa[i+1] + k];
            //this->wT_ptr[cont++] = this->bias_ptr[i_capa[i+1] + k];
        }
    }



    /*
    for(int i = 0; i < n_capas - 1; i++) {
    for(int k = 0; k < capas[i + 1]; k++) { // Por cada neurona de la siguiente capa
        for(int j = 0; j < capas[i]; j++) { // Por cada neurona de la capa actual
            // Calculate the index directly
            this->wT_ptr[(i * (capas[i + 1] * (capas[i] + 1))) + (k * (capas[i] + 1)) + j] = this->w_ptr[i_w_ptr[i] + j * capas[i + 1] + k];
        }
        // Añadir bias
        this->wT_ptr[(i * (capas[i + 1] * (capas[i] + 1))) + (k * (capas[i] + 1)) + capas[i]] = this->bias_ptr[i_capa[i + 1] + k];
    }
}

    
    */

    //mostrar_pesos_Traspuestos_ptr();
    

    // Punteros device
    cudaMalloc((void **) &d_wT, n_pesos * n_neuronas * sizeof(float));      
    cudaMalloc((void **) &d_w, n_pesos * n_neuronas * sizeof(float));      
    cudaMalloc((void **) &d_grad_w, n_pesos * n_neuronas * sizeof(float));      
    cudaMemcpy(d_wT, wT_ptr,  n_pesos * n_neuronas * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w_ptr,  n_pesos * n_neuronas * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_z, n_neuronas_GEMM * mini_batch * sizeof(float));      
    cudaMalloc((void **) &d_a, n_neuronas_GEMM * mini_batch * sizeof(float));      
    cudaMalloc((void **) &d_aT, n_neuronas_GEMM * mini_batch * sizeof(float));      
    cudaMalloc((void **) &d_grad_b, n_neuronas_GEMM * mini_batch * sizeof(float));      
    cudaMalloc((void **) &d_b, n_neuronas * sizeof(float));      
    cudaMalloc((void **) &d_y, capas[n_capas-1] * mini_batch * sizeof(float));      

    
    block_size = 4;
    this->block.x = block_size;
    this->block.y = block_size;

    // Tamaño de grid para propagación hacia delante
    this->grid_forward.x = (mini_batch  + block_size -1) / block_size; 
    this->grid_forward.y = (capas[1] + block_size -1) / block_size;

    this->smem = (2*block.x * block.y) *sizeof(float);
    

    // Capa SoftMax ------------
    this->block_softmax.x = this->block.x;
    this->block_softmax.y = 1;
    this->grid_softmax.x = (mini_batch  + block_softmax.x -1) / block_softmax.x; 
    this->grid_softmax.y = 1; 

    cudaMalloc((void **) &d_softmax, capas[n_capas-1] * mini_batch * sizeof(float)); 


    cudaMalloc((void **) &d_sum_acc_entr, 2*mini_batch * sizeof(float));      

};


void FullyConnected::set_biasGEMM(float *bias)
{
    cudaMemcpy(d_b, bias, n_neuronas * sizeof(float), cudaMemcpyHostToDevice);
}

void FullyConnected::set_wGEMM(float *w)
{
    cudaMemcpy(d_w, w, n_pesos * sizeof(float), cudaMemcpyHostToDevice);
}


void FullyConnected::mostrar_pesos_ptr()
{
    // Mostrar pesos
    cout << "Pesos" << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        cout << "Capa " << i << endl;
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
            {
                cout << "W(" << j << "," << k << "): ";
                cout << this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
}


void FullyConnected::mostrar_pesos_Traspuestos_ptr()
{
    // Mostrar pesos
    cout << "Pesos ^T" << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        cout << "Capa " << i << endl;
        for(int j=0; j<capas[i+1]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas_wT[i]; k++)     // Por cada neurona de la siguiente capa
            {
                cout << "W(" << j << "," << k << "): ";
                cout << this->wT_ptr[i_wT[i] + j*capas_wT[i] + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
}

void FullyConnected::mostrar_pesos()
{
    cout << "Pesos" << endl;
    for(int i=0; i<w.size(); i++)
    {
        cout << "Capa " << i << endl;
        for(int j=0; j<w[i].size(); j++)
        {
            for(int k=0; k<w[i][j].size(); k++)
            {
                cout << "W(" << j << "," << k << "): ";
                cout << w[i][j][k] << " ";

            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
}

void FullyConnected::mostrar_neuronas_ptr(float *z_ptr)
{
    // Mostrar neuronas
    cout << "Neuronas" << endl;
    for(int i=0; i<n_capas; i++)
    {
        cout << "Capa " << i << endl;
        for(int j=0; j<capas[i]; j++)
            cout << z_ptr[i_capa[i] + j] << " ";
        cout << endl;
    }
    cout << endl;
}


void FullyConnected::mostrar_neuronas(const vector<vector<float>> &z)
{
    // Mostrar neuronas
    cout << "Neuronas" << endl;
    for(int i=0; i<a.size(); i++)
    {
        cout << "Capa " << i << endl;
        for(int j=0; j<a[i].size(); j++)
            cout << z[i][j] << " ";
        cout << endl;
    }
    cout << endl;
}

/*
    @brief  Genera los pesos entre 2 capas de neuronas
    @capa   Capa a generar los pesos con la siguiente
    @return Se actualizan los valores de w (pesos de la red)
*/
void FullyConnected::generar_pesos(const int &capa)
{
    // Inicialización He
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0, sqrt(2.0 / this->a[capa].size())); 

    int cont = 0;
    for(int i=0; i<this->a[capa].size(); ++i)
        for(int j=0; j<this->a[capa+1].size(); ++j)
            this->w[capa][i][j] = cont++;
            //this->w[capa][i][j] = distribution(gen);
}

void FullyConnected::copiar_w_de_vector_a_ptr(vector<vector<vector<float>>> w_)
{
    for(int i=0; i<w_.size(); i++)
        for(int j=0; j<w_[i].size(); j++)
            for(int k=0; k<w_[i][j].size(); k++)
                w_ptr[i_w_ptr[i] + j*capas[i+1] + k] = w_[i][j][k];
}


/*
    @brief  Genera los pesos entre 2 capas de neuronas
    @capa   Capa a generar los pesos con la siguiente
    @return Se actualizan los valores de w (pesos de la red)
*/
void FullyConnected::generar_pesos_ptr(const int &capa)
{
    // Inicialización He
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0, sqrt(2.0 / this->capas[capa])); 

    int cont = 0;
    for(int i=0; i<this->capas[capa]; ++i)
        for(int j=0; j<this->capas[capa+1]; ++j)
            this->w_ptr[i_w_ptr[capa] + i*capas[capa+1] + j] = distribution(gen);
            //this->w_ptr[i_w_ptr[capa] + i*capas[capa+1] + j] = cont++;
            //this->w_ptr[i_w_ptr[capa] + i*capas[capa+1] + j] = (float) (i+1)/10;
}


/*
    @brief      Función de activación ReLU
    @x          Valor sobre el cual aplicar ReLU
    @return     @x tras aplicar ReLU sobre él
*/
float FullyConnected::relu(const float &x)
{
    float result = 0.0;

    if(x > 0)
        result = x;
    
    return result;
}

/*
    @brief      Derivada de la función de activación ReLU
    @x          Valor sobre el cual aplicar la derivada de ReLU
    @return     @x tras aplicar la derivada de ReLU sobre él
*/
float FullyConnected::deriv_relu(const float &x)
{
    float result = 0.0;

    if(x > 0)
        result = 1;
    
    return result;
}

/*
    @brief      Función de activación Sigmoid
    @x          Valor sobre el cual aplicar Sigmoid
    @return     @x tras aplicar Sigmoid sobre él
*/
float FullyConnected::sigmoid(const float &x)
{
    return 1/(1+exp(-x));
}

/*
    @brief  Propagación hacia delante por toda la red totalmente conectada
    @x      Valores de entrada a la red
    @a      Neuronas de la red antes de aplicar la función de activación
    @z      Neuronas de la red después de aplicar la función de activación

    @return Se actualizan los valores de @a y @z
*/
void FullyConnected::forwardPropagation(const vector<float> &x, vector<vector<float>> &a, vector<vector<float>> &z)
{
    float max, sum = 0.0, epsilon = 0.000000001;

    cout << "PESOS NORMALES: " << endl;
    this->mostrar_pesos_ptr();

    cout << "PESOS ^T: " << endl;


    // Introducimos input -------------------------------------------------------
    for(int i=0; i<x.size(); ++i)
    {
        z[0][i] = x[i];
        a[0][i] = x[i];
    }
        
    
    // Forward Propagation ------------------------------------------------------------
    // Por cada capa
    for(int i=0; i<this->a.size()-1; ++i)
    {

        // Por cada neurona de la capa siguiente
        for(int k=0; k<this->a[i+1].size(); ++k)
        {
            // Reset siguiente capa
            a[i+1][k] = 0.0;

            // w[i][j][k] indica el peso que conecta la neurona j de la capa i con la neurona k de la capa i+1
            for(int j=0; j<this->a[i].size(); ++j)
                a[i+1][k] += z[i][j] * this->w[i][j][k];
            
            // Aplicar bias o sesgo
            a[i+1][k] += this->bias[i+1][k];
        }

        // Aplicamos función de activación asociada a la capa actual -------------------------------

        // En capas ocultas se emplea ReLU como función de activación
        if(i < this->a.size() - 2)
        {
            for(int k=0; k<this->a[i+1].size(); ++k)
                z[i+1][k] = relu(a[i+1][k]);
            
        }else
        {
            // En la capa output se emplea softmax como función de activación
            sum = 0.0;
            max = a[i+1][0];

            // Normalizar -----------------------------------------------------------------
            // Encontrar el máximo
            for(int k=0; k<this->a[i+1].size(); ++k)
                if(max < this->a[i+1][k])
                    max = a[i+1][k];
            
            // Normalizar
            for(int k=0; k<this->a[i+1].size(); ++k)
                a[i+1][k] = a[i+1][k] - max;
            
            // Calculamos la suma exponencial de todas las neuronas de la capa output ---------------------
            for(int k=0; k<this->a[i+1].size(); ++k)
                sum += exp(a[i+1][k]);

            for(int k=0; k<this->a[i+1].size(); ++k)
                z[i+1][k] = exp(a[i+1][k]) / sum;
            
        } 
    }
}


/*
    @brief  Propagación hacia delante por toda la red totalmente conectada
    @x      Valores de entrada a la red
    @a      Neuronas de la red antes de aplicar la función de activación
    @z      Neuronas de la red después de aplicar la función de activación

    @return Se actualizan los valores de @a y @z
*/
void FullyConnected::forwardPropagationGEMM(float *x, float *y)
{
    float * capa = capasGEMM[0];


    // Copiar entrada X en capasGEMM[0]
    for(int i=0; i<this->capas[0]; i++)
        for(int j=0; j<this->mini_batch; j++)
            capa[i*mini_batch + j] = x[i*mini_batch + j];

    // Pasar valores de primera capa a GPU
    cudaMemcpy(d_z, capa,  mini_batch * (capas[0]+1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, capa,  mini_batch * (capas[0]+1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, capas[n_capas-1] * mini_batch * sizeof(float), cudaMemcpyHostToDevice);

    // Capas intermedias
    for(int c=0; c<n_capas-1; c++)
    {
        // Tamaño de grid para propagación hacia delante
        this->grid_forward.x = (mini_batch  + block_size -1) / block_size; 
        this->grid_forward.y = (capas[c+1] + block_size -1) / block_size;

        // Aplicar ReLU en capas intermedias, pero en SoftMax no
        forward_capas_intermedias<<<grid_forward, block, smem>>>(capas[c+1], mini_batch, capas[c]+1, d_wT + i_wT[c], d_z + i_capasGEMM[c], d_z + i_capasGEMM[c+1], d_a + i_capasGEMM[c+1], (c!=n_capas-2));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
    }

    // Transpuesta para tener 1 fila por dato de minibatch 
    matrizTranspuesta_GPU<<<grid_softmax, block_softmax>>>(d_z + i_capasGEMM[n_capas-1], d_softmax, capas[n_capas-1], mini_batch);
    kernel_sofmax<<<grid_softmax, block_softmax>>>(mini_batch, capas[n_capas-1], d_softmax, d_a + i_capasGEMM[n_capas-1]);

    // ---------------------------------------------------------------
    // ---------------------------------------------------------------
    /*
    cout << " ---------------------------- A ---------------------------- " << endl;
    capa = capasGEMM[0];
    for(int c=0; c<n_capas-1; c++)
    {        
        capa = capasGEMM[c];
        cudaMemcpy(capa, d_a + i_capasGEMM[c],  capas[c] * mini_batch * sizeof(float), cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
        

        cout << "CAPA " << c << " A" << endl;
        for(int i=0; i<this->capas[c]; i++)
        {
            for(int j=0; j<this->mini_batch; j++)
                cout << capa[i*mini_batch + j] << " ";
            cout << endl;
        }
        cout << endl;
    }

    capa = capasGEMM[n_capas-1];
    cudaMemcpy(capa, d_a + i_capasGEMM[n_capas-1],  capas[n_capas-1] * mini_batch * sizeof(float), cudaMemcpyDeviceToHost);
    cout << "CAPA SoftMax " << endl;
    for(int i=0; i<this->mini_batch; i++)
    {
        for(int j=0; j<this->capas[n_capas-1]; j++)
            cout << capa[i*this->capas[n_capas-1] + j]<< " ";
        cout << endl;
    }
    cout << endl;
    
    
    cout << " ---------------------------- Z ---------------------------- " << endl;
    capa = capasGEMM[0];
    for(int c=0; c<n_capas-2; c++)
    {        
        capa = capasGEMM[c+1];
        cudaMemcpy(capa, d_z + i_capasGEMM[c+1],  capas[c+1] * mini_batch * sizeof(float), cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
        

        cout << "CAPA " << c+1 << " A" << endl;
        for(int i=0; i<this->capas[c+1]; i++)
        {
            for(int j=0; j<this->mini_batch; j++)
                cout << capa[i*mini_batch + j] << " ";
            cout << endl;
        }
        cout << endl;
    }
    
    
    capa = capasGEMM[n_capas-1];
    cudaMemcpy(capa, d_softmax,  capas[n_capas-1] * mini_batch * sizeof(float), cudaMemcpyDeviceToHost);
    cout << "CAPA SoftMax " << endl;
    for(int i=0; i<this->mini_batch; i++)
    {
        for(int j=0; j<this->capas[n_capas-1]; j++)
            cout << capa[i*this->capas[n_capas-1] + j]<< " ";
        cout << endl;
    }
    cout << endl;
    */
    
}



/*
    @brief  Propagación hacia delante por toda la red totalmente conectada
    @x      Valores de entrada a la red
    @a      Neuronas de la red antes de aplicar la función de activación
    @z      Neuronas de la red después de aplicar la función de activación

    @return Se actualizan los valores de @a y @z
*/
void FullyConnected::forwardPropagation_ptr(float *x, float *a_ptr, float *z_ptr)
{
    float max, sum = 0.0, epsilon = 0.000000001;

    // Introducimos input -------------------------------------------------------
    for(int i=0; i<capas[0]; i++)
    {
        z_ptr[i] = x[i];
        a_ptr[i] = x[i];
    }

    
    // Forward Propagation ------------------------------------------------------------
    // Por cada capa
    for(int i=0; i<n_capas -1; ++i)
    {

        // Por cada neurona de la capa siguiente
        for(int k=0; k<capas[i+1]; ++k)
        {
            // Reset siguiente capa
            a_ptr[i_capa[i+1] + k] = 0.0;

            for(int j=0; j<capas[i]; ++j)
                a_ptr[i_capa[i+1] + k] += z_ptr[i_capa[i] + j] * w_ptr[i_w_ptr[i] + j*capas[i+1] + k];

            // Aplicar bias o sesgo
            a_ptr[i_capa[i+1] + k] += this->bias_ptr[i_capa[i+1] + k];
        }

        // Aplicamos función de activación asociada a la capa actual -------------------------------

        // En capas ocultas se emplea ReLU como función de activación
        if(i < n_capas - 2)
        {
            for(int k=0; k<capas[i+1]; ++k)
                z_ptr[i_capa[i+1] + k] = relu(a_ptr[i_capa[i+1] + k]);
        }else
        {
            
            // En la capa output se emplea softmax como función de activación
            sum = 0.0;
            max = a_ptr[i_capa[i+1]];

            // Normalizar -----------------------------------------------------------------
            // Encontrar el máximo
            for(int k=0; k<capas[i+1]; ++k)
                if(max < a_ptr[i_capa[i+1] + k])
                    max = a_ptr[i_capa[i+1] + k];
            
            // Normalizar
            for(int k=0; k<capas[i+1]; ++k)
                a_ptr[i_capa[i+1] + k] = a_ptr[i_capa[i+1] + k] - max;
            
            // Calculamos la suma exponencial de todas las neuronas de la capa output ---------------------
            for(int k=0; k<capas[i+1]; ++k)
                sum += exp(a_ptr[i_capa[i+1] + k]);

            for(int k=0; k<capas[i+1]; ++k)
            {
                z_ptr[i_capa[i+1] + k] = exp(a_ptr[i_capa[i+1] + k]) / sum;
                //cout << z_ptr[i_capa[i+1] + k] << " ";
            }
            //cout << endl;
        } 
    }
    
}


/*
    @brief  Realiza la medida de Entropía Cruzada sobre un conjunto de datos
    @x      Conjunto de datos de entrada
    @y      Etiquetas de los datos de entrada
    @ini    Primera posición i en cumplir {y[i] corresponde a x[i], y[i+1] corresponde a x[i+1], ...} 
    @return Valor de entropía cruzada sobre el conjunto de datos de entrada x
*/
float FullyConnected::cross_entropy(vector<vector<float>> x, vector<vector<float>> y)
{
    float sum = 0.0, prediccion = 0.0, epsilon = 0.000000001;
    int n=this->a.size()-1;

    vector<vector<float>> a, z;
    a = this->a;
    z = this->a;

    for(int i=0; i<x.size(); ++i)
    {
        forwardPropagation(x[i], a, z);

        for(int c=0; c<this->a[n].size(); ++c)
            if(y[i][c] == 1)
                prediccion = z[n][c];
            
        sum += log(prediccion+epsilon);
    }

    //sum = -sum / x.size();

    return sum;
}

/*
    @brief  Realiza la medida de Entropía Cruzada sobre un conjunto de datos
    @x      Conjunto de datos de entrada
    @y      Etiquetas de los datos de entrada
    @ini    Primera posición i en cumplir {y[i] corresponde a x[i], y[i+1] corresponde a x[i+1], ...} 
    @return Valor de entropía cruzada sobre el conjunto de datos de entrada x
*/
float FullyConnected::cross_entropy_ptr(float *x, float *y, int n_datos, float *a_ptr, float *z_ptr)
{
    float sum = 0.0, prediccion = 0.0, epsilon = 0.000000001;
    int n=n_capas-1;

    for(int i=0; i<n_datos; ++i)
    {
        float *x_i = x + i*capas[0];                                // Cada x[i] tiene tantos valores como neuronas hay en la primera capa
        forwardPropagation_ptr(x_i, a_ptr, z_ptr);

        for(int c=0; c<capas[n]; ++c)
            if(y[i*capas[n] + c] == 1)                      // Cada y[i] tiene valores como neuronas hay en la última capa. 1 neurona y 1 valor por clase. One-hot.
                prediccion = z_ptr[i_capa[n] + c];
            
        sum += log(prediccion+epsilon);
    }

    sum = -sum / n_datos;

    return sum;
}


/*
    @brief  Realiza la medida de Entropía Cruzada sobre un conjunto de datos
    @x      Conjunto de datos de entrada
    @y      Etiquetas de los datos de entrada
    @return Evaluación del modelo sobre el conjunto de datos de entrada x
*/
void FullyConnected::evaluar_modelo_GEMM(float *x, float *y)
{
    forwardPropagationGEMM(x, y);
    kernel_evaluacion_modelo<<<grid_softmax, this->block_softmax>>>(mini_batch, this->capas[n_capas-1], d_softmax, d_y, d_sum_acc_entr); 
}


/*
    @brief  Realiza la medida de Accuracy sobre un conjunto de datos
    @x      Conjunto de datos de entrada
    @y      Etiquetas de los datos de entrada
    @return Valor de accuracy sobre el conjunto de datos de entrada x
*/
float FullyConnected::accuracy_ptr(float *x, float *y, int n_datos, float *a_ptr, float *z_ptr)
{
    float sum =0.0, max;
    int prediccion, n=n_capas-1;
    float *x_i = nullptr;

    for(int i=0; i<n_datos; i++)
    {
        // Propagación hacia delante de la entrada x[i] a lo largo de la red
        x_i = x + i*capas[0];                                // Cada x[i] tiene tantos valores como neuronas hay en la primera capa
        forwardPropagation_ptr(x_i, a_ptr, z_ptr);

        // Inicialización
        max = z_ptr[i_capa[n]];
        prediccion = 0;

        // Obtener valor más alto de la capa output
        for(int c=0; c<capas[n]; ++c)
        {
            if(max < z_ptr[i_capa[n] + c])
            {
                max = z_ptr[i_capa[n] + c];
                prediccion = c;
            }
        }

        // Ver si etiqueta real y predicción coindicen
        sum += y[i*capas[n] + prediccion];
    }
    cout << endl;

    sum = sum / n_datos * 100;

    return sum;
}


/*
    @brief  Realiza la medida de Accuracy sobre un conjunto de datos
    @x      Conjunto de datos de entrada
    @y      Etiquetas de los datos de entrada
    @return Valor de accuracy sobre el conjunto de datos de entrada x
*/
float FullyConnected::accuracy(vector<vector<float>> x, vector<vector<float>> y)
{
    float sum =0.0, max;
    int prediccion, n=this->a.size()-1;

    vector<vector<float>> a, z;
    a = this->a;
    z = this->a;

    for(int i=0; i<x.size(); ++i)
    {
        // Propagación hacia delante de la entrada x[i] a lo largo de la red
        forwardPropagation(x[i], a, z);

        // Inicialización
        max = z[n][0];
        prediccion = 0;

        // Obtener valor más alto de la capa output
        for(int c=1; c<this->a[n].size(); ++c)
        {
            if(max < z[n][c])
            {
                max = z[n][c];
                prediccion = c;
            }
        }

        // Ver si etiqueta real y predicción coindicen
        sum += y[i][prediccion];
    }

    //sum = sum / x.size() * 100;

    return sum;
}


/*
    @brief      Entrenamiento de la red
    @x          Conjunto de datos de entrada
    @y          Etiquetas asociadas a cada dato de entrada
    @batch      Conjunto de índices desordenados tal que para cada dato de entrada x[i], este está asociado con y[batch[i]]
    @n_datos    Número de datos con los que entrenar
    @grad_pesos Gradiente de cada peso de la red
    @grad_b     Gradiente de cada bias de la red
    @grad_x     Gradiente respecto a la entrada x
    @a          Neuronas de la red antes de aplicar la función de activación
    @z          Neuronas de la red después de aplicar la función de activación
    @grad_a     Gradiente respecto a cada neurona de la red antes de aplicar la función de activación
    @return     Se actualizan los valores de @grad_pesos, @grad_b, @grad_x, @a, @z, y @grad_a
*/
//void FullyConnected::train(const vector<vector<float>> &x, const vector<vector<float>> &y, const vector<int> &batch, const int &n_datos, vector<vector<vector<float>>> &grad_pesos, vector<vector<float>> &grad_b, vector<vector<float>> &grad_x, vector<vector<float>> &a, vector<vector<float>> &z, vector<vector<float>> &grad_a)
void FullyConnected::trainGEMM(float *x, float *y, int *batch, const int &n_datos, float * grad_w_ptr, float * grad_bias_ptr, float *grad_x, float *a_ptr, float *z_ptr, float *grad_a_ptr)
{
    float epsilon = 0.000000001;
    int i_output = n_capas -1, i_last_h = i_output-1; // índice de la capa output, Índice de la capa h1 respectivamente
    int n_clases = capas[n_capas-1];
    float *grad_a = (float *)malloc(capas[n_capas-1] * mini_batch * sizeof(float));
    dim3 grid_back_softmax((mini_batch  + block_softmax.x -1) / block_softmax.x, 1); 
    dim3 grid_transpuesta_back; 
    dim3 grid_grad_w; 
    dim3 grid_grad_b; 
    cudaMemcpy(d_y, y, n_clases * mini_batch * sizeof(float), cudaMemcpyHostToDevice);

    // Propagación hacia delante de cada dato perteneciente al minibatch
    forwardPropagationGEMM(x, y);

    // Cálculo del gradiente respecto a la entrada de la capa SoftMax
    kernel_back_softmax<<<grid_back_softmax, block_softmax>>>(mini_batch, n_clases, d_z + i_capasGEMM[n_capas-1], d_softmax, d_y, n_clases);

    // Capas intermedias
    // Transpuesta para tener 1 columna por dato de minibatch 
    matrizTranspuesta_GPU<<<grid_back_softmax, block_softmax>>>(d_z + i_capasGEMM[n_capas-1], d_a + i_capasGEMM[n_capas-1], mini_batch, capas[n_capas-1]);
    
    // Capas intermedias
    for(int c=n_capas-1; c>0; c--)
    {
        // Tamaño de grid
        this->grid_forward.x = (mini_batch  + block_size -1) / block_size; 
        this->grid_forward.y = (capas[c-1] + block_size -1) / block_size;

        // Aplicar ReLU en capas intermedias, pero en SoftMax no
        backprop_capas_intermedias<<<grid_forward, block, smem>>>(capas[c-1], mini_batch, capas[c], d_w + i_w_ptr[c-1], d_a + i_capasGEMM[c], d_a + i_capasGEMM[c-1], (c>1));

        // Tamaño de grid
        grid_grad_b.x = (mini_batch  + block_size -1) / block_size; 
        grid_grad_b.y = (capas[c]  + block_size -1) / block_size;
        kernel_grad_bias<<<grid_grad_b, block_softmax>>>(capas[c], mini_batch, d_a + i_capasGEMM[c], d_grad_b + i_capasGEMM[c]);

        // Tamaño de grid
        grid_transpuesta_back.x = (capas[c]  + block_size -1) / block_size; 
        grid_transpuesta_back.y = 1;

        matrizTranspuesta_GPU<<<grid_transpuesta_back, block_softmax>>>(d_a + i_capasGEMM[c], d_aT + i_capasGEMM[c], capas[c], mini_batch);

        // Tamaño de grid
        grid_grad_w.x = (capas[c]  + block_size -1) / block_size; 
        grid_grad_w.y = (capas[c-1] + block_size -1) / block_size;

        multiplicarMatricesGPU<<<grid_grad_w, block, smem>>>(capas[c-1], capas[c], mini_batch, d_z + i_capasGEMM[c-1], d_aT + i_capasGEMM[c], d_grad_w + i_w_ptr[c-1]);
    }

    int c = 0;
    float *capa = capasGEMM[0];

    /*
    cout << " ---------------------------- Gradiente de bias GEMM ---------------------------- " << endl;
    //float *capa = capasGEMM[0];
    for(int c=0; c<n_capas; c++)
    {        
        capa = capasGEMM[c];
        cudaMemcpy(capa, d_grad_b + i_capasGEMM[c], capas[c] * sizeof(float), cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));

        for(int i=0; i<this->capas[c]; i++)
            cout << capa[i] << " ";
        cout << endl;
    }    

    /*
    cout << " ---------------------------- A ---------------------------- " << endl;
    //float *capa = capasGEMM[0];
    for(int c=0; c<n_capas; c++)
    {        
        capa = capasGEMM[c];
        cudaMemcpy(capa, d_a + i_capasGEMM[c],  capas[c] * mini_batch * sizeof(float), cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));

        cout << "CAPA " << c << " A" << endl;
        for(int i=0; i<this->capas[c]; i++)
        {
            for(int j=0; j<this->mini_batch; j++)
                cout << capa[i*mini_batch + j] << " ";
            cout << endl;
        }
        cout << endl;
    }
    */
    
    /*
    // Mostrar pesos
    float * h_grad_w = (float *)malloc(n_pesos * n_neuronas * sizeof(float));
    cudaMemcpy(h_grad_w, d_grad_w, n_pesos * n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);
    cout << "Gradiente de Pesos GEMM" << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        cout << "Capa " << i << endl;
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
            {
                cout << "W(" << j << "," << k << "): ";
                cout << h_grad_w[i_w_ptr[i] + j*capas[i+1] + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    */
}


/*
    @brief      Entrenamiento de la red
    @x          Conjunto de datos de entrada
    @y          Etiquetas asociadas a cada dato de entrada
    @batch      Conjunto de índices desordenados tal que para cada dato de entrada x[i], este está asociado con y[batch[i]]
    @n_datos    Número de datos con los que entrenar
    @grad_pesos Gradiente de cada peso de la red
    @grad_b     Gradiente de cada bias de la red
    @grad_x     Gradiente respecto a la entrada x
    @a          Neuronas de la red antes de aplicar la función de activación
    @z          Neuronas de la red después de aplicar la función de activación
    @grad_a     Gradiente respecto a cada neurona de la red antes de aplicar la función de activación
    @return     Se actualizan los valores de @grad_pesos, @grad_b, @grad_x, @a, @z, y @grad_a
*/
//void FullyConnected::train(const vector<vector<float>> &x, const vector<vector<float>> &y, const vector<int> &batch, const int &n_datos, vector<vector<vector<float>>> &grad_pesos, vector<vector<float>> &grad_b, vector<vector<float>> &grad_x, vector<vector<float>> &a, vector<vector<float>> &z, vector<vector<float>> &grad_a)
void FullyConnected::train_ptr(float *x, float *y, int *batch, const int &n_datos, float * grad_w_ptr, float * grad_bias_ptr, float *grad_x, float *a_ptr, float *z_ptr, float *grad_a_ptr)
{
    float epsilon = 0.000000001;
    int i_output = n_capas -1, i_last_h = i_output-1; // índice de la capa output, Índice de la capa h1 respectivamente

    for(int i=0; i<n_datos; ++i)
    {
        // Propagación hacia delante
        float *x_i = x + i*capas[0];                                // Cada x[i] tiene tantos valores como neuronas hay en la primera capa
        forwardPropagation_ptr(x_i, a_ptr, z_ptr);
        
        // ---------------------
        // ---------------------
        /*
        cout << "z_ptr " << i << endl;
        for(int c=0; c<n_capas; c++)
        {
            for(int j=0; j<capas[c]; j++)
                cout << z_ptr[i_capa[c] + j] << " ";
            cout << endl;
        }
        cout << endl;
        */
        // ---------------------
        // ---------------------

        // Propagación hacia detrás
        // Inicializar a 0 gradiente respecto a input
        for(int _i = 0; _i < n_capas; ++_i)
            for(int j = 0; j < capas[_i]; ++j)
                grad_a_ptr[i_capa[_i] + j] = 0.0;

        // Capa SoftMax -----------------------------------------------
        // Se calcula gradiente del error respecto a cada Z_k
        for(int k=0; k<capas[i_output]; ++k)
            grad_a_ptr[i_capa[i_output] + k] = z_ptr[i_capa[i_output] + k] - y[batch[i]*capas[i_output] + k];
            // grad_Zk = O_k - y_k

        // Pesos h_last - Softmax
        for(int p=0; p<capas[i_last_h]; ++p)
            for(int k=0; k<capas[i_output]; ++k)
                grad_w_ptr[i_w_ptr[i_last_h] + p*capas[i_last_h+1] + k] += grad_a_ptr[i_capa[i_output] + k] * z_ptr[i_capa[i_last_h] + p];
                //                                 grad_Zk                  *  z^i_last_h_p
        
        // Sesgos capa softmax
        for(int k=0; k<capas[i_output]; ++k)
            grad_bias_ptr[i_capa[i_output] + k] += grad_a_ptr[i_capa[i_output] + k];
            // bk = grad_Zk

        // Última capa oculta -----------------------------------------------
        for(int p=0; p<capas[i_last_h]; ++p)      
            for(int k=0; k<capas[i_output]; ++k)
                grad_a_ptr[i_capa[i_last_h] + p] += grad_a_ptr[i_capa[i_output] + k] * w_ptr[i_w_ptr[i_last_h] + p*capas[i_last_h+1] + k] * deriv_relu(a_ptr[i_capa[i_last_h] + p]);
                //                              grad_Zk           *  w^i_last_h_pk          * ...
                
        // Capas ocultas intermedias
        for(int capa= i_last_h; capa > 1; capa--)
        {
            // Pesos
            for(int i_act = 0; i_act < capas[capa]; ++i_act)       // Por cada neurona de la capa actual
                for(int i_ant = 0; i_ant < capas[capa-1]; ++i_ant)     // Por cada neurona de la capa anterior
                    grad_w_ptr[i_w_ptr[capa-1] + i_ant*capas[capa] + i_act] += grad_a_ptr[i_capa[capa] + i_act] * z_ptr[i_capa[capa-1] + i_ant];

            // Bias
            for(int i_act = 0; i_act < capas[capa]; ++i_act)
                grad_bias_ptr[i_capa[capa] + i_act] += grad_a_ptr[i_capa[capa] + i_act];
            
            // Grad input
            for(int i_ant = 0; i_ant < capas[capa-1]; ++i_ant)     // Por cada neurona de la capa anterior
                for(int i_act = 0; i_act < capas[capa]; ++i_act)       // Por cada neurona de la capa actual
                    grad_a_ptr[i_capa[capa-1] + i_ant] += grad_a_ptr[i_capa[capa] + i_act] * w_ptr[i_w_ptr[capa-1] + i_ant*capas[capa] + i_act] * deriv_relu(a_ptr[i_capa[capa-1] + i_ant]);
        }
        
        
        // Capa input
        // Pesos
        int capa=1;
        for(int i_act = 0; i_act < capas[capa]; ++i_act)       // Por cada neurona de la capa actual
            for(int i_ant = 0; i_ant < capas[capa-1]; ++i_ant)     // Por cada neurona de la capa anterior
                grad_w_ptr[i_w_ptr[capa-1] + i_ant*capas[capa] + i_act] += grad_a_ptr[i_capa[capa] + i_act] * z_ptr[i_capa[capa-1] + i_ant];
        
        // Grad input
        for(int i_ant = 0; i_ant < capas[capa-1]; ++i_ant)     // Por cada neurona de la capa anterior
            for(int i_act = 0; i_act < capas[capa]; ++i_act)       // Por cada neurona de la capa actual
                grad_a_ptr[i_capa[capa-1] + i_ant] += grad_a_ptr[i_capa[capa] + i_act] * w_ptr[i_w_ptr[capa-1] + i_ant*capas[capa] + i_act];

        // Bias
        for(int i_act = 0; i_act < capas[capa]; ++i_act)
            grad_bias_ptr[i_capa[capa] + i_act] += grad_a_ptr[i_capa[capa] + i_act];

        // Copiar gradiente respecto a input de primera capa en grad_x[i]
        for(int j=0; j<capas[0]; j++)
            grad_x[i*capas[0] + j] = grad_a_ptr[j];


        // ---------------------
        // ---------------------
        /*
        cout << "grad_a_ptr " << i << endl;
        for(int c=0; c<n_capas; c++)
        {
            for(int j=0; j<capas[c]; j++)
                cout << grad_a_ptr[i_capa[c] + j] << " ";
            cout << endl;
        }
        cout << endl;
        */
        // ---------------------
        // ---------------------

    } 

    // Mostrar pesos
    /*
    cout << "Gradiente de Pesos" << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        cout << "Capa " << i << endl;
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
            {
                cout << "W(" << j << "," << k << "): ";
                cout << grad_w_ptr[i_w_ptr[i] + j*capas[i+1] + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    */
    /*
    cout << "grad_bias_ptr " << endl;
    for(int c=0; c<n_capas; c++)
    {
        for(int j=0; j<capas[c]; j++)
            cout << grad_bias_ptr[i_capa[c] + j] << " ";
        cout << endl;
    }
    cout << endl;
    */    
}



/*
    @brief      Entrenamiento de la red
    @x          Conjunto de datos de entrada
    @y          Etiquetas asociadas a cada dato de entrada
    @batch      Conjunto de índices desordenados tal que para cada dato de entrada x[i], este está asociado con y[batch[i]]
    @n_datos    Número de datos con los que entrenar
    @grad_pesos Gradiente de cada peso de la red
    @grad_b     Gradiente de cada bias de la red
    @grad_x     Gradiente respecto a la entrada x
    @a          Neuronas de la red antes de aplicar la función de activación
    @z          Neuronas de la red después de aplicar la función de activación
    @grad_a     Gradiente respecto a cada neurona de la red antes de aplicar la función de activación
    @return     Se actualizan los valores de @grad_pesos, @grad_b, @grad_x, @a, @z, y @grad_a
*/
void FullyConnected::train(const vector<vector<float>> &x, const vector<vector<float>> &y, const vector<int> &batch, const int &n_datos, vector<vector<vector<float>>> &grad_pesos, vector<vector<float>> &grad_b, vector<vector<float>> &grad_x, vector<vector<float>> &a, vector<vector<float>> &z, vector<vector<float>> &grad_a)
{
    float epsilon = 0.000000001;
    int i_output = this->a.size()-1, i_last_h = i_output-1; // índice de la capa output, Índice de la capa h1 respectivamente
    vector<vector<float>> vector_2d(n_datos);
    grad_x = vector_2d;


    for(int i=0; i<n_datos; ++i)
    {
        // Propagación hacia delante
        forwardPropagation(x[i], a, z);
        
        // Propagación hacia detrás
        // Inicializar a 0 gradiente respecto a input
        for(int _i = 0; _i < grad_a.size(); ++_i)
            for(int j = 0; j < grad_a[_i].size(); ++j)
                grad_a[_i][j] = 0.0;


        // Capa SoftMax -----------------------------------------------
        // Se calcula gradiente del error respecto a cada Z_k
        for(int k=0; k<this->a[i_output].size(); ++k)
            grad_a[i_output][k] = z[i_output][k] - y[batch[i]][k];
            // grad_Zk = O_k - y_k
        
        // Pesos h_last - Softmax
        for(int p=0; p<this->a[i_last_h].size(); ++p)
            for(int k=0; k<this->a[i_output].size(); ++k)
                grad_pesos[i_last_h][p][k] += grad_a[i_output][k] * z[i_last_h][p];
                //                                 grad_Zk                  *  z^i_last_h_p
        
        // Sesgos capa softmax
        for(int k=0; k<this->a[i_output].size(); ++k)
            grad_b[i_output][k] += grad_a[i_output][k];
            // bk = grad_Zk

        // Última capa oculta -----------------------------------------------
        for(int p=0; p<this->a[i_last_h].size(); ++p)      
            for(int k=0; k<this->a[i_output].size(); ++k)
                grad_a[i_last_h][p] += grad_a[i_output][k] * this->w[i_last_h][p][k] * deriv_relu(a[i_last_h][p]);
                //                              grad_Zk           *  w^i_last_h_pk          * ...
                
        // Capas ocultas intermedias
        for(int capa= i_last_h; capa > 1; capa--)
        {
            // Pesos
            for(int i_act = 0; i_act < this->a[capa].size(); ++i_act)       // Por cada neurona de la capa actual
                for(int i_ant = 0; i_ant < this->a[capa-1].size(); ++i_ant)     // Por cada neurona de la capa anterior
                    grad_pesos[capa-1][i_ant][i_act] += grad_a[capa][i_act] * z[capa-1][i_ant];

            // Bias
            for(int i_act = 0; i_act < this->a[capa].size(); ++i_act)
                grad_b[capa][i_act] += grad_a[capa][i_act];
            
            // Grad input
            for(int i_ant = 0; i_ant < this->a[capa-1].size(); ++i_ant)     // Por cada neurona de la capa anterior
                for(int i_act = 0; i_act < this->a[capa].size(); ++i_act)       // Por cada neurona de la capa actual
                    grad_a[capa-1][i_ant] += grad_a[capa][i_act] * this->w[capa-1][i_ant][i_act] * deriv_relu(a[capa-1][i_ant]);
        }
        
        
        // Capa input
        // Pesos
        int capa=1;
        for(int i_act = 0; i_act < this->a[capa].size(); ++i_act)       // Por cada neurona de la capa actual
            for(int i_ant = 0; i_ant < this->a[capa-1].size(); ++i_ant)     // Por cada neurona de la capa anterior
                grad_pesos[capa-1][i_ant][i_act] += grad_a[capa][i_act] * z[capa-1][i_ant];
        
        // Grad input
        for(int i_ant = 0; i_ant < this->a[capa-1].size(); ++i_ant)     // Por cada neurona de la capa anterior
            for(int i_act = 0; i_act < this->a[capa].size(); ++i_act)       // Por cada neurona de la capa actual
                grad_a[capa-1][i_ant] += grad_a[capa][i_act] * this->w[capa-1][i_ant][i_act];


        grad_x[i] = grad_a[0];
        
    } 
}

/*
    @brief          Escalar los pesos para evitar que los gradientes "exploten"
    @clip_value     Valor a emplear para realizar el "clip" o escalado
    @return         Se actualizan los valores de w (pesos de la red)
*/
void FullyConnected::escalar_pesos_GEMM(float clip_value)
{
    //float *bias_copy = (float *)malloc(n_neuronas * sizeof(float));       // Cada neurona tiene asociado un bias
    dim3 bloque_reduce(32, 1);
    float *pesos_copy = (float *)malloc(n_pesos * n_neuronas * sizeof(float)),
          *max = (float *)malloc(sizeof(float)),    
          *min = (float *)malloc(sizeof(float));    
    dim3 grid_reduce((n_pesos * n_neuronas + bloque_reduce.x -1) / bloque_reduce.x, 1);
    size_t smem_reduce = (2*bloque_reduce.x * bloque_reduce.y) *sizeof(float);

    float *d_max_por_bloque, *d_min_por_bloque, *d_max, *d_min;
    cudaMalloc((void **) &d_max_por_bloque, grid_reduce.x * sizeof(float));      
    cudaMalloc((void **) &d_min_por_bloque, grid_reduce.x * sizeof(float));      
    cudaMalloc((void **) &d_max, sizeof(float));      
    cudaMalloc((void **) &d_min, sizeof(float));      
    
    cudaMemcpy(pesos_copy, d_w, n_pesos * n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);

    // Mostrar pesos
    /*
    cout << "Pesos antes " << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
                cout << pesos_copy[i_w_ptr[i] + j*capas[i+1] + k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    */
    
    reduceMax<<<grid_reduce, bloque_reduce, smem_reduce>>>(d_w, d_max_por_bloque, n_pesos * n_neuronas, d_max); 
    reduceMin<<<grid_reduce, bloque_reduce, smem_reduce>>>(d_w, d_min_por_bloque, n_pesos * n_neuronas, d_min); 
    kernel_escalar_pesos<<<grid_reduce, bloque_reduce>>>(d_w, n_pesos * n_neuronas, d_max, d_min, clip_value); 

    cudaMemcpy(max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(min, d_min, sizeof(float), cudaMemcpyDeviceToHost);

    /*
    cout << "Máximo GEMM: " << max[0] << endl;
    cout << "Mínimo GEMM: " << min[0] << endl;
    

    cudaMemcpy(pesos_copy, d_w, n_pesos * n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);

    // Mostrar pesos
    cout << "Pesos después " << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
                cout << pesos_copy[i_w_ptr[i] + j*capas[i+1] + k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;    
    */
}


/*
    @brief          Escalar los pesos para evitar que los gradientes "exploten"
    @clip_value     Valor a emplear para realizar el "clip" o escalado
    @return         Se actualizan los valores de w (pesos de la red)
*/
void FullyConnected::escalar_pesos_ptr(float clip_value)
{
    /*
    // Mostrar pesos
    cout << "Pesos antes " << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
                cout << w_ptr[i_w_ptr[i] + j*capas[i+1] + k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    */

    // Calcular el máximo y el mínimo de los pesos
    float max = this->w_ptr[0], min = this->w_ptr[0];
    
    for(int i=0; i<n_capas-1; i++)
        for(int j=0; j<capas[i]; j++)
            for(int k=0; k<capas[i+1]; k++)
            {
                if(max < this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k])
                    max = this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k];
                
                if(min > this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k])
                    min = this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k];
            }
    
    //cout << "Máximo: " << max << endl;
    //cout << "Mínimo: " << min << endl;
    // Realizar gradient clipping
    float scaling_factor = clip_value / std::max(std::abs(max), std::abs(min));
    for(int i=0; i<n_capas-1; i++)
        for(int j=0; j<capas[i]; j++)
            for(int k=0; k<capas[i+1]; k++)
                this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k] = std::max(std::min(this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k], clip_value), -clip_value);

    // Mostrar pesos
    /*
    cout << "Pesos después " << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
                cout << w_ptr[i_w_ptr[i] + j*capas[i+1] + k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    */
}

/*
    @brief          Escalar los pesos para evitar que los gradientes "exploten"
    @clip_value     Valor a emplear para realizar el "clip" o escalado
    @return         Se actualizan los valores de w (pesos de la red)
*/
void FullyConnected::escalar_pesos(float clip_value)
{
    // Calcular el máximo y el mínimo de los pesos
    float max = this->w[0][0][0], min = this->w[0][0][0];

    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
            {
                if(max < this->w[i][j][k])
                    max = this->w[i][j][k];
                
                if(min > this->w[i][j][k])
                    min = this->w[i][j][k];
            }
    
    // Realizar gradient clipping
    float scaling_factor = clip_value / std::max(std::abs(max), std::abs(min));
    for(int i=0; i<this->w.size(); i++)
        for(int j=0; j<this->w[i].size(); j++)
            for(int k=0; k<this->w[i][j].size(); k++)
                this->w[i][j][k] = std::max(std::min(this->w[i][j][k], clip_value), -clip_value);
}


/*
    @brief          Actualizar los pesos y sesgos de la red
    @grad_pesos     Gradiente de cada peso de la red
    @grad_b         Gradiente de cada sesgo de la red
    @return         Se actualizar los valores de w y bias (pesos y sesgos de la red)
*/
void FullyConnected::actualizar_parametros_gpu(float *grad_pesos, float *grad_b)
{
    float *bias_copy = (float *)malloc(n_neuronas * sizeof(float));       // Cada neurona tiene asociado un bias
    float *grad_pesos_copy = (float *)malloc(n_pesos * n_neuronas * sizeof(float));       
    float *pesos_copy = (float *)malloc(n_pesos * n_neuronas * sizeof(float));       
    dim3 grid_act_grad_b;
    dim3 grid_act_grad_w;
    cudaMemcpy(pesos_copy, d_w, n_pesos * n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);
    
    /*
    // Mostrar pesos
    cout << "Pesos GEMM antes" << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
                cout << pesos_copy[i_w_ptr[i] + j*capas[i+1] + k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    */


    int * d_capas_wT, *d_capas;
    cudaMalloc((void **) &d_capas_wT, n_capas * sizeof(int));      
    cudaMalloc((void **) &d_capas, n_capas * sizeof(int));

    cudaMemcpy(d_capas_wT, capas_wT, n_capas * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capas, capas, n_capas * sizeof(int), cudaMemcpyHostToDevice);
    

    /*
    cudaMemcpy(bias_copy, d_b, n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);

    
    cout << "bias antes GEMM " << endl;
    for(int c=0; c<n_capas; c++)
    {
        for(int j=0; j<capas[c]; j++)
            cout << bias_copy[i_capa[c] + j] << " ";
        cout << endl;
    }
    cout << endl;
    
    */
    for(int i=0; i<n_capas; i++)
    {
        // Tamaño de grid
        grid_act_grad_b.x = (capas[i]  + block_softmax.x -1) / block_softmax.x; 
        grid_act_grad_b.y = 1;
        kernel_actualizar_bias<<<grid_act_grad_b, block_softmax>>>(capas[i], d_b + i_capa[i], d_grad_b + i_capasGEMM[i], this->lr);
    }

    // Actualizar pesos
    for(int i=0; i<n_capas-1; i++)
    {
        // Tamaño de grid
        grid_act_grad_w.x = (capas[i+1]  + block_softmax.x -1) / block_softmax.x; 
        grid_act_grad_w.y = (capas[i]  + block_softmax.x -1) / block_softmax.x;
        kernel_actualizar_pesos<<<grid_act_grad_w, block>>>(capas[i], capas[i+1], d_w + i_w_ptr[i], d_grad_w + i_w_ptr[i], this->lr);   
        

        kernel_transpuesta_pesos<<<grid_act_grad_w, block>>>(d_capas_wT, d_capas, d_w + i_w_ptr[i], d_wT + i_wT[i], i, d_b + i_capa[i+1]);
    }

    cudaMemcpy(pesos_copy, d_w, n_pesos * n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);
    
    /*
    // Mostrar pesos
    cout << "Pesos GEMM después" << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
                cout << pesos_copy[i_w_ptr[i] + j*capas[i+1] + k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    */

    cudaMemcpy(pesos_copy, d_wT, n_pesos * n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);
    
    /*
    // Mostrar pesos
    cout << "Pesos ^T" << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        for(int j=0; j<capas[i+1]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas_wT[i]; k++)     // Por cada neurona de la siguiente capa
            {
                cout << pesos_copy[i_wT[i] + j*capas_wT[i] + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    */
    


    /*
    cudaMemcpy(bias_copy, d_b, n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "bias después GEMM " << endl;
    for(int c=0; c<n_capas; c++)
    {
        for(int j=0; j<capas[c]; j++)
            cout << bias_copy[i_capa[c] + j] << " ";
        cout << endl;
    }
    cout << endl;
    */
}

/*
    @brief          Actualizar los pesos y sesgos de la red
    @grad_pesos     Gradiente de cada peso de la red
    @grad_b         Gradiente de cada sesgo de la red
    @return         Se actualizar los valores de w y bias (pesos y sesgos de la red)
*/
void FullyConnected::actualizar_parametros_ptr(float *grad_pesos, float *grad_b)
{
    /*
    // Mostrar pesos
    cout << "Pesos antes " << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
                cout << w_ptr[i_w_ptr[i] + j*capas[i+1] + k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    */

    // Actualizar pesos
    for(int i=0; i<n_capas-1; i++)
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
                w_ptr[i_w_ptr[i] + j*capas[i+1] + k] -= this->lr * grad_pesos[i_w_ptr[i] + j*capas[i+1] + k];

    /*
    // Mostrar pesos
    cout << "Pesos después " << endl;
    for(int i=0; i<n_capas-1; i++)
    {
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
        {
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
                cout << w_ptr[i_w_ptr[i] + j*capas[i+1] + k] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    */

    /*
    cout << "bias antes " << endl;
    for(int c=0; c<n_capas; c++)
    {
        for(int j=0; j<capas[c]; j++)
            cout << bias_ptr[i_capa[c] + j] << " ";
        cout << endl;
    }
    cout << endl;
    */

    for(int i=0; i<n_capas; i++)
        for(int j=0; j<capas[i]; j++)
            bias_ptr[i_capa[i] + j] -= this->lr * grad_b[i_capa[i] + j];

    /*
    cout << "bias después " << endl;
    for(int c=0; c<n_capas; c++)
    {
        for(int j=0; j<capas[c]; j++)
            cout << bias_ptr[i_capa[c] + j] << " ";
        cout << endl;
    }
    cout << endl;   
    */ 
}

/*
    @brief          Actualizar los pesos y sesgos de la red
    @grad_pesos     Gradiente de cada peso de la red
    @grad_b         Gradiente de cada sesgo de la red
    @return         Se actualizar los valores de w y bias (pesos y sesgos de la red)
*/
void FullyConnected::actualizar_parametros(const vector<vector<vector<float>>> &grad_pesos, const vector<vector<float>> &grad_b)
{
    // Actualizar pesos
    for(int j=0; j<this->w.size(); j++)
        for(int k=0; k<this->w[j].size(); k++)
            for(int p=0; p<this->w[j][k].size(); p++)
                this->w[j][k][p] -= this->lr * grad_pesos[j][k][p];

    // Actualizar bias
    for(int i=0; i<this->bias.size(); i++)
        for(int j=0; j<this->bias[i].size(); j++)
            this->bias[i][j] -= this->lr * grad_b[i][j];
}


void FullyConnected::matrizTranspuesta(float* X, float *Y, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            Y[j * rows + i] = X[i * cols + j];
}

void FullyConnected::matrizTranspuesta(float* matrix, int rows, int cols)
{
    // Allocate a new matrix to hold the transposed data
    float* transposedMatrix = (float*)malloc(cols * rows * sizeof(float));
    
    if (transposedMatrix == nullptr) {
        cerr << "Memory allocation for transposed matrix failed" << endl;
        exit(1);
    }

    // Transpose the matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Copy the element from original matrix at position (i, j) to
            // transposed matrix at position (j, i)
            transposedMatrix[j * rows + i] = matrix[i * cols + j];
        }
    }

    // Copy the transposed data back to the original matrix
    // This requires the original matrix to be able to hold the new dimensions
    // i.e., it must have space for 'cols * rows' elements
    for (int i = 0; i < cols * rows; i++) {
        matrix[i] = transposedMatrix[i];
    }

    // Free the allocated memory for the transposed matrix
    free(transposedMatrix);
}

void mostrar_ptr_2D(float *x, int H, int W)
{
    for(int i=0; i<H; i++)
    {
        for(int j=0; j<W; j++)
            cout << x[i*W + j] << " ";
        cout << endl;
    }
}

void mostrar_vector_2D(vector<vector<float>> x)
{
    for(int i=0; i<x.size(); i++)
    {
        for(int j=0; j<x[i].size(); j++)
            cout << x[i][j] << " ";
        cout << endl;
    }
}



int main()
{
    // CPU --------------
    cout << " ---------- CPU ---------- " << endl; 
    int tam_x = 3, n_datos = 10, n_clases = 2;
    //int tam_x = 3, n_datos = 3, n_clases = 2;
    vector<int> capas = {tam_x, 5, 3, 7, n_clases};
    //vector<int> capas = {tam_x, 2, 3, 2, 4, n_clases};
    vector<vector<float>> a_cpu, z_cpu, grad_a_cpu, y_cpu = {{0.0, 1.0}, {0.0, 1.0}};
    vector<vector<vector<float>>> grad_w_cpu;
    vector<float> x_cpu;
    vector<vector<float>> X_cpu, grad_b_cpu, grad_x_cpu;
    vector<int> batch_cpu(n_datos);

    

    /*
    FullyConnected fully_cpu(capas, 0.1);
    a_cpu = fully_cpu.get_a();
    z_cpu = fully_cpu.get_a();
    grad_a_cpu = fully_cpu.get_a();
    grad_w_cpu = fully_cpu.get_pesos();
    grad_b_cpu = fully_cpu.get_bias();

    for(int i=0; i<n_datos; i++)
        batch_cpu[i] = i;

    for(int i=0; i<tam_x; i++)
        x_cpu.push_back(i);

    for(int i=0; i<n_datos; i++)
        X_cpu.push_back(x_cpu);

    
    for(int i=0; i<grad_w_cpu.size(); i++)
        for(int j=0; j<grad_w_cpu[i].size(); j++)
            for(int k=0; k<grad_w_cpu[i][j].size(); k++)
                grad_w_cpu[i][j][k] = 0.0;

    for(int i=0; i<grad_b_cpu.size(); i++)
        for(int j=0; j<grad_b_cpu[i].size(); j++)
            grad_b_cpu[i][j] = 0.0;
    /*
    //fully_cpu.forwardPropagation(x_cpu, a_cpu, z_cpu);
    //fully_cpu.mostrar_neuronas(z_cpu);
    mostrar_vector_2D(X_cpu);
    mostrar_vector_2D(y_cpu);
    cout << "Entr: " << fully_cpu.cross_entropy(X_cpu, y_cpu) << endl;
    cout << "Acc: " << fully_cpu.accuracy(X_cpu, y_cpu) << endl;
    
    vector<vector<vector<float>>> w_cpu_copy = fully_cpu.get_pesos();
    
    fully_cpu.train(X_cpu, y_cpu, batch_cpu, n_datos, grad_w_cpu, grad_b_cpu, grad_x_cpu, a_cpu, z_cpu, grad_a_cpu);
    fully_cpu.actualizar_parametros(grad_w_cpu, grad_b_cpu);
    fully_cpu.escalar_pesos(2);
    //fully_cpu.mostrar_pesos();
    */
    



    // GPU --------------
    cout << " ---------- GPU ---------- " << endl; 
    int n_capas = 5;
    int *capas_ptr = (int *)malloc(n_capas * sizeof(int));
    int *i_capa = (int *)malloc(n_capas * sizeof(int));

    for(int i=0; i<n_capas; i++)
        capas_ptr[i] = capas[i];
    FullyConnected fully_gpu(capas_ptr, n_capas, 0.1, n_datos);

    
    int n_neuronas = 0, n_pesos = 0;

    // Pesos ------------------------------------------------------------------
    int *i_w_ptr = (int *)malloc(n_capas * sizeof(int));

    // Neuronas ------------------------------------------------------------------
    for(int i=0; i<n_capas; i++)
    {
        i_capa[i] = n_neuronas;
        n_neuronas += capas_ptr[i];
        capas_ptr[i] = capas[i];
    }

    for(int i=0; i<n_capas-1; i++)
    {
        i_w_ptr[i] = n_pesos;
        n_pesos += capas_ptr[i] * capas_ptr[i+1];   // Cada neurona de la capa actual se conecta a cada neurona de la capa siguiente
    }
    
    
    float *grad_w_ptr = (float *)malloc(n_pesos * sizeof(float));

    // Inicializar gradientes de pesos a 0.0
    for(int i=0; i<n_pesos; i++)
        grad_w_ptr[i] = 0.0;

    // Entrada y salida ------------------------------------------------------------------
    float *X_gpu = (float *)malloc(n_datos * tam_x * sizeof(float));
    float *X_gpuT = (float *)malloc(n_datos * tam_x * sizeof(float));
    float *y_gpu = (float *)malloc(n_datos * n_clases * sizeof(float));
    int *batch_gpu = (int *)malloc(n_datos * sizeof(int));
    
    for(int i=0; i<n_datos; i++)
        batch_gpu[i] = i;

    
    int cont = -2;
    for(int i=0; i<n_datos; i++)
        for(int j=0; j<tam_x; j++)
        {
            X_gpu[i*tam_x + j] = cont;
            X_gpuT[i*tam_x + j] = cont++;
            //X_gpu[i*tam_x + j] = X_cpu[i][j];
        }

    
    for(int i=0; i<n_datos; i++)
        for(int j=0; j<n_clases; j++)
            (j == 1) ? y_gpu[i*n_clases + j] = 1.0 : y_gpu[i*n_clases + j] = 0.0;

    fully_gpu.matrizTranspuesta(X_gpuT, n_datos, tam_x);
    /*
    cout << "DATOS" << endl;
    for(int i=0; i<n_datos; i++)
    {
        for(int j=0; j<tam_x; j++)
        {
            cout << X_gpu[i*tam_x + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "DATOS ^T" << endl;
    for(int i=0; i<tam_x; i++)
    {
        for(int j=0; j<n_datos; j++)
        {
            cout << X_gpuT[i*n_datos + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    */
    
    
    float *a_ptr = (float *)malloc(n_neuronas * sizeof(float));
    float *z_ptr = (float *)malloc(n_neuronas * sizeof(float));
    float *grad_bias_ptr = (float *)malloc(n_neuronas * sizeof(float));       
    float * bias_copy = (float *)malloc(n_neuronas * sizeof(float)),       
           * bias_GEMM = (float *)malloc(n_neuronas * sizeof(float)),       
           * w_copy = (float *)malloc(n_pesos * sizeof(float)),       
           * w_GEMM = (float *)malloc(n_pesos * sizeof(float));       
    float *grad_a_ptr = (float *)malloc(n_neuronas * sizeof(float));

    // Inicializar gradientes de sesgos a 0.0
    for(int i=0; i<n_neuronas; i++)
        grad_bias_ptr[i] = 0.0;

    
    //fully_gpu.copiar_w_de_vector_a_ptr(w_cpu_copy);
    //fully_gpu.mostrar_neuronas_ptr();
    //mostrar_ptr_2D(X_gpu, n_datos, tam_x);
    //mostrar_ptr_2D(y_gpu, n_datos, n_clases);
    //cout << "Entr: " << fully_gpu.cross_entropy_ptr(X_gpu, y_gpu, n_datos, a_ptr, z_ptr) << endl;
    //cout << "Acc: " << fully_gpu.accuracy_ptr(X_gpu, y_gpu, n_datos, a_ptr, z_ptr) << endl;

    bias_copy = fully_gpu.get_bias_ptr();
    for(int i=0; i<n_neuronas; i++)
        bias_GEMM[i] = bias_copy[i];        

    w_copy = fully_gpu.get_pesos_ptr();
    for(int i=0; i<n_pesos; i++)
        w_GEMM[i] = w_copy[i];        

    float *grad_x_gpu = (float *)malloc(tam_x * n_datos * sizeof(float));
    //fully_gpu.train_ptr(X_gpu, y_gpu, batch_gpu, n_datos, grad_w_ptr, grad_bias_ptr, grad_x_gpu, a_ptr, z_ptr, grad_a_ptr);
    //fully_gpu.actualizar_parametros_ptr(grad_w_ptr, grad_bias_ptr);
    //fully_gpu.escalar_pesos_ptr(0.5);
    //fully_gpu.mostrar_pesos_ptr();
    printf("Accuracy: %f, Entropía Cruzada: %f\n", fully_gpu.accuracy_ptr(X_gpu, y_gpu, n_datos, a_ptr, z_ptr), fully_gpu.cross_entropy_ptr(X_gpu, y_gpu, n_datos, a_ptr, z_ptr));
    


    /*    
    cout << "a_ptr" << endl;
    for(int i=0; i<n_datos; i++)
    {
        fully_gpu.forwardPropagation_ptr(X_gpu + i*capas[0], a_ptr, z_ptr);
        for(int c=0; c<n_capas; c++)
        {
            for(int j=0; j<capas[c]; j++)
            {
                cout << a_ptr[i_capa[c] + j] << " ";
            }
            cout << endl;
        }
        cout << endl;

    }
    cout << endl;
    
    cout << "z_ptr" << endl;
    for(int i=0; i<n_datos; i++)
    {
        fully_gpu.forwardPropagation_ptr(X_gpu + i*capas[0], a_ptr, z_ptr);
        for(int c=0; c<n_capas; c++)
        {
            for(int j=0; j<capas[c]; j++)
            {
                cout << z_ptr[i_capa[c] + j] << " ";
            }
            cout << endl;
        }
        cout << endl;

    }
    cout << endl;
    */

    // -------------------------------------------------------------
    fully_gpu.set_biasGEMM(bias_GEMM);
    fully_gpu.set_wGEMM(w_GEMM);
    //fully_gpu.forwardPropagationGEMM(X_gpuT, y_gpu);
    
    //fully_gpu.trainGEMM(X_gpuT, y_gpu, batch_gpu, n_datos, grad_w_ptr, grad_bias_ptr, grad_x_gpu, a_ptr, z_ptr, grad_a_ptr);
    //fully_gpu.actualizar_parametros_gpu(grad_w_ptr, grad_bias_ptr);
    //fully_gpu.escalar_pesos_GEMM(0.5);
    
    fully_gpu.evaluar_modelo_GEMM(X_gpuT, y_gpu);
    
    free(capas_ptr); free(i_w_ptr); free(grad_w_ptr); free(X_gpu); free(X_gpuT); free(y_gpu); free(a_ptr); free(z_ptr); free(grad_bias_ptr); free(grad_a_ptr);
    free(grad_x_gpu);

    free(batch_gpu);      

    return 0;
}
