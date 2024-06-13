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
		int i = blockIdx.x * blockDim.x + threadIdx.x; // tid = threadIdx.x,


    float maximo = -9999999, sum = 0.0;

    if(i < M)
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
		int i = blockIdx.x * blockDim.x + threadIdx.x; // tid = threadIdx.x

    // Cada hebra se encarga de una fila
    if(i < rows)
        for (int j = 0; j < cols; j++)
            Y[j * rows + i] = X[i * cols + j];
}


__global__ void kernel_back_softmax(int M, int K, float *grad_a, float *Z, float *Y, int n_clases)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // tid = threadIdx.x

    if(i < M)
    {
        // Cada hebra se encarga de una fila
        for(int j=0; j<K; ++j)
						grad_a[i*K + j] = Z[i*K + j] - Y[i*n_clases + j];
    }
}

__global__ void kernel_grad_bias(int M, int K, float *grad_a, float *grad_b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // tid = threadIdx.x

    if(i < M)
    {
        // Inicializar valor del gradiente de bias a 0.0
        grad_b[i] = 0.0f;

        // Cada hebra se encarga de una fila
        for(int j=0; j<K; ++j)
            grad_b[i] += grad_a[i*K + j];
    }
}


__global__ void kernel_actualizar_bias(int N, float *bias, float *grad_bias, float lr, int mini_batch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // tid = threadIdx.x

    if(i < N)
        bias[i] -= lr * (grad_bias[i] / mini_batch);
}

__global__ void kernel_actualizar_pesos(int M, int K, float *w, float *grad_w, float lr, int mini_batch)
{
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;
        // tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);

    if(iy < M && ix <K)
    {
        // Cada hebra se encarga de un peso
        w[iy*K + ix] -= lr * (grad_w[iy*K + ix] / mini_batch);
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
__global__ void multiplicarMatricesGPU_fully(int M, int N, int K, const float *A, const float *B, float *C)
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
		int i = blockIdx.x * blockDim.x + threadIdx.x, predic_acc = 0; // tid = threadIdx.x

    float predic_entr = 0.0, epsilon = 0.000000001, maximo = -9999999, entr = 0.0, acc = 0.0;

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

__global__ void reduceMax(float * X, float * Y, const int N)
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
}

__global__ void reduceMin(float * X, float * Y, const int N)
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
}


__global__ void min_max(float *maximos, float *minimos, const int N)
{
	extern __shared__ float sdata[];

	// int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Obtener el máximo total en todos los bloques
    if(i == 0)
    {
        for(int j=1; j<gridDim.x; j++)
        {
            maximos[0] = max(maximos[0], maximos[j]);
            minimos[0] = min(minimos[0], minimos[j]);
        }
    }
}

__global__ void kernel_escalar_pesos(float * X, const int N, float *maxi, float *mini, float valor_clip)
{
	// int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    float maximo = abs(maxi[0]), minimo = abs(mini[0]),
          factor_de_escala = valor_clip / max(maximo, minimo);

    // Escalar los pesos
    if(i < N)
        X[i] = max(min(X[i], valor_clip), -valor_clip);
}


__global__ void kernel_transpuesta_pesos(int *capas_wT, int *capas, float *w, float *wT, int capa, float *bias)
{
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;
        // tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);

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
FullyConnected::FullyConnected(int *capas, int n_capas, float lr, int mini_batch)
{

    liberar_memoria = true;
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
		n_neuronas_GEMM = capas[0] * mini_batch;
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
    for(int i=0; i<n_capas-1; i++)
    {
        for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
        {
            for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
            {
                this->wT_ptr[i_wT[i] + k*capas_wT[i] + j] = this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k];
            }

            // Añadir bias
            this->wT_ptr[i_wT[i] + k*capas_wT[i] + capas[i]] = this->bias_ptr[i_capa[i+1] + k];
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

    //mostrar_pesos_Traspuestos_ptr();
    */

    // Tamaño de bloques ----------------
    block_size = 8;

    // Bloque 1D
    this->block_1D.x = block_size;
    this->block_1D.y = 1;

    // Bloque 2D
    this->block_2D.x = block_size;
    this->block_2D.y = block_size;

    // Tamaño de grids ----------------
    // Grid 1D
    this->grid_1D.x = (mini_batch  + block_2D.x -1) / block_2D.x;
    this->grid_1D.y = 1;

    // Grid 2D
    this->grid_2D.x = (mini_batch  + block_2D.x -1) / block_2D.x;
    this->grid_2D.y = (capas[1] + block_2D.y -1) / block_2D.y;

    // Memoria compartida a nivel de bloque ----------------
    // Memoria compartida en bloques 1D
    smem_1D = (2*block_1D.x) *sizeof(float);

    // Memoria compartida en bloques 2D
    smem_2D = (2*block_2D.x * block_2D.y) *sizeof(float);

    // Reserva de memoria en device ----------------
    // Capa SoftMax
    cudaMalloc((void **) &d_softmax, capas[n_capas-1] * mini_batch * sizeof(float));

    // Sumas para Accuracy y Entropía Cruzada
    cudaMalloc((void **) &d_sum_acc_entr, 2*mini_batch * sizeof(float));

    // Capas Transpuestas
    cudaMalloc((void **) &d_capas_wT, n_capas * sizeof(int));

    // Capas
    cudaMalloc((void **) &d_capas, n_capas * sizeof(int));

    // Valor máximo
    cudaMalloc((void **) &d_max, sizeof(float));

    // Valor mínimo
    cudaMalloc((void **) &d_min, sizeof(float));

    // Matriz de  pesos transpuesta
    cudaMalloc((void **) &d_wT, n_pesos * n_neuronas * sizeof(float));

    // Matriz de pesos
    cudaMalloc((void **) &d_w, n_pesos * sizeof(float));

    // Gradiente de pesos
    cudaMalloc((void **) &d_grad_w, n_pesos * sizeof(float));

    // Valor de cada neurona tras aplicar su función de activación asociada
    cudaMalloc((void **) &d_z, n_neuronas_GEMM * sizeof(float));

    // Valor de cada neurona antes de aplicar su función de activación asociada
    cudaMalloc((void **) &d_a, n_neuronas_GEMM * sizeof(float));

    // Matriz transpuesta de valores de cada neurona antes aplicar su función de activación asociada
    cudaMalloc((void **) &d_aT, n_neuronas_GEMM * sizeof(float));

    // Gradiente de cada bias o sesgo
    cudaMalloc((void **) &d_grad_b, n_neuronas_GEMM * sizeof(float));

    // Bias o sesgos
    cudaMalloc((void **) &d_b, n_neuronas * sizeof(float));

    // Etiquetas
    cudaMalloc((void **) &d_y, capas[n_capas-1] * mini_batch * sizeof(float));

    // Copiar información de host a device ----------------
    cudaMemcpy(d_capas_wT, capas_wT, n_capas * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capas, capas, n_capas * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wT, wT_ptr,  n_pesos * n_neuronas * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w_ptr,  n_pesos * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, bias_ptr,  n_neuronas * sizeof(float), cudaMemcpyHostToDevice);
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

    for(int i=0; i<this->capas[capa]; ++i)
        for(int j=0; j<this->capas[capa+1]; ++j)
            this->w_ptr[i_w_ptr[capa] + i*capas[capa+1] + j] = distribution(gen);
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
void FullyConnected::forwardPropagationGEMM()
{
    // Capas intermedias
    for(int c=0; c<n_capas-1; c++)
    {
        // Tamaño de grid para propagación hacia delante
        this->grid_2D.x = (mini_batch  + block_2D.x -1) / block_2D.x;
        this->grid_2D.y = (capas[c+1] + block_2D.y -1) / block_2D.y;

        // Aplicar ReLU en capas intermedias, pero en SoftMax no
        forward_capas_intermedias<<<grid_2D, block_2D, smem_2D>>>(capas[c+1], mini_batch, capas[c]+1, d_wT + i_wT[c], d_z + i_capasGEMM[c], d_z + i_capasGEMM[c+1], d_a + i_capasGEMM[c+1], (c!=n_capas-2));
    }

    // Transpuesta para tener 1 fila por dato de minibatch
    this->grid_1D.x = (capas[n_capas-1]  + block_1D.x -1) / block_1D.x;
    matrizTranspuesta_GPU<<<grid_1D, block_1D>>>(d_z + i_capasGEMM[n_capas-1], d_softmax, capas[n_capas-1], mini_batch);

    this->grid_1D.x = (mini_batch  + block_1D.x -1) / block_1D.x;
    kernel_sofmax<<<grid_1D, block_1D>>>(mini_batch, capas[n_capas-1], d_softmax, d_a + i_capasGEMM[n_capas-1]);

    // ---------------------------------------------------------------
    // ---------------------------------------------------------------
    /*
    float * capa = capasGEMM[0];

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
    float *capa = capasGEMM[0];
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
    float max, sum = 0.0;

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
                z_ptr[i_capa[i+1] + k] = exp(a_ptr[i_capa[i+1] + k]) / sum;

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
void FullyConnected::evaluar_modelo_GEMM()
{
    this->grid_1D.x = (mini_batch  + block_1D.x -1) / block_1D.x;

    forwardPropagationGEMM();
    kernel_evaluacion_modelo<<<grid_1D, this->block_1D>>>(mini_batch, this->capas[n_capas-1], d_softmax, d_y, d_sum_acc_entr);
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
void FullyConnected::trainGEMM(float *grad_x)
{
    int n_clases = capas[n_capas-1];

    grid_1D.x = (mini_batch  + block_1D.x -1) / block_1D.x;

    // Propagación hacia delante de cada dato perteneciente al minibatch
    forwardPropagationGEMM();

    // Cálculo del gradiente respecto a la entrada de la capa SoftMax
    kernel_back_softmax<<<grid_1D, block_1D>>>(mini_batch, n_clases, d_z + i_capasGEMM[n_capas-1], d_softmax, d_y, n_clases);

    // Capas intermedias
    // Transpuesta para tener 1 columna por dato de minibatch
    matrizTranspuesta_GPU<<<grid_1D, block_1D>>>(d_z + i_capasGEMM[n_capas-1], d_a + i_capasGEMM[n_capas-1], mini_batch, capas[n_capas-1]);

    // Capas intermedias
    for(int c=n_capas-1; c>0; c--)
    {
        // Tamaño de grid
        this->grid_2D.x = (mini_batch  + block_2D.x -1) / block_2D.x;
        this->grid_2D.y = (capas[c-1] + block_2D.y -1) / block_2D.y;

        // Aplicar ReLU en capas intermedias, pero en SoftMax no
        backprop_capas_intermedias<<<grid_2D, block_2D, smem_2D>>>(capas[c-1], mini_batch, capas[c], d_w + i_w_ptr[c-1], d_a + i_capasGEMM[c], d_a + i_capasGEMM[c-1], (c>1));

        // Tamaño de grid
        grid_1D.x = (capas[c]  + block_1D.x -1) / block_1D.x;
        kernel_grad_bias<<<grid_1D, block_1D>>>(capas[c], mini_batch, d_a + i_capasGEMM[c], d_grad_b + i_capasGEMM[c]);

        // Tamaño de grid
        grid_1D.x = (capas[c]  + block_1D.x -1) / block_1D.x;

        matrizTranspuesta_GPU<<<grid_1D, block_1D>>>(d_a + i_capasGEMM[c], d_aT + i_capasGEMM[c], capas[c], mini_batch);

        // Tamaño de grid
        grid_2D.x = (capas[c]  + block_2D.x -1) / block_2D.x;
        grid_2D.y = (capas[c-1] + block_2D.y -1) / block_2D.y;

        multiplicarMatricesGPU_fully<<<grid_2D, block_2D, smem_2D>>>(capas[c-1], capas[c], mini_batch, d_z + i_capasGEMM[c-1], d_aT + i_capasGEMM[c], d_grad_w + i_w_ptr[c-1]);
    }

    int c = 0;

    // Tamaño de grid
    grid_1D.x = (capas[c]  + block_size -1) / block_size;

    matrizTranspuesta_GPU<<<grid_1D, block_1D>>>(d_a + i_capasGEMM[c], d_aT + i_capasGEMM[c], capas[c], mini_batch);

    cudaMemcpy(grad_x, d_aT + i_capasGEMM[c], capas[c] * mini_batch * sizeof(float), cudaMemcpyDeviceToHost);



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

    cout << " ---------------------------- A ---------------------------- " << endl;
    //float *capa = capasGEMM[0];
    for(int c=0; c<n_capas; c++)
    {
        capa = capasGEMM[c];
        cudaMemcpy(capa, d_a + i_capasGEMM[c], capas[c] * mini_batch * sizeof(float), cudaMemcpyDeviceToHost);
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


    // Mostrar gradiente de pesos
    float * h_grad_w = (float *)malloc(n_pesos * n_neuronas * sizeof(float));
    cudaMemcpy(h_grad_w, d_grad_w, n_pesos * sizeof(float), cudaMemcpyDeviceToHost);
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
    /*
    float * h_grad_w = (float *)malloc(n_pesos * sizeof(float));
    cudaMemcpy(h_grad_w, d_grad_w, n_pesos * sizeof(float), cudaMemcpyDeviceToHost);
    cout << "Gradiente de Pesos GEMM" << endl;
    for(int j=0; j<10; j++)   // Por cada neurona de la capa actual
    {
        for(int k=0; k<10; k++)     // Por cada neurona de la siguiente capa
            cout << h_grad_w[i_w_ptr[1] + j*capas[1+1] + k] << " ";
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
    @brief          Escalar los pesos para evitar que los gradientes "exploten"
    @clip_value     Valor a emplear para realizar el "clip" o escalado
    @return         Se actualizan los valores de w (pesos de la red)
*/
void FullyConnected::escalar_pesos_GEMM(float clip_value)
{
    /*
    //float *bias_copy = (float *)malloc(n_neuronas * sizeof(float));       // Cada neurona tiene asociado un bias

    float *pesos_copy = (float *)malloc(n_pesos * n_neuronas * sizeof(float)),
          *max = (float *)malloc(sizeof(float)),
          *min = (float *)malloc(sizeof(float));

    cudaMemcpy(pesos_copy, d_w, n_pesos * n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);

    // Mostrar pesos
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
    /*
    cudaMemcpy(pesos_copy, d_wT, n_pesos * n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);


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


    float *pesos_copy = (float *)malloc(n_pesos * sizeof(float)),
          *max = (float *)malloc(sizeof(float)),
          *min = (float *)malloc(sizeof(float));

    this->grid_1D.x = (n_pesos + block_1D.x -1) / block_1D.x;

    reduceMax<<<grid_1D, block_1D, smem_1D>>>(d_w, d_max, n_pesos);
    reduceMin<<<grid_1D, block_1D, smem_1D>>>(d_w, d_min, n_pesos);
    min_max<<<grid_1D, block_1D>>>(d_max, d_min, grid_1D.x);
    kernel_escalar_pesos<<<grid_1D, block_1D>>>(d_w, n_pesos, d_max, d_min, clip_value);

    // Actualizar matriz de pesos transpuesta
    for(int i=0; i<n_capas-1; i++)
    {
        // Tamaño de grid
        grid_2D.x = (capas[i+1]  + block_1D.x -1) / block_1D.x;
        grid_2D.y = (capas[i]  + block_1D.x -1) / block_1D.x;

        kernel_transpuesta_pesos<<<grid_2D, block_2D>>>(d_capas_wT, d_capas, d_w + i_w_ptr[i], d_wT + i_wT[i], i, d_b + i_capa[i+1]);
    }

    // Pasar cambios a CPU, TEMPORAL --------------------------------------------------------------
    cudaMemcpy(w_ptr, d_w, n_pesos * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bias_ptr, d_b, n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);

    /*
    cudaMemcpy(max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(min, d_min, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Máximo GEMM: " << max[0] << endl;
    cout << "Mínimo GEMM: " << min[0] << endl;


    cudaMemcpy(pesos_copy, d_w, n_pesos * sizeof(float), cudaMemcpyDeviceToHost);

    // Mostrar pesos
    cout << "Pesos GEMM " << endl;
    for(int j=0; j<10; j++)   // Por cada neurona de la capa actual
    {
        for(int k=0; k<10; k++)     // Por cada neurona de la siguiente capa
            cout << pesos_copy[i_w_ptr[1] + j*capas[1+1] + k] << " ";
        cout << endl;
    }
    cout << endl;

    /*
    // Mostrar pesos
    cudaMemcpy(pesos_copy, d_w, n_pesos * n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);
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


    cudaMemcpy(pesos_copy, d_wT, n_pesos * n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);


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
    /*
    float scaling_factor = clip_value / std::max(std::abs(max), std::abs(min));
    for(int i=0; i<n_capas-1; i++)
        for(int j=0; j<capas[i]; j++)
            for(int k=0; k<capas[i+1]; k++)
                this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k] = std::max(std::min(this->w_ptr[i_w_ptr[i] + j*capas[i+1] + k], clip_value), -clip_value);
    */
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
    @brief          Actualizar los pesos y sesgos de la red
    @grad_pesos     Gradiente de cada peso de la red
    @grad_b         Gradiente de cada sesgo de la red
    @return         Se actualizar los valores de w y bias (pesos y sesgos de la red)
*/
void FullyConnected::actualizar_parametros_gpu()
{

    float *bias_copy = (float *)malloc(n_neuronas * sizeof(float));       // Cada neurona tiene asociado un bias
    float *grad_pesos_copy = (float *)malloc(n_pesos * sizeof(float));
    float *pesos_copy = (float *)malloc(n_pesos * n_neuronas * sizeof(float));

    /*
    cudaMemcpy(pesos_copy, d_w, n_pesos * sizeof(float), cudaMemcpyDeviceToHost);

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
        grid_1D.x = (capas[i]  + block_1D.x -1) / block_1D.x;
        kernel_actualizar_bias<<<grid_1D, block_1D>>>(capas[i], d_b + i_capa[i], d_grad_b + i_capasGEMM[i], this->lr, mini_batch);
    }

    // Actualizar pesos
    for(int i=0; i<n_capas-1; i++)
    {
        // Tamaño de grid
        grid_2D.x = (capas[i+1]  + block_1D.x -1) / block_1D.x;
        grid_2D.y = (capas[i]  + block_1D.x -1) / block_1D.x;
        kernel_actualizar_pesos<<<grid_2D, block_2D>>>(capas[i], capas[i+1], d_w + i_w_ptr[i], d_grad_w + i_w_ptr[i], this->lr, mini_batch);
        kernel_transpuesta_pesos<<<grid_2D, block_2D>>>(d_capas_wT, d_capas, d_w + i_w_ptr[i], d_wT + i_wT[i], i, d_b + i_capa[i+1]);
    }

    /*
    cudaMemcpy(pesos_copy, d_w, n_pesos * sizeof(float), cudaMemcpyDeviceToHost);

    // Mostrar pesos
    cout << "Pesos GEMM " << endl;
    for(int j=0; j<10; j++)   // Por cada neurona de la capa actual
    {
        for(int k=0; k<10; k++)     // Por cada neurona de la siguiente capa
            cout << pesos_copy[i_w_ptr[1] + j*capas[1+1] + k] << " ";
        cout << endl;
    }
    cout << endl;
    */

    /*
    cudaMemcpy(pesos_copy, d_w, n_pesos * sizeof(float), cudaMemcpyDeviceToHost);

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

    int k1;
    cin >> k1;

    /*
    cudaMemcpy(pesos_copy, d_wT, n_pesos * n_neuronas * sizeof(float), cudaMemcpyDeviceToHost);

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
    /*
    for(int i=0; i<n_capas-1; i++)
        for(int j=0; j<capas[i]; j++)   // Por cada neurona de la capa actual
            for(int k=0; k<capas[i+1]; k++)     // Por cada neurona de la siguiente capa
                w_ptr[i_w_ptr[i] + j*capas[i+1] + k] -= this->lr * grad_pesos[i_w_ptr[i] + j*capas[i+1] + k];
    */
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

    cout << "bias antes " << endl;
    for(int c=0; c<n_capas; c++)
    {
        for(int j=0; j<capas[c]; j++)
            cout << bias_ptr[i_capa[c] + j] << " ";
        cout << endl;
    }
    cout << endl;
    */
    /*
    for(int i=0; i<n_capas; i++)
        for(int j=0; j<capas[i]; j++)
            bias_ptr[i_capa[i] + j] -= this->lr * grad_b[i_capa[i] + j];
    */
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

void FullyConnected::set_train(float *x, float *y, int mini_batch)
{
    this->mini_batch = mini_batch;

    // Pasar valores de primera capa a GPU
    cudaMemcpy(d_z, x,  mini_batch * (capas[0]+1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, x,  mini_batch * (capas[0]+1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, capas[n_capas-1] * mini_batch * sizeof(float), cudaMemcpyHostToDevice);
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

/*
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
    int tam_x = 3, n_datos = 2, n_clases = 5;
    //int tam_x = 3, n_datos = 2, n_clases = 2;
    vector<int> capas = {tam_x, 5, n_clases};
    //vector<int> capas = {tam_x, 5, n_clases};
    vector<vector<float>> a_cpu, z_cpu, grad_a_cpu, y_cpu = {{0.0, 1.0}, {0.0, 1.0}};
    vector<vector<vector<float>>> grad_w_cpu;
    vector<float> x_cpu;
    vector<vector<float>> X_cpu, grad_b_cpu, grad_x_cpu;
    vector<int> batch_cpu(n_datos);

    // GPU --------------
    cout << " ---------- GPU ---------- " << endl;
    int n_capas = 3;
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

    bias_copy = fully_gpu.get_bias_ptr();
    for(int i=0; i<n_neuronas; i++)
        bias_GEMM[i] = bias_copy[i];

    w_copy = fully_gpu.get_pesos_ptr();
    for(int i=0; i<n_pesos; i++)
        w_GEMM[i] = w_copy[i];

    float *grad_x_gpu = (float *)malloc(tam_x * n_datos * sizeof(float));
    fully_gpu.train_ptr(X_gpu, y_gpu, batch_gpu, n_datos, grad_w_ptr, grad_bias_ptr, grad_x_gpu, a_ptr, z_ptr, grad_a_ptr);

    //fully_gpu.actualizar_parametros_ptr(grad_w_ptr, grad_bias_ptr);
    //fully_gpu.escalar_pesos_ptr(0.5);
    //fully_gpu.mostrar_pesos_ptr();
    printf("Accuracy: %f, Entropía Cruzada: %f\n", fully_gpu.accuracy_ptr(X_gpu, y_gpu, n_datos, a_ptr, z_ptr), fully_gpu.cross_entropy_ptr(X_gpu, y_gpu, n_datos, a_ptr, z_ptr));


    cout << "grad_x_gpu: " << endl;
    for(int i=0; i<n_datos; i++)
    {
        for(int j=0; j<tam_x; j++)
            cout << grad_x_gpu[i*tam_x + j] << " ";
        cout << endl;
    }
    cout << endl;


    for(int i=0; i<n_datos; i++)
        for(int j=0; j<tam_x; j++)
            grad_x_gpu[i*tam_x + j] = 0.0;


    // -------------------------------------------------------------
    fully_gpu.set_biasGEMM(bias_GEMM);
		fully_gpu.set_wGEMM(w_GEMM);
		fully_gpu.set_train(X_gpuT, y_gpu, n_datos);


    //fully_gpu.forwardPropagationGEMM();
    fully_gpu.trainGEMM(grad_x_gpu);


		//fully_gpu.actualizar_parametros_gpu();
    //fully_gpu.escalar_pesos_GEMM(0.5);
    fully_gpu.evaluar_modelo_GEMM();


    cout << "grad_x_gpu: " << endl;
    for(int i=0; i<n_datos; i++)
    {
        for(int j=0; j<tam_x; j++)
            cout << grad_x_gpu[i*tam_x + j] << " ";
        cout << endl;
    }
    cout << endl;



    free(capas_ptr); free(i_w_ptr); free(grad_w_ptr); free(X_gpu); free(X_gpuT); free(y_gpu); free(a_ptr); free(z_ptr); free(grad_bias_ptr); free(grad_a_ptr);
    free(grad_x_gpu);
    free(batch_gpu);


    return 0;
}
*/
