#include "convolutional.h"

using namespace std::chrono;



/*
    @brief          Multiplicación matricial
    @M              Filas de la matriz A
    @N              Columnas de la matriz B
    @K              Columnas de la matriz A
    @A              Primera matriz a multiplicar
    @B              Segunda matriz a multiplicar
    @C              Matriz resultado
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


/*
    @brief          Multiplicación matricial para calcular el gradiente con respecto a los pesos
    @M              Filas de la matriz A
    @N              Columnas de la matriz B
    @K              Columnas de la matriz A
    @A              Primera matriz a multiplicar
    @B              Segunda matriz a multiplicar
    @C              Matriz resultado
*/
__global__ void multiplicarMatricesGPU_calcular_grad_w(int M, int N, int K, const float *A, const float *B, float *C)
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

		__syncthreads();
    if(iy < M && ix < N)
        C[iy*N + ix] += sum;
}

/*
    @brief          Propagación hacia delante mediane GEMM
    @M              Filas de la matriz A
    @N              Columnas de la matriz B
    @K              Columnas de la matriz A
    @A              Primera matriz a multiplicar
    @B              Segunda matriz a multiplicar
    @C              Matriz resultado sin aplicar función de activación
    @bias           Sesgos
    @output         Resultado (con función de activación aplicada)
*/
__global__ void forward_propagation_GEMM(int M, int N, int K, const float *A, const float *B, float *C, const float *bias, float *output)
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

    float sum = 0.0f, result = 0.0f;

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
        result = sum + bias[iy];
        C[iy*N + ix] = result;

        // ReLU
        if(result > 0)
            output[iy*N + ix] = result;
        else
            output[iy*N + ix] = 0;
    }
    
}

/*
    @brief          Matriz transpuesta
    @X              Matriz de entrada
    @Y              Matriz transpuesta resultado
    @rows           Número de filsa
    @cols           Número de columnas
*/
__global__ void matrizTranspuesta_conv(float* X, float *Y, int rows, int cols)
{
    int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;

    // Cada hebra se encarga de una fila
    if(iy < rows && ix < cols)
        Y[iy * rows + ix] = X[ix * cols + iy];
}


/*
    @brief  Cálculo del máximo mediante una reducción
    @X      Valores iniciales
    @Y      Máximos
    @N      Número de valores
*/
__global__ void reduceMax_conv(float * X, float * Y, const int N)
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

/*
    @brief  Cálculo del mínimo mediante una reducción
    @X      Valores iniciales
    @Y      Máximos
    @N      Número de valores
*/
__global__ void reduceMin_conv(float * X, float * Y, const int N)
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

/*
    @brief      Cálculo del máximo y mínimo totales 
    @maximos    Valores máximos
    @minimos    Valores mínimos
    @N      Número de valores
*/
__global__ void min_max_conv(float *maximos, float *minimos, const int N)
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

/*
    @brief          Escalar los pesos
    @X              Pesos
    @N              Número de pesos
    @maxi           Valor máximo
    @mini           Valor mínimo
    @valor_clip     Valor del clip
*/
__global__ void kernel_escalar_pesos_conv(float * X, const int N, float *maxi, float *mini, float valor_clip)
{
	// int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    float maximo = abs(maxi[0]), minimo = abs(mini[0]),
          factor_de_escala = valor_clip / max(maximo, minimo);

    // Escalar los pesos
    if(i < N)
        X[i] = max(min(X[i] * factor_de_escala, valor_clip), -valor_clip);
}

/*
    @brief      Desenrrollado
    @C          Número de canales de profundiad
    @H          Número de filas
    @W          Número de columnas
    @K          Tamaño del kernel de pesos
    @X          Matriz de entrada
    @X_unroll   Matriz desenrrollada
*/
__global__ void unrollGPU(int C, int H, int W, int K, const float *X, float *X_unroll)
{
		int tid = blockIdx.x * blockDim.x + threadIdx.x,
				H_out = H-K+1,
				W_out = W-K+1,
				W_unroll = H_out * W_out;
		int c, h_unroll, w_unroll, h_out, w_out, w_base;

		if(tid < C * W_unroll)
		{
			c = tid / W_unroll;
			w_unroll = tid % W_unroll;
			h_out = w_unroll / W_out;
			w_out = w_unroll % W_out;
			w_base = c*K*K;

			for(int p=0; p<K; p++)
				for(int q=0; q<K; q++)
				{
					h_unroll = w_base + p*K + q;
					X_unroll[h_unroll*W_unroll + w_unroll] = X[c*H*W + (h_out+p)*W + (w_out+q)];
				}
		}

}


/*
    @brief      Derivada de la función de activación
    @C          Número de canales de profundiad
    @H          Número de filas
    @W          Número de columnas
    @output     Salida de la capa
    @a          Salida de la capa antes de aplicar la función de activación
*/
__global__ void deriv_activ_func_output(int C, int H, int W, float *output, const float *a)
{
		int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;
		int pos;

		// Realizar derivada Y_out/Y_in
		if(iy < H && ix < W)
		{
				for(int i=0; i<C; i++)
				{
						pos = i*H*W + iy*W + ix;

						// Derivada de ReLU, (output[pos] = output[pos] * deriv_activationFunction(a[pos]);)
						if(a[pos] <= 0)
							output[pos] = 0.0;
				}
		}
}

/*
    @brief      Centra los valores de una matriz
    @C          Número de canales de profundiad
    @H          Número de filas
    @W          Número de columnas
    @H_pad      Número de filas (con padding)
    @W_pad      Número de columnas (con padding)
    @output     Salida de la capa
    @output_pad Salida de la capa (con padding)
*/
__global__ void centrar(int C, int H, int W, int H_pad, int W_pad, const float *output, float *output_pad)
{
		int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;

		// Realizar derivada Y_out/Y_in
		if(iy < H_pad && ix < W_pad)
            for(int i=0; i<C; i++)
                output_pad[i*H_pad*W_pad + iy*W_pad + ix] = output[i*H*W + iy*W + ix];
}

/*
    @brief      Aplica padding
    @C          Número de canales de profundiad
    @H          Número de filas
    @W          Número de columnas
    @X          Matriz de entrada
    @Y          Matriz de salida
    @pad        Padding
*/
__global__ void aplicar_padding_gpu(int C, int H, int W, const float *X, float *Y, int pad)
{
		int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;

		// Inicializar salida a 0.0
		if(iy < H && ix < W)
			for(int c=0; c<C; c++)
				Y[c*H*W + iy*W + ix] = 0.0;

		__syncthreads();

		// Traslado horizontal y vertical en "pad unidades"
		if(iy < H-2*pad && ix < W-2*pad)
				for(int c=0; c<C; c++)
						Y[c*H*W + (iy + pad)*W + (ix + pad)] = X[c*H*W + iy*W + ix];
}

/*
    @brief      Desenrrolla una matriz en 3 dimensiones
    @C          Número de canales de profundiad
    @H          Número de filas
    @W          Número de columnas
    @X          Matriz de entrada
    @X_unroll   Matriz desenrrollada
*/
__global__ void unroll_3dim_gpu(int C, int H, int W, int K, float *X, float *X_unroll)
{
		int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;
		int i_out, i_in;

		// Calcular el tamaño de salida
		int H_out = H - K + 1;
		int W_out = W - K + 1;

		if(iy < H_out && ix < W_out)
		{
				i_out = iy * W_out + ix;
				i_in = 0;

				for (int d = 0; d < C; ++d)
						for (int ki = 0; ki < K; ++ki)
								for (int kj = 0; kj < K; ++kj)
										X_unroll[i_out * (K * K * C) + i_in++] = X[d * (H * W) + (iy + ki) * W + (ix + kj)];
		}
}

/*
    @brief      Desenrrolla una matriz en 1 dimensión
    @C          Número de canales de profundiad
    @H          Número de filas
    @W          Número de columnas
    @X          Matriz de entrada
    @X_unroll   Matriz desenrrollada
*/
__global__ void unroll_1dim_gpu(int C, int H, int W, int K, float *X, float *X_unroll)
{
		int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;

		// Calcular el tamaño de salida
		int H_out = H - K+1, W_out = W -K +1;

		if(iy < H_out && ix < W_out)
				for(int c=0; c<C; c++)
            for(int ky=0; ky < K; ky++)		// Guardar K*K elementos de "convolución"
                for(int kx=0; kx<K; kx++)
                    X_unroll[(((c * H_out + iy) * W_out + ix) * K + ky) * K + kx] = X[c*H*W + (iy+ky)*W + (ix+kx)];
}

/*
    @brief      Desenrrolla la matriz de pesos
    @n_kernels  Número de kernels de pesos
    @K          Tamaño de los kernels de pesos
    @X          Matriz de entrada
    @Y          Matriz desenrrollada
*/
__global__ void unroll_matriz_pesos(int C, int n_kernels, int K, float *X, float *Y)
{
		int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;

		if(iy < C && ix < n_kernels)
				for(int kx = K-1; kx >= 0; kx--)
						for(int ky = K-1; ky >=0; ky--)
								Y[iy*n_kernels*K*K + ix*K*K + kx*K + ky] = X[ix*C*K*K + iy*K*K + kx*K + ky];
}

/*
    @brief      Acumula el gradiente con respecto a los pesos
    @n_kernels  Número de kernels de pesos
    @C          Número de canales de profundidad
    @K          Tamaño de los kernels de pesos
    @grad_w     Gradientes con respecto a los pesos
    @grad_w_it  Gradientes con respecto a los pesos en una iteración determinada
*/
__global__ void acumular_grad_w(int n_kernels, int C, int K, float *grad_w, float *grad_w_it)
{
		int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;

		if(iy < n_kernels && ix < C)
				for(int kx=0; kx<K; kx++)
						for(int ky=0; ky<K; ky++)
								grad_w[iy*C*K*K + ix*K*K + kx*K + ky] += grad_w_it[iy*C*K*K + ix*K*K + kx*K + ky];
}

/*
    @brief  Calcula una suma mediante reducción
    @X      Valores de entrada
    @Y      Valores de salida
    @N      Número de valores
*/
__global__ void reduce_suma(float * X, float * Y, const int N)
{
	extern __shared__ float sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;


    // Cargar los datos en memoria compartida
	sdata[tid] = ((i < N) ? X[i] : 0.0f);
	__syncthreads();


  // Obtener la suma local de cada bloque
	for(int s = blockDim.x/2; s > 0; s >>= 1)
  {
	  if (tid < s)
	        sdata[tid] += + sdata[tid+s];
	  __syncthreads();
	}

	if (tid == 0)
        Y[blockIdx.x] = sdata[0];

}

/*
    @brief      Acumula el gradiente con respecto a los sesgos
    @grad_bias  Gradientes con respecto a los sesgos
    @i_kernel   Número de kernel
*/
__global__ void acumular_grad_bias(float *X, float *grad_bias, int i_kernel)
{
	// int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Obtener la suma total en todos los bloques
    if(i == 0)
        for(int j=0; j<gridDim.x; j++)
            grad_bias[i_kernel] += X[j];

}

/*
    @brief      Actualiza los parámetros
    @n_kernels  Número de kernels de pesos
    @C          Número de canales de profundidad
    @K          Tamaño de los kernels de pesos
    @w          Pesos
    @grad_W     Gradiente respecto a los pesos
    @bias       Sesgos
    @grad_bias  Gradiente respecto a los sesgos
    @lr         Learning rate
    @mini_batch Tamaño del minibatch
*/
__global__ void kernel_actualizar_parametros(int n_kernels, int C, int K, float *w, float *grad_w, float *bias, float *grad_bias, float lr, int mini_batch)
{
  	int iy = threadIdx.y + blockIdx.y * blockDim.y, ix = threadIdx.x + blockIdx.x * blockDim.x;
        // tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);

    if(iy < n_kernels && ix <C)
    {
            // Actualizar pesos
            for(int k=0; k<K; ++k)
                    for(int p=0; p<K; ++p)
                            w[iy*C*K*K + ix*K*K + k*K + p] -= lr * (grad_w[iy*C*K*K + ix*K*K + k*K + p] / mini_batch);

            // Actualizar bias
            if(ix == 0)
                bias[iy] -= lr * (grad_bias[iy] / mini_batch);
    }
}


/*
    CONSTRUCTOR de la clase Convolutional
    --------------------------------------

    @n_kernels      Número de kernels a aplicar
    @kernel_fils    Número de filas por kernel
    @kernel_cols    Número de columnas por kernel
    @C              Número de canales de profundiad de la entrada
    @lr             Learning Rate o Tasa de Aprendizaje
*/
Convolutional::Convolutional(int n_kernels, int kernel_fils, int kernel_cols, int C, int H, int W, float lr, int pad)
{
    // Kernels de pesos
    this->n_kernels = n_kernels;
    this->kernel_fils = kernel_fils;
    this->kernel_cols = kernel_cols;

    // Imagen de entrada
    this->C = C;
    this->H = H;
    this->W = W;

    // Imagen de salida
    this->H_out = H - kernel_fils + 1 + 2*pad;
    this->W_out = W - kernel_cols + 1 + 2*pad;

    // Dimensiones de los volúmenes "desenrrollados" ------------------
    // Dimensiones de la entrada 'desenrrollada'
    this->fils_input_unroll = kernel_fils*kernel_cols*C;
    this->cols_input_unroll = H_out * W_out;

    // Dimensiones de los pesos como matriz 2D
    this->fils_w = this->n_kernels;
    this->cols_w = kernel_fils*kernel_cols*C;

    // Tamaños de los volúmenes "desenrrollados"
    this->bytes_input_unroll = fils_input_unroll * cols_input_unroll *sizeof(float);    // Espacio para la entrada 'desenrrollada'
    this->bytes_output = cols_input_unroll * fils_w *sizeof(float);              // Espacio para la salida
    this->bytes_w = fils_w * cols_w * sizeof(float);
		this->bytes_bias = this->n_kernels * sizeof(float);

    // Learning Rate
    this->lr = lr;

    // Pesos
    float *w_ptr = (float *)malloc(fils_w * cols_w * sizeof(float));

    // Inicializar pesos mediante Inicialización He
    this->generar_pesos_ptr(w_ptr);

    // Bias
    float *bias_ptr = (float *)malloc(this->n_kernels * sizeof(float));

    // Un bias por filtro, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
    for(int i=0; i<n_kernels; i++)
        bias_ptr[i] = 0.0;


    // GPU -------------------------
    // Tamaño de bloque
    this->block.x = BLOCK_SIZE;
    this->block.y = BLOCK_SIZE;

		block_1D.x = BLOCK_SIZE;
		block_1D.y = 1;

    // Memoria compartida a nivel de bloque
    this->smem = (2*block.x * block.y) *sizeof(float);
		this->smem_1D = (2*block_1D.x) *sizeof(float);


    // Punteros device
    checkCudaErrors(cudaMalloc((void **) &d_input_unroll, bytes_input_unroll));
    checkCudaErrors(cudaMalloc((void **) &d_w, bytes_w));
    checkCudaErrors(cudaMalloc((void **) &d_bias, bytes_bias));

    this->pad = kernel_fils-1;
    this->H_out_pad = H_out +2*pad;
    this->W_out_pad = W_out + 2*pad;

    this->fils_output_unroll = kernel_fils*kernel_cols*n_kernels;
    this->cols_output_unroll = H*W;
    this->fils_matriz_pesos = C;
    this->cols_matriz_pesos = kernel_fils*kernel_cols*n_kernels;
    this->fils_input_back_unroll = H_out * W_out;
    this->cols_input_back_unroll = kernel_fils*kernel_cols*C;
    this->bytes_output_unroll = fils_output_unroll * cols_output_unroll * sizeof(float);
    this->bytes_matriz_pesos = fils_matriz_pesos * cols_matriz_pesos * sizeof(float);
    this->bytes_input_back_unroll = fils_input_unroll * cols_input_unroll * sizeof(float);

    // Reserva de memoria en device
    checkCudaErrors(cudaMalloc((void **) &d_sum_local, (cols_output_unroll  + BLOCK_SIZE -1) / BLOCK_SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_output_unroll_T, bytes_output_unroll));
    checkCudaErrors(cudaMalloc((void **) &d_input_back_unroll_T, bytes_input_back_unroll));
    checkCudaErrors(cudaMalloc((void **) &d_output_centrado, this->n_kernels * H_out_pad * W_out_pad * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_output_pad, this->n_kernels * H_out_pad * W_out_pad * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_output_unroll, bytes_output_unroll));
    checkCudaErrors(cudaMalloc((void **) &d_matriz_pesos, bytes_matriz_pesos));
    checkCudaErrors(cudaMalloc((void **) &d_input_back_unroll, bytes_input_back_unroll));

    int n_pesos = n_kernels * C * kernel_fils * kernel_cols;
    checkCudaErrors(cudaMalloc((void **) &d_max, ((n_pesos + block_1D.x -1) / block_1D.x) * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_min, ((n_pesos + block_1D.x -1) / block_1D.x) * sizeof(float)));

    cudaMemcpy(d_w, w_ptr, bytes_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias_ptr, this->n_kernels * sizeof(float), cudaMemcpyHostToDevice);

    free(bias_ptr);
    free(w_ptr);
};


/*
    @brief  Copia de una capa convolucional a otra
    @conv   Capa convolucional a copiar
*/
void Convolutional::copiar(const Convolutional & conv)
{
    // Kernels de pesos
    this->n_kernels = conv.n_kernels;
    this->kernel_fils = conv.kernel_fils;
    this->kernel_cols = conv.kernel_cols;

    // Imagen de entrada
    this->C = conv.C;
    this->H = conv.H;
    this->W = conv.W;

    // Imagen de salida
    this->H_out = conv.H_out;
    this->W_out = conv.W_out;

    // Dimensiones de los volúmenes "desenrrollados" ------------------
    // Dimensiones de la entrada 'desenrrollada'
    this->fils_input_unroll = conv.fils_input_unroll;
    this->cols_input_unroll = conv.cols_input_unroll;

    // Dimensiones de los pesos como matriz 2D
    this->fils_w = conv.fils_w;
    this->cols_w = conv.cols_w;

    // Tamaños de los volúmenes "desenrrollados"
    this->bytes_input_unroll = conv.bytes_input_unroll;    // Espacio para la entrada 'desenrrollada'
    this->bytes_output = conv.bytes_output;              // Espacio para la salida
    this->bytes_w = conv.bytes_w;
		this->bytes_bias = conv.bytes_bias;

    // Learning Rate
    this->lr = conv.lr;

    // Pesos
    float *w_ptr = (float *)malloc(fils_w * cols_w * sizeof(float));

    // Inicializar pesos mediante Inicialización He
    this->generar_pesos_ptr(w_ptr);

    // Bias
    float *bias_ptr = (float *)malloc(this->n_kernels * sizeof(float));

    // Un bias por filtro, https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
    for(int i=0; i<n_kernels; i++)
        bias_ptr[i] = 0.0;

    // GPU -------------------------
    // Tamaño de bloque
    this->block.x = BLOCK_SIZE;
    this->block.y = BLOCK_SIZE;

    block_1D.x = BLOCK_SIZE;
    block_1D.y = 1;

    // Memoria compartida a nivel de bloque
    this->smem = (2*block.x * block.y) *sizeof(float);
    this->smem_1D = (2*block_1D.x) *sizeof(float);



    this->pad = conv.pad;
    this->H_out_pad = conv.H_out_pad;
    this->W_out_pad = conv.W_out_pad;

    this->fils_output_unroll = conv.fils_output_unroll;
    this->cols_output_unroll = conv.cols_output_unroll;
    this->fils_matriz_pesos = conv.fils_matriz_pesos;
    this->cols_matriz_pesos = conv.cols_matriz_pesos;
    this->fils_input_back_unroll = conv.fils_input_back_unroll;
    this->cols_input_back_unroll = conv.cols_input_back_unroll;
    this->bytes_output_unroll = conv.bytes_output_unroll;
    this->bytes_matriz_pesos = conv.bytes_matriz_pesos;
    this->bytes_input_back_unroll = conv.bytes_input_back_unroll;

    // Reserva de memoria en device
    checkCudaErrors(cudaMalloc((void **) &d_sum_local, (cols_output_unroll  + BLOCK_SIZE -1) / BLOCK_SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_output_unroll_T, bytes_output_unroll));
    checkCudaErrors(cudaMalloc((void **) &d_input_back_unroll_T, bytes_input_back_unroll));
    checkCudaErrors(cudaMalloc((void **) &d_output_centrado, this->n_kernels * H_out_pad * W_out_pad * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_output_pad, this->n_kernels * H_out_pad * W_out_pad * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_w, bytes_w));
    checkCudaErrors(cudaMalloc((void **) &d_bias, bytes_bias));
    checkCudaErrors(cudaMalloc((void **) &d_input_unroll, bytes_input_unroll));
    checkCudaErrors(cudaMalloc((void **) &d_output_unroll, bytes_output_unroll));
    checkCudaErrors(cudaMalloc((void **) &d_matriz_pesos, bytes_matriz_pesos));
    checkCudaErrors(cudaMalloc((void **) &d_input_back_unroll, bytes_input_back_unroll));
    checkCudaErrors(cudaMemcpy(d_w, w_ptr, bytes_w, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_bias, bias_ptr, bytes_bias, cudaMemcpyHostToDevice));

    int n_pesos = n_kernels * C * kernel_fils * kernel_cols;
    checkCudaErrors(cudaMalloc((void **) &d_max, ((n_pesos + block_1D.x -1) / block_1D.x) * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_min, ((n_pesos + block_1D.x -1) / block_1D.x) * sizeof(float)));

    free(bias_ptr);
    free(w_ptr);
}

/*
    @brief  Destructor
*/
Convolutional::~Convolutional()
{
    cudaFree(d_input_unroll); cudaFree(d_w); cudaFree(d_output_unroll);
    cudaFree(d_matriz_pesos); cudaFree(d_input_back_unroll);
		cudaFree(d_bias); cudaFree(d_output_centrado); cudaFree(d_output_pad); cudaFree(d_input_back_unroll_T);
		cudaFree(d_output_unroll_T); cudaFree(d_sum_local);
		cudaFree(d_max); cudaFree(d_min);

    checkCudaErrors(cudaGetLastError());

};

/*
    @brief      Inicializa los pesos de la capa convolucional según la inicialización He
    @return     Se modifica w (los pesos de la capa)
*/
void Convolutional::generar_pesos_ptr(float *w)
{
    // Inicialización He
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> distribution(0.0, sqrt(2.0 / (this->n_kernels * this->C * this->kernel_fils * this->kernel_fils)));

    for(int i=0; i<n_kernels; ++i)
        for(int j=0; j<C; ++j)
            for(int k=0; k<kernel_fils; ++k)
                for(int p=0; p<kernel_cols; ++p)
                    w[i*C*kernel_fils*kernel_cols + j*kernel_fils*kernel_cols + k*kernel_cols + p ] = distribution(gen);
                    // w[i*C*kernel_fils*kernel_cols + j*kernel_fils*kernel_cols + k*kernel_cols + p ] = 1;
}




/*
    @brief      Función de activación ReLU
    @x          Valor sobre el cual aplicar ReLU
    @return     @x tras aplicar ReLU sobre él
*/
float Convolutional::activationFunction(float x)
{
	// RELU
	return (x > 0.0) ? x : 0;
};

/*
    @brief      Derivada de la función de activación ReLU
    @x          Valor sobre el cual aplicar la derivada de ReLU
    @return     @x tras aplicar la derivada de ReLU sobre él
*/
float Convolutional::deriv_activationFunction(float x)
{
    float result = 0.0;

    if(x > 0)
        result = 1;

    return result;
}

void Convolutional::checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}

/*
    @brief      Propagación hacia delante a lo largo de toda la capa convolucional
    @input      Volumen de entrada 3D
    @output     Volumen de salida 3D
    @a          Valor de las neuronas antes de aplicar la función de activación
    @return     Se modifica @output y @a
*/
void Convolutional::forwardPropagation_vectores_externos(float *input, float *output, float *a)
{
    dim3 block_1D(BLOCK_SIZE, 1);
    dim3 grid_1D((C*H*W + block_1D.x -1) / block_1D.x, 1);
    unrollGPU<<<grid_1D, block_1D>>>(C, H, W, this->kernel_fils, input, d_input_unroll);

    // Multiplicación de matrices
    this->grid.x = (cols_input_unroll  + block.x -1) / block.x;
    this->grid.y = (fils_w + block.y -1) / block.y;

    forward_propagation_GEMM<<<grid, block, smem>>>(this->n_kernels, cols_input_unroll, cols_w, d_w, d_input_unroll, a, d_bias, output);
};


/*
    @brief      Aplica padding 
    @imagen_3D  Imagen sobre la cual se aplica padding
    @C          Canales de profundidad
    @H          Filas
    @W          Columnas
    @pad        Padding
*/
void Convolutional::aplicar_padding_ptr(float *imagen_3D, int C, int H, int W, int pad)
{

    for(int c_=0; c_<C; c_++)
    {
        // Traslado vertical en "pad unidades"
        for(int p=0; p<pad; p++)
            for(int j=H-1; j>0; j--)
                for(int k=0; k<W; k++)
                    imagen_3D[c_*H*W + j*W + k] = imagen_3D[c_*H*W + (j-1)*W + k];


        // Inicializar a 0.0 las "pad" primeras filas
        for(int i=0; i<pad; i++)
            for(int j=0; j<W; j++)
                imagen_3D[c_*H*W + i*W + j] = 0.0;

        // Inicializar a 0.0 las "pad" últimas filas
        for(int i=H-1; i>H-1-pad; i--)
            for(int j=0; j<W; j++)
                imagen_3D[c_*H*W + i*W + j] = 0.0;

        // Traslado horizontal en "pad unidades"
        for(int p=0; p<pad; p++)
            for(int j=0; j<H; j++)
                for(int k=W-1; k>0; k--)
                    imagen_3D[c_*H*W + j*W + k] = imagen_3D[c_*H*W + j*W + k-1];


        // Inicializar a 0.0 las "pad" primeras columnas
        for(int i=0; i<H; i++)
            for(int j=0; j<pad; j++)
                imagen_3D[c_*H*W + i*W + j] = 0.0;

        // Inicializar a 0.0 las "pad" últimas columnas
        for(int i=0; i<H; i++)
            for(int j=W-1; j>W-1-pad; j--)
                imagen_3D[c_*H*W + i*W + j] = 0.0;

    }
};


/*
    @brief      Retropropagación de la capa convolucional
    @input      Volumen 3D de entrada de la capa
    @output     Volumen 3D de salida de la capa
    @a          Valor de las neuronas antes de aplicar la función de activación
    @grad_w     Gradientes respecto a pesos
    @grad_bias  Gradientes respecto a sesgos
    @pad        Nivel de padding que se aplicó anteriormente
*/
void Convolutional::backPropagation_vectores_externos(float *input, float *output, float *a, float *grad_w, float *grad_bias)
{
    grid.x = (H_out  + BLOCK_SIZE -1) / BLOCK_SIZE;
    grid.y = (W_out  + BLOCK_SIZE -1) / BLOCK_SIZE;
    deriv_activ_func_output<<<grid, block>>>(n_kernels, H_out, W_out, output, a);

    grid.x = (H_out_pad  + BLOCK_SIZE -1) / BLOCK_SIZE;
    grid.y = (W_out_pad  + BLOCK_SIZE -1) / BLOCK_SIZE;
    centrar<<<grid, block>>>(n_kernels, H_out, W_out, H_out_pad, W_out_pad, output, d_output_centrado);
    aplicar_padding_gpu<<<grid, block>>>(this->n_kernels, H_out_pad, W_out_pad, d_output_centrado, d_output_pad, pad);
    unroll_3dim_gpu<<<grid, block>>>(n_kernels, H_out_pad, W_out_pad, kernel_fils, d_output_pad, d_output_unroll);		// "Desenrrollar" volumen de salida

    // Transpuesta del volumen de salida desenrrollado
    grid.x = (cols_output_unroll + BLOCK_SIZE -1) / BLOCK_SIZE;
    grid.y = (fils_output_unroll + BLOCK_SIZE -1) / BLOCK_SIZE;
    matrizTranspuesta_conv<<<grid, block>>>(d_output_unroll, d_output_unroll_T, cols_output_unroll, fils_output_unroll);

    // "Desenrrollar" volumen de entrada
    grid.x = (H + BLOCK_SIZE -1) / BLOCK_SIZE;
    grid.y = (W + BLOCK_SIZE -1) / BLOCK_SIZE;
    unroll_1dim_gpu<<<grid, block>>>(C, H, W, H_out, input, d_input_back_unroll);

    // Transpuesta del volumen de entrada desenrrollado
    grid.x = (kernel_fils*kernel_cols*C + BLOCK_SIZE -1) / BLOCK_SIZE;
    grid.y = (H_out*W_out + BLOCK_SIZE -1) / BLOCK_SIZE;
    matrizTranspuesta_conv<<<grid, block>>>(d_input_back_unroll, d_input_back_unroll_T, kernel_fils*kernel_cols*C, H_out*W_out);

    // Desenrrollar la matriz de pesos
    grid.x = (this->n_kernels + BLOCK_SIZE -1) / BLOCK_SIZE;
    grid.y = (C + BLOCK_SIZE -1) / BLOCK_SIZE;
    unroll_matriz_pesos<<<grid, block>>>(C, this->n_kernels, kernel_fils, d_w, d_matriz_pesos);

    // Multiplicación de matrices
    this->grid.x = (cols_input_back_unroll  + BLOCK_SIZE -1) / BLOCK_SIZE;
    this->grid.y = (this->n_kernels + BLOCK_SIZE -1) / BLOCK_SIZE;
    multiplicarMatricesGPU_calcular_grad_w<<<grid, block, smem>>>(this->n_kernels, cols_input_back_unroll, H_out * W_out, output, d_input_back_unroll_T, grad_w);    // Gradiente respecto a pesos

    this->grid.x = (cols_output_unroll  + BLOCK_SIZE -1) / BLOCK_SIZE;
    this->grid.y = (C + BLOCK_SIZE -1) / BLOCK_SIZE;
    multiplicarMatricesGPU<<<grid, block, smem>>>(fils_matriz_pesos, cols_output_unroll, cols_matriz_pesos, d_matriz_pesos, d_output_unroll_T, input);  // Gradiente respecto a entrada

    // Gradiente respecto a cada sesgo
    grid.x = (H_out*W_out + BLOCK_SIZE -1) / BLOCK_SIZE;
    grid.y = 1;

    for(int i=0; i<this->n_kernels; i++)
    {
        reduce_suma<<<grid, block_1D, smem_1D>>>(output + i*H_out*W_out, d_sum_local, H_out*W_out);
        acumular_grad_bias<<<grid, block_1D>>>(d_sum_local, grad_bias, i);
    }
		

};

/*
    @brief          Escalar los pesos para evitar que los gradientes "exploten"
    @clip_value     Valor a emplear para realizar el "clip" o escalado
    @return         Se actualizan los valores de w (pesos de la capa)
*/
void Convolutional::escalar_pesos_vectores_externos(float clip_value)
{
    int n_pesos = n_kernels * C * kernel_fils * kernel_cols;
    grid.x = (n_pesos + block_1D.x -1) / block_1D.x;
    grid.y = 1;

    reduceMax_conv<<<grid, block_1D, smem_1D>>>(d_w, d_max, n_pesos);
    reduceMin_conv<<<grid, block_1D, smem_1D>>>(d_w, d_min, n_pesos);
    min_max_conv<<<grid, block_1D>>>(d_max, d_min, grid.x);
    kernel_escalar_pesos_conv<<<grid, block_1D>>>(d_w, n_pesos, d_max, d_min, clip_value);
}

/*
    @brief          Actualizar los pesos y sesgos de la capa
    @grad_w         Gradiente de cada peso de la capa
    @grad_b         Gradiente de cada sesgo de la capa
    @return         Se actualizar los valores de w y bias (pesos y sesgos de la capa)
*/
void Convolutional::actualizar_grads_vectores_externos(float *grad_w, float *grad_bias, int mini_batch)
{
		grid.x = (C  + BLOCK_SIZE -1) / BLOCK_SIZE;
		grid.y = (n_kernels  + BLOCK_SIZE -1) / BLOCK_SIZE;
		kernel_actualizar_parametros<<<grid, block>>>(n_kernels, C, kernel_fils, d_w, grad_w, d_bias, grad_bias, lr, mini_batch);
}
