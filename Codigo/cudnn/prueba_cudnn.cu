#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

/*
    OBJETIVO: 2 capas, primero una convolucional y luego una MaxPool.
              Se busca realizar una propagación hacia delante y luego una propagación hacia detrás
*/

// Error checking macro
#define checkCUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "CUDNN error: " << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

// CUDA error checking
#define checkCUDA(status) { \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

// Print the contents of a tensor
void printTensor(float* tensor, int n, int c, int h, int w) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < c; ++j) {
            for (int k = 0; k < h; ++k) {
                for (int l = 0; l < w; ++l) {
                    std::cout << tensor[i * c * h * w + j * h * w + k * w + l] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

int main() {

    // Establecer dimensiones de trabajo ------------------------------------------------------------
    int pad = 1;
    int H_conv=4, W_conv=4, C_conv=1, mini_batch=1,    // Dimensiones imagen de entrada
        K_conv=2, n_kernels=2,
        H_out_conv=H_conv-K_conv+1 + 2*pad, W_out_conv=W_conv-K_conv+1 + 2*pad,
        H_pool=H_out_conv, W_pool=W_out_conv, C_pool=n_kernels,
        K_pool=2,
        H_out_pool=H_pool/K_pool, W_out_pool=W_pool/K_pool;

    // Reserva de memoria ------------------------------------------------------------------------------
    int datos_conv = mini_batch * C_conv * H_conv * W_conv, 
        datos_conv_kernel = n_kernels * C_conv * K_conv * K_conv,
        datos_pool = mini_batch * C_pool * H_pool * W_pool;
    float h_input[datos_conv], h_kernel[datos_conv_kernel],
          h_conv_output[mini_batch * n_kernels * H_out_conv * W_out_conv], 
          h_pool_output[mini_batch * C_pool * H_out_pool * W_out_pool];

    float *d_input, *d_conv_output, *d_pool_output, *d_kernel;
    cudaMalloc(&d_input, sizeof(h_input));
    cudaMalloc(&d_conv_output, mini_batch * n_kernels * H_out_conv * W_out_conv * sizeof(float));
    cudaMalloc(&d_pool_output, mini_batch * C_pool * H_out_pool * W_out_pool * sizeof(float));
    cudaMalloc(&d_kernel, sizeof(h_kernel));


    // Inicializar input y kernels -----------------------------------
    for(int i=0; i<datos_conv; i++)
        h_input[i] = i;

    for(int i=0; i<datos_conv_kernel; i++)
        h_kernel[i] = 1.0;

    cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);



    // Creación de descriptores -------------------------------------------------------------------------
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // Set up tensor descriptors for input
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        mini_batch,  // batch size
        C_conv,  // channels
        H_conv, // height (reduced for simplicity)
        W_conv  // width (reduced for simplicity)
    ));

    // Set up tensor descriptors for convolution output
    cudnnTensorDescriptor_t conv_output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&conv_output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        conv_output_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        mini_batch,  // batch size
        n_kernels, // channels
        H_out_conv, // height (calculated manually)
        W_out_conv  // width (calculated manually)
    ));

    // Set up convolution descriptor
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(
        kernel_descriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        n_kernels, // out_channels
        C_conv,  // in_channels
        K_conv,  // kernel height
        K_conv   // kernel width
    ));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
        convolution_descriptor,
        pad, pad,  // padding
        1, 1,  // strides
        1, 1,  // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    // Set up tensor descriptors for maxpool output
    cudnnTensorDescriptor_t pool_output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&pool_output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        pool_output_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        mini_batch,  // batch size
        C_pool,  // channels
        H_out_pool, // height
        W_out_pool  // width
    ));

    cudnnPoolingDescriptor_t pooling_descriptor;
    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    checkCUDNN(cudnnSetPooling2dDescriptor(
        pooling_descriptor,
        CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN,
        K_pool, K_pool, // window height and width
        0, 0, // padding height and width
        K_pool, K_pool  // stride height and width
    ));

    // Mostrar por pantalla input y kernels -----------------------------------------
    printf("Input\n");
    printTensor(h_input, mini_batch, C_conv, H_conv, W_conv);

    printf("Kernels\n");
    printTensor(h_kernel, mini_batch, n_kernels, K_conv, K_conv);

    // Perform the convolution forward pass
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(
        cudnn,
        &alpha,
        input_descriptor,
        d_input,
        kernel_descriptor,
        d_kernel,
        convolution_descriptor,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        nullptr, 0,
        &beta,
        conv_output_descriptor,
        d_conv_output
    ));

    // Perform the maxpool forward pass
    checkCUDNN(cudnnPoolingForward(
        cudnn,
        pooling_descriptor,
        &alpha,
        conv_output_descriptor,
        d_conv_output,
        &beta,
        pool_output_descriptor,
        d_pool_output
    ));

    // Copy the result back to the host
    cudaMemcpy(h_conv_output, d_conv_output, sizeof(h_conv_output), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pool_output, d_pool_output, sizeof(h_pool_output), cudaMemcpyDeviceToHost);

    // Print the convolution output
    std::cout << "Convolution output tensor:" << std::endl;
    printTensor(h_conv_output, mini_batch, n_kernels, H_out_conv, W_out_conv);

    // Print the maxpool output
    std::cout << "Maxpool output tensor:" << std::endl;
    printTensor(h_pool_output, mini_batch, C_pool, H_out_pool, W_out_pool);


    // Clean up
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(conv_output_descriptor);
    cudnnDestroyTensorDescriptor(pool_output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroyPoolingDescriptor(pooling_descriptor);
    cudaFree(d_input);
    cudaFree(d_conv_output);
    cudaFree(d_pool_output);
    cudaFree(d_kernel);
    cudnnDestroy(cudnn);

    return 0;
}
