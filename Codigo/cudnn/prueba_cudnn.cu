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
    int H_conv=5, W_conv=5, C_conv=1, mini_batch=1,    // Dimensiones imagen de entrada
        K_conv=2, n_kernels=2,
        H_out_conv=H_conv-K_conv+1 + 2*pad, W_out_conv=W_conv-K_conv+1 + 2*pad,
        H_pool=H_out_conv, W_pool=W_out_conv, C_pool=n_kernels,
        K_pool=2,
        H_out_pool=H_pool/K_pool, W_out_pool=W_pool/K_pool;

    // Reserva de memoria ------------------------------------------------------------------------------
    int datos_conv = mini_batch * C_conv * H_conv * W_conv, 
        datos_conv_out = mini_batch * n_kernels * H_out_conv * W_out_conv, 
        datos_conv_kernel = n_kernels * C_conv * K_conv * K_conv,
        datos_pool = mini_batch * C_pool * H_pool * W_pool,
        datos_pool_out = mini_batch * C_pool * H_out_pool * W_out_pool;
    float h_input[datos_conv], h_kernel[datos_conv_kernel],
          h_conv_output[datos_conv_out], h_relu_output[datos_conv_out],
          h_pool_output[datos_pool_out];

    float *d_input, *d_conv_output, *d_relu_output, *d_pool_output, *d_kernel;
    cudaMalloc(&d_input, sizeof(h_input));
    cudaMalloc(&d_conv_output, datos_conv_out * sizeof(float));
    cudaMalloc(&d_relu_output, datos_conv_out * sizeof(float));
    cudaMalloc(&d_pool_output, datos_pool_out * sizeof(float));
    cudaMalloc(&d_kernel, sizeof(h_kernel));

    // Reserva de memoria para los gradientes ---------------------------------------------------
    float h_dconv[datos_conv_out], h_drelu[datos_conv_out], h_dinput[datos_conv],
          h_dkernel[datos_conv_kernel], h_dpool[datos_pool_out]; // gradient of the loss with respect to maxpool output
    
    for(int i=0; i<datos_pool_out; ++i) 
        h_dpool[i] = 1.0; // Initialize dpool to ones for simplicity
    
    float *d_dpool, *d_drelu, *d_dconv, *d_dinput, *d_dkernel;
    cudaMalloc(&d_dpool, sizeof(h_dpool));
    cudaMalloc(&d_drelu, sizeof(h_relu_output));
    cudaMalloc(&d_dconv, sizeof(h_conv_output));
    cudaMalloc(&d_dinput, sizeof(h_input));
    cudaMalloc(&d_dkernel, sizeof(h_kernel));
    cudaMemcpy(d_dpool, h_dpool, sizeof(h_dpool), cudaMemcpyHostToDevice);

    // Inicializar input y kernels -----------------------------------
    int cont=-10;
    for(int i=0; i<datos_conv; i++)
        h_input[i] = cont++;

    for(int i=0; i<datos_conv_kernel; i++)
        h_kernel[i] = 1.0;

    cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);



    // Creación de descriptores -------------------------------------------------------------------------
    cudnnHandle_t cudnn_handle;
    checkCUDNN(cudnnCreate(&cudnn_handle));

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

    // Set up tensor descriptors for ReLU output
    cudnnTensorDescriptor_t relu_output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&relu_output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        relu_output_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        mini_batch,  // batch size
        n_kernels, // channels
        H_out_conv, // height (same as convolution output)
        W_out_conv  // width (same as convolution output)
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

    // Set up activation descriptor for ReLU
    cudnnActivationDescriptor_t activation_descriptor;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(
        activation_descriptor,
        CUDNN_ACTIVATION_RELU,
        CUDNN_PROPAGATE_NAN,
        0.0
    ));

    // Mostrar por pantalla input y kernels -----------------------------------------
    printf("Input\n");
    printTensor(h_input, mini_batch, C_conv, H_conv, W_conv);

    printf("Kernels\n");
    printTensor(h_kernel, mini_batch, n_kernels, K_conv, K_conv);

    // Perform the convolution forward pass
    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(
        cudnn_handle,
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

    // Perform the ReLU forward pass
    checkCUDNN(cudnnActivationForward(
        cudnn_handle,
        activation_descriptor,
        &alpha,
        conv_output_descriptor,
        d_conv_output,
        &beta,
        relu_output_descriptor,
        d_relu_output
    ));

    // Perform the maxpool forward pass
    checkCUDNN(cudnnPoolingForward(
        cudnn_handle,
        pooling_descriptor,
        &alpha,
        relu_output_descriptor,
        d_relu_output,
        &beta,
        pool_output_descriptor,
        d_pool_output
    ));

    // Copy the result back to the host
    cudaMemcpy(h_conv_output, d_conv_output, sizeof(h_conv_output), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_output, d_relu_output, sizeof(h_relu_output), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pool_output, d_pool_output, sizeof(h_pool_output), cudaMemcpyDeviceToHost);

    // Print the convolution output
    std::cout << "Convolution output tensor:" << std::endl;
    printTensor(h_conv_output, mini_batch, n_kernels, H_out_conv, W_out_conv);

    // Print the ReLU output
    std::cout << "ReLU output tensor:" << std::endl;
    printTensor(h_relu_output, mini_batch, n_kernels, H_out_conv, W_out_conv);

    // Print the maxpool output
    std::cout << "Maxpool output tensor:" << std::endl;
    printTensor(h_pool_output, mini_batch, C_pool, H_out_pool, W_out_pool);


    printf("BACKPROP\n");
    // Backpropagation -----------------------------------------------------------------------------

    // Perform the maxpool backward pass
    checkCUDNN(cudnnPoolingBackward(
        cudnn_handle,                       // handle
        pooling_descriptor,                 // poolingDesc
        &alpha,                             // *alpha
        pool_output_descriptor,             // yDesc
        d_pool_output,                      // *y
        pool_output_descriptor,             // dyDesc
        d_dpool,                            // *dy
        relu_output_descriptor,             // xDesc
        d_relu_output,                      // *xData
        &beta,                              // *beta
        relu_output_descriptor,             // dxDesc
        d_drelu                             // *dx
    ));

    // Perform the ReLU backward pass
    checkCUDNN(cudnnActivationBackward(
        cudnn_handle,
        activation_descriptor,
        &alpha,
        relu_output_descriptor,
        d_relu_output,
        relu_output_descriptor,
        d_drelu,
        conv_output_descriptor,
        d_conv_output,
        &beta,
        conv_output_descriptor,
        d_dconv
    ));

    // Perform the convolution backward pass
    checkCUDNN(cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        kernel_descriptor,
        d_kernel,
        conv_output_descriptor,
        d_dconv,
        convolution_descriptor,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        nullptr, 0,
        &beta,
        input_descriptor,
        d_dinput
    ));

    checkCUDNN(cudnnConvolutionBackwardFilter(
        cudnn_handle,
        &alpha,
        input_descriptor,
        d_input,
        conv_output_descriptor,
        d_dconv,
        convolution_descriptor,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        nullptr, 0,
        &beta,
        kernel_descriptor,
        d_dkernel
    ));

    // Copy the gradients back to the host
    cudaMemcpy(h_drelu, d_drelu, sizeof(h_drelu), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dconv, d_dconv, sizeof(h_dconv), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dinput, d_dinput, sizeof(h_dinput), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dkernel, d_dkernel, sizeof(h_dkernel), cudaMemcpyDeviceToHost);

    // Print the gradients
    std::cout << "Gradient with respect to ReLU output (dReLU):" << std::endl;
    printTensor(h_drelu, mini_batch, n_kernels, H_out_conv, W_out_conv);

    std::cout << "Gradient with respect to convolution output (dConv):" << std::endl;
    printTensor(h_dconv, mini_batch, n_kernels, H_out_conv, W_out_conv);

    std::cout << "Gradient with respect to input (dInput):" << std::endl;
    printTensor(h_dinput, mini_batch, C_conv, H_conv, W_conv);

    std::cout << "Gradient with respect to kernel (dKernel):" << std::endl;
    printTensor(h_dkernel, n_kernels, C_conv, K_conv, K_conv);

    // Clean up
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(conv_output_descriptor);
    cudnnDestroyTensorDescriptor(relu_output_descriptor);
    cudnnDestroyTensorDescriptor(pool_output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroyPoolingDescriptor(pooling_descriptor);
    cudnnDestroyActivationDescriptor(activation_descriptor);
    cudaFree(d_input);
    cudaFree(d_conv_output);
    cudaFree(d_relu_output);
    cudaFree(d_pool_output);
    cudaFree(d_kernel);
    cudaFree(d_dpool);
    cudaFree(d_drelu);
    cudaFree(d_dconv);
    cudaFree(d_dinput);
    cudaFree(d_dkernel);
    cudnnDestroy(cudnn_handle);

    return 0;
}
