#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>

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
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // Set up tensor descriptors for input
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        1,  // batch size
        3,  // channels
        4, // height (reduced for simplicity)
        4  // width (reduced for simplicity)
    ));

    // Set up tensor descriptors for convolution output
    cudnnTensorDescriptor_t conv_output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&conv_output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        conv_output_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        1,  // batch size
        2, // channels
        2, // height (calculated manually)
        2  // width (calculated manually)
    ));

    // Set up convolution descriptor
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(
        kernel_descriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        2, // out_channels
        3,  // in_channels
        3,  // kernel height
        3   // kernel width
    ));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
        convolution_descriptor,
        0, 0,  // padding
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
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        1,  // batch size
        2,  // channels
        1, // height
        1  // width
    ));

    cudnnPoolingDescriptor_t pooling_descriptor;
    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    checkCUDNN(cudnnSetPooling2dDescriptor(
        pooling_descriptor,
        CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN,
        2, 2, // window height and width
        0, 0, // padding height and width
        2, 2  // stride height and width
    ));

    // Allocate memory for tensors
    float h_input[1 * 3 * 4 * 4] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    float h_kernel[2 * 3 * 3 * 3] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,

        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
    };

    float* d_input;
    float* d_conv_output;
    float* d_pool_output;
    float* d_kernel;
    cudaMalloc(&d_input, sizeof(h_input));
    cudaMalloc(&d_conv_output, 1 * 2 * 2 * 2 * sizeof(float));
    cudaMalloc(&d_pool_output, 1 * 2 * 1 * 1 * sizeof(float));
    cudaMalloc(&d_kernel, sizeof(h_kernel));

    cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

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
    float h_conv_output[1 * 2 * 2 * 2];
    float h_pool_output[1 * 2 * 1 * 1];
    cudaMemcpy(h_conv_output, d_conv_output, sizeof(h_conv_output), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pool_output, d_pool_output, sizeof(h_pool_output), cudaMemcpyDeviceToHost);

    // Print the convolution output
    std::cout << "Convolution output tensor:" << std::endl;
    printTensor(h_conv_output, 1, 2, 2, 2);

    // Print the maxpool output
    std::cout << "Maxpool output tensor:" << std::endl;
    printTensor(h_pool_output, 1, 2, 1, 1);

    // Fully connected layer with cuDNN
    float h_fc_weights[2 * 2] = {
        1, 1,
        1, 1
    };
    float h_fc_biases[2] = {0, 0};
    float h_fc_output[2];

    float* d_fc_weights;
    float* d_fc_biases;
    float* d_fc_output;
    float* d_fc_input;
    cudaMalloc(&d_fc_weights, sizeof(h_fc_weights));
    cudaMalloc(&d_fc_biases, sizeof(h_fc_biases));
    cudaMalloc(&d_fc_output, sizeof(h_fc_output));
    cudaMalloc(&d_fc_input, sizeof(h_pool_output));

    cudaMemcpy(d_fc_weights, h_fc_weights, sizeof(h_fc_weights), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc_biases, h_fc_biases, sizeof(h_fc_biases), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc_input, h_pool_output, sizeof(h_pool_output), cudaMemcpyHostToDevice);

    cudnnTensorDescriptor_t fc_input_descriptor;
    cudnnTensorDescriptor_t fc_output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&fc_input_descriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&fc_output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        fc_input_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        1,  // batch size
        2,  // channels
        1,  // height
        1   // width
    ));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        fc_output_descriptor,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        1,  // batch size
        2,  // channels (output size)
        1,  // height
        1   // width
    ));

    // Perform the fully connected layer forward pass using cuDNN
    const float alpha_fc = 1.0f, beta_fc = 0.0f;

    // Create a tensor operation descriptor
    cudnnOpTensorDescriptor_t op_tensor_desc;
    checkCUDNN(cudnnCreateOpTensorDescriptor(&op_tensor_desc));
    checkCUDNN(cudnnSetOpTensorDescriptor(
        op_tensor_desc,
        CUDNN_OP_TENSOR_MUL,
        CUDNN_DATA_FLOAT,
        CUDNN_NOT_PROPAGATE_NAN
    ));

    // Multiply inputs with weights
    checkCUDNN(cudnnOpTensor(
        cudnn,
        op_tensor_desc,
        &alpha_fc,
        fc_input_descriptor,
        d_fc_input,
        &alpha_fc,
        fc_input_descriptor,
        d_fc_weights,
        &beta_fc,
        fc_output_descriptor,
        d_fc_output
    ));

    // Add biases
    checkCUDNN(cudnnAddTensor(
        cudnn,
        &alpha_fc,
        fc_input_descriptor,
        d_fc_biases,
        &alpha_fc,
        fc_output_descriptor,
        d_fc_output
    ));

    // Copy the result back to the host
    cudaMemcpy(h_fc_output, d_fc_output, sizeof(h_fc_output), cudaMemcpyDeviceToHost);

    // Print the fully connected layer output
    std::cout << "Fully connected layer output:" << std::endl;
    for (int i = 0; i < 2; ++i) {
        std::cout << h_fc_output[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(conv_output_descriptor);
    cudnnDestroyTensorDescriptor(pool_output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroyPoolingDescriptor(pooling_descriptor);
    cudnnDestroyTensorDescriptor(fc_input_descriptor);
    cudnnDestroyTensorDescriptor(fc_output_descriptor);
    cudnnDestroyOpTensorDescriptor(op_tensor_desc);
    cudaFree(d_input);
    cudaFree(d_conv_output);
    cudaFree(d_pool_output);
    cudaFree(d_kernel);
    cudaFree(d_fc_weights);
    cudaFree(d_fc_biases);
    cudaFree(d_fc_output);
    cudaFree(d_fc_input);
    cudnnDestroy(cudnn);

    return 0;
}
