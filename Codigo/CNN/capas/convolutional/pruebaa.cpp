#include <iostream>
#include <vector>

int main() {
    // Input tensor
    std::vector<std::vector<std::vector<float>>> x =
        {
            {
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12},
                {13, 14, 15, 16}
            },
            {
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12},
                {13, 14, 15, 16}
            }
        };

       // Filter tensor
    std::vector<std::vector<std::vector<std::vector<float>>>> weight(1, std::vector<std::vector<std::vector<float>>>(2, std::vector<std::vector<float>>(4, std::vector<float>(4, 1.0f))));

    // Bias tensor
    std::vector<float> b = {1.0f};

    // Padding, stride
    int pad = 1;
    int stride = 1;

    // Padding input tensor
    int C = x.size();
    int H = x[0].size();
    int W = x[0][0].size();

    std::vector<std::vector<std::vector<float>>> x_pad(std::vector<std::vector<std::vector<float>>>(C, std::vector<std::vector<float>>(H + 2 * pad, std::vector<float>(W + 2 * pad, 0.0f))));

        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    x_pad[c][i + pad][j + pad] = x[c][i][j];
                }
            }
        }
    

    // Output tensor dimensions
    int F = weight.size();
    int Hf = weight[0][0].size();
    int Wf = weight[0][0][0].size();

    int H_out = 1 + (H + 2 * pad - Hf) / stride;
    int W_out = 1 + (W + 2 * pad - Wf) / stride;

    // Output tensor
    std::vector<std::vector<std::vector<float>>> y(std::vector<std::vector<std::vector<float>>>(F, std::vector<std::vector<float>>(H_out, std::vector<float>(W_out, 0.0f))));

    // Convolution
        for (int f = 0; f < F; ++f) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    int i = h * stride;
                    int j = w * stride;
                    float conv_sum = 0.0f;

                    for (int c = 0; c < C; ++c) {
                        for (int p = 0; p < Hf; ++p) {
                            for (int q = 0; q < Wf; ++q) {
                                conv_sum += x_pad[c][i + p][j + q] * weight[f][c][p][q];
                            }
                        }
                    }

                    y[f][h][w] = conv_sum + b[f];
                }
            }
        }
    

    // Print the padded input for a given image on a given color channel
    std::cout << "Padded input for a given image on a given color channel\n";
    for (int i = 0; i < H + 2 * pad; ++i) {
        for (int j = 0; j < W + 2 * pad; ++j) {
            std::cout << x_pad[0][i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Print the expected output tensor shape
    std::cout << "\nExpected output tensor to have shape {" << F << ", " << H_out << ", " << W_out << "}\n";


    // Print the output tensor y
    std::cout << "\nOutput tensor y:\n";
        for (int f = 0; f < F; ++f) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    std::cout << y[f][h][w] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

    
    // ------------------------------ BACKPROP -----------------------------------------

    // Gradient variables
    std::vector<std::vector<std::vector<std::vector<float>>>> grad_weight(1, std::vector<std::vector<std::vector<float>>>(2, std::vector<std::vector<float>>(4, std::vector<float>(4, 0.0f))));
    std::vector<std::vector<std::vector<float>>> grad_x(std::vector<std::vector<std::vector<float>>>(C, std::vector<std::vector<float>>(H, std::vector<float>(W, 0.0f))));
    std::vector<float> grad_b(1, 0.0f);
    std::vector<std::vector<std::vector<float>>> grad_x_pad(std::vector<std::vector<std::vector<float>>>(C, std::vector<std::vector<float>>(H + 2 * pad, std::vector<float>(W + 2 * pad, 0.0f))));
    std::vector<std::vector<std::vector<float>>> grad_y(std::vector<std::vector<std::vector<float>>>(F, std::vector<std::vector<float>>(H_out, std::vector<float>(W_out, 1.0f))));

    // Compute gradients
    for (int f = 0; f < F; ++f) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                int i = h * stride;
                int j = w * stride;

                // Gradient for weight
                for (int c = 0; c < C; ++c) {
                    for (int p = 0; p < Hf; ++p) {
                        for (int q = 0; q < Wf; ++q) {
                            grad_weight[f][c][p][q] += x_pad[c][i + p][j + q] * grad_y[f][h][w];
                        }
                    }
                }

                // Gradient for x_pad
                for (int c = 0; c < C; ++c) {
                    for (int p = 0; p < Hf; ++p) {
                        for (int q = 0; q < Wf; ++q) {
                            grad_x_pad[c][i + p][j + q] += weight[f][c][p][q] * grad_y[f][h][w];
                        }
                    }
                }
            }
        }
    }
    

    // Get rid of padding for grad_x
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                grad_x[c][i][j] = grad_x_pad[c][i + pad][j + pad];
            }
        }
    }
    

    // Compute gradient of bias
    for (int f = 0; f < F; ++f) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                grad_b[f] += grad_y[f][h][w];
            }
        }
    }


    std::cout << "\nGradient of x (grad_x):\n";
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    std::cout << grad_x[c][i][j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    

    // Print the grad_b variable
    std::cout << "\nGradient of bias (grad_b):\n";
    for (int f = 0; f < F; ++f) {
        std::cout << grad_b[f] << " ";
    }
    std::cout << std::endl;
    

    return 0;
}
