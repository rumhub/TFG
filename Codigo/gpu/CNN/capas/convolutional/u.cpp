#include <iostream>
#include <vector>

// Function to perform convolution with loop unrolling
std::vector<std::vector<int>> convolutionUnrolled(const std::vector<std::vector<std::vector<int>>>& input,
                                                  const std::vector<std::vector<int>>& kernel) {
    // Assuming input dimensions are [depth][height][width]
    int depth = input.size();
    int height = input[0].size();
    int width = input[0][0].size();
    int kernelHeight = kernel.size();
    int kernelWidth = kernel[0].size();

    // Calculate the size of output
    int outputHeight = height - kernelHeight + 1;
    int outputWidth = width - kernelWidth + 1;

    std::vector<std::vector<int>> output(outputHeight * outputWidth,
                                          std::vector<int>(kernelHeight * kernelWidth * depth));

    int outputIndex = 0;
    for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
            int inputIndex = 0;
            for (int d = 0; d < depth; ++d) {
                for (int ki = 0; ki < kernelHeight; ++ki) {
                    for (int kj = 0; kj < kernelWidth; ++kj) {
                        output[outputIndex][inputIndex++] = input[d][i + ki][j + kj];
                    }
                }
            }
            ++outputIndex;
        }
    }

    return output;
}

int main() {
    // 3D input tensor
    std::vector<std::vector<std::vector<int>>> input = {
        {{1, 2, 3}, 
         {4, 5, 6}, 
         {7, 8, 9}},
        {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}},
        {{19, 20, 21}, {22, 23, 24}, {25, 26, 27}}
    };

    // 2D kernel
    std::vector<std::vector<int>> kernel = {
        {1, 2},
        {3, 4}
    };

    std::vector<std::vector<int>> result = convolutionUnrolled(input, kernel);

    std::cout << "Unrolled matrix:" << std::endl;
    for (const auto& row : result) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
