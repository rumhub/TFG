cmake_minimum_required(VERSION 3.9)
project(cuda_test LANGUAGES CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 50)
add_executable(cuda_test test.cu)

