cmake_minimum_required(VERSION 3.12)

project(gpu_merkle_trees VERSION 0.0.1 LANGUAGES CXX)
enable_language(CUDA)

find_package(stdgpu 1.0.0 REQUIRED)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB SOURCES gpu_merkle_tree.cu )

add_executable(gpu_test gpu_main.cu gpu_merkle_tree.cu)
target_link_libraries(gpu_test PUBLIC stdgpu::stdgpu)
set_property(TARGET gpu_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)


