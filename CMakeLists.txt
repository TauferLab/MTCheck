cmake_minimum_required(VERSION 3.12)

##project(dedup VERSION 0.0.1 LANGUAGES CXX CUDA)
#project(dedup VERSION 0.0.1 LANGUAGES CXX)
#
#find_package(Kokkos REQUIRED)
#find_package(stdgpu 1.0.0 REQUIRED)
#
#option(DEDUP_ENABLE_CUDA "Enable CUDA" OFF)
#if(DEDUP_ENABLE_CUDA)
#    enable_language(CUDA)
#endif()
#
#set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
#
#file(GLOB SOURCES src/cpu_dedup.cpp src/gpu_dedup.cu src/dedup.cu)
#
#add_library(dedup ${SOURCES})
#
#set_target_properties(dedup PROPERTIES
#    CXX_STANDARD 14
#    CXX_STANDARD_REQUIRED YES
#    CXX_EXTENSIONS NO
#)
#
#set_target_properties(dedup PROPERTIES PUBLIC_HEADER include/dedup.hpp)
#set_target_properties(dedup PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#
#target_include_directories(dedup PUBLIC ${CMAKE_SOURCE_DIR}/include)
#
#include(GNUInstallDirs)
#
#install(TARGETS dedup 
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
#        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#
#add_executable(cpu_test src/cpu_test.cpp)
#target_link_libraries(cpu_test PUBLIC dedup)
#target_link_libraries(cpu_test PRIVATE Kokkos::kokkos)
#
#add_executable(gpu_test src/gpu_test.cu)
#set_property(TARGET gpu_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(gpu_test PUBLIC dedup)
#target_link_libraries(gpu_test PRIVATE Kokkos::kokkos)
#target_link_libraries(gpu_test PUBLIC stdgpu::stdgpu)
#
##project(dedup VERSION 0.0.1 LANGUAGES CXX CUDA)
#project(dedup VERSION 0.0.1 LANGUAGES CXX)

project(dedup VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Kokkos REQUIRED)
find_package(OpenSSL REQUIRED)

include(GNUInstallDirs)

#add_executable(merkle_tree_test src/test_dedup.cpp)
#target_include_directories(merkle_tree_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
#target_link_libraries(merkle_tree_test PRIVATE Kokkos::kokkos)
#
#set_target_properties(merkle_tree_test PROPERTIES
#    CXX_STANDARD 14
#    CXX_STANDARD_REQUIRED YES
#    CXX_EXTENSIONS OFF
#)

add_executable(merkle_tree_test_files src/test_dedup_files.cpp)
target_include_directories(merkle_tree_test_files PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(merkle_tree_test_files PRIVATE Kokkos::kokkos)
target_link_libraries(merkle_tree_test_files PRIVATE OpenSSL::SSL)

set_target_properties(merkle_tree_test_files PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

#add_executable(merkle_tree_test_files_mpi src/dedup_files_mpi.cpp)
#target_include_directories(merkle_tree_test_files_mpi PUBLIC ${CMAKE_SOURCE_DIR}/include)
#target_link_libraries(merkle_tree_test_files_mpi PRIVATE Kokkos::kokkos)
#target_link_libraries(merkle_tree_test_files_mpi PRIVATE OpenSSL::SSL)
#
#set_target_properties(merkle_tree_test_files_mpi PROPERTIES
#    CXX_STANDARD 14
#    CXX_STANDARD_REQUIRED YES
#    CXX_EXTENSIONS OFF
#)

add_executable(checkpoint_generator src/data_generation.cpp)
target_include_directories(checkpoint_generator PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(checkpoint_generator PRIVATE Kokkos::kokkos)

set_target_properties(checkpoint_generator PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_restart src/test_restart.cpp)
target_include_directories(test_restart PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_restart PRIVATE Kokkos::kokkos)

set_target_properties(test_restart PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

