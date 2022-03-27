cmake_minimum_required(VERSION 3.12)

#project(dedup VERSION 0.0.1 LANGUAGES CXX CUDA)
project(dedup VERSION 0.0.1 LANGUAGES CXX)
option(DEDUP_ENABLE_CUDA "Enable CUDA" OFF)
if(DEDUP_ENABLE_CUDA)
    enable_language(CUDA)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB SOURCES src/cpu_dedup.cpp src/gpu_dedup.cu src/dedup.cu)

add_library(dedup ${SOURCES})

set_target_properties(dedup PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

set_target_properties(dedup PROPERTIES PUBLIC_HEADER include/dedup.hpp)
set_target_properties(dedup PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(dedup PUBLIC ${CMAKE_SOURCE_DIR}/include)

include(GNUInstallDirs)

install(TARGETS dedup 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_executable(cpu_test src/cpu_test.cpp)
target_link_libraries(cpu_test PUBLIC dedup)

add_executable(gpu_test src/gpu_test.cu)
set_property(TARGET gpu_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_test PUBLIC dedup)

