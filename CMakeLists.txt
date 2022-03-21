cmake_minimum_required(VERSION 3.12)

project(dedup VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

#find_package(Kokkos REQUIRED)

file(GLOB SOURCES src/*.cpp src/*.cu)

add_library(dedup ${SOURCES})

#target_link_libraries(dedup PRIVATE Kokkos::kokkos)

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

add_executable(test src/main.cu)
#add_executable(test src/main.cpp)
#set_property(TARGET test
#             PROPERTY CUDA_SEPARABLE_CROMPILATION ON)
target_link_libraries(test PUBLIC dedup)
