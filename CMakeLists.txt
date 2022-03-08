cmake_minimum_required(VERSION 3.12)

project(dedup VERSION 0.0.1)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

#find_package(Kokkos REQUIRED)

file(GLOB SOURCES src/*.cpp)

add_library(dedup ${SOURCES})

#target_link_libraries(dedup PRIVATE Kokkos::kokkos)

set_target_properties(dedup PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

set_target_properties(dedup PROPERTIES PUBLIC_HEADER include/dedup.hpp)

target_include_directories(dedup PUBLIC ${CMAKE_SOURCE_DIR}/include)

include(GNUInstallDirs)

install(TARGETS dedup 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_executable(test src/main.cpp)
target_link_libraries(test PUBLIC dedup)
