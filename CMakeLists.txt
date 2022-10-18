cmake_minimum_required(VERSION 3.12)

project(dedup VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Kokkos REQUIRED)
find_package(OpenSSL REQUIRED)

include(GNUInstallDirs)
file(GLOB DEDUP_SOURCES src/test_dedup_files.cpp src/hash_functions.cpp)

#add_executable(merkle_tree_test src/test_dedup.cpp)
#target_include_directories(merkle_tree_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
#target_link_libraries(merkle_tree_test PRIVATE Kokkos::kokkos)
#
#set_target_properties(merkle_tree_test PROPERTIES
#    CXX_STANDARD 14
#    CXX_STANDARD_REQUIRED YES
#    CXX_EXTENSIONS OFF
#)

#add_executable(merkle_tree_test_files src/test_dedup_files.cpp)
add_executable(merkle_tree_test_files ${DEDUP_SOURCES})
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
#add_executable(checkpoint_generator ${SOURCES})
target_include_directories(checkpoint_generator PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(checkpoint_generator PRIVATE Kokkos::kokkos)

set_target_properties(checkpoint_generator PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_restart src/test_restart.cpp)
#add_executable(test_restart ${SOURCES})
target_include_directories(test_restart PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_restart PRIVATE Kokkos::kokkos)
target_link_libraries(test_restart PRIVATE OpenSSL::SSL)

set_target_properties(test_restart PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

