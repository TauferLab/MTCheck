cmake_minimum_required(VERSION 3.12)

project(dedup VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Kokkos REQUIRED)
find_package(OpenSSL REQUIRED)

include(GNUInstallDirs)

add_executable(dedup_chkpt_files src/dedup_files.cpp)
target_include_directories(dedup_chkpt_files PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dedup_chkpt_files PRIVATE Kokkos::kokkos)
target_link_libraries(dedup_chkpt_files PRIVATE OpenSSL::SSL)
set_target_properties(dedup_chkpt_files PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(dedup_chkpt_files_mpi src/dedup_files_mpi.cpp)
target_include_directories(dedup_chkpt_files_mpi PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dedup_chkpt_files_mpi PRIVATE Kokkos::kokkos)
target_link_libraries(dedup_chkpt_files_mpi PRIVATE OpenSSL::SSL)
set_target_properties(dedup_chkpt_files_mpi PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(checkpoint_generator src/data_generation.cpp)
target_include_directories(checkpoint_generator PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(checkpoint_generator PRIVATE Kokkos::kokkos)
set_target_properties(checkpoint_generator PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(restart_chkpt_files src/restart_chkpt_files.cpp)
target_include_directories(restart_chkpt_files PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(restart_chkpt_files PRIVATE Kokkos::kokkos)
target_link_libraries(restart_chkpt_files PRIVATE OpenSSL::SSL)
set_target_properties(restart_chkpt_files PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(full_chkpt_test src/full_chkpt.cpp)
target_include_directories(full_chkpt_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(full_chkpt_test PRIVATE Kokkos::kokkos)
target_link_libraries(full_chkpt_test PRIVATE OpenSSL::SSL)
set_target_properties(full_chkpt_test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(naive_chkpt_test src/naive_chkpt.cpp)
target_include_directories(naive_chkpt_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(naive_chkpt_test PRIVATE Kokkos::kokkos)
target_link_libraries(naive_chkpt_test PRIVATE OpenSSL::SSL)
set_target_properties(naive_chkpt_test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(list_chkpt_test src/list_chkpt.cpp)
target_include_directories(list_chkpt_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(list_chkpt_test PRIVATE Kokkos::kokkos)
target_link_libraries(list_chkpt_test PRIVATE OpenSSL::SSL)
set_target_properties(list_chkpt_test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

#add_executable(tree_chkpt_test src/tree_chkpt.cpp)
#target_include_directories(tree_chkpt_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
#target_link_libraries(tree_chkpt_test PRIVATE Kokkos::kokkos)
#target_link_libraries(tree_chkpt_test PRIVATE OpenSSL::SSL)
#set_target_properties(tree_chkpt_test PROPERTIES
#    CXX_STANDARD 14
#    CXX_STANDARD_REQUIRED YES
#    CXX_EXTENSIONS OFF
#)

add_executable(test_case_1 src/test_case_1.cpp)
target_include_directories(test_case_1 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_1 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_1 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_1 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_2 src/test_case_2.cpp)
target_include_directories(test_case_2 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_2 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_2 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_2 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_3 src/test_case_3.cpp)
target_include_directories(test_case_3 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_3 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_3 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_3 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_4 src/test_case_4.cpp)
target_include_directories(test_case_4 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_4 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_4 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_4 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_5 src/test_case_5.cpp)
target_include_directories(test_case_5 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_5 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_5 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_5 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_6 src/test_case_6.cpp)
target_include_directories(test_case_6 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_6 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_6 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_6 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_7 src/test_case_7.cpp)
target_include_directories(test_case_7 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_7 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_7 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_7 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_8 src/test_case_8.cpp)
target_include_directories(test_case_8 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_8 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_8 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_8 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_9 src/test_case_9.cpp)
target_include_directories(test_case_9 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_9 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_9 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_9 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

enable_testing()
add_test(NAME full_chkpt_test COMMAND full_chkpt_test 128 10 --kokkos-threads=2 --kokkos-num-devices=1)
add_test(NAME naive_chkpt_test COMMAND naive_chkpt_test 128 10 --kokkos-threads=2 --kokkos-num-devices=1)
add_test(NAME list_chkpt_test COMMAND list_chkpt_test 128 10 --kokkos-threads=2 --kokkos-num-devices=1)
#add_test(NAME tree_chkpt_test COMMAND tree_chkpt_test 128 10 --kokkos-threads=2 --kokkos-num-devices=1)

