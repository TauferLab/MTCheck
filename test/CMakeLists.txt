add_executable(full_chkpt_test full_chkpt.cpp)
target_include_directories(full_chkpt_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(full_chkpt_test PRIVATE Kokkos::kokkos)
target_link_libraries(full_chkpt_test PRIVATE OpenSSL::SSL)
set_target_properties(full_chkpt_test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(basic_chkpt_test basic_chkpt.cpp)
target_include_directories(basic_chkpt_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(basic_chkpt_test PRIVATE Kokkos::kokkos)
target_link_libraries(basic_chkpt_test PRIVATE OpenSSL::SSL)
set_target_properties(basic_chkpt_test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(list_chkpt_test list_chkpt.cpp)
target_include_directories(list_chkpt_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(list_chkpt_test PRIVATE Kokkos::kokkos)
target_link_libraries(list_chkpt_test PRIVATE OpenSSL::SSL)
set_target_properties(list_chkpt_test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(tree_chkpt_test tree_chkpt.cpp)
target_include_directories(tree_chkpt_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tree_chkpt_test PRIVATE Kokkos::kokkos)
target_link_libraries(tree_chkpt_test PRIVATE OpenSSL::SSL)
set_target_properties(tree_chkpt_test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_01 test_case_01.cpp)
target_include_directories(test_case_01 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_01 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_01 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_01 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_02 test_case_02.cpp)
target_include_directories(test_case_02 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_02 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_02 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_02 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_03 test_case_03.cpp)
target_include_directories(test_case_03 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_03 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_03 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_03 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_04 test_case_04.cpp)
target_include_directories(test_case_04 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_04 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_04 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_04 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_05 test_case_05.cpp)
target_include_directories(test_case_05 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_05 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_05 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_05 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_06 test_case_06.cpp)
target_include_directories(test_case_06 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_06 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_06 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_06 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_07 test_case_07.cpp)
target_include_directories(test_case_07 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_07 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_07 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_07 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_08 test_case_08.cpp)
target_include_directories(test_case_08 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_08 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_08 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_08 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_09 test_case_09.cpp)
target_include_directories(test_case_09 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_09 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_09 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_09 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_executable(test_case_10 test_case_10.cpp)
target_include_directories(test_case_10 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_case_10 PRIVATE Kokkos::kokkos)
target_link_libraries(test_case_10 PRIVATE OpenSSL::SSL)
set_target_properties(test_case_10 PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS OFF
)

add_test(NAME full_chkpt_test COMMAND full_chkpt_test 128 10 --kokkos-threads=2 --kokkos-num-devices=1)
add_test(NAME basic_chkpt_test COMMAND basic_chkpt_test 128 10 --kokkos-threads=2 --kokkos-num-devices=1)
add_test(NAME list_chkpt_test COMMAND list_chkpt_test 128 10 --kokkos-threads=2 --kokkos-num-devices=1)
add_test(NAME tree_chkpt_test COMMAND tree_chkpt_test 128 10 --kokkos-threads=2 --kokkos-num-devices=1)
add_test(NAME tree_test_case_01 COMMAND test_case_01 --kokkos-num-threads=2 --kokkos-num-devices=1)
add_test(NAME tree_test_case_02 COMMAND test_case_02 --kokkos-num-threads=2 --kokkos-num-devices=1)
add_test(NAME tree_test_case_03 COMMAND test_case_03 --kokkos-num-threads=2 --kokkos-num-devices=1)
add_test(NAME tree_test_case_04 COMMAND test_case_04 --kokkos-num-threads=2 --kokkos-num-devices=1)
add_test(NAME tree_test_case_05 COMMAND test_case_05 --kokkos-num-threads=2 --kokkos-num-devices=1)
add_test(NAME tree_test_case_06 COMMAND test_case_06 --kokkos-num-threads=2 --kokkos-num-devices=1)
add_test(NAME tree_test_case_07 COMMAND test_case_07 --kokkos-num-threads=2 --kokkos-num-devices=1)
add_test(NAME tree_test_case_08 COMMAND test_case_08 --kokkos-num-threads=2 --kokkos-num-devices=1)
add_test(NAME tree_test_case_09 COMMAND test_case_09 --kokkos-num-threads=2 --kokkos-num-devices=1)
add_test(NAME tree_test_case_10 COMMAND test_case_10 --kokkos-num-threads=2 --kokkos-num-devices=1)
