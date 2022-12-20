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

add_library(deduplicator INTERFACE)
target_include_directories(deduplicator INTERFACE 
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>
)
#target_compile_features(deduplicator INTERFACE cxx_std 14)
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_BINARY_DIR}/deduplicatorConfigVersion.cmake"
                                 VERSION 0.0.0
                                 COMPATIBILITY SameMajorVersion
                                 )
install(TARGETS deduplicator
        EXPORT deduplicatorTargets
        LIBRARY DESTINATION lib COMPONENT Runtime 
        ARCHIVE DESTINATION lib COMPONENT Development
        RUNTIME DESTINATION bin COMPONENT Runtime
        PUBLIC_HEADER DESTINATION include COMPONENT Development
        BUNDLE DESTINATION bin COMPONENT Runtime
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/deduplicatorConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/deduplicatorConfig.cmake"
  INSTALL_DESTINATION lib/cmake/deduplicator)

install(EXPORT deduplicatorTargets FILE deduplicatorTargets.cmake NAMESPACE Deduplicator:: DESTINATION lib/cmake/deduplicator)
install(FILES "${PROJECT_BINARY_DIR}/deduplicatorConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/deduplicatorConfig.cmake"
        DESTINATION lib/cmake/deduplicator)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

enable_testing()
add_subdirectory(test)
