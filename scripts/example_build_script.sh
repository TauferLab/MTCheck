#!/usr/bin/env bash

cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=$HOME/kokkos/bin/nvcc_wrapper \
  -DCMAKE_INSTALL_PREFIX=$HOME/Src_Deduplication_Module/build/install \
  -DKokkos_DIR=$HOME/kokkos/build/install/lib64/cmake/Kokkos \
  ..

#  -DKokkos_DIR=$HOME/kokkos-gpu/build/install/lib64/cmake/Kokkos \
