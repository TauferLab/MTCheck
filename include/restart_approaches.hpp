#ifndef DEDUP_APPROACHES_HPP
#define DEDUP_APPROACHES_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include "stdio.h"
#include <libgen.h>
#include <openssl/md5.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <utility>
#include <string>
//#include <map>
#include <fstream>
#include "hash_functions.hpp"
#include "restart_merkle_tree.hpp"
#include "kokkos_hash_list.hpp"
#include "utils.hpp"

#define VERIFY_OUTPUT

std::pair<double,double> 
restart_full_chkpt(
                   std::vector<std::string>& chkpt_files, 
                   uint32_t chunk_size, 
                   uint32_t select_chkpt
                   )  {
  std::pair<double,double> full_times;
  std::fstream file;
  // Full checkpoint
  file.open(chkpt_files[select_chkpt], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
  size_t filesize = file.tellg();
  file.seekg(0);
  Kokkos::View<uint8_t*> reference_d("Reference", filesize);
  Kokkos::deep_copy(reference_d, 0);
  auto reference_h = Kokkos::create_mirror_view(reference_d);
  // Total time
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  // Read checkpoint
  std::chrono::high_resolution_clock::time_point r1 = std::chrono::high_resolution_clock::now();
  file.read((char*)(reference_h.data()), filesize);
  std::chrono::high_resolution_clock::time_point r2 = std::chrono::high_resolution_clock::now();
  // Copy checkpoint to GPU
  std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();
  Kokkos::deep_copy(reference_d, reference_h);
  std::chrono::high_resolution_clock::time_point c2 = std::chrono::high_resolution_clock::now();
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  // Update timers
  full_times.first = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
  full_times.second = 0.0;

  file.close();

  #ifdef VERIFY_OUTPUT
  std::string digest = calculate_digest_host(reference_h);
  std::cout << "Full chkpt digest:     " << digest << std::endl;
  #endif

  return full_times;
}

//std::pair<double,double> 
//restart_naive_list_chkpt(
//                   std::vector<Kokkos::View<uint8_t*>>& incr_chkpts, 
//                   uint32_t chunk_size, 
//                   uint32_t select_chkpt
//                   )  {
//  std::pair<double,double> naive_list_times;
//  size_t filesize = incr_chkpts[select_chkpt].extent(0);
//  Kokkos::View<uint8_t*> reference_d("Reference", filesize);
//  Kokkos::deep_copy(reference_d, 0);
//  std::chrono::high_resolution_clock::time_point n1 = std::chrono::high_resolution_clock::now();
////  naive_list_times = restart_incr_chkpt_naivehashlist(chkpt_files, select_chkpt, reference_d);
//  naive_list_times = restart_incr_chkpt_naivehashlist(chkpt_files, select_chkpt, reference_d);
//  std::chrono::high_resolution_clock::time_point n2 = std::chrono::high_resolution_clock::now();
//
//  #ifdef VERIFY_OUTPUT
//  auto reference_h = Kokkos::create_mirror_view(reference_d);
//  std::string digest = calculate_digest_host(reference_h);
//  std::cout << "Naive Hashlist digest: " << digest << std::endl;
//  #endif
//  return naive_list_times;
//}

std::pair<double,double> 
restart_naive_list_chkpt(
                   std::vector<std::string>& chkpt_files, 
                   uint32_t chunk_size, 
                   uint32_t select_chkpt
                   )  {
  std::pair<double,double> naive_list_times;
  std::fstream file;
  file.open(chkpt_files[select_chkpt], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
  size_t filesize = file.tellg();
  file.seekg(0);
  Kokkos::View<uint8_t*> reference_d("Reference", filesize);
  Kokkos::deep_copy(reference_d, 0);
  std::chrono::high_resolution_clock::time_point n1 = std::chrono::high_resolution_clock::now();
  naive_list_times = restart_incr_chkpt_naivehashlist(chkpt_files, select_chkpt, reference_d);
  std::chrono::high_resolution_clock::time_point n2 = std::chrono::high_resolution_clock::now();

  #ifdef VERIFY_OUTPUT
  auto reference_h = Kokkos::create_mirror_view(reference_d);
  std::string digest = calculate_digest_host(reference_h);
  std::cout << "Naive Hashlist digest: " << digest << std::endl;
  #endif
  return naive_list_times;
}

#endif // DEDUP_APPROACHES_HPP

