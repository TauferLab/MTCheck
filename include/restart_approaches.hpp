#ifndef RESTART_APPROACHES_HPP
#define RESTART_APPROACHES_HPP

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
                   uint32_t select_chkpt,
                   bool verify
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

  if(verify) {
    std::string digest = calculate_digest_host(reference_h);
    std::cout << "Full chkpt digest:     " << digest << std::endl;
  }

  return full_times;
}

std::pair<double,double> 
restart_naive_list_chkpt(
                         std::vector<std::vector<uint8_t>>& incr_chkpts, 
                         Kokkos::View<uint8_t*>& reference_d,
                         uint32_t chunk_size, 
                         uint32_t select_chkpt,
                         int verify)  {
  std::pair<double,double> naive_list_times;
  size_t filesize = incr_chkpts[select_chkpt].size();
  Kokkos::deep_copy(reference_d, 0);
  std::chrono::high_resolution_clock::time_point n1 = std::chrono::high_resolution_clock::now();
//  naive_list_times = restart_incr_chkpt_naivehashlist(chkpt_files, select_chkpt, reference_d);
  naive_list_times = restart_incr_chkpt_naivehashlist(incr_chkpts, select_chkpt, reference_d);
  std::chrono::high_resolution_clock::time_point n2 = std::chrono::high_resolution_clock::now();

  if(verify) {
    auto reference_h = Kokkos::create_mirror_view(reference_d);
    std::string digest = calculate_digest_host(reference_h);
    std::cout << "Naive Hashlist digest: " << digest << std::endl;
  }
  return naive_list_times;
}

std::pair<double,double> 
restart_naive_list_chkpt(
                   std::vector<std::string>& chkpt_files, 
                   uint32_t chunk_size, 
                   uint32_t select_chkpt,
                   bool verify
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

  if(verify) {
    auto reference_h = Kokkos::create_mirror_view(reference_d);
    std::string digest = calculate_digest_host(reference_h);
    std::cout << "Naive Hashlist digest: " << digest << std::endl;
  }
  return naive_list_times;
}

std::pair<double,double> 
restart_list_chkpt(
                         std::vector<std::vector<uint8_t>>& incr_chkpts, 
                         Kokkos::View<uint8_t*>& reference_d,
                         uint32_t chunk_size, 
                         uint32_t select_chkpt,
                         int verify)  {
  std::pair<double,double> list_times;
  size_t filesize = incr_chkpts[select_chkpt].size();
  Kokkos::deep_copy(reference_d, 0);
  std::chrono::high_resolution_clock::time_point n1 = std::chrono::high_resolution_clock::now();
  list_times = restart_incr_chkpt_hashlist(incr_chkpts, select_chkpt, reference_d);
  std::chrono::high_resolution_clock::time_point n2 = std::chrono::high_resolution_clock::now();

  if(verify) {
    auto reference_h = Kokkos::create_mirror_view(reference_d);
    std::string digest = calculate_digest_host(reference_h);
    std::cout << "Hash list digest: " << digest << std::endl;
  }
  return list_times;
}

std::pair<double,double> 
restart_tree_chkpt(
                         std::vector<std::vector<uint8_t>>& incr_chkpts, 
                         Kokkos::View<uint8_t*>& reference_d,
                         uint32_t chunk_size, 
                         uint32_t select_chkpt,
                         int verify)  {
  std::pair<double,double> list_times;
  size_t filesize = incr_chkpts[select_chkpt].size();
  Kokkos::deep_copy(reference_d, 0);
  std::chrono::high_resolution_clock::time_point n1 = std::chrono::high_resolution_clock::now();
  list_times = restart_incr_chkpt_hashtree(incr_chkpts, select_chkpt, reference_d);
  std::chrono::high_resolution_clock::time_point n2 = std::chrono::high_resolution_clock::now();

  if(verify) {
    auto reference_h = Kokkos::create_mirror_view(reference_d);
    std::string digest = calculate_digest_host(reference_h);
    std::cout << "Hash list digest: " << digest << std::endl;
  }
  return list_times;
}

#endif // RESTART_APPROACHES_HPP

