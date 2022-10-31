#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <libgen.h>
#include <iostream>
#include "stdio.h"
#include "mpi.h"
#include "dedup_approaches.hpp"

#define WRITE_CHKPT

// Clear caches by doing unrelated work on GPU/CPU
void flush_cache() {
  uint32_t GB = 268435456;
  Kokkos::View<uint32_t*> a("A", GB);
  Kokkos::View<uint32_t*> b("B", GB);
  Kokkos::View<uint32_t*> c("C", GB);

  Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
  Kokkos::parallel_for(GB, KOKKOS_LAMBDA(const uint32_t i) {
    auto rand_gen = rand_pool.get_state();
    a(i) = static_cast<uint32_t>(rand_gen.urand() % UINT_MAX);
    b(i) = static_cast<uint32_t>(rand_gen.urand() % UINT_MAX);
    rand_pool.free_state(rand_gen);
  });
  Kokkos::parallel_for(GB, KOKKOS_LAMBDA(const uint32_t i) {
    c(i) = a(i)*b(i);
  });
}

// Read checkpoints from files and deduplicate using various approaches
// Expects full file paths
// Usage:
//   ./dedup_chkpt_files chunk_size num_chkpts [approaches] [chkpt files]
// Possible approaches
//   --run-full-chkpt   :   Simple full checkpoint strategy
//   --run-naive-chkpt  :   Deduplicate using a list of hashes. Only leverage time dimension.
//                          Compare chunks with chunks from prior chkpts at the same offset.
//   --run-list-chkpt   :   Deduplicate using a list of hashes. Deduplicates with current and
//                          past checkpoints. Compares with all possible chunks, not just at
//                          the same offset.
//   --run-tree-chkpt   :   Our deduplication approach. Takes into account time and space
//                          dimension for deduplication. Compacts metadata using forests of 
//                          Merkle trees
int main(int argc, char** argv) {
  DEBUG_PRINT("Sanity check\n");
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    using Timer = std::chrono::high_resolution_clock;
    STDOUT_PRINT("------------------------------------------------------\n");

    // Read input flags
    uint32_t chunk_size = static_cast<uint32_t>(atoi(argv[1]));
    DEBUG_PRINT("Loaded chunk size\n");
    uint32_t num_chkpts = static_cast<uint32_t>(atoi(argv[2]));
    bool run_full = false;
    bool run_naive = false;
    bool run_list = false;
    bool run_tree = false;
    uint32_t arg_offset = 0;
    for(uint32_t i=0; i<argc; i++) {
      if((strcmp(argv[i], "--run-full-chkpt") == 0)) {
        run_full = true;
        arg_offset += 1;
      } else if(strcmp(argv[i], "--run-naive-chkpt") == 0) {
        run_naive = true;
        arg_offset += 1;
      } else if(strcmp(argv[i], "--run-list-chkpt") == 0) {
        run_list = true;
        arg_offset += 1;
      } else if(strcmp(argv[i], "--run-tree-chkpt") == 0) {
        run_tree = true;
        arg_offset += 1;
      }
    }
    std::vector<std::string> chkpt_files;
    std::vector<std::string> full_chkpt_files;
    std::vector<std::string> chkpt_filenames;
    for(uint32_t i=0; i<argc; i++) {
      // Filter out input checkpoint files so that each rank only reads checkpoints for their rank
      std::string rank_str = std::string("Rank") + std::to_string(rank);
      size_t found = std::string(argv[i]).find(rank_str);
      if(found != std::string::npos) {
        DEBUG_PRINT("Argv[%u]: %s\n", i, argv[i]);
        full_chkpt_files.push_back(std::string(argv[i]));
        std::cout << "Full path: " << std::string(argv[i]) << std::endl;
        chkpt_files.push_back(std::string(argv[i]));
        chkpt_filenames.push_back(std::string(argv[i]));
        uint32_t last = chkpt_filenames.size()-1;
        size_t name_start = chkpt_filenames[last].rfind('/') + 1;
        std::cout << "Name start: " << name_start << std::endl;
        chkpt_filenames[last].erase(chkpt_filenames[last].begin(), chkpt_filenames[last].begin()+name_start);
        std::cout << "Filename: " << chkpt_filenames[last] << std::endl;
      }
    }
    DEBUG_PRINT("Read checkpoint files\n");
    DEBUG_PRINT("Number of checkpoints: %u\n", num_chkpts);
    num_chkpts = full_chkpt_files.size();

    // Hash function
//    SHA1 hasher;
//    Murmur3C hasher;
    MD5Hash hasher;

    // Full checkpoint
    if(run_full) {
      printf("=======================Full Checkpoint=======================\n");
      full_chkpt(hasher, full_chkpt_files, chkpt_filenames, chunk_size, num_chkpts);
      printf("=======================Full Checkpoint=======================\n");
      if(arg_offset > 1) {
        flush_cache();
        arg_offset -= 1;
      }
    }
    // Naive list checkpoint
    if(run_naive) {
      printf("====================Naive List Checkpoint====================\n");
      naive_chkpt(hasher, full_chkpt_files, chkpt_filenames, chunk_size, num_chkpts);
      printf("====================Naive List Checkpoint====================\n");
      if(arg_offset > 1) {
        flush_cache();
        arg_offset -= 1;
      }
    }
    // Hash list checkpoint
    if(run_list) {
      printf("====================Hash List Checkpoint ====================\n");
      list_chkpt(hasher, full_chkpt_files, chkpt_filenames, chunk_size, num_chkpts);
      printf("====================Hash List Checkpoint ====================\n");
      if(arg_offset > 1) {
        flush_cache();
        arg_offset -= 1;
      }
    }
    // Tree checkpoint
    if(run_tree) {
      printf("====================Hash Tree Checkpoint ====================\n");
      tree_chkpt(hasher, full_chkpt_files, chkpt_filenames, chunk_size, num_chkpts);
      printf("====================Hash Tree Checkpoint ====================\n");
    }
  }
  STDOUT_PRINT("------------------------------------------------------\n");
  Kokkos::finalize();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}



