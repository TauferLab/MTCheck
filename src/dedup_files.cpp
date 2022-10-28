#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <string>
#include <vector>
#include <fstream>
//#include <utility>
#include "stdio.h"
#include "dedup_approaches.hpp"

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

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Timer = std::chrono::high_resolution_clock;
    STDOUT_PRINT("------------------------------------------------------\n");

    // Process data from checkpoint files
    DEBUG_PRINT("Argv[1]: %s\n", argv[1]);
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
    for(uint32_t i=0; i<num_chkpts; i++) {
      full_chkpt_files.push_back(std::string(argv[3+arg_offset+i]));
      chkpt_files.push_back(std::string(argv[3+arg_offset+i]));
      chkpt_filenames.push_back(std::string(argv[3+arg_offset+i]));
      size_t name_start = chkpt_filenames[i].rfind('/') + 1;
      chkpt_filenames[i].erase(chkpt_filenames[i].begin(), chkpt_filenames[i].begin()+name_start);
    }
    DEBUG_PRINT("Read checkpoint files\n");
    DEBUG_PRINT("Number of checkpoints: %u\n", num_chkpts);

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
  Kokkos::finalize();
}

