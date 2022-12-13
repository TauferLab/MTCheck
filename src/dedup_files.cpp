#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <string>
#include <vector>
#include <fstream>
#include "stdio.h"
#include "deduplicator.hpp"
//#include "dedup_approaches.hpp"

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
    // Read checkpoint files and store full paths and file names 
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

    // Hash Function
//    SHA1 hasher;
//    Murmur3C hasher;
    MD5Hash hasher;

    using Timer = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;
    Deduplicator<MD5Hash> deduplicator(chunk_size);
    // Iterate through num_chkpts
    for(uint32_t idx=0; idx<num_chkpts; idx++) {
      // Open file and read/calc important values
      std::ifstream f;
      f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
      f.open(full_chkpt_files[idx], std::ifstream::in | std::ifstream::binary);
      f.seekg(0, f.end);
      size_t data_len = f.tellg();
      f.seekg(0, f.beg);
      uint32_t num_chunks = data_len/chunk_size;
      if(num_chunks*chunk_size < data_len)
        num_chunks += 1;

      // Read checkpoint file and load it into the device
      Kokkos::View<uint8_t*> current("Current region", data_len);
      Kokkos::View<uint8_t*, Kokkos::CudaHostPinnedSpace> current_h("Current region mirror", data_len);
      f.read((char*)(current_h.data()), data_len);
      Kokkos::deep_copy(current, current_h);
      f.close();

      std::string logname = chkpt_filenames[idx];
      if(run_full) {
        std::string filename = full_chkpt_files[idx] + ".full_chkpt";
        deduplicator.checkpoint(Full, (uint8_t*)(current.data()), current.size(), filename, logname, idx==0);
      }
      if(run_naive) {
        std::string filename = full_chkpt_files[idx] + ".naivehashlist.incr_chkpt";
        deduplicator.checkpoint(Naive, (uint8_t*)(current.data()), current.size(), filename, logname, idx==0);
      }
      if(run_list) {
        std::string filename = full_chkpt_files[idx] + ".hashlist.incr_chkpt";
        deduplicator.checkpoint(List, (uint8_t*)(current.data()), current.size(), filename, logname, idx==0);
      }
      if(run_tree) {
        std::string filename = full_chkpt_files[idx] + ".hashtree.incr_chkpt";
        deduplicator.checkpoint(Tree, (uint8_t*)(current.data()), current.size(), filename, logname, idx==0);
      }
      Kokkos::fence();
    }
  }
  Kokkos::finalize();
}

