#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include "stdio.h"
#include <string>
#include <map>
#include <fstream>
#include "hash_functions.hpp"
//#include "kokkos_merkle_tree.hpp"
#include "restart_merkle_tree.hpp"
#include "kokkos_hash_list.hpp"
#include "update_pattern_analysis.hpp"
#include "restart_approaches.hpp"
#include "deduplicator.hpp"
#include <libgen.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <utility>
#include <openssl/md5.h>
#include "utils.hpp"

#define VERIFY_OUTPUT

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
  DEBUG_PRINT("Sanity check\n");
  Kokkos::initialize(argc, argv);
  {
    using Timer = std::chrono::high_resolution_clock;
    STDOUT_PRINT("------------------------------------------------------\n");

    // Process data from checkpoint files
    uint32_t restart_id = static_cast<uint32_t>(atoi(argv[1]));
    uint32_t num_chkpts = static_cast<uint32_t>(atoi(argv[2]));
    uint32_t num_tests = static_cast<uint32_t>(atoi(argv[3]));
    uint32_t chunk_size = static_cast<uint32_t>(atoi(argv[4]));
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
    std::vector<std::string> chkpt_files_trim;
    for(uint32_t i=0; i<num_chkpts; i++) {
      chkpt_files.push_back(std::string(argv[5+arg_offset+i]));
      std::string filename = std::string(argv[5+arg_offset+i]);
      size_t name_start = filename.rfind('/') + 1;
      filename.erase(filename.begin(), filename.begin()+name_start);
      chkpt_files_trim.push_back(filename);
    }
    std::vector<std::string> full_chkpt_files;
    for(uint32_t i=0; i<num_chkpts; i++) {
      full_chkpt_files.push_back(chkpt_files[i]+".full_chkpt");
    }
    std::vector<std::string> naivehashlist_chkpt_files;
    for(uint32_t i=0; i<num_chkpts; i++) {
      naivehashlist_chkpt_files.push_back(chkpt_files[i]+".naivehashlist.incr_chkpt");
    }
    std::vector<std::string> hashlist_chkpt_files;
    for(uint32_t i=0; i<num_chkpts; i++) {
      hashlist_chkpt_files.push_back(chkpt_files[i]+".hashlist.incr_chkpt");
    }
    std::vector<std::string> hashtree_chkpt_files;
    for(uint32_t i=0; i<num_chkpts; i++) {
      hashtree_chkpt_files.push_back(chkpt_files[i]+".hashtree.incr_chkpt");
    }
    STDOUT_PRINT("Read checkpoint files\n");
    STDOUT_PRINT("Number of checkpoints: %u\n", num_chkpts);


    uint32_t select_chkpt = restart_id;

    std::fstream timing_file;
    timing_file.open(chkpt_files_trim[select_chkpt]+".chunk_size."+std::to_string(chunk_size)+".restart_timing.csv", std::fstream::out | std::fstream::app);

    for(uint32_t j=0; j<num_tests; j++) {
      std::ifstream file;
      file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

      // Full checkpoint
      file.open(chkpt_files[select_chkpt], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
      size_t filesize = file.tellg();
      file.seekg(0);
      Kokkos::View<uint8_t*> reference_d("Reference View", filesize);
      Kokkos::deep_copy(reference_d, 0);
      auto reference_h = Kokkos::create_mirror_view(reference_d);
      file.close();

      if(run_full) {
        // Full checkpoint
        std::vector<Kokkos::View<uint8_t*>::HostMirror> chkpts;
        for(uint32_t i=0; i<num_chkpts; i++) {
          std::fstream file;
          auto fileflags = std::ifstream::in | std::ifstream::binary | std::ifstream::ate;
          file.open(full_chkpt_files[i], fileflags);
          size_t filesize = file.tellg();
          file.seekg(0);
          Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace> chkpt("Chkpt", filesize);
          Kokkos::deep_copy(chkpt, 0);
          file.read((char*)(chkpt.data()), filesize);
          file.close();
          chkpts.push_back(chkpt);
        }
        std::string logname = chkpt_files_trim[select_chkpt]+".chunk_size."+std::to_string(chunk_size)+".restart_timing.csv";
        Deduplicator<MD5Hash> deduplicator(chunk_size);
        deduplicator.restart(Full, reference_d, chkpts, logname, select_chkpt);
#ifdef VERIFY_OUTPUT
        Kokkos::deep_copy(reference_h, reference_d);
        std::string digest = calculate_digest_host(reference_h);
        std::cout << "Full chkpt digest:     " << digest << std::endl;
#endif
      }
      //====================================================================
      // Incremental checkpoint (Hash list)
      if(run_naive) {
        std::vector<Kokkos::View<uint8_t*>::HostMirror> chkpts;
        for(uint32_t i=0; i<num_chkpts; i++) {
          std::fstream file;
          auto fileflags = std::ifstream::in | std::ifstream::binary | std::ifstream::ate;
          file.open(naivehashlist_chkpt_files[i], fileflags);
          size_t filesize = file.tellg();
          file.seekg(0);
          Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace> chkpt("Chkpt", filesize);
          Kokkos::deep_copy(chkpt, 0);
          file.read((char*)(chkpt.data()), filesize);
          file.close();
          chkpts.push_back(chkpt);
        }

        std::string logname = chkpt_files_trim[select_chkpt]+
                              ".chunk_size."+std::to_string(chunk_size)+".restart_timing.csv";
        Deduplicator<MD5Hash> deduplicator(chunk_size);
        deduplicator.restart(Naive, reference_d, chkpts, logname, select_chkpt);
#ifdef VERIFY_OUTPUT
        Kokkos::deep_copy(reference_h, reference_d);
        std::string digest = calculate_digest_host(reference_h);
        std::cout << "Naive Hashlist digest: " << digest << std::endl;
#endif
      }
      //====================================================================
      // Incremental checkpoint (Hash list)
      if(run_list) {
        std::vector<Kokkos::View<uint8_t*>::HostMirror> chkpts;
        for(uint32_t i=0; i<num_chkpts; i++) {
          std::fstream file;
          auto fileflags = std::ifstream::in | std::ifstream::binary | std::ifstream::ate;
          file.open(hashlist_chkpt_files[i], fileflags);
          size_t filesize = file.tellg();
          file.seekg(0);
          Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace> chkpt("Chkpt", filesize);
          Kokkos::deep_copy(chkpt, 0);
          file.read((char*)(chkpt.data()), filesize);
          file.close();
          chkpts.push_back(chkpt);
        }

        std::string logname = chkpt_files_trim[select_chkpt]+".chunk_size."+std::to_string(chunk_size)+".restart_timing.csv";
        Deduplicator<MD5Hash> deduplicator(chunk_size);
        deduplicator.restart(List, reference_d, chkpts, logname, select_chkpt);
#ifdef VERIFY_OUTPUT
        Kokkos::deep_copy(reference_h, reference_d);
        std::string digest = calculate_digest_host(reference_h);
        std::cout << "Hashlist digest:       " << digest << std::endl;
#endif
      }
      if(run_tree) {
        // Incremental checkpoint (Hash tree)
        std::vector<Kokkos::View<uint8_t*>::HostMirror> chkpts;
        for(uint32_t i=0; i<num_chkpts; i++) {
          std::fstream file;
          auto fileflags = std::ifstream::in | std::ifstream::binary | std::ifstream::ate;
          file.open(hashtree_chkpt_files[i], fileflags);
          size_t filesize = file.tellg();
          file.seekg(0);
          Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace> chkpt("Chkpt", filesize);
          Kokkos::deep_copy(chkpt, 0);
          file.read((char*)(chkpt.data()), filesize);
          file.close();
          chkpts.push_back(chkpt);
        }

        std::string logname = chkpt_files_trim[select_chkpt]+".chunk_size."+std::to_string(chunk_size)+".restart_timing.csv";
        Deduplicator<MD5Hash> deduplicator(chunk_size);
        deduplicator.restart(Tree, reference_d, chkpts, logname, select_chkpt);
#ifdef VERIFY_OUTPUT
        Kokkos::deep_copy(reference_h, reference_d);
        std::string digest = calculate_digest_host(reference_h);
        std::cout << "Hashtree digest:       " << digest << std::endl;
#endif
      }
      STDOUT_PRINT("Restarted checkpoint\n");
    }
    timing_file.close();
  }
  Kokkos::finalize();
}
