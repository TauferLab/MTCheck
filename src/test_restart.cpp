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


//    uint32_t num_tests = 5;
    num_chkpts=1;
    uint32_t num_timers = (3+1+1);
    uint32_t select_chkpt = restart_id;
    std::vector<double> times(num_chkpts*num_timers, 0.0);

//    std::fstream result_data;
//    result_data.open(chkpt_files_trim[select_chkpt]+".timings.csv", std::fstream::out | std::fstream::app);
//    result_data << "Copy full chkpt to GPU,Restart full chkpt,Copy list chkpt to GPU,Restart list chkpt,Copy tree chkpt to GPU,Restart tree chkpt\n";

    std::fstream timing_file;
    timing_file.open(chkpt_files_trim[select_chkpt]+".chunk_size."+std::to_string(chunk_size)+".restart_timing.csv", std::fstream::out | std::fstream::app);

    for(uint32_t j=0; j<num_tests; j++) {
//      for(uint32_t i=0; i<num_chkpts; i++) {
        std::pair<double,double> full_times;
        std::pair<double,double> naive_list_times;
        std::pair<double,double> list_times;
        std::pair<double,double> tree_times;
 
        std::ifstream file;
        file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        // Full checkpoint
        file.open(chkpt_files[select_chkpt], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
        size_t filesize = file.tellg();
        file.seekg(0);
        Kokkos::View<uint8_t*> reference_d("Reference", filesize);
        Kokkos::deep_copy(reference_d, 0);
        auto reference_h = Kokkos::create_mirror_view(reference_d);
        file.close();

      if(run_full) {
        // Full checkpoint
        file.open(chkpt_files[select_chkpt], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
        size_t filesize = file.tellg();
//        printf("File size: %lu\n", filesize);
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
//        times[num_timers*i+0] += (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count());
//        times[num_timers*i+1] += (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(r2-r1).count());
//        times[num_timers*i+2] += (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
       timing_file << "Full" << "," << select_chkpt << "," << chunk_size << "," << full_times.first << "," << full_times.second << std::endl;

        file.close();

#ifdef VERIFY_OUTPUT
        HashDigest correct;
        MD5((uint8_t*)(reference_h.data()), filesize, correct.digest);
        static const char hexchars[] = "0123456789ABCDEF";
        std::string ref_digest;
        for(int k=0; k<16; k++) {
          unsigned char b = correct.digest[k];
          char hex[3];
          hex[0] = hexchars[b >> 4];
          hex[1] = hexchars[b & 0xF];
          hex[2] = 0;
          ref_digest.append(hex);
          if(k%4 == 3)
            ref_digest.append(" ");
        }
        std::cout << "Correct digest:        " << ref_digest << std::endl;
#endif
       
       Kokkos::deep_copy(reference_d, 0);
       Kokkos::deep_copy(reference_h, 0);
      if(arg_offset > 1) {
        flush_cache();
        arg_offset -= 1;
      }
     }
       //====================================================================
       // Incremental checkpoint (Hash list)
     if(run_naive) {
       Kokkos::deep_copy(reference_d, 0);
       std::chrono::high_resolution_clock::time_point n1 = std::chrono::high_resolution_clock::now();
       naive_list_times = restart_incr_chkpt_naivehashlist(naivehashlist_chkpt_files, select_chkpt, reference_d);
       std::chrono::high_resolution_clock::time_point n2 = std::chrono::high_resolution_clock::now();
       timing_file << "Naive" << "," << select_chkpt << "," << chunk_size << "," << naive_list_times.first << "," << naive_list_times.second << std::endl;

#ifdef VERIFY_OUTPUT
        reference_h = Kokkos::create_mirror_view(reference_d);
        Kokkos::deep_copy(reference_h, reference_d);
        HashDigest naive_hashlist;
        MD5((uint8_t*)(reference_h.data()), filesize, naive_hashlist.digest);
        static const char hexchars[] = "0123456789ABCDEF";
        std::string naive_list_digest;
        for(int k=0; k<16; k++) {
          unsigned char b = naive_hashlist.digest[k];
          char hex[3];
          hex[0] = hexchars[b >> 4];
          hex[1] = hexchars[b & 0xF];
          hex[2] = 0;
          naive_list_digest.append(hex);
          if(k%4 == 3)
            naive_list_digest.append(" ");
        }
        std::cout << "Naive Hashlist digest: " << naive_list_digest << std::endl;
#endif
      if(arg_offset > 1) {
        flush_cache();
        arg_offset -= 1;
      }
    }
       //====================================================================
       // Incremental checkpoint (Hash list)
    if(run_list) {
       Kokkos::deep_copy(reference_d, 0);
       std::chrono::high_resolution_clock::time_point l1 = std::chrono::high_resolution_clock::now();
       list_times = restart_incr_chkpt_hashlist(hashlist_chkpt_files, select_chkpt, reference_d);
       std::chrono::high_resolution_clock::time_point l2 = std::chrono::high_resolution_clock::now();
//       times[num_timers*i+3] += (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(l2-l1).count());
        timing_file << "List" << "," << select_chkpt << "," << chunk_size << "," << list_times.first << "," << list_times.second << std::endl;

#ifdef VERIFY_OUTPUT
//        printf("Memory size: %lu\n", reference_d.size());
        reference_h = Kokkos::create_mirror_view(reference_d);
        Kokkos::deep_copy(reference_h, reference_d);
//printf("Copied data to host\n");
        HashDigest hashlist;
        MD5((uint8_t*)(reference_h.data()), filesize, hashlist.digest);
        static const char hexchars[] = "0123456789ABCDEF";
//printf("Computed list digest\n");
        std::string list_digest;
        for(int k=0; k<16; k++) {
          unsigned char b = hashlist.digest[k];
          char hex[3];
          hex[0] = hexchars[b >> 4];
          hex[1] = hexchars[b & 0xF];
          hex[2] = 0;
          list_digest.append(hex);
          if(k%4 == 3)
            list_digest.append(" ");
        }
        std::cout << "Hashlist digest:       " << list_digest << std::endl;
#endif
      if(arg_offset > 1) {
        flush_cache();
        arg_offset -= 1;
      }
      }
      if(run_tree) {
//printf("Incremental checkpoint hash tree\n");
        // Incremental checkpoint (Hash tree)
        Kokkos::deep_copy(reference_d, 0);
        std::chrono::high_resolution_clock::time_point m1 = std::chrono::high_resolution_clock::now();
        tree_times = restart_incr_chkpt_hashtree(hashtree_chkpt_files, select_chkpt, reference_d);
        std::chrono::high_resolution_clock::time_point m2 = std::chrono::high_resolution_clock::now();
//        times[num_timers*i+4] += (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(m2-m1).count());

#ifdef VERIFY_OUTPUT
//        printf("Memory size: %lu\n", reference_d.size());
        Kokkos::deep_copy(reference_h, reference_d);
        HashDigest hashtree;
        MD5((uint8_t*)(reference_h.data()), filesize, hashtree.digest);
        static const char hexchars[] = "0123456789ABCDEF";
        std::string tree_digest;
        for(int k=0; k<16; k++) {
          unsigned char b = hashtree.digest[k];
          char hex[3];
          hex[0] = hexchars[b >> 4];
          hex[1] = hexchars[b & 0xF];
          hex[2] = 0;
          tree_digest.append(hex);
          if(k%4 == 3)
            tree_digest.append(" ");
        }
        std::cout << "Hashtree digest:       " << tree_digest << std::endl;
#endif
        timing_file << "Tree" << "," << select_chkpt << "," << chunk_size << "," << tree_times.first << "," << tree_times.second << std::endl;
      if(arg_offset > 1) {
        flush_cache();
        arg_offset -= 1;
      }
      }
      STDOUT_PRINT("Restarted checkpoint\n");
      
//      result_data << full_times.first << "," << full_times.second << "," 
//                  << naive_list_times.first << "," << naive_list_times.second << "," 
//                  << list_times.first << "," << list_times.second << "," 
//                  << tree_times.first << "," << tree_times.second << std::endl;
    }
  timing_file.close();
  }
  Kokkos::finalize();
}
