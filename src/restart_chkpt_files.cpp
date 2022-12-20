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
    DedupMode mode = get_mode(argc, argv);
    if(mode == Unknown) {
      printf("ERROR: Incorrect mode\n");
      print_mode_help();
    }
    uint32_t arg_offset = 1;
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
    std::vector<std::string> basic_chkpt_files;
    for(uint32_t i=0; i<num_chkpts; i++) {
      basic_chkpt_files.push_back(chkpt_files[i]+".basic.incr_chkpt");
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

      if(mode == Full) {
        //====================================================================
        // Full checkpoint
        //====================================================================
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
        std::string logname = chkpt_files_trim[select_chkpt];
        Deduplicator<MD5Hash> deduplicator(chunk_size);
        deduplicator.restart(Full, reference_d, chkpts, logname, select_chkpt);
#ifdef VERIFY_OUTPUT
        Kokkos::deep_copy(reference_h, reference_d);
        std::string digest = calculate_digest_host(reference_h);
        std::cout << "Full chkpt digest:     " << digest << std::endl;
#endif
      } else if(mode == Basic) {
        //====================================================================
        // Basic Incremental checkpoint 
        //====================================================================
        std::vector<Kokkos::View<uint8_t*>::HostMirror> chkpts;
        for(uint32_t i=0; i<num_chkpts; i++) {
          std::fstream file;
          auto fileflags = std::ifstream::in | std::ifstream::binary | std::ifstream::ate;
          file.open(basic_chkpt_files[i], fileflags);
          size_t filesize = file.tellg();
          file.seekg(0);
          Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace> chkpt("Chkpt", filesize);
          Kokkos::deep_copy(chkpt, 0);
          file.read((char*)(chkpt.data()), filesize);
          file.close();
          chkpts.push_back(chkpt);
        }

        std::string logname = chkpt_files_trim[select_chkpt];
        Deduplicator<MD5Hash> deduplicator(chunk_size);
        deduplicator.restart(Basic, reference_d, chkpts, logname, select_chkpt);
#ifdef VERIFY_OUTPUT
        Kokkos::deep_copy(reference_h, reference_d);
        std::string digest = calculate_digest_host(reference_h);
        std::cout << "Basic Hashlist digest: " << digest << std::endl;
#endif
      } else if(mode == List) {
        //====================================================================
        // Incremental checkpoint (Hash list)
        //====================================================================
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

        std::string logname = chkpt_files_trim[select_chkpt];
        Deduplicator<MD5Hash> deduplicator(chunk_size);
        deduplicator.restart(List, reference_d, chkpts, logname, select_chkpt);
#ifdef VERIFY_OUTPUT
        Kokkos::deep_copy(reference_h, reference_d);
        std::string digest = calculate_digest_host(reference_h);
        std::cout << "Hashlist digest:       " << digest << std::endl;
#endif
      } else {
        //====================================================================
        // Incremental checkpoint (Hash tree)
        //====================================================================
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

        std::string logname = chkpt_files_trim[select_chkpt];
        Deduplicator<MD5Hash> deduplicator(chunk_size);
        deduplicator.restart(mode, reference_d, chkpts, logname, select_chkpt);
#ifdef VERIFY_OUTPUT
        Kokkos::deep_copy(reference_h, reference_d);
        std::string digest = calculate_digest_host(reference_h);
        std::cout << "Hashtree digest:       " << digest << std::endl;
#endif
      }
      STDOUT_PRINT("Restarted checkpoint\n");
    }
  }
  Kokkos::finalize();
}
