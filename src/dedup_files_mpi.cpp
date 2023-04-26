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

// Read checkpoints from files and deduplicate using various approaches
// Expects full file paths
// Usage:
//   ./dedup_chkpt_files chunk_size num_chkpts [approaches] [chkpt files]
// Possible approaches
//   --run-full-chkpt   :   Simple full checkpoint strategy
//   --run-basic-chkpt  :   Deduplicate using a list of hashes. Only leverage time dimension.
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
MPI_Barrier(MPI_COMM_WORLD);
    printf("MPI COMM SIZE: %d, MPI_RANK: %d\n", comm_size, rank);

//    using Timer = std::chrono::high_resolution_clock;
//    STDOUT_PRINT("------------------------------------------------------\n");
//
//    // Read input flags
//    uint32_t chunk_size = static_cast<uint32_t>(atoi(argv[1]));
//    DEBUG_PRINT("Loaded chunk size\n");
//    uint32_t num_chkpts = static_cast<uint32_t>(atoi(argv[2]));
//    bool run_full = false;
//    bool run_basic = false;
//    bool run_list = false;
//    bool run_tree = false;
//    uint32_t arg_offset = 0;
//    for(uint32_t i=0; i<argc; i++) {
//      if((strcmp(argv[i], "--run-full-chkpt") == 0)) {
//        run_full = true;
//        arg_offset += 1;
//      } else if(strcmp(argv[i], "--run-basic-chkpt") == 0) {
//        run_basic = true;
//        arg_offset += 1;
//      } else if(strcmp(argv[i], "--run-list-chkpt") == 0) {
//        run_list = true;
//        arg_offset += 1;
//      } else if(strcmp(argv[i], "--run-tree-chkpt") == 0) {
//        run_tree = true;
//        arg_offset += 1;
//      }
//    }
//    std::vector<std::string> chkpt_files;
//    std::vector<std::string> full_chkpt_files;
//    std::vector<std::string> chkpt_filenames;
//    for(uint32_t i=0; i<argc; i++) {
//      // Filter out input checkpoint files so that each rank only reads checkpoints for their rank
//      std::string rank_str = std::string("Rank") + std::to_string(rank);
//      size_t found = std::string(argv[i]).find(rank_str);
//      if(found != std::string::npos) {
//        DEBUG_PRINT("Argv[%u]: %s\n", i, argv[i]);
//        full_chkpt_files.push_back(std::string(argv[i]));
//        std::cout << "Full path: " << std::string(argv[i]) << std::endl;
//        chkpt_files.push_back(std::string(argv[i]));
//        chkpt_filenames.push_back(std::string(argv[i]));
//        uint32_t last = chkpt_filenames.size()-1;
//        size_t name_start = chkpt_filenames[last].rfind('/') + 1;
//        std::cout << "Name start: " << name_start << std::endl;
//        chkpt_filenames[last].erase(chkpt_filenames[last].begin(), chkpt_filenames[last].begin()+name_start);
//        std::cout << "Filename: " << chkpt_filenames[last] << std::endl;
//      }
//    }
//    DEBUG_PRINT("Read checkpoint files\n");
//    DEBUG_PRINT("Number of checkpoints: %u\n", num_chkpts);
//    num_chkpts = full_chkpt_files.size();
//
//    // Hash function
////    SHA1 hasher;
////    Murmur3C hasher;
//    MD5Hash hasher;
//
//    // Full checkpoint
//    if(run_full) {
//      printf("=======================Full Checkpoint=======================\n");
//      full_chkpt(hasher, full_chkpt_files, chkpt_filenames, chunk_size, num_chkpts);
//      printf("=======================Full Checkpoint=======================\n");
//    }
//    // Basic list checkpoint
//    if(run_basic) {
//      printf("====================Basic List Checkpoint====================\n");
//      basic_chkpt(hasher, full_chkpt_files, chkpt_filenames, chunk_size, num_chkpts);
//      printf("====================Basic List Checkpoint====================\n");
//    }
//    // Hash list checkpoint
//    if(run_list) {
//      printf("====================Hash List Checkpoint ====================\n");
//      list_chkpt(hasher, full_chkpt_files, chkpt_filenames, chunk_size, num_chkpts);
//      printf("====================Hash List Checkpoint ====================\n");
//    }
//    // Tree checkpoint
//    if(run_tree) {
//      printf("====================Hash Tree Checkpoint ====================\n");
//      tree_chkpt(hasher, full_chkpt_files, chkpt_filenames, chunk_size, num_chkpts);
//      printf("====================Hash Tree Checkpoint ====================\n");
//    }
    using Timer = std::chrono::high_resolution_clock;
    STDOUT_PRINT("------------------------------------------------------\n");

    // Process data from checkpoint files
    printf("Argv[1]: %s\n", argv[1]);
    printf("Argv[2]: %s\n", argv[2]);
    uint32_t chunk_size = static_cast<uint32_t>(atoi(argv[1]));
    uint32_t num_chkpts = static_cast<uint32_t>(atoi(argv[2]));
    DedupMode mode = get_mode(argc, argv);
    if(mode == Unknown) {
      printf("ERROR: Incorrect mode\n");
      print_mode_help();
    }
    uint32_t arg_offset = 1;
    // Read checkpoint files and store full paths and file names 
    std::vector<std::string> chkpt_files;
    std::vector<std::string> full_chkpt_files;
    std::vector<std::string> chkpt_filenames;
    MPI_Barrier(MPI_COMM_WORLD);
    for(uint32_t i=0; i<num_chkpts; i++) {
      // Filter out input checkpoint files so that each rank only reads checkpoints for their rank
      std::string rank_str = std::string("Rank") + std::to_string(rank);
      size_t found = std::string(argv[3+arg_offset+i]).find(rank_str);
      if(found != std::string::npos) {
        full_chkpt_files.push_back(std::string(argv[3+arg_offset+i]));
        chkpt_files.push_back(std::string(argv[3+arg_offset+i]));
        std::string file_str = std::string(argv[3+arg_offset+i]);
        size_t name_start = file_str.rfind('/') + 1;
        chkpt_filenames.push_back(std::string(argv[3+arg_offset+i]+name_start));
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    DEBUG_PRINT("Read checkpoint files\n");
    DEBUG_PRINT("Number of checkpoints: %u\n", num_chkpts);

    // Hash Function
//    SHA1 hasher;
//    Murmur3C hasher;
    MD5Hash hasher;

MPI_Barrier(MPI_COMM_WORLD);
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
      Kokkos::View<uint8_t*>::HostMirror current_h("Current region mirror", data_len);
      f.read((char*)(current_h.data()), data_len);
      Kokkos::deep_copy(current, current_h);
      f.close();

      std::string logname = chkpt_filenames[idx];
      std::string filename = full_chkpt_files[idx];
printf("Logname: %s\nFilename: %s\n", logname.c_str(), filename.c_str());
      if(mode == Full) {
        filename = filename + ".full_chkpt";
      } else if(mode == Basic) {
        filename = filename + ".basic.incr_chkpt";
      } else if(mode == List) {
        filename = filename + ".hashlist.incr_chkpt";
      } else {
        filename = filename + ".hashtree.incr_chkpt";
      }
      deduplicator.checkpoint(mode, (uint8_t*)(current.data()), current.size(), filename, logname, idx==0);
      Kokkos::fence();
    }
  }
  STDOUT_PRINT("------------------------------------------------------\n");
  Kokkos::finalize();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}



