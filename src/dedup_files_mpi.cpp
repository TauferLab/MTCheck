#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <libgen.h>
#include <iostream>
#include "stdio.h"
#include "mpi.h"
#include "deduplicator.hpp"

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
    printf("MPI COMM SIZE: %d, MPI_RANK: %d\n", comm_size, rank);

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

    Deduplicator deduplicator(chunk_size);
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



