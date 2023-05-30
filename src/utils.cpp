#include "utils.hpp"

void print_mode_help() {
  printf("Modes: \n");
  printf("Full checkpoint:                                   --run-full-chkpt\n");
  printf("Basic checkpoint:                                  --run-basic-chkpt\n");
  printf("List checkpoint:                                   --run-list-chkpt\n");
  printf("Tree checkpoint:                                   --run-tree-chkpt\n");
  printf("Tree checkpoint lowest offset reference algorithm: --run-tree-low-offset-ref-chkpt\n");
  printf("Tree checkpoint lowest offset algorithm:           --run-tree-low-offset-chkpt\n");
  printf("Tree checkpoint lowest root reference algorithm:   --run-tree-low-root-ref-chkpt\n");
  printf("Tree checkpoint lowest root algorithm:             --run-tree-low-root-chkpt\n");
}

DedupMode get_mode(int argc, char** argv) {
  for(int i=0; i<argc; i++) {
    if((strcmp(argv[i], "--run-full-chkpt") == 0)) {
      return Full;
    } else if(strcmp(argv[i], "--run-basic-chkpt") == 0) {
      return Basic;
    } else if(strcmp(argv[i], "--run-list-chkpt") == 0) {
      return List;
    } else if(strcmp(argv[i], "--run-tree-chkpt") == 0) {
      return Tree;
    } else if(strcmp(argv[i], "--run-tree-low-offset-ref-chkpt") == 0) {
      return TreeLowOffsetRef;
    } else if(strcmp(argv[i], "--run-tree-low-offset-chkpt") == 0) {
      return TreeLowOffset;
    } else if(strcmp(argv[i], "--run-tree-low-root-ref-chkpt") == 0) {
      return TreeLowRootRef;
    } else if(strcmp(argv[i], "--run-tree-low-root-chkpt") == 0) {
      return TreeLowRoot;
    }
  }
  return Unknown;
}

void write_metadata_breakdown(std::fstream& fs, 
                              DedupMode mode,
                              header_t& header, 
                              Kokkos::View<uint8_t*>::HostMirror& buffer, 
                              uint32_t num_chkpts) {
  // Print header
  STDOUT_PRINT("==========Header==========\n");
  STDOUT_PRINT("Baseline chkpt          : %u\n" , header.ref_id);
  STDOUT_PRINT("Current chkpt           : %u\n" , header.chkpt_id);
  STDOUT_PRINT("Memory size             : %lu\n", header.datalen);
  STDOUT_PRINT("Chunk size              : %u\n" , header.chunk_size);
  STDOUT_PRINT("Num first ocur          : %u\n" , header.num_first_ocur);
  STDOUT_PRINT("Num shift dupl          : %u\n" , header.num_shift_dupl);
  STDOUT_PRINT("Num prior chkpts        : %u\n" , header.num_prior_chkpts);
  STDOUT_PRINT("==========Header==========\n");
  // Print repeat map
  STDOUT_PRINT("==========Repeat Map==========\n");
  for(uint32_t i=0; i<header.num_prior_chkpts; i++) {
    uint32_t chkpt = 0, num = 0;
    uint64_t header_offset = sizeof(header_t) + 
                             header.num_first_ocur*sizeof(uint32_t) + 
                             i*2*sizeof(uint32_t);
    memcpy(&chkpt, buffer.data()+header_offset, sizeof(uint32_t));
    memcpy(&num, buffer.data()+header_offset+sizeof(uint32_t), sizeof(uint32_t));
    STDOUT_PRINT("%u:%u\n", chkpt, num);
  }
  STDOUT_PRINT("==========Repeat Map==========\n");
  STDOUT_PRINT("Header bytes: %lu\n", sizeof(header_t));
  STDOUT_PRINT("Distinct bytes: %lu\n", header.num_first_ocur*sizeof(uint32_t));
  // Write size of header and metadata for First occurrence chunks
  fs << sizeof(header_t) << "," << header.num_first_ocur*sizeof(uint32_t) << ",";
  uint64_t distinct_bytes = 0;
  uint32_t num_chunks = header.datalen/header.chunk_size;
  if(header.chunk_size*num_chunks < header.datalen)
    num_chunks += 1;
  uint32_t num_nodes = 2*num_chunks-1;
  for(uint32_t i=0; i<header.num_first_ocur; i++) {
    uint32_t node;
    memcpy(&node, buffer.data()+sizeof(header_t)+i*sizeof(uint32_t), sizeof(uint32_t));
    uint32_t size;
    if(mode == Basic || mode == List) {
      size = 1;
    } else {
      size = num_leaf_descendents(node, num_nodes);
    }
    distinct_bytes += size*header.chunk_size;
  }
  STDOUT_PRINT("Size of Data region: %lu\n", distinct_bytes);
  // Check whether this is the reference checkpoint. Reference is a special case
  if(header.ref_id != header.chkpt_id) {
    // Write size of repeat map
    STDOUT_PRINT("Repeat map bytes: %lu\n", 2*sizeof(uint32_t)*header.num_prior_chkpts);
    fs << 2*sizeof(uint32_t)*header.num_prior_chkpts;
    // Write bytes associated with each checkpoint
    for(uint32_t i=0; i<num_chkpts; i++) {
      if(i < header.num_prior_chkpts) {
        // Write bytes for shifted duplicates from checkpoint i
        uint32_t chkpt = 0, num = 0;
        uint64_t repeat_map_offset = sizeof(header_t) + 
                                     header.num_first_ocur*sizeof(uint32_t) + 
                                     i*2*sizeof(uint32_t);
        memcpy(&chkpt, buffer.data()+repeat_map_offset, sizeof(uint32_t));
        memcpy(&num, buffer.data()+repeat_map_offset+sizeof(uint32_t), sizeof(uint32_t));
        STDOUT_PRINT("Repeat bytes for %u: %lu\n", chkpt, num*2*sizeof(uint32_t));
        fs << "," << num*2*sizeof(uint32_t);
      } else {
        // No bytes associated with checkpoint i
        STDOUT_PRINT("Repeat bytes for %u: %u\n", i, 0);;
        fs << "," << 0;
      }
    }
    fs << std::endl;
  } else {
    // Repeat map is unnecessary for the baseline
    STDOUT_PRINT("Repeat map bytes: %u\n", 0);
    fs << 0 << ",";
    // Write amount of metadata for shifted duplicates
    STDOUT_PRINT("Repeat bytes for %u: %lu\n", header.chkpt_id, header.num_shift_dupl*2*sizeof(uint32_t));
    fs << header.num_shift_dupl*2*sizeof(uint32_t);
    // Write 0s for remaining checkpoints
    for(uint32_t i=1; i<num_chkpts; i++) {
      STDOUT_PRINT("Repeat bytes for %u: %u\n", i, 0);;
      fs << "," << 0;
    }
    fs << std::endl;
  }
}
