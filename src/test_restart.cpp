#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include "stdio.h"
#include <string>
#include <map>
#include <fstream>
#include "kokkos_merkle_tree.hpp"
#include "kokkos_hash_list.hpp"
#include "update_pattern_analysis.hpp"
#include <libgen.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <utility>
#include "utils.hpp"

//  file << static_cast<uint32_t>(prior_chkpt_id) << static_cast<uint32_t>(chkpt_id) << (static_cast<uint64_t>(data.size()) << static_cast<uint32_t>(chunk_size) << static_cast<uint32_t>(shared.size()) << static_cast<uint32_t>(distinct.size());
typedef struct header_t {
  uint32_t ref_id;
  uint32_t chkpt_id;
  uint64_t datalen;
  uint32_t chunk_size;
  uint32_t repeat_size;
  uint32_t distinct_size;
} header_t;

void read_header(header_t& header, std::ifstream& file) {
  if(file.is_open()) {
    file.seekg(0);
    file.read((char*)&header, sizeof(header_t));
    printf("Ref ID: %u\n",        header.ref_id);
    printf("Chkpt ID: %u\n",      header.chkpt_id);
    printf("Data len: %lu\n",     header.datalen);
    printf("Chunk size: %u\n",    header.chunk_size);
    printf("Repeat size: %u\n",   header.repeat_size);
    printf("Distinct size: %u\n", header.distinct_size);
  } else {
    printf("Error (read_header): file was not open\n");
  }
}

//void read_distinct_global_metadata(DistinctHostNodeIDMap& distinct, header_t& header, std::ifstream& file) {
//  if(file.is_open()) {
//    if(header.ref_id == header.chkpt_id) {
//      distinct.rehash(distinct.size()+header.distinct_size);
//      file.seekg(sizeof(header_t)+(sizeof(uint32_t)+sizeof(NodeID))*header.repeat_size);
//      for(uint32_t i=0; i<header.distinct_size; i++) {
//        NodeID id;
////        HashDigest digest;
//        file.read((char*)&id, sizeof(uint32_t)*2);
////        file.read((char*)&digest.digest, sizeof(HashDigest));
////        distinct.insert(digest, id);
//        file.seekg(file.tellg()+static_cast<long>(header.chunk_size));
//      }
//      printf("Read %d entries\n", header.distinct_size);
//    } else {
//      printf("File is not a reference checkpoint\n");
//    }
//  } else {
//    printf("Error (read_distinct_metadata): file was not open\n");
//  }
//}
//
//void read_distinct_metadata(CompactHostTable& distinct, header_t& header, std::ifstream& file) {
//  if(file.is_open()) {
//    uint32_t num_chunks = header.datalen/header.chunk_size;
//    if(num_chunks*header.chunk_size < header.datalen)
//      num_chunks += 1;
//    uint32_t num_nodes = 2*num_chunks-1;
//    distinct.rehash(distinct.size()+header.distinct_size);
//    file.seekg(sizeof(header_t)+(sizeof(uint32_t)+sizeof(NodeID))*header.repeat_size);
//    for(uint32_t i=0; i<header.distinct_size; i++) {
//      uint32_t node;
//      NodeID prev;
//      file.read((char*)&node, sizeof(uint32_t));
//      file.read((char*)&prev, sizeof(NodeID));
//      distinct.insert(node, prev);
//      uint32_t size = num_leaf_descendents(node, num_nodes);
//      if(node == prev.node && header.chkpt_id == prev.tree) {
//        file.seekg(file.tellg()+static_cast<long>(size*header.chunk_size));
//      }
//    }
//    printf("Read %d entries\n", header.distinct_size);
//  } else {
//    printf("Error (read_distinct_metadata): file was not open\n");
//  }
//}
//
//void read_distinct_metadata(DistinctHostMap& distinct, Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace>& buffer, header_t& header, std::ifstream& file) {
//  if(file.is_open()) {
//    uint32_t num_chunks = header.datalen/header.chunk_size;
//    if(num_chunks*header.chunk_size < header.datalen)
//      num_chunks += 1;
//    uint32_t num_nodes = 2*num_chunks-1;
//    if(header.ref_id == header.chkpt_id) {
//      distinct.rehash(distinct.size()+header.distinct_size);
//      file.seekg(sizeof(header_t)+(sizeof(uint32_t)+sizeof(NodeID))*header.repeat_size);
//      for(uint32_t i=0; i<header.distinct_size; i++) {
//        NodeID id;
//        HashDigest digest;
//        file.read((char*)&id, sizeof(uint32_t)*2);
//        file.read((char*)&digest.digest, sizeof(HashDigest));
//        distinct.insert(digest, id);
//        if(id.node >= num_chunks-1) {
//          uint32_t writesize = header.chunk_size;
//          if(id.node == num_nodes-1) {
//            writesize = header.datalen-(id.node-num_chunks+1)*header.chunk_size;
//          }
//          file.read((char*)(buffer.data()+(id.node-num_chunks+1)*header.chunk_size), header.chunk_size);
//        }
////        file.seekg(file.tellg()+static_cast<long>(header.chunk_size));
//      }
//      printf("Read %d entries\n", header.distinct_size);
//    } else {
//      printf("File is not a reference checkpoint\n");
//    }
//  } else {
//    printf("Error (read_distinct_metadata): file was not open\n");
//  }
//}
//
//void read_distinct_metadata(CompactHostTable& distinct, Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace>& buffer, header_t& header, std::ifstream& file) {
//  if(file.is_open()) {
//    uint32_t num_chunks = header.datalen/header.chunk_size;
//    if(num_chunks*header.chunk_size < header.datalen)
//      num_chunks += 1;
//    uint32_t num_nodes = 2*num_chunks-1;
//    distinct.rehash(distinct.size()+header.distinct_size);
//    file.seekg(sizeof(header_t)+(sizeof(uint32_t)+sizeof(NodeID))*header.repeat_size);
//    for(uint32_t i=0; i<header.distinct_size; i++) {
//      uint32_t node;
//      NodeID prev;
//      file.read((char*)&node, sizeof(uint32_t));
//      file.read((char*)&prev, sizeof(NodeID));
//      distinct.insert(node, prev);
//      uint32_t size = num_leaf_descendents(node, num_nodes);
//      uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
//      uint32_t end   = rightmost_leaf(node, num_nodes) - (num_chunks-1);
//      if(node == prev.node && header.chkpt_id == prev.tree) {
//        uint32_t writesize = header.chunk_size*size;
//        if(end == num_chunks-1) {
//          writesize = header.datalen-start*header.chunk_size;
//        }
//        file.read((char*)(buffer.data()+start*header.chunk_size), header.chunk_size);
////        file.seekg(file.tellg()+static_cast<long>(size*header.chunk_size));
//      }
//    }
//    printf("Read %d entries\n", header.distinct_size);
//  } else {
//    printf("Error (read_distinct_metadata): file was not open\n");
//  }
//}
//
////template<>
//void read_repeat_reference_metadata(SharedHostTreeMap& repeat, header_t& header, std::ifstream& file) {
//  if(file.is_open()) {
//    if(header.ref_id == header.chkpt_id) {
//      repeat.rehash(repeat.size()+header.repeat_size);
//      file.seekg(sizeof(header_t));
//      for(uint32_t i=0; i<header.repeat_size; i++) {
//        uint32_t node;
//        NodeID id;
//        file.read((char*)&node, sizeof(uint32_t));
//        file.read((char*)&id, sizeof(NodeID));
//        repeat.insert(node, id);
//      }
//      printf("Read %d entries\n", header.repeat_size);
//    } else {
//    }
//  } else {
//    printf("Error (read_repeat_metadata): file was not open\n");
//  }
//}
//
////template<>
//void read_repeat_metadata(CompactHostTable& repeat, header_t& header, std::ifstream& file) {
//  if(file.is_open()) {
//    uint32_t num_chunks = header.datalen/header.chunk_size;
//    if(num_chunks*header.chunk_size < header.datalen)
//      num_chunks += 1;
//    uint32_t num_nodes = 2*num_chunks-1;
//    repeat.rehash(repeat.size()+header.repeat_size);
//    file.seekg(sizeof(header_t));
//    for(uint32_t i=0; i<header.repeat_size; i++) {
//      uint32_t node;
//      NodeID prev;
//      file.read((char*)&node, sizeof(uint32_t));
//      file.read((char*)&prev, sizeof(NodeID));
//      repeat.insert(node, prev);
//      uint32_t size = num_leaf_descendents(node, num_nodes);
//    }
//    printf("Read %d entries\n", header.repeat_size);
//  } else {
//    printf("Error (read_repeat_metadata): file was not open\n");
//  }
//}
//
//void restart_baseline(std::vector<std::string>& chkpt_files, uint32_t file_idx, Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace>& buffer) {
//  std::ifstream file;
//  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
//  file.open(chkpt_files[file_idx], std::ifstream::in | std::ifstream::binary);
//  header_t header;
//  SharedHostTreeMap repeat_map;
//  read_header(header, file);
//  Kokkos::resize(buffer, header.datalen);
//  if(header.ref_id == header.chkpt_id) {
//    DistinctHostMap distinct_map;
//    SharedHostTreeMap repeat_map;
//    read_repeat_reference_metadata(repeat_map, header, file);
//    read_distinct_metadata(distinct_map, buffer, header, file);
//  } else {
//    CompactHostTable distinct_map;
//    CompactHostTable repeat_map;
//    read_repeat_reference_metadata(repeat_map, header, file);
//    read_distinct_metadata(distinct_map, buffer, header, file);
//  }
//  file.close();
//}

int main(int argc, char** argv) {
  DEBUG_PRINT("Sanity check\n");
  Kokkos::initialize(argc, argv);
  {
//    using Timer = std::chrono::high_resolution_clock;
//    STDOUT_PRINT("------------------------------------------------------\n");
//
//    // Process data from checkpoint files
//    DEBUG_PRINT("Argv[1]: %s\n", argv[1]);
//    uint32_t chunk_size = static_cast<uint32_t>(atoi(argv[1]));
//    DEBUG_PRINT("Loaded chunk size\n");
//    uint32_t num_chkpts = static_cast<uint32_t>(atoi(argv[2]));
//    std::vector<std::string> chkpt_files;
//    std::vector<std::string> full_chkpt_files;
//    for(uint32_t i=0; i<num_chkpts; i++) {
//      full_chkpt_files.push_back(std::string(argv[3+i]));
//      chkpt_files.push_back(std::string(argv[3+i]));
//    }
//    STDOUT_PRINT("Read checkpoint files\n");
//    STDOUT_PRINT("Number of checkpoints: %u\n", num_chkpts);
//
//    uint32_t num_tests = 5;
//    std::vector<double> times(num_chkpts, 0.0);
//    for(uint32_t j=0; j<num_tests; j++) {
//      for(uint32_t i=0; i<num_chkpts; i++) {
//        if(i == 0) {
//          DistinctHostMap distinct_map;
//          SharedHostTreeMap repeat_map;
//          double elapsed = 0.0;
////          std::ifstream file;
////          file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
//          std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
////          file.open(full_chkpt_files[i], std::ifstream::in | std::ifstream::binary);
//          Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace> buffer("Restart data", 1);
//          restart_baseline(full_chkpt_files, i, buffer);
////          header_t header;
////          read_header(header, file);
////          read_repeat_reference_metadata(repeat_map, header, file);
////          read_distinct_metadata(distinct_map, header, file);
//          std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
//          elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
////          file.close();
//          times[i] += (elapsed*1e-9);
//        } else {
//          CompactHostTable distinct_map;
//          CompactHostTable repeat_map;
//          double elapsed = 0.0;
////          std::ifstream file;
////          file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
//          std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
////          file.open(full_chkpt_files[i], std::ifstream::in | std::ifstream::binary);
//          Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace> buffer("Restart data", 1);
//          restart_baseline(full_chkpt_files, i, buffer);
////          header_t header;
////          read_header(header, file);
////          read_repeat_metadata(repeat_map, header, file);
////          read_distinct_metadata(distinct_map, header, file);
//          std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
//          elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
////          file.close();
//          times[i] += (elapsed*1e-9);
//        }
//      }
//    }
//    for(uint32_t i=0; i<num_chkpts; i++) {
//      std::cout << "Average time spent for checkpoint " << i << ": " << times[i]/num_tests << std::endl;
//    }
  }
  Kokkos::finalize();
}
