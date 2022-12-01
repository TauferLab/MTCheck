#ifndef DEDUPLICATOR_HPP
#define DEDUPLICATOR_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <utility>
#include "stdio.h"
#include "dedup_merkle_tree.hpp"
#include "write_merkle_tree_chkpt.hpp"
#include "kokkos_hash_list.hpp"
//#include "dedup_approaches.hpp"
#include "utils.hpp"

void write_metadata_breakdown2(std::fstream& fs, 
                              header_t& header, 
                              Kokkos::View<uint8_t*>::HostMirror& buffer, 
                              uint32_t num_chkpts) {
  // Print header
  STDOUT_PRINT("==========Header==========\n");
  STDOUT_PRINT("Baseline chkpt          : %u\n" , header.ref_id);
  STDOUT_PRINT("Current chkpt           : %u\n" , header.chkpt_id);
  STDOUT_PRINT("Memory size             : %lu\n", header.datalen);
  STDOUT_PRINT("Chunk size              : %u\n" , header.chunk_size);
  STDOUT_PRINT("Window size             : %u\n" , header.window_size);
  STDOUT_PRINT("Num distinct            : %u\n" , header.distinct_size);
  STDOUT_PRINT("Num repeats for current : %u\n" , header.curr_repeat_size);
  STDOUT_PRINT("Num repeats for previous: %u\n" , header.prev_repeat_size);
  STDOUT_PRINT("Num prior chkpts        : %u\n" , header.num_prior_chkpts);
  STDOUT_PRINT("==========Header==========\n");
  // Print repeat map
  STDOUT_PRINT("==========Repeat Map==========\n");
  for(uint32_t i=0; i<header.num_prior_chkpts; i++) {
    uint32_t chkpt = 0, num = 0;
    uint64_t header_offset = header.distinct_size*sizeof(uint32_t)+i*2*sizeof(uint32_t);
    memcpy(&chkpt, buffer.data()+header_offset, sizeof(uint32_t));
    memcpy(&num, buffer.data()+header_offset+sizeof(uint32_t), sizeof(uint32_t));
    STDOUT_PRINT("%u:%u\n", chkpt, num);
  }
  STDOUT_PRINT("==========Repeat Map==========\n");
  STDOUT_PRINT("Header bytes: %lu\n", 40);
  STDOUT_PRINT("Distinct bytes: %lu\n", header.distinct_size*sizeof(uint32_t));
  // Write size of header and metadata for First occurrence chunks
  fs << 40 << "," << header.distinct_size*sizeof(uint32_t) << ",";
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
        uint64_t repeat_map_offset = header.distinct_size*sizeof(uint32_t)+i*2*sizeof(uint32_t);
        memcpy(&chkpt, buffer.data()+repeat_map_offset, sizeof(uint32_t));
        memcpy(&num, buffer.data()+repeat_map_offset+sizeof(uint32_t), sizeof(uint32_t));
        STDOUT_PRINT("Repeat bytes for %u: %lu\n", chkpt, num*2*sizeof(uint32_t));
        fs << "," << num*2*sizeof(uint32_t);
      } else {
        // No bytes associated with checkpoint i
        STDOUT_PRINT("Repeat bytes for %u: %lu\n", i, 0);;
        fs << "," << 0;
      }
    }
    fs << std::endl;
  } else {
    // Repeat map is unnecessary for the baseline
    STDOUT_PRINT("Repeat map bytes: %lu\n", 0);
    fs << 0 << ",";
    // Write amount of metadata for shifted duplicates
    STDOUT_PRINT("Repeat bytes for %u: %lu\n", header.chkpt_id, header.curr_repeat_size*2*sizeof(uint32_t));
    fs << header.curr_repeat_size*2*sizeof(uint32_t);
    // Write 0s for remaining checkpoints
    for(uint32_t i=1; i<num_chkpts; i++) {
      STDOUT_PRINT("Repeat bytes for %u: %lu\n", i, 0);;
      fs << "," << 0;
    }
    fs << std::endl;
  }
}

enum DedupMode {
  Full,
  Naive,
  List,
  Tree
};

template<typename HashFunc>
class Deduplicator {
  public:
    HashFunc hash_func;
    MerkleTree tree;
    HashList leaves;
    DigestNodeIDMap first_ocur_d;
//    DistinctNodeIDMap first_ocur_d;
    CompactTable first_ocur_updates_d;
    CompactTable shift_dupl_updates_d;
    uint32_t chunk_size;
    uint32_t num_chunks;
    uint32_t num_nodes;
    uint32_t current_id;
    uint32_t baseline_id;
    uint64_t data_len;
    DedupMode mode;
    std::pair<uint64_t,uint64_t> datasizes;
//    std::chrono::duration<double> timers[3];
    double timers[3];

    Deduplicator() {
      tree = MerkleTree(1);
      first_ocur_d = DigestNodeIDMap(1);
//      first_ocur_d = DistinctNodeIDMap(1);
      first_ocur_updates_d = CompactTable(1);
      shift_dupl_updates_d = CompactTable(1);
      chunk_size = 4096;
      current_id = 0;
      mode = Tree;
    }

    Deduplicator(uint32_t bytes_per_chunk) {
      tree = MerkleTree(1);
      first_ocur_d = DigestNodeIDMap(1);
//      first_ocur_d = DistinctNodeIDMap(1);
      first_ocur_updates_d = CompactTable(1);
      shift_dupl_updates_d = CompactTable(1);
      chunk_size = bytes_per_chunk;
      current_id = 0;
      mode = Tree;
    }

    void checkpoint(Kokkos::View<uint8_t*>& data, std::string& filename, std::string& logname, bool make_baseline) {
      // ==========================================================================================
      // Deduplicate data
      // ==========================================================================================
      data_len = data.size();
      num_chunks = data_len/chunk_size;
      if(num_chunks*chunk_size < data_len)
        num_chunks += 1;
      num_nodes = 2*num_chunks-1;

      if(current_id == 0) {
        tree = MerkleTree(num_chunks);
        first_ocur_d = DigestNodeIDMap(num_nodes);
        first_ocur_updates_d = CompactTable(num_chunks);
        shift_dupl_updates_d = CompactTable(num_chunks);
      }
      if(tree.tree_d.size() < num_nodes) {
        Kokkos::resize(tree.tree_d, num_nodes);
        Kokkos::resize(tree.tree_h, num_nodes);
      }
      if(first_ocur_d.capacity() < first_ocur_d.size()+num_nodes)
        first_ocur_d.rehash(first_ocur_d.size()+num_nodes);
      if(num_chunks != first_ocur_updates_d.capacity()) {
        first_ocur_updates_d.rehash(num_nodes);
        shift_dupl_updates_d.rehash(num_nodes);
      }

      first_ocur_updates_d.clear();
      shift_dupl_updates_d.clear();
      using Timer = std::chrono::high_resolution_clock;
      std::string dedup_region_name = std::string("Deduplication chkpt ") + std::to_string(current_id);
      Timer::time_point start_create_tree0 = Timer::now();
      Kokkos::Profiling::pushRegion(dedup_region_name.c_str());
      if((current_id == 0) || make_baseline) {
        create_merkle_tree_deterministic(hash_func, tree, data, chunk_size, current_id, first_ocur_d, shift_dupl_updates_d);
        baseline_id = current_id;
      } else {
        deduplicate_data_deterministic(data, chunk_size, hash_func, tree, current_id, first_ocur_d, shift_dupl_updates_d, first_ocur_updates_d);
//        num_subtree_roots(data, chunk_size, tree, current_id, first_ocur_d, shift_dupl_updates_d, first_ocur_updates_d);
      }
      Kokkos::Profiling::popRegion();
      Timer::time_point end_create_tree0 = Timer::now();
      timers[0] = std::chrono::duration_cast<std::chrono::duration<double>>(end_create_tree0 - start_create_tree0).count();
      printf("First occurrence map capacity:    %lu, size: %lu\n", first_ocur_d.capacity(), first_ocur_d.size());
      printf("First occurrence update capacity: %lu, size: %lu\n", first_ocur_updates_d.capacity(), first_ocur_updates_d.size());
      printf("Shift duplicate update capacity:  %lu, size: %lu\n", shift_dupl_updates_d.capacity(), shift_dupl_updates_d.size());

      // ==========================================================================================
      // Create Diff
      // ==========================================================================================
      Kokkos::View<uint8_t*> diff;
      header_t header;
      std::string collect_region_name = std::string("Start writing incremental checkpoint ") 
                                + std::to_string(current_id);
      Timer::time_point start_collect = Timer::now();
      Kokkos::Profiling::pushRegion(collect_region_name.c_str());
      if((current_id == 0) || make_baseline) {
        datasizes = write_incr_chkpt_hashtree_local_mode(data, diff, chunk_size, 
                                                          first_ocur_d, shift_dupl_updates_d, 
                                                          baseline_id, current_id, header);
      } else {
        datasizes = write_incr_chkpt_hashtree_global_mode(data, diff, chunk_size, 
                                                          first_ocur_updates_d, shift_dupl_updates_d, 
                                                          baseline_id, current_id, header);
      }
      Kokkos::Profiling::popRegion();
      Timer::time_point end_collect = Timer::now();
      timers[1] = std::chrono::duration_cast<std::chrono::duration<double>>(end_collect - start_collect).count();
      
      // ==========================================================================================
      // Write diff to file
      // ==========================================================================================
      auto diff_h = Kokkos::create_mirror_view(diff);
      Timer::time_point start_write = Timer::now();
      std::string write_region_name = std::string("Copy diff to host ") 
                                      + std::to_string(current_id);
      Kokkos::Profiling::pushRegion(write_region_name.c_str());
      Kokkos::deep_copy(diff_h, diff);
      Timer::time_point end_write = Timer::now();
      timers[2] = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();

      std::fstream result_data, timing_file, size_file;
      std::string result_logname = logname+".chunk_size."+std::to_string(chunk_size)+".csv";
      std::string size_logname = logname+".chunk_size."+std::to_string(chunk_size)+".size.csv";
      std::string timing_logname = logname+".chunk_size."+std::to_string(chunk_size)+".timing.csv";
      result_data.open(result_logname, std::fstream::out | std::fstream::app);
      size_file.open(size_logname, std::fstream::out | std::fstream::app);
      timing_file.open(timing_logname, std::fstream::out | std::fstream::app);

      uint32_t num_chkpts = 10;
      result_data << timers[0] << ',' << timers[1] << ',' << timers[2] << ',' << datasizes.first << ',' << datasizes.second << std::endl;
      timing_file << "Tree" << "," << current_id << "," << chunk_size << "," << timers[0] << "," << timers[1] << "," << timers[2] << std::endl;
      size_file << "Tree" << "," << current_id << "," << chunk_size << "," << datasizes.first << "," << datasizes.second << ",";
      write_metadata_breakdown2(size_file, header, diff_h, num_chkpts);
      std::ofstream file;
      file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
      file.open(filename, std::ofstream::out | std::ofstream::binary);
      file.write((char*)(&header), sizeof(header_t));
      file.write((const char*)(diff_h.data()), diff_h.size());
      file.flush();
      file.close();
      result_data.close();
      size_file.close();
      timing_file.close();
      current_id += 1;
    }

    void dedup(Kokkos::View<uint8_t*>& data, bool make_baseline) {
      data_len = data.size();
      num_chunks = data_len/chunk_size;
      if(num_chunks*chunk_size < data_len)
        num_chunks += 1;
      num_nodes = 2*num_chunks-1;

      if(current_id == 0) {
        tree = MerkleTree(num_chunks);
        first_ocur_d = DigestNodeIDMap(num_nodes);
//        first_ocur_d = DistinctNodeIDMap(1);
        first_ocur_updates_d = CompactTable(num_chunks);
        shift_dupl_updates_d = CompactTable(num_chunks);
      }
      if(tree.tree_d.size() < num_nodes) {
        Kokkos::resize(tree.tree_d, num_nodes);
        Kokkos::resize(tree.tree_h, num_nodes);
      }
      if(first_ocur_d.capacity() < first_ocur_d.size()+num_nodes)
        first_ocur_d.rehash(first_ocur_d.size()+num_nodes);
      if(num_chunks != first_ocur_updates_d.capacity()) {
        first_ocur_updates_d.rehash(num_nodes);
        shift_dupl_updates_d.rehash(num_nodes);
      }

      first_ocur_updates_d.clear();
      shift_dupl_updates_d.clear();
      if((current_id == 0) || make_baseline) {
        create_merkle_tree_deterministic(hash_func, tree, data, chunk_size, current_id, first_ocur_d, shift_dupl_updates_d);
        baseline_id = current_id;
      } else {
        deduplicate_data_deterministic(data, chunk_size, hash_func, tree, current_id, first_ocur_d, shift_dupl_updates_d, first_ocur_updates_d);
//        num_subtree_roots(data, chunk_size, tree, current_id, first_ocur_d, shift_dupl_updates_d, first_ocur_updates_d);
      }
      printf("First occurrence map capacity:    %lu, size: %lu\n", first_ocur_d.capacity(), first_ocur_d.size());
      printf("First occurrence update capacity: %lu, size: %lu\n", first_ocur_updates_d.capacity(), first_ocur_updates_d.size());
      printf("Shift duplicate update capacity:  %lu, size: %lu\n", shift_dupl_updates_d.capacity(), shift_dupl_updates_d.size());

      Kokkos::deep_copy(tree.tree_h, tree.tree_d);
    }

    std::pair<uint64_t,uint64_t> 
    create_diff(Kokkos::View<uint8_t*>& data, 
                header_t& header, 
                Kokkos::View<uint8_t*>& diff, 
                bool make_baseline) {
      if((current_id == 0) || make_baseline) {
        datasizes = write_incr_chkpt_hashtree_local_mode(data, diff, chunk_size, 
                                                          first_ocur_d, shift_dupl_updates_d, 
                                                          baseline_id, current_id, header);
      } else {
        datasizes = write_incr_chkpt_hashtree_global_mode(data, diff, chunk_size, 
                                                          first_ocur_updates_d, shift_dupl_updates_d, 
                                                          baseline_id, current_id, header);
      }
      return datasizes;
    }

    void write_diff(header_t& header, Kokkos::View<uint8_t*>& diff, std::string& filename) {
      auto diff_h = Kokkos::create_mirror_view(diff);
      Kokkos::deep_copy(diff_h, diff);
      std::fstream timing_file, size_file;
      std::string size_logname = filename+".chunk_size."+std::to_string(chunk_size)+".size.csv";
      size_file.open(size_logname, std::fstream::out | std::fstream::app);
      uint32_t num_chkpts = 10;
      size_file << "Tree" << "," << current_id << "," << chunk_size << "," << datasizes.first << "," << datasizes.second << ",";
      write_metadata_breakdown2(size_file, header, diff_h, num_chkpts);
      std::ofstream file;
      file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
      file.open(filename, std::ofstream::out | std::ofstream::binary);
      file.write((char*)(&header), sizeof(header_t));
      file.write((const char*)(diff_h.data()), diff_h.size());
      file.flush();
      file.close();
      size_file.close();
      current_id += 1;
    }
};

#endif // DEDUPLICATOR_HPP
