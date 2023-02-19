#ifndef DEDUPLICATOR_HPP
#define DEDUPLICATOR_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Bitset.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <utility>
#include "stdio.h"
#include "dedup_merkle_tree.hpp"
#include "write_merkle_tree_chkpt.hpp"
#include "kokkos_hash_list.hpp"
#include "restart_merkle_tree.hpp"
//#include "dedup_approaches.hpp"
#include "utils.hpp"

#define VERIFY_OUTPUT

void write_metadata_breakdown2(std::fstream& fs, 
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
    uint64_t header_offset = sizeof(header_t)+header.num_first_ocur*sizeof(uint32_t)+i*2*sizeof(uint32_t);
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
        uint64_t repeat_map_offset = sizeof(header_t)+header.num_first_ocur*sizeof(uint32_t)+i*2*sizeof(uint32_t);
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
    STDOUT_PRINT("Repeat bytes for %u: %lu\n", header.chkpt_id, header.num_shift_dupl*2*sizeof(uint32_t));
    fs << header.num_shift_dupl*2*sizeof(uint32_t);
    // Write 0s for remaining checkpoints
    for(uint32_t i=1; i<num_chkpts; i++) {
      STDOUT_PRINT("Repeat bytes for %u: %lu\n", i, 0);;
      fs << "," << 0;
    }
    fs << std::endl;
  }
}

template<typename HashFunc>
class Deduplicator {
  public:
    HashFunc hash_func;
    MerkleTree tree;
    HashList leaves;
    DigestNodeIDMap first_ocur_d;
    CompactTable first_ocur_updates_d;
    CompactTable shift_dupl_updates_d;
    // Hash list
    DistinctNodeIDMap first_ocur_chunks_d;
    SharedNodeIDMap shift_dupl_chunks_d;
    SharedNodeIDMap fixed_dupl_chunks_d;
    Vector<uint32_t> first_ocur_vec;
    Vector<uint32_t> shift_dupl_vec;
    uint32_t chunk_size;
    uint32_t num_chunks;
    uint32_t num_nodes;
    uint32_t current_id;
    uint32_t baseline_id;
    uint64_t data_len;
    DedupMode mode;
    std::pair<uint64_t,uint64_t> datasizes;
    double timers[3];
    double restart_timers[2];
    Kokkos::Bitset<Kokkos::DefaultExecutionSpace> changes_bitset;

    Deduplicator() {
      tree = MerkleTree(1);
      first_ocur_d = DigestNodeIDMap(1);
      first_ocur_updates_d = CompactTable(1);
      shift_dupl_updates_d = CompactTable(1);
      chunk_size = 4096;
      current_id = 0;
      mode = Tree;
    }

    Deduplicator(uint32_t bytes_per_chunk) {
      tree = MerkleTree(1);
      first_ocur_d = DigestNodeIDMap(1);
      first_ocur_updates_d = CompactTable(1);
      shift_dupl_updates_d = CompactTable(1);
      chunk_size = bytes_per_chunk;
      current_id = 0;
      mode = Tree;
    }

    void write_chkpt_log(header_t& header, Kokkos::View<uint8_t*>::HostMirror& diff_h, std::string& logname) {
      std::fstream result_data, timing_file, size_file;
      std::string result_logname = logname+".chunk_size."+std::to_string(chunk_size)+".csv";
      std::string size_logname = logname+".chunk_size."+std::to_string(chunk_size)+".size.csv";
      std::string timing_logname = logname+".chunk_size."+std::to_string(chunk_size)+".timing.csv";
      result_data.open(result_logname, std::fstream::out | std::fstream::app);
      size_file.open(size_logname, std::fstream::out | std::fstream::app);
      timing_file.open(timing_logname, std::fstream::out | std::fstream::app);

      uint32_t num_chkpts = 10;

      if(mode == Full) {
        result_data << "0.0" << ","          // Comparison time
                    << "0.0" << ","          // Collection time
                    << timers[2] << ","     // Write time
                    << datasizes.first << ',' // Size of data
                    << "0" << ',';           // Size of metadata
        timing_file << "Full" << ","     // Approach
                    << current_id << ","        // Checkpoint ID
                    << chunk_size << "," // Chunk size
                    << "0.0" << ","      // Comparison time
                    << "0.0" << ","      // Collection time
                    << timers[2]        // Write time
                    << std::endl;
        size_file << "Full" << ","         // Approach
                  << current_id << ","            // Checkpoint ID
                  << chunk_size << ","     // Chunk size
                  << datasizes.first << "," // Size of data
                  << "0,"                  // Size of metadata
                  << "0,"                  // Size of header
                  << "0,"                  // Size of distinct metadata
                  << "0";                  // Size of repeat map
        for(uint32_t j=0; j<num_chkpts; j++) {
          size_file << ",0"; // Size of metadata for checkpoint j
        }
        size_file << std::endl;
      } else if(mode == Basic) {
        result_data << timers[0] << ','      // Comparison time
                    << timers[1] << ','      // Collection time  
                    << timers[2] << ','        // Write time
                    << datasizes.first << ','   // Size of data     
                    << datasizes.second << ','; // Size of metadata 
        timing_file << "Basic" << ","      // Approach
                    << current_id << ","          // Checkpoint ID
                    << chunk_size << ","   // Chunk size      
                    << timers[0] << "," // Comparison time 
                    << timers[1] << "," // Collection time 
                    << timers[2]          // Write time
                    << std::endl;
        size_file << "Basic" << ","           // Approach
                  << current_id << ","               // Checkpoint ID
                  << chunk_size << ","        // Chunk size
                  << datasizes.first << ","   // Size of data
                  << datasizes.second << ","; // Size of metadata
        write_metadata_breakdown2(size_file, mode, header, diff_h, num_chkpts);
      } else if(mode == List) {
        result_data << timers[0] << ',' 
                    << timers[1] << ',' 
                    << timers[2] << ',' 
                    << datasizes.first << ',' 
                    << datasizes.second << ',';
        timing_file << "List" << "," 
                    << current_id << "," 
                    << chunk_size << "," 
                    << timers[0] << "," // Comparison time 
                    << timers[1] << "," // Collection time 
                    << timers[2]          // Write time
                    << std::endl;
        size_file << "List" << "," 
                  << current_id << "," 
                  << chunk_size << "," 
                  << datasizes.first << "," 
                  << datasizes.second << ",";
        write_metadata_breakdown2(size_file, mode, header, diff_h, num_chkpts);
      } else if((mode == Tree) || (mode == TreeLowOffsetRef) || (mode == TreeLowOffset) || 
                (mode == TreeLowRootRef) || (mode == TreeLowRoot)) {
        std::string approach("Tree");
        if(mode == TreeLowOffsetRef) {
          approach = std::string("TreeLowOffsetRef");
        } else if(mode == TreeLowOffset) {
          approach = std::string("TreeLowOffset");
        } else if(mode == TreeLowRootRef) {
          approach = std::string("TreeLowRootRef");
        } else if(mode == TreeLowRoot) {
          approach = std::string("TreeLowRoot");
        }

        result_data << timers[0] << ',' 
                    << timers[1] << ',' 
                    << timers[2] << ',' 
                    << datasizes.first << ',' 
                    << datasizes.second << std::endl;
        timing_file << approach << "," 
                    << current_id << "," 
                    << chunk_size << "," 
                    << timers[0] << "," 
                    << timers[1] << "," 
                    << timers[2] << std::endl;
        size_file << approach << "," 
                  << current_id << "," 
                  << chunk_size << "," 
                  << datasizes.first << "," 
                  << datasizes.second << ",";
        write_metadata_breakdown2(size_file, mode, header, diff_h, num_chkpts);
      }
    }

    void write_chkpt(header_t& header, Kokkos::View<uint8_t*>::HostMirror& diff_h, std::string& filename) {
      std::ofstream file;
      file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
      file.open(filename, std::ofstream::out | std::ofstream::binary);

      file.write((const char*)(diff_h.data()), diff_h.size());

      file.flush();
      file.close();
    }

    template<typename DataView>
    void checkpoint(DedupMode dedup_mode, header_t& header, DataView& data, Kokkos::View<uint8_t*>::HostMirror& diff_h, bool make_baseline) {
      // ==========================================================================================
      // Deduplicate data
      // ==========================================================================================
      mode = dedup_mode;
      data_len = data.size();
      num_chunks = data_len/chunk_size;
      if(num_chunks*chunk_size < data_len)
        num_chunks += 1;
      num_nodes = 2*num_chunks-1;

      if(mode == Basic) {
        if(make_baseline) {
          leaves = HashList(num_chunks);
          first_ocur_vec = Vector<uint32_t>(num_chunks);
          changes_bitset = Kokkos::Bitset<Kokkos::DefaultExecutionSpace>(num_chunks);
        }
        if(leaves.list_d.size() < num_chunks) {
          Kokkos::resize(leaves.list_d, num_chunks);
          Kokkos::resize(leaves.list_h, num_chunks);
        }
//        changes_bitset = Kokkos::Bitset<Kokkos::DefaultExecutionSpace>(num_chunks);
//        changes_bitset.reset();
      } else if(mode == List) {
        if(make_baseline) {
          leaves = HashList(num_chunks);
          first_ocur_chunks_d = DistinctNodeIDMap(num_chunks);
          first_ocur_vec = Vector<uint32_t>(num_chunks);
          shift_dupl_vec = Vector<uint32_t>(num_chunks);
        }
        if(leaves.list_d.size() < num_chunks) {
          Kokkos::resize(leaves.list_d, num_chunks);
          Kokkos::resize(leaves.list_h, num_chunks);
        }
        if(first_ocur_chunks_d.capacity() < first_ocur_chunks_d.size()+num_chunks)
          first_ocur_chunks_d.rehash(first_ocur_chunks_d.size()+num_chunks);
      } else if((mode == Tree) || (mode == TreeLowOffsetRef) || (mode == TreeLowOffset) || 
                (mode == TreeLowRootRef) || (mode == TreeLowRoot)) {
        if(make_baseline) {
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
          first_ocur_updates_d.rehash(num_chunks);
          shift_dupl_updates_d.rehash(num_chunks);
        }
        first_ocur_updates_d.clear();
        shift_dupl_updates_d.clear();
      }

      using Timer = std::chrono::high_resolution_clock;
      using Duration = std::chrono::duration<double>;
      std::string dedup_region_name = std::string("Deduplication chkpt ") + std::to_string(current_id);
      Timer::time_point start_create_tree0 = Timer::now();
      Kokkos::Profiling::pushRegion(dedup_region_name.c_str());

      if(mode == Basic) {
//        compare_lists_basic(hash_func, leaves, changes_bitset, current_id, data, chunk_size);
        compare_lists_basic(hash_func, leaves, first_ocur_vec, current_id, data, chunk_size);
printf("Number of changed chunks: %u\n", first_ocur_vec.size());
//changes_bitset.set();
//printf("(Bitset) Post set. Number of changed chunks: %u\n", changes_bitset.count());
if(current_id == 0) {
////changes_bitset.set();
//  auto bitset = Kokkos::Bitset<Kokkos::DefaultExecutionSpace>(num_chunks);
//  bitset.reset();
//  first_ocur_vec.clear();
//  Kokkos::sort(first_ocur_vec.vector_d, 0, first_ocur_vec.size());
//  Kokkos::parallel_for(first_ocur_vec.size(), KOKKOS_LAMBDA(const uint32_t i) {
//    bitset.set(i);
//    first_ocur_vec.vector_d(i) = i;
//    if(i == 0)
//      first_ocur_vec.len_d(0) = first_ocur_vec.capacity();
//  });
//  printf("Bitset count: %u\n", bitset.count());
}
      } else if(mode == List) {
        compare_lists_global(leaves, current_id, data, chunk_size, first_ocur_chunks_d, first_ocur_vec, shift_dupl_vec);
      } else if((mode == Tree) || (mode == TreeLowOffsetRef) || (mode == TreeLowOffset) || 
                (mode == TreeLowRootRef) || (mode == TreeLowRoot)) {
        if((current_id == 0) || make_baseline) {
          deduplicate_data_deterministic_baseline(data, chunk_size, hash_func, tree, current_id, 
                                         first_ocur_d, shift_dupl_updates_d, first_ocur_updates_d);
          baseline_id = current_id;
        } else {
          if((mode == TreeLowOffset)) {
            deduplicate_data_deterministic(data, chunk_size, hash_func, tree, current_id, 
                                           first_ocur_d, shift_dupl_updates_d, first_ocur_updates_d);
          } else if((mode == TreeLowOffsetRef)) {
            dedup_low_offset_ref(data, chunk_size, tree, current_id, 
                                 first_ocur_d, shift_dupl_updates_d, first_ocur_updates_d);
          } else if(mode == TreeLowRootRef) {
            dedup_low_root_ref(data, chunk_size, tree, current_id, 
                               first_ocur_d, shift_dupl_updates_d, first_ocur_updates_d);
          } else if((mode == Tree) || (mode == TreeLowRoot)) {
            dedup_low_root(data, chunk_size, hash_func, tree, current_id, 
                           first_ocur_d, shift_dupl_updates_d, first_ocur_updates_d);
          }
          if(current_id == 0 || make_baseline)
            baseline_id = current_id;
          STDOUT_PRINT("First occurrence update capacity: %lu, size: %lu\n", 
                 first_ocur_updates_d.capacity(), first_ocur_updates_d.size());
          STDOUT_PRINT("Shift duplicate update capacity:  %lu, size: %lu\n", 
                 shift_dupl_updates_d.capacity(), shift_dupl_updates_d.size());
        }
      }

      Kokkos::Profiling::popRegion();
      Timer::time_point end_create_tree0 = Timer::now();
      timers[0] = std::chrono::duration_cast<Duration>(end_create_tree0 - start_create_tree0).count();

      // ==========================================================================================
      // Create Diff
      // ==========================================================================================
      Kokkos::View<uint8_t*> diff;
      std::string collect_region_name = std::string("Start writing incremental checkpoint ") 
                                + std::to_string(current_id);
      Timer::time_point start_collect = Timer::now();
      Kokkos::Profiling::pushRegion(collect_region_name.c_str());

      if(mode == Full) {
        datasizes = std::make_pair(data.size(), 0);
      } else if(mode == Basic) {
        datasizes = write_incr_chkpt_hashlist_basic(data, diff, chunk_size, first_ocur_vec, 
                                                    0, current_id, header);
//        datasizes = write_incr_chkpt_hashlist_basic(data, diff, chunk_size, changes_bitset, 
//                                                    0, current_id, header);
//        datasizes = write_incr_chkpt_hashlist_global(data, diff, chunk_size, leaves, first_ocur_chunks_d, first_ocur_vec, shift_dupl_vec, 0, current_id, header);
//        datasizes = write_incr_chkpt_hashlist_basic_test(data, diff, chunk_size, first_ocur_vec, 
//                                                    0, current_id, header);
      } else if(mode == List) {
          datasizes = write_incr_chkpt_hashlist_global(data, diff, chunk_size, leaves, first_ocur_chunks_d, first_ocur_vec, shift_dupl_vec, 0, current_id, header);
//        }
      } else if((mode == Tree) || (mode == TreeLowOffsetRef) || (mode == TreeLowOffset) || 
                (mode == TreeLowRootRef) || (mode == TreeLowRoot)) {
          datasizes = write_incr_chkpt_hashtree_global_mode(data, diff, chunk_size, 
                                                            first_ocur_updates_d, 
                                                            shift_dupl_updates_d, 
                                                            baseline_id, current_id, header);
      }

      Kokkos::Profiling::popRegion();
      Timer::time_point end_collect = Timer::now();
      timers[1] = std::chrono::duration_cast<Duration>(end_collect - start_collect).count();

      if(mode == Full) {
        Kokkos::resize(diff, data.size());
        Kokkos::deep_copy(diff, data);
      }
      
      // ==========================================================================================
      // Copy diff to host 
      // ==========================================================================================
//      auto diff_h = Kokkos::create_mirror_view(diff);
      Kokkos::resize(diff_h, diff.size());
      Timer::time_point start_write = Timer::now();
      std::string write_region_name = std::string("Copy diff to host ") 
                                      + std::to_string(current_id);
      Kokkos::Profiling::pushRegion(write_region_name.c_str());

      if(mode == Full) {
        Kokkos::deep_copy(diff_h, data);
      } else if((mode == Basic) || (mode == List) || (mode == Tree) || 
                (mode == TreeLowOffsetRef) || (mode == TreeLowOffset) || 
                (mode == TreeLowRootRef) || (mode == TreeLowRoot)) {
        Kokkos::deep_copy(diff_h, diff);
        memcpy(diff_h.data(), &header, sizeof(header_t));
      }

      Kokkos::Profiling::popRegion();
      Timer::time_point end_write = Timer::now();
      timers[2] = std::chrono::duration_cast<Duration>(end_write - start_write).count();
    }

    void checkpoint(DedupMode dedup_mode, uint8_t* data_ptr, size_t len, 
                    std::string& filename, std::string& logname, bool make_baseline) {
      Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
      Kokkos::View<uint8_t*>::HostMirror diff_h;
      header_t header;
      checkpoint(dedup_mode, header, data, diff_h, make_baseline);
      write_chkpt_log(header, diff_h, logname);
      write_chkpt(header, diff_h, filename);
      current_id += 1;
    }

    void checkpoint(DedupMode dedup_mode, uint8_t* data_ptr, size_t len, 
                    Kokkos::View<uint8_t*>::HostMirror& diff_h, bool make_baseline) {
      Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
      header_t header;
      checkpoint(dedup_mode, header, data, diff_h, make_baseline);
      current_id += 1;
    }

    void checkpoint(DedupMode dedup_mode, uint8_t* data_ptr, size_t len, 
                    Kokkos::View<uint8_t*>::HostMirror& diff_h, std::string& logname, 
                    bool make_baseline) {
      Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
      header_t header;
      checkpoint(dedup_mode, header, data, diff_h, make_baseline);
      write_chkpt_log(header, diff_h, logname);
      current_id += 1;
    }

    void write_restart_log(uint32_t select_chkpt, std::string& logname) {
      std::fstream timing_file;
      timing_file.open(logname, std::fstream::out | std::fstream::app);
      if(mode == Full) {
        timing_file << "Full" << ",";
      } else if(mode == Basic) {
        timing_file << "Basic" << ","; 
      } else if(mode == List) {
        timing_file << "List" << ","; 
      } else if(mode == Tree) {
        timing_file << "Tree" << ","; 
      } else if(mode == TreeLowOffsetRef) {
        timing_file << "TreeLowOffsetRef" << ","; 
      } else if(mode == TreeLowOffset) {
        timing_file << "TreeLowOffset" << ","; 
      } else if(mode == TreeLowRootRef) {
        timing_file << "TreeLowRootRef" << ","; 
      } else if(mode == TreeLowRoot) {
        timing_file << "TreeLowRoot" << ","; 
      }
      timing_file << select_chkpt << "," 
                  << chunk_size << "," 
                  << restart_timers[0] << "," 
                  << restart_timers[1] << std::endl;
      timing_file.close();
    }

    void restart(DedupMode dedup_mode, Kokkos::View<uint8_t*> data, 
                 std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                 std::string& logname, uint32_t chkpt_id) {
      using Nanoseconds = std::chrono::nanoseconds;
      using Timer = std::chrono::high_resolution_clock;
      mode = dedup_mode;
      if(dedup_mode == Full) {
        // Full checkpoint
        Kokkos::resize(data, chkpts[chkpt_id].size());
        // Total time
        Timer::time_point t1 = Timer::now();
        // Copy checkpoint to GPU
        Timer::time_point c1 = Timer::now();
        Kokkos::deep_copy(data, chkpts[chkpt_id]);
        Timer::time_point c2 = Timer::now();
        Timer::time_point t2 = Timer::now();
        // Update timers
        restart_timers[0] = (1e-9)*(std::chrono::duration_cast<Nanoseconds>(c2-c1).count());
        restart_timers[1] = 0.0;
      } else if(dedup_mode == Basic) {
        auto basic_list_times = restart_incr_chkpt_basic(chkpts, chkpt_id, data);
//        auto basic_list_times = restart_list(chkpts, chkpt_id, data);
        restart_timers[0] = basic_list_times.first;
        restart_timers[1] = basic_list_times.second;
      } else if(dedup_mode == List) {
//        auto list_times = restart_incr_chkpt_hashlist(chkpts, chkpt_id, data);
        auto list_times = restart_list(chkpts, chkpt_id, data);
        restart_timers[0] = list_times.first;
        restart_timers[1] = list_times.second;
      } else if((mode == Tree) || (mode == TreeLowOffsetRef) || (mode == TreeLowOffset) || 
                (mode == TreeLowRootRef) || (mode == TreeLowRoot)) {
        auto tree_times = restart_incr_chkpt_hashtree(chkpts, chkpt_id, data);
        restart_timers[0] = tree_times.first;
        restart_timers[1] = tree_times.second;
      }
      std::string restart_logname = logname+".chunk_size."+std::to_string(chunk_size)+".restart_timing.csv";
      write_restart_log(chkpt_id, restart_logname);
    } 

    void restart(DedupMode dedup_mode, uint8_t* data_ptr, size_t len, 
                 std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                 std::string& logname, uint32_t chkpt_id) {
      Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
      restart(dedup_mode, data, chkpts, logname, chkpt_id);
    }

    void restart(DedupMode dedup_mode, Kokkos::View<uint8_t*> data, 
                 std::vector<std::string>& chkpt_filenames, 
                 std::string& logname, uint32_t chkpt_id) {
      using Nanoseconds = std::chrono::nanoseconds;
      using Timer = std::chrono::high_resolution_clock;
      mode = dedup_mode;
      if(dedup_mode == Full) {
        // Full checkpoint
        std::fstream file;
        auto fileflags = std::ifstream::in | std::ifstream::binary | std::ifstream::ate;
        file.open(chkpt_filenames[chkpt_id], fileflags);
        size_t filesize = file.tellg();
        file.seekg(0);
        Kokkos::resize(data, filesize);
        auto data_h = Kokkos::create_mirror_view(data);
        // Read checkpoint
        Timer::time_point r1 = Timer::now();
        file.read((char*)(data_h.data()), filesize);
        file.close();
        Timer::time_point r2 = Timer::now();
        // Total time
        Timer::time_point t1 = Timer::now();
        // Copy checkpoint to GPU
        Timer::time_point c1 = Timer::now();
        Kokkos::deep_copy(data, data_h);
        Timer::time_point c2 = Timer::now();
        Timer::time_point t2 = Timer::now();
        // Update timers
        restart_timers[0] = (1e-9)*(std::chrono::duration_cast<Nanoseconds>(c2-c1).count());
        restart_timers[1] = 0.0;
      } else if(dedup_mode == Basic) {
        std::vector<std::string> basiclist_chkpt_files;
        for(uint32_t i=0; i<chkpt_filenames.size(); i++) {
          basiclist_chkpt_files.push_back(chkpt_filenames[i]+".basic.incr_chkpt");
        }
        auto basic_list_times = restart_incr_chkpt_basic(basiclist_chkpt_files, chkpt_id, data);
        restart_timers[0] = basic_list_times.first;
        restart_timers[1] = basic_list_times.second;
      } else if(dedup_mode == List) {
        std::vector<std::string> hashlist_chkpt_files;
        for(uint32_t i=0; i<chkpt_filenames.size(); i++) {
          hashlist_chkpt_files.push_back(chkpt_filenames[i]+".hashlist.incr_chkpt");
        }
        auto list_times = restart_incr_chkpt_hashlist(hashlist_chkpt_files, chkpt_id, data);
        restart_timers[0] = list_times.first;
        restart_timers[1] = list_times.second;
      } else if((mode == Tree) || (mode == TreeLowOffsetRef) || (mode == TreeLowOffset) || 
                (mode == TreeLowRootRef) || (mode == TreeLowRoot)) {
        std::vector<std::string> hashtree_chkpt_files;
        for(uint32_t i=0; i<chkpt_filenames.size(); i++) {
          hashtree_chkpt_files.push_back(chkpt_filenames[i]+".hashtree.incr_chkpt");
        }
        auto tree_times = restart_incr_chkpt_hashtree(hashtree_chkpt_files, chkpt_id, data);
        restart_timers[0] = tree_times.first;
        restart_timers[1] = tree_times.second;
      }
      write_restart_log(chkpt_id, logname);
    } 
};

#endif // DEDUPLICATOR_HPP
