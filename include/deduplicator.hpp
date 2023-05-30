#ifndef DEDUPLICATOR_HPP
#define DEDUPLICATOR_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Bitset.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <utility>
#include "stdio.h"
#include "full_approach.hpp"
#include "basic_approach.hpp"
#include "list_approach.hpp"
#include "tree_approach.hpp"
#include "utils.hpp"

class Deduplicator {
  public:
    MerkleTree tree;
    HashList leaves;
    DigestNodeIDDeviceMap first_ocur_d; // Map of first occurrences
    Vector<uint32_t> first_ocur_vec; // First occurrence root offsets
    Vector<uint32_t> shift_dupl_vec; // Shifted duplicate root offsets
    Kokkos::Bitset<Kokkos::DefaultExecutionSpace> changes_bitset;
    uint32_t chunk_size;
    uint32_t num_chunks;
    uint32_t num_nodes;
    uint32_t current_id;
    uint32_t baseline_id;
    uint64_t data_len;
    DedupMode mode;
    // timers and data sizes
    std::pair<uint64_t,uint64_t> datasizes;
    double timers[4];
    double restart_timers[2];

    Deduplicator() {
      tree = MerkleTree(1);
      first_ocur_d = DigestNodeIDDeviceMap(1);
      chunk_size = 4096;
      current_id = 0;
      mode = Tree;
    }

    Deduplicator(uint32_t bytes_per_chunk) {
      tree = MerkleTree(1);
      first_ocur_d = DigestNodeIDDeviceMap(1);
      chunk_size = bytes_per_chunk;
      current_id = 0;
      mode = Tree;
    }

    /**
     * Write logs for the checkpoint metadata/data breakdown, runtimes, and the overall summary.
     * The data breakdown log shows the proportion of data and metadata as well as how much 
     * metadata corresponds to each prior checkpoint.
     * The timing log contains the time spent comparing chunks, gathering scattered chunks,
     * and the time spent copying the resulting checkpoint from the device to host.
     *
     * \param header  The checkpoint header
     * \param diff_h  The incremental checkpoint
     * \param logname Base filename for the logs
     */
    void write_chkpt_log(header_t& header, 
                         Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                         std::string& logname) {
      std::fstream result_data, timing_file, size_file;
      std::string result_logname = logname+".chunk_size."+std::to_string(chunk_size)+".csv";
      std::string size_logname = logname+".chunk_size."+std::to_string(chunk_size)+".size.csv";
      std::string timing_logname = logname+".chunk_size."+std::to_string(chunk_size)+".timing.csv";
      result_data.open(result_logname, std::fstream::ate | std::fstream::out | std::fstream::app);
      size_file.open(size_logname, std::fstream::out | std::fstream::app);
      timing_file.open(timing_logname, std::fstream::out | std::fstream::app);
      if(result_data.tellp() == 0) {
        result_data << "Approach,Chkpt ID,Chunk Size,Uncompressed Size,Compressed Size,Data Size,Metadata Size,Setup Time,Comparison Time,Gather Time,Write Time" << std::endl;
      }

      // TODO make this more generic 
      uint32_t num_chkpts = 10;

      if(mode == Full) {
        result_data << "Full" << "," // Approach
                    << current_id << "," // Chkpt ID
                    << chunk_size << "," // Chunk size
                    << data_len << "," // Uncompressed size
                    << datasizes.first+datasizes.second << "," // Compressed size
                    << datasizes.first << "," // Compressed data size
                    << datasizes.second << "," // Compressed metadata size
                    << timers[0] << "," // Compression setup time
                    << timers[1] << "," // Compression comparison time
                    << timers[2] << "," // Compression gather chunks time
                    << timers[3] << std::endl; // Compression copy diff to host
        timing_file << "Full" << ","     // Approach
                    << current_id << ","        // Checkpoint ID
                    << chunk_size << "," // Chunk size
                    << "0.0" << ","      // Comparison time
                    << "0.0" << ","      // Collection time
                    << timers[3]        // Write time
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
        result_data << "Basic" << "," // Approach
                    << current_id << "," // Chkpt ID
                    << chunk_size << "," // Chunk size
                    << data_len << "," // Uncompressed size
                    << datasizes.first+datasizes.second << "," // Compressed size
                    << datasizes.first << "," // Compressed data size
                    << datasizes.second << "," // Compressed metadata size
                    << timers[0] << "," // Compression setup time
                    << timers[1] << "," // Compression comparison time
                    << timers[2] << "," // Compression gather chunks time
                    << timers[3] << std::endl; // Compression copy diff to host
        timing_file << "Basic" << ","      // Approach
                    << current_id << ","          // Checkpoint ID
                    << chunk_size << ","   // Chunk size      
                    << timers[1] << "," // Comparison time 
                    << timers[2] << "," // Collection time 
                    << timers[3]          // Write time
                    << std::endl;
        size_file << "Basic" << ","           // Approach
                  << current_id << ","               // Checkpoint ID
                  << chunk_size << ","        // Chunk size
                  << datasizes.first << ","   // Size of data
                  << datasizes.second << ","; // Size of metadata
        write_metadata_breakdown(size_file, mode, header, diff_h, num_chkpts);
      } else if(mode == List) {
        result_data << "List" << "," // Approach
                    << current_id << "," // Chkpt ID
                    << chunk_size << "," // Chunk size
                    << data_len << "," // Uncompressed size
                    << datasizes.first+datasizes.second << "," // Compressed size
                    << datasizes.first << "," // Compressed data size
                    << datasizes.second << "," // Compressed metadata size
                    << timers[0] << "," // Compression setup time
                    << timers[1] << "," // Compression comparison time
                    << timers[2] << "," // Compression gather chunks time
                    << timers[3] << std::endl; // Compression copy diff to host
        timing_file << "List" << "," 
                    << current_id << "," 
                    << chunk_size << "," 
                    << timers[1] << "," // Comparison time 
                    << timers[2] << "," // Collection time 
                    << timers[3]          // Write time
                    << std::endl;
        size_file << "List" << "," 
                  << current_id << "," 
                  << chunk_size << "," 
                  << datasizes.first << "," 
                  << datasizes.second << ",";
        write_metadata_breakdown(size_file, mode, header, diff_h, num_chkpts);
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

        result_data << approach << "," // Approach
                    << current_id << "," // Chkpt ID
                    << chunk_size << "," // Chunk size
                    << data_len << "," // Uncompressed size
                    << datasizes.first+datasizes.second << "," // Compressed size
                    << datasizes.first << "," // Compressed data size
                    << datasizes.second << "," // Compressed metadata size
                    << timers[0] << "," // Compression setup time
                    << timers[1] << "," // Compression comparison time
                    << timers[2] << "," // Compression gather chunks time
                    << timers[3] << std::endl; // Compression copy diff to host
        timing_file << approach << "," 
                    << current_id << "," 
                    << chunk_size << "," 
                    << timers[1] << "," 
                    << timers[2] << "," 
                    << timers[3] << std::endl;
        size_file << approach << "," 
                  << current_id << "," 
                  << chunk_size << "," 
                  << datasizes.first << "," 
                  << datasizes.second << ",";
        write_metadata_breakdown(size_file, mode, header, diff_h, num_chkpts);
      }
    }

    /**
     * Helper function for writing the checkpoint (on Host) to file
     *
     * \param header   The incremental checkpoint header
     * \param diff_h   The incremental checkpoint
     * \param filename Output filename for the checkpoint
     */
    void write_chkpt( header_t& header, 
                      Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                      std::string& filename) {
      std::ofstream file;
      file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
      file.open(filename, std::ofstream::out | std::ofstream::binary);

      file.write((const char*)(diff_h.data()), diff_h.size());

      file.flush();
      file.close();
    }

    /**
     * Main checkpointing function. Given a Kokkos View, create an incremental checkpoint using 
     * the chosen checkpoint strategy. The deduplication mode can be one of the following:
     *   - Full: No deduplication
     *   - Basic: Remove chunks that have not changed since the previous checkpoint
     *   - List: Save a single copy of each unique chunk and use metadata to handle duplicates
     *   - Tree: Save minimal set of chunks and use a compact metadata representations
     *
     * \param dedup_mode    Checkpoint approach (Full|Basic|List|Tree)
     * \param header        The checkpoint header
     * \param data          Data View to be checkpointed
     * \param diff_h        The output incremental checkpoint on the Host
     * \param make_baseline Flag determining whether to make a baseline checkpoint
     */
    template<typename DataView>
    void checkpoint(DedupMode dedup_mode, 
                    header_t& header, 
                    DataView& data, 
                    Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                    bool make_baseline) {
      using Timer = std::chrono::high_resolution_clock;
      using Duration = std::chrono::duration<double>;
      // ==========================================================================================
      // Deduplicate data
      // ==========================================================================================
      Timer::time_point beg_chkpt = Timer::now();
      std::string setup_region_name = std::string("Deduplication chkpt ") + 
                                      std::to_string(current_id) + std::string(": Setup");
      Kokkos::Profiling::pushRegion(setup_region_name.c_str());

      // Set important values
      mode = dedup_mode;
      data_len = data.size();
      num_chunks = data_len/chunk_size;
      if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(chunk_size) < data_len)
        num_chunks += 1;
      num_nodes = 2*num_chunks-1;

      // Allocate or resize necessary variables for each approach
      if(mode == Basic) {
        if(make_baseline) {
          leaves = HashList(num_chunks);
          changes_bitset = Kokkos::Bitset<Kokkos::DefaultExecutionSpace>(num_chunks);
        }
        if(leaves.list_d.size() < num_chunks) {
          Kokkos::resize(leaves.list_d, num_chunks);
          Kokkos::resize(leaves.list_h, num_chunks);
        }
      } else if(mode == List) {
        if(make_baseline) {
          leaves = HashList(num_chunks);
          first_ocur_d = DigestNodeIDDeviceMap(num_chunks);
          first_ocur_vec = Vector<uint32_t>(num_chunks);
          shift_dupl_vec = Vector<uint32_t>(num_chunks);
        }
        first_ocur_vec.clear();
        shift_dupl_vec.clear();
        if(leaves.list_d.size() < num_chunks) {
          Kokkos::resize(leaves.list_d, num_chunks);
          Kokkos::resize(leaves.list_h, num_chunks);
        }
        if(first_ocur_d.capacity() < first_ocur_d.size()+num_chunks)
          first_ocur_d.rehash(first_ocur_d.size()+num_chunks);
      } else if((mode == Tree) || (mode == TreeLowOffsetRef) || (mode == TreeLowOffset) || 
                (mode == TreeLowRootRef) || (mode == TreeLowRoot)) {
        if(make_baseline) {
          tree = MerkleTree(num_chunks);
          first_ocur_d = DigestNodeIDDeviceMap(num_nodes);
          first_ocur_vec = Vector<uint32_t>(num_chunks);
          shift_dupl_vec = Vector<uint32_t>(num_chunks);
        }
        first_ocur_vec.clear();
        shift_dupl_vec.clear();
        std::string resize_tree_label = std::string("Deduplication chkpt ") + 
                                        std::to_string(current_id) + 
                                        std::string(": Setup: Resize Tree");
        Kokkos::Profiling::pushRegion(resize_tree_label.c_str());
        if(tree.tree_d.size() < num_nodes) {
          Kokkos::resize(tree.tree_d, num_nodes);
          Kokkos::resize(tree.tree_h, num_nodes);
        }
        Kokkos::Profiling::popRegion();
        std::string resize_map_label = std::string("Deduplication chkpt ") + 
                                       std::to_string(current_id) + 
                                       std::string(": Setup: Resize First Ocur Map");
        Kokkos::Profiling::pushRegion(resize_map_label.c_str());
        if(first_ocur_d.capacity() < first_ocur_d.size()+num_nodes)
          first_ocur_d.rehash(first_ocur_d.size()+num_nodes);
        Kokkos::Profiling::popRegion();
        std::string resize_updates_label = std::string("Deduplication chkpt ") + 
                                           std::to_string(current_id) + 
                                           std::string(": Setup: Resize Update Map");
        Kokkos::Profiling::pushRegion(resize_updates_label.c_str());
        Kokkos::Profiling::popRegion();
        std::string clear_updates_label = std::string("Deduplication chkpt ") + 
                                          std::to_string(current_id) + 
                                          std::string(": Setup: Clear Update Map");
        Kokkos::Profiling::pushRegion(clear_updates_label.c_str());
        Kokkos::Profiling::popRegion();
      }
      Kokkos::Profiling::popRegion();

      std::string dedup_region_name = std::string("Deduplication chkpt ") + 
                                      std::to_string(current_id);
      Timer::time_point start_create_tree0 = Timer::now();
      timers[0] = std::chrono::duration_cast<Duration>(start_create_tree0 - beg_chkpt).count();
      Kokkos::Profiling::pushRegion(dedup_region_name.c_str());

      // Deduplicate data and identify nodes and chunks needed for the incremental checkpoint
      if((current_id == 0) || make_baseline) {
        baseline_id = current_id;
      }
      const uint8_t* data_ptr = reinterpret_cast<uint8_t*>(data.data());
      const size_t data_size = data.span()*sizeof(typename DataView::value_type);
      if(mode == Basic) {
        dedup_data_basic(leaves, changes_bitset, current_id, data_ptr, data_size, chunk_size);
      } else if(mode == List) {
        dedup_data_list(leaves, current_id, data, chunk_size, 
                             first_ocur_d, first_ocur_vec, shift_dupl_vec);
      } else if((mode == Tree) || (mode == TreeLowOffsetRef) || (mode == TreeLowOffset) || 
                (mode == TreeLowRootRef) || (mode == TreeLowRoot)) {
        if((current_id == 0) || make_baseline) {
          dedup_data_tree_baseline(data, chunk_size, tree, current_id, 
                                         first_ocur_d, shift_dupl_vec, first_ocur_vec);
          baseline_id = current_id;
        } else {
          // Different variations of the metadata compaction
          if((mode == Tree) || (mode == TreeLowOffset)) { 
            // Use the lowest offset to determine which node is the first occurrence
            dedup_data_tree_low_offset(data, chunk_size, tree, current_id, 
                                           first_ocur_d, shift_dupl_vec, first_ocur_vec);
          } else if((mode == TreeLowOffsetRef)) {
            // Reference code for the lowest offset
            dedup_low_offset_ref(data, chunk_size, tree, current_id, 
                                 first_ocur_d, shift_dupl_vec, first_ocur_vec);
          } else if(mode == TreeLowRootRef) {
            // Reference code for the lowest root
            dedup_low_root_ref(data, chunk_size, tree, current_id, 
                               first_ocur_d, shift_dupl_vec, first_ocur_vec);
          } else if((mode == TreeLowRoot)) {
            // Break ties for first occurrence based on which chunk produces the largest subtree
            dedup_data_tree_low_root(data, chunk_size, tree, current_id, 
                           first_ocur_d, shift_dupl_vec, first_ocur_vec);
          }
        }
      }

      Kokkos::Profiling::popRegion();
      Timer::time_point end_create_tree0 = Timer::now();
      timers[1] = std::chrono::duration_cast<Duration>(end_create_tree0 - start_create_tree0).count();

      // ==========================================================================================
      // Create Diff
      // ==========================================================================================
      Kokkos::View<uint8_t*> diff;
      std::string collect_region_name = std::string("Start writing incremental checkpoint ") 
                                + std::to_string(current_id);
      Timer::time_point start_collect = Timer::now();
      Kokkos::Profiling::pushRegion(collect_region_name.c_str());

      if(mode == Full) {
        // No need to create diff for full approach
        datasizes = std::make_pair(data.size(), 0);
      } else if(mode == Basic) {
        datasizes = write_diff_basic(data_ptr, data_size, diff, chunk_size, changes_bitset, 
                                     baseline_id, current_id, header);
      } else if(mode == List) {
          datasizes = write_diff_list(data, diff, chunk_size, leaves, 
            first_ocur_d, first_ocur_vec, shift_dupl_vec, baseline_id, current_id, header);
      } else if((mode == Tree) || (mode == TreeLowOffsetRef) || (mode == TreeLowOffset) || 
                (mode == TreeLowRootRef) || (mode == TreeLowRoot)) {
          datasizes = write_diff_tree(data, diff, chunk_size, 
                                                            tree, first_ocur_d, 
                                                            first_ocur_vec, shift_dupl_vec,
                                                            baseline_id, current_id, header);
      }

      Kokkos::Profiling::popRegion();
      Timer::time_point end_collect = Timer::now();
      timers[2] = std::chrono::duration_cast<Duration>(end_collect - start_collect).count();

      // ==========================================================================================
      // Copy diff to host 
      // ==========================================================================================
      Timer::time_point start_write = Timer::now();
      if(mode == Full) {
        Kokkos::resize(diff_h, data.size());
      } else {
        Kokkos::resize(diff_h, diff.size());
      }
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
      timers[3] = std::chrono::duration_cast<Duration>(end_write - start_write).count();
    }

    /**
     * Main checkpointing function. Given a raw device pointer, create an incremental checkpoint 
     * using the chosen checkpoint strategy. The deduplication mode can be one of the following:
     *   - Full: No deduplication
     *   - Basic: Remove chunks that have not changed since the previous checkpoint
     *   - List: Save a single copy of each unique chunk and use metadata to handle duplicates
     *   - Tree: Save minimal set of chunks and use a compact metadata representations
     *
     * \param dedup_mode    Checkpoint approach (Full|Basic|List|Tree)
     * \param data_ptr      Raw data pointer that needs to be deduplicated
     * \param len           Length of data
     * \param filename      Filename to save checkpoint
     * \param logname       Base filename for logs
     * \param make_baseline Flag determining whether to make a baseline checkpoint
     */
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

    /**
     * Main checkpointing function. Given a raw device pointer, create an incremental checkpoint 
     * using the chosen checkpoint strategy. Save checkpoint to host view. 
     * The deduplication mode can be one of the following:
     *   - Full: No deduplication
     *   - Basic: Remove chunks that have not changed since the previous checkpoint
     *   - List: Save a single copy of each unique chunk and use metadata to handle duplicates
     *   - Tree: Save minimal set of chunks and use a compact metadata representations
     *
     * \param dedup_mode    Checkpoint approach (Full|Basic|List|Tree)
     * \param data_ptr      Raw data pointer that needs to be deduplicated
     * \param len           Length of data
     * \param diff_h        Host View to store incremental checkpoint
     * \param make_baseline Flag determining whether to make a baseline checkpoint
     */
    void checkpoint(DedupMode dedup_mode, uint8_t* data_ptr, size_t len, 
                    Kokkos::View<uint8_t*>::HostMirror& diff_h, bool make_baseline) {
      Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
      header_t header;
      checkpoint(dedup_mode, header, data, diff_h, make_baseline);
      current_id += 1;
    }

    /**
     * Main checkpointing function. Given a raw device pointer, create an incremental checkpoint 
     * using the chosen checkpoint strategy. Save checkpoint to host view and write logs.
     * The deduplication mode can be one of the following:
     *   - Full: No deduplication
     *   - Basic: Remove chunks that have not changed since the previous checkpoint
     *   - List: Save a single copy of each unique chunk and use metadata to handle duplicates
     *   - Tree: Save minimal set of chunks and use a compact metadata representations
     *
     * \param dedup_mode    Checkpoint approach (Full|Basic|List|Tree)
     * \param data_ptr      Raw data pointer that needs to be deduplicated
     * \param len           Length of data
     * \param diff_h        Host View to store incremental checkpoint
     * \param logname       Base filename for logs
     * \param make_baseline Flag determining whether to make a baseline checkpoint
     */
    void checkpoint(DedupMode dedup_mode, uint8_t* data_ptr, size_t len, 
                    Kokkos::View<uint8_t*>::HostMirror& diff_h, std::string& logname, 
                    bool make_baseline) {
      Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
      header_t header;
      checkpoint(dedup_mode, header, data, diff_h, make_baseline);
      write_chkpt_log(header, diff_h, logname);
      current_id += 1;
    }

    /**
     * Function for writing the restart log.
     *
     * \param select_chkpt Which checkpoint to write the log
     * \param logname      Filename for writing log
     */
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

    /**
     * Restart checkpoint from vector of incremental checkpoints loaded on the Host.
     *
     * \param dedup_mode Deduplication approach
     * \param data       Data View to restart checkpoint into
     * \param chkpts     Vector of prior incremental checkpoints stored on the Host
     * \param logname    Filename for restart logs
     * \param chkpt_id   ID of checkpoint to restart
     */
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
        // Copy checkpoint to GPU
        Timer::time_point c1 = Timer::now();
        Kokkos::deep_copy(data, chkpts[chkpt_id]);
        Timer::time_point c2 = Timer::now();
        // Update timers
        restart_timers[0] = (1e-9)*(std::chrono::duration_cast<Nanoseconds>(c2-c1).count());
        restart_timers[1] = 0.0;
      } else if(dedup_mode == Basic) {
        auto basic_list_times = restart_chkpt_basic(chkpts, chkpt_id, data);
        restart_timers[0] = basic_list_times.first;
        restart_timers[1] = basic_list_times.second;
      } else if(dedup_mode == List) {
        auto list_times = restart_chkpt_list(chkpts, chkpt_id, data);
        restart_timers[0] = list_times.first;
        restart_timers[1] = list_times.second;
      } else if((mode == Tree) || (mode == TreeLowOffsetRef) || (mode == TreeLowOffset) || 
                (mode == TreeLowRootRef) || (mode == TreeLowRoot)) {
        auto tree_times = restart_chkpt_tree(chkpts, chkpt_id, data);
        restart_timers[0] = tree_times.first;
        restart_timers[1] = tree_times.second;
      }
      std::string restart_logname = logname + ".chunk_size." + std::to_string(chunk_size) +
                                    ".restart_timing.csv";
      write_restart_log(chkpt_id, restart_logname);
    } 

    /**
     * Restart checkpoint from vector of incremental checkpoints loaded on the Host. 
     * Store result into raw device pointer.
     *
     * \param dedup_mode Deduplication approach
     * \param data_ptr   Device pointer to save checkpoint in
     * \param len        Length of data
     * \param chkpts     Vector of prior incremental checkpoints stored on the Host
     * \param logname    Filename for restart logs
     * \param chkpt_id   ID of checkpoint to restart
     */
    void restart(DedupMode dedup_mode, uint8_t* data_ptr, size_t len, 
                 std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                 std::string& logname, uint32_t chkpt_id) {
      Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
      restart(dedup_mode, data, chkpts, logname, chkpt_id);
    }

    /**
     * Restart checkpoint from checkpoint files
     *
     * \param dedup_mode Deduplication approach
     * \param data       Data View to restart checkpoint into
     * \param filenames  Vector of prior incremental checkpoints stored in files
     * \param logname    Filename for restart logs
     * \param chkpt_id   ID of checkpoint to restart
     */
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
        //Timer::time_point r1 = Timer::now();
        file.read((char*)(data_h.data()), filesize);
        file.close();
        //Timer::time_point r2 = Timer::now();
        // Total time
        //Timer::time_point t1 = Timer::now();
        // Copy checkpoint to GPU
        Timer::time_point c1 = Timer::now();
        Kokkos::deep_copy(data, data_h);
        Timer::time_point c2 = Timer::now();
        //Timer::time_point t2 = Timer::now();
        // Update timers
        restart_timers[0] = (1e-9)*(std::chrono::duration_cast<Nanoseconds>(c2-c1).count());
        restart_timers[1] = 0.0;
      } else if(dedup_mode == Basic) {
        std::vector<std::string> basiclist_chkpt_files;
        for(uint32_t i=0; i<chkpt_filenames.size(); i++) {
          basiclist_chkpt_files.push_back(chkpt_filenames[i]+".basic.incr_chkpt");
        }
        auto basic_list_times = restart_chkpt_basic(basiclist_chkpt_files, chkpt_id, data);
        restart_timers[0] = basic_list_times.first;
        restart_timers[1] = basic_list_times.second;
      } else if(dedup_mode == List) {
        std::vector<std::string> hashlist_chkpt_files;
        for(uint32_t i=0; i<chkpt_filenames.size(); i++) {
          hashlist_chkpt_files.push_back(chkpt_filenames[i]+".hashlist.incr_chkpt");
        }
        auto list_times = restart_chkpt_list(hashlist_chkpt_files, chkpt_id, data);
        restart_timers[0] = list_times.first;
        restart_timers[1] = list_times.second;
      } else if((mode == Tree) || (mode == TreeLowOffsetRef) || (mode == TreeLowOffset) || 
                (mode == TreeLowRootRef) || (mode == TreeLowRoot)) {
        std::vector<std::string> hashtree_chkpt_files;
        for(uint32_t i=0; i<chkpt_filenames.size(); i++) {
          hashtree_chkpt_files.push_back(chkpt_filenames[i]+".hashtree.incr_chkpt");
        }
        auto tree_times = restart_chkpt_tree(hashtree_chkpt_files, chkpt_id, data);
        restart_timers[0] = tree_times.first;
        restart_timers[1] = tree_times.second;
      }
      write_restart_log(chkpt_id, logname);
    } 
};

#endif // DEDUPLICATOR_HPP
