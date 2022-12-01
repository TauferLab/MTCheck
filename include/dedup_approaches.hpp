#ifndef DEDUP_APPROACHES_HPP
#define DEDUP_APPROACHES_HPP
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
#include "utils.hpp"
//#include "update_pattern_analysis.hpp"
#include "deduplicator.hpp"

#define WRITE_CHKPT

void write_metadata_breakdown(std::fstream& fs, 
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

// Read checkpoints from files and "deduplicate" using the full approach 
// to measure time spent copying data as a point of comparison.
void full_chkpt(Hasher& hasher, 
                std::vector<std::string>& full_chkpt_files, 
                std::vector<std::string>& chkpt_filenames, 
                uint32_t chunk_size, 
                uint32_t num_chkpts) 
{
// TODO Probably could reduce arguments. Could get filenames 
// and number of checkpoints from single vector of strings
  using Timer = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double>;
  // Iterate through num_chkpts
  for(uint32_t idx=0; idx<num_chkpts; idx++) {
    // Open file and read/calc important values
    DEBUG_PRINT("Processing checkpoint %u\n", idx);
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    DEBUG_PRINT("Set exceptions\n");
    f.open(full_chkpt_files[idx], std::ifstream::in | std::ifstream::binary);
    DEBUG_PRINT("Opened file\n");
    f.seekg(0, f.end);
    DEBUG_PRINT("Seek end of file\n");
    size_t data_len = f.tellg();
    DEBUG_PRINT("Measure length of file\n");
    f.seekg(0, f.beg);
    DEBUG_PRINT("Seek beginning of file\n");
    DEBUG_PRINT("Length of checkpoint %u: %zd\n", idx, data_len);
    uint32_t num_chunks = data_len/chunk_size;
    if(num_chunks*chunk_size < data_len)
      num_chunks += 1;
    DEBUG_PRINT("Number of chunks: %u\n", num_chunks);

    // Open log file for results
    std::fstream result_data;
    std::string log_name = chkpt_filenames[idx]+".chunk_size."+std::to_string(chunk_size);
    result_data.open(log_name+".csv", std::fstream::out | std::fstream::app);

    // Open logfile for timing and checkpoint size
    std::fstream timing_file, size_file;
    timing_file.open(log_name+".timing.csv", std::fstream::out | std::fstream::app);
    size_file.open(log_name+".size.csv", std::fstream::out | std::fstream::app);

    // Read checkpoint file and load it into the device
    Kokkos::View<uint8_t*> current("Current region", data_len);
    Kokkos::View<uint8_t*>::HostMirror current_h = Kokkos::create_mirror_view(current);
    f.read((char*)(current_h.data()), data_len);
    Kokkos::deep_copy(current, current_h);
    DEBUG_PRINT("Size of full checkpoint: %zd\n", current.size());
    f.close();

    // Full checkpoint
    // Open output checkpoint file
    std::string filename = full_chkpt_files[idx] + ".full_chkpt";
    std::ofstream file;
    file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    file.open(filename, std::ofstream::out | std::ofstream::binary);
    DEBUG_PRINT("Opened full checkpoint file\n");

    // Record time spent on copying the data into the Device
    std::string region_name = std::string("Write full checkpoint ") + std::to_string(idx);
    Timer::time_point start_write = Timer::now();
    Kokkos::Profiling::pushRegion(region_name.c_str());
    Kokkos::deep_copy(current_h, current);
    Kokkos::Profiling::popRegion();
    Timer::time_point end_write = Timer::now();
    auto elapsed = end_write-start_write;
    auto write_time = std::chrono::duration_cast<Duration>(elapsed).count();
    DEBUG_PRINT("Wrote full checkpoint\n");

    // Write checkpoint to file
    file.write((char*)(current_h.data()), current_h.size());
    STDOUT_PRINT("Time spent writing full checkpoint: %f\n", write_time);

    // Write timing and checkpoint size to logs
    result_data << "0.0" << ","          // Comparison time
                << "0.0" << ","          // Collection time
                << write_time << ","     // Write time
                << current.size() << ',' // Size of data
                << "0" << ',';           // Size of metadata
    timing_file << "Full" << ","     // Approach
                << idx << ","        // Checkpoint ID
                << chunk_size << "," // Chunk size
                << "0.0" << ","      // Comparison time
                << "0.0" << ","      // Collection time
                << write_time        // Write time
                << std::endl;
    size_file << "Full" << ","         // Approach
              << idx << ","            // Checkpoint ID
              << chunk_size << ","     // Chunk size
              << current.size() << "," // Size of data
              << "0,"                  // Size of metadata
              << "0,"                  // Size of header
              << "0,"                  // Size of distinct metadata
              << "0";                  // Size of repeat map
    for(uint32_t j=0; j<num_chkpts; j++) {
      size_file << ",0"; // Size of metadata for checkpoint j
    }
    size_file << std::endl;

    // Close files
    DEBUG_PRINT("Closing files\n");
    file.close();
    result_data.close();
    timing_file.close();
    size_file.close();
    STDOUT_PRINT("------------------------------------------------------\n");
  }
  STDOUT_PRINT("------------------------------------------------------\n");
}

// Read checkpoints from files and deduplicate using the incremental matching approach.
// Divide data into chunks and compute hashes for chunks. Use hashes to detect when a
// chunk has been changed and collect those changes into a diff. Note that chunk i only
// compares hashes with chunks i in previous checkpoints
template<typename Hasher>
void naive_chkpt(Hasher& hasher, 
                Kokkos::View<uint8_t**,Kokkos::LayoutLeft,Kokkos::DefaultHostExecutionSpace>& chkpts_h,
                std::vector<std::vector<uint8_t>>& incr_chkpts,
                std::string& log_filename,
                const uint32_t chunk_size, 
                const uint32_t num_chkpts) 
{
  using Duration = std::chrono::duration<double>;
  using Timer = std::chrono::high_resolution_clock;

  // Keep copy of prvious hash list to detect when things have changed
  HashList prev_list(0);

  // Iterate through checkpoints and deduplicate
  for(uint32_t idx=0; idx<num_chkpts; idx++) {
    size_t data_len = chkpts_h.extent(0);
    uint32_t num_chunks = data_len/chunk_size;
    if(num_chunks*chunk_size < data_len)
      num_chunks += 1;
    Kokkos::resize(prev_list.list_d, num_chunks);
    Kokkos::resize(prev_list.list_h, num_chunks);

    Kokkos::View<uint8_t*> current("Current region", data_len);

    std::string log_name = log_filename+".chunk_size."+std::to_string(chunk_size);
    std::fstream result_data;
    std::fstream timing_file, size_file;

    if(log_filename.size() > 0) {
      // Open result file
      std::string result_name = log_name +".csv";
      result_data.open(result_name, std::fstream::out | std::fstream::app);

      // Open log files for timings and checkpoint sizes
      std::string timing_name = log_name +".timing.csv";
      std::string size_name = log_name +".size.csv";
      timing_file.open(timing_name, std::fstream::out | std::fstream::app);
      size_file.open(size_name, std::fstream::out | std::fstream::app);
    }

    // Read data into memory and copy to Device
    auto chkpt_view = Kokkos::subview(chkpts_h, Kokkos::ALL, idx);
    Kokkos::deep_copy(current, chkpt_view);

    // Create list for hashes and bitset for identifying changes
    HashList list0 = HashList(num_chunks);
    Kokkos::Bitset<Kokkos::DefaultExecutionSpace> changes_bitset(num_chunks);
    DEBUG_PRINT("initialized local maps and list\n");
    Kokkos::fence();

    // Compare lists and record time taken
    Timer::time_point start_compare = Timer::now();
    Kokkos::Profiling::pushRegion((std::string("Find distinct chunks ") + std::to_string(idx)).c_str());
    compare_lists_naive(hasher, prev_list, list0, changes_bitset, idx, current, chunk_size);
    Kokkos::Profiling::popRegion();
    Timer::time_point end_compare = Timer::now();
    Kokkos::fence();
    auto elapsed = end_compare-start_compare;
    auto compare_time = std::chrono::duration_cast<Duration>(elapsed).count();
    // Update previous list with new hashes
  	Kokkos::deep_copy(prev_list.list_d, list0.list_d);

    // Create buffer from diff
    uint32_t prior_idx = 0;
    Kokkos::fence();
    std::string region_name = std::string("Writing incremental checkpoint ") + std::to_string(idx);
    Timer::time_point start_collect = Timer::now();
    Kokkos::Profiling::pushRegion(region_name.c_str());
    Kokkos::View<uint8_t*> buffer_d;
    header_t header;
    std::pair<uint64_t,uint64_t> datasizes = write_incr_chkpt_hashlist_naive(current, 
                                                                             buffer_d, 
                                                                             chunk_size, 
                                                                             changes_bitset, 
                                                                             prior_idx, 
                                                                             idx, 
                                                                             header);
    Kokkos::Profiling::popRegion();
    Timer::time_point end_collect = Timer::now();
    auto collect_time = std::chrono::duration_cast<Duration>(end_collect - start_collect).count();
    STDOUT_PRINT("Time spect collecting updates: %f\n", collect_time);

    // Record time taken to copy deduplicated data to host
    Timer::time_point start_write = Timer::now();
    Kokkos::Profiling::pushRegion(region_name.c_str());
    auto buffer_h = Kokkos::create_mirror_view(buffer_d);
    Kokkos::deep_copy(buffer_h, buffer_d);
    memcpy(buffer_h.data(), &header, sizeof(header_t));
    Kokkos::fence();
    Kokkos::Profiling::popRegion();
    Timer::time_point end_write = Timer::now();
    auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
    STDOUT_PRINT("Time spect copying updates: %f\n", write_time);

    // Write timing data and checkpoint breakdown to logs
    if(log_filename.size() > 0) {
      result_data << compare_time << ','      // Comparison time
                  << collect_time << ','      // Collection time  
                  << write_time << ','        // Write time
                  << datasizes.first << ','   // Size of data     
                  << datasizes.second << ','; // Size of metadata 
      timing_file << "Naive" << ","      // Approach
                  << idx << ","          // Checkpoint ID
                  << chunk_size << ","   // Chunk size      
                  << compare_time << "," // Comparison time 
                  << collect_time << "," // Collection time 
                  << write_time          // Write time
                  << std::endl;
      size_file << "Naive" << ","           // Approach
                << idx << ","               // Checkpoint ID
                << chunk_size << ","        // Chunk size
                << datasizes.first << ","   // Size of data
                << datasizes.second << ","; // Size of metadata
      write_metadata_breakdown(size_file, header, buffer_h, num_chkpts);
    }

    // Store incremental checkpoints
    if(idx == num_chkpts-1) {
      std::vector<uint8_t> vector_buf = std::vector<uint8_t>(buffer_h.extent(0));
      memcpy(vector_buf.data(), buffer_h.data(), buffer_h.extent(0));
      incr_chkpts.push_back(vector_buf);
    }

    Kokkos::fence();
    // Close files
    DEBUG_PRINT("Closing files\n");
    if(log_filename.size() > 0) {
      result_data.close();
      timing_file.close();
      size_file.close();
    }
    STDOUT_PRINT("------------------------------------------------------\n");
  }
  STDOUT_PRINT("------------------------------------------------------\n");
}

// Read checkpoints from files and deduplicate using the incremental matching approach.
// Divide data into chunks and compute hashes for chunks. Use hashes to detect when a
// chunk has been changed and collect those changes into a diff. Note that chunk i only
// compares hashes with chunks i in previous checkpoints
template<typename Hasher>
void naive_chkpt(Hasher& hasher, 
                std::vector<std::string>& full_chkpt_files, 
                std::vector<std::string>& chkpt_filenames, 
                uint32_t chunk_size, 
                uint32_t num_chkpts) 
{
  using Duration = std::chrono::duration<double>;
  using Timer = std::chrono::high_resolution_clock;

  // Keep copy of prvious hash list to detect when things have changed
  HashList prev_list(0);

  // Iterate through checkpoints and deduplicate
  for(uint32_t idx=0; idx<num_chkpts; idx++) {
    // Open file and setup important values
    DEBUG_PRINT("Processing checkpoint %u\n", idx);
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    DEBUG_PRINT("Set exceptions\n");
    f.open(full_chkpt_files[idx], std::ifstream::in | std::ifstream::binary);
    DEBUG_PRINT("Opened file\n");
    f.seekg(0, f.end);
    DEBUG_PRINT("Seek end of file\n");
    size_t data_len = f.tellg();
    DEBUG_PRINT("Measure length of file\n");
    f.seekg(0, f.beg);
    DEBUG_PRINT("Seek beginning of file\n");
    DEBUG_PRINT("Length of checkpoint %u: %zd\n", idx, data_len);
    uint32_t num_chunks = data_len/chunk_size;
    if(num_chunks*chunk_size < data_len)
      num_chunks += 1;
    DEBUG_PRINT("Number of chunks: %u\n", num_chunks);
    if(idx == 0) {
      Kokkos::resize(prev_list.list_d, num_chunks);
      Kokkos::resize(prev_list.list_h, num_chunks);
    }

    std::string log_name = chkpt_filenames[idx]+".chunk_size."+std::to_string(chunk_size);
    // Open result file
    std::fstream result_data;
    std::string result_name = log_name +".csv";
    result_data.open(result_name, std::fstream::out | std::fstream::app);

    // Open log files for timings and checkpoint sizes
    std::fstream timing_file, size_file;
    std::string timing_name = log_name +".timing.csv";
    std::string size_name = log_name +".size.csv";
    timing_file.open(timing_name, std::fstream::out | std::fstream::app);
    size_file.open(size_name, std::fstream::out | std::fstream::app);

    // Read data into memory and copy to Device
    Kokkos::View<uint8_t*> current("Current region", data_len);
    Kokkos::View<uint8_t*>::HostMirror current_h = Kokkos::create_mirror_view(current);
    f.read((char*)(current_h.data()), data_len);
    Kokkos::deep_copy(current, current_h);
    DEBUG_PRINT("Size of full checkpoint: %zd\n", current.size());
    f.close();

    // Create list for hashes and bitset for identifying changes
    HashList list0 = HashList(num_chunks);
    Kokkos::Bitset<Kokkos::DefaultExecutionSpace> changes_bitset(num_chunks);
    DEBUG_PRINT("initialized local maps and list\n");
    Kokkos::fence();

    // Compare lists and record time taken
    Timer::time_point start_compare = Timer::now();
    Kokkos::Profiling::pushRegion((std::string("Find distinct chunks ") + std::to_string(idx)).c_str());
    compare_lists_naive(hasher, prev_list, list0, changes_bitset, idx, current, chunk_size);
    Kokkos::Profiling::popRegion();
    Timer::time_point end_compare = Timer::now();
    Kokkos::fence();
    auto elapsed = end_compare-start_compare;
    auto compare_time = std::chrono::duration_cast<Duration>(elapsed).count();
    // Update previous list with new hashes
  	Kokkos::deep_copy(prev_list.list_d, list0.list_d);

    // Create buffer from diff
    uint32_t prior_idx = 0;
    Kokkos::fence();
    std::string output_name = full_chkpt_files[idx]+".naivehashlist.incr_chkpt";
    std::string region_name = std::string("Writing incremental checkpoint ") + std::to_string(idx);
    Timer::time_point start_collect = Timer::now();
    Kokkos::Profiling::pushRegion(region_name.c_str());
    Kokkos::View<uint8_t*> buffer_d;
    header_t header;
    std::pair<uint64_t,uint64_t> datasizes = write_incr_chkpt_hashlist_naive(current, 
                                                                             buffer_d, 
                                                                             chunk_size, 
                                                                             changes_bitset, 
                                                                             prior_idx, 
                                                                             idx, 
                                                                             header);
    Kokkos::Profiling::popRegion();
    Timer::time_point end_collect = Timer::now();
    auto collect_time = std::chrono::duration_cast<Duration>(end_collect - start_collect).count();
    STDOUT_PRINT("Time spect collecting updates: %f\n", collect_time);

    // Record time taken to copy deduplicated data to host
    Timer::time_point start_write = Timer::now();
    Kokkos::Profiling::pushRegion(region_name.c_str());
    auto buffer_h = Kokkos::create_mirror_view(buffer_d);
    Kokkos::deep_copy(buffer_h, buffer_d);
    memcpy(buffer_h.data(), &header, sizeof(header_t));
    Kokkos::fence();
    Kokkos::Profiling::popRegion();
    Timer::time_point end_write = Timer::now();
    auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
    STDOUT_PRINT("Time spect copying updates: %f\n", write_time);

    // Write timing data and checkpoint breakdown to logs
    result_data << compare_time << ','      // Comparison time
                << collect_time << ','      // Collection time  
                << write_time << ','        // Write time
                << datasizes.first << ','   // Size of data     
                << datasizes.second << ','; // Size of metadata 
    timing_file << "Naive" << ","      // Approach
                << idx << ","          // Checkpoint ID
                << chunk_size << ","   // Chunk size      
                << compare_time << "," // Comparison time 
                << collect_time << "," // Collection time 
                << write_time          // Write time
                << std::endl;
    size_file << "Naive" << ","           // Approach
              << idx << ","               // Checkpoint ID
              << chunk_size << ","        // Chunk size
              << datasizes.first << ","   // Size of data
              << datasizes.second << ","; // Size of metadata
    write_metadata_breakdown(size_file, header, buffer_h, num_chkpts);

#ifdef WRITE_CHKPT
    // Write deduplicate data to file
    std::ofstream file;
    file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    file.open(full_chkpt_files[idx]+".naivehashlist.incr_chkpt", std::ofstream::out | std::ofstream::binary);
//    file.write((char*)(&header), sizeof(header_t));
    file.write((const char*)(buffer_h.data()), buffer_h.size());
    file.flush();
    file.close();
#endif
    Kokkos::fence();
    // Close files
    DEBUG_PRINT("Closing files\n");
    result_data.close();
    timing_file.close();
    size_file.close();
    STDOUT_PRINT("------------------------------------------------------\n");
  }
  STDOUT_PRINT("------------------------------------------------------\n");
}

template<typename Hasher>
void list_chkpt(Hasher& hasher, 
                Kokkos::View<uint8_t**,Kokkos::LayoutLeft,Kokkos::DefaultHostExecutionSpace>& chkpts_h,
                std::vector<std::vector<uint8_t>>& incr_chkpts,
                std::string& log_filename,
                uint32_t chunk_size, 
                uint32_t num_chkpts,
                int lineage_based) 
{
  using Timer = std::chrono::high_resolution_clock;
  DistinctNodeIDMap g_distinct_chunks = DistinctNodeIDMap(1);
  SharedNodeIDMap g_shared_chunks = SharedNodeIDMap(1);
  SharedNodeIDMap g_identical_chunks = SharedNodeIDMap(1);

  // Iterate through checkpoints and deduplicate
  for(uint32_t idx=0; idx<num_chkpts; idx++) {
    size_t data_len = chkpts_h.extent(0);
    uint32_t num_chunks = data_len/chunk_size;
    if(num_chunks*chunk_size < data_len)
      num_chunks += 1;
    if(idx == 0) {
      g_shared_chunks.rehash(num_chunks);
      g_identical_chunks.rehash(num_chunks);
    }
    g_distinct_chunks.rehash(g_distinct_chunks.size()+num_chunks);

    std::string log_name = log_filename+".chunk_size."+std::to_string(chunk_size);
    std::fstream result_data;
    std::fstream timing_file, size_file;

    if(log_filename.size() > 0) {
      // Open result file
      std::string result_name = log_name +".csv";
      result_data.open(result_name, std::fstream::out | std::fstream::app);

      // Open log files for timings and checkpoint sizes
      std::string timing_name = log_name +".timing.csv";
      std::string size_name = log_name +".size.csv";
      timing_file.open(timing_name, std::fstream::out | std::fstream::app);
      size_file.open(size_name, std::fstream::out | std::fstream::app);
    }

    // Read data into memory and copy to Device
    Kokkos::View<uint8_t*> current("Current region", data_len);
    auto chkpt_view = Kokkos::subview(chkpts_h, Kokkos::ALL, idx);
    Kokkos::deep_copy(current, chkpt_view);

    // Create tables for hash list deduplication
    DistinctNodeIDMap l_distinct_chunks = DistinctNodeIDMap(num_chunks);
    SharedNodeIDMap l_shared_chunks     = SharedNodeIDMap(num_chunks);
    SharedNodeIDMap l_identical_chunks     = SharedNodeIDMap(num_chunks);
//    g_distinct_chunks.rehash(num_chunks);
//    g_shared_chunks.rehash(num_chunks);

    HashList list0 = HashList(num_chunks);
    Kokkos::fence();

    uint32_t num_distinct = g_distinct_chunks.size();
    Kokkos::fence();
    Timer::time_point start_compare = Timer::now();
    Kokkos::Profiling::pushRegion((std::string("Find distinct chunks ") + std::to_string(idx)).c_str());
    if(lineage_based) {
      compare_lists_global(hasher, list0, idx, current, chunk_size, l_identical_chunks, l_shared_chunks, g_distinct_chunks, g_identical_chunks, g_shared_chunks, g_distinct_chunks);
    } else {
      compare_lists_local(hasher, list0, idx, current, chunk_size, l_shared_chunks, l_distinct_chunks, g_shared_chunks, g_distinct_chunks);
    }
    Kokkos::Profiling::popRegion();
    Timer::time_point end_compare = Timer::now();

    Kokkos::fence();

    auto compare_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_compare - start_compare).count();
    if(lineage_based) {
      // Update global repeat map
      Kokkos::deep_copy(g_shared_chunks, l_shared_chunks);
      Kokkos::deep_copy(g_identical_chunks, l_identical_chunks);
    } else if(idx == 0) {
      // Update global distinct map
      Kokkos::deep_copy(g_distinct_chunks, l_distinct_chunks);
      // Update global shared map
      Kokkos::deep_copy(g_shared_chunks, l_shared_chunks);
      DEBUG_PRINT("Updated global lists\n");
    }

    uint32_t prior_idx = 0;
    Kokkos::fence();
    Timer::time_point start_collect = Timer::now();
    Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
    Kokkos::View<uint8_t*> buffer_d;
    header_t header;
    std::pair<uint64_t,uint64_t> datasizes;
    if(lineage_based) {
      datasizes = write_incr_chkpt_hashlist_global(current, buffer_d, chunk_size, g_distinct_chunks, l_shared_chunks, prior_idx, idx, header);
    } else {
      datasizes = write_incr_chkpt_hashlist_local(current, buffer_d, chunk_size, l_distinct_chunks, l_shared_chunks, prior_idx, idx, header);
    }
    Kokkos::Profiling::popRegion();
    Timer::time_point end_collect = Timer::now();
    auto collect_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_collect - start_collect).count();
    STDOUT_PRINT("Time spect collecting updates: %f\n", collect_time);

    Timer::time_point start_write = Timer::now();
    Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
    auto buffer_h = Kokkos::create_mirror_view(buffer_d);
    Kokkos::deep_copy(buffer_h, buffer_d);
    memcpy(buffer_h.data(), &header, sizeof(header_t));
    Kokkos::fence();
    Kokkos::Profiling::popRegion();
    Timer::time_point end_write = Timer::now();
    auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
    STDOUT_PRINT("Time spect copying updates: %f\n", write_time);
    if(log_filename.size() > 0) {
      result_data << compare_time << ','      // Comparison time
                  << collect_time << ','      // Collection time  
                  << write_time << ','        // Write time
                  << datasizes.first << ','   // Size of data     
                  << datasizes.second << ','; // Size of metadata 
      timing_file << "List" << ","      // Approach
                  << idx << ","          // Checkpoint ID
                  << chunk_size << ","   // Chunk size      
                  << compare_time << "," // Comparison time 
                  << collect_time << "," // Collection time 
                  << write_time          // Write time
                  << std::endl;
      size_file << "List" << ","           // Approach
                << idx << ","               // Checkpoint ID
                << chunk_size << ","        // Chunk size
                << datasizes.first << ","   // Size of data
                << datasizes.second << ","; // Size of metadata
      write_metadata_breakdown(size_file, header, buffer_h, num_chkpts);
    }

    if(idx == num_chkpts-1) {
      std::vector<uint8_t> vector_buf = std::vector<uint8_t>(buffer_h.extent(0));
      memcpy(vector_buf.data(), buffer_h.data(), buffer_h.extent(0));
      incr_chkpts.push_back(vector_buf);
    }

    Kokkos::fence();
    if(log_filename.size() > 0) {
      result_data.close();
      timing_file.close();
      size_file.close();
    }
    STDOUT_PRINT("------------------------------------------------------\n");
  }
  STDOUT_PRINT("------------------------------------------------------\n");
}

template<typename Hasher>
void list_chkpt(Hasher& hasher, 
                std::vector<std::string>& full_chkpt_files, 
                std::vector<std::string>& chkpt_filenames, 
                uint32_t chunk_size, 
                uint32_t num_chkpts) 
{
  using Timer = std::chrono::high_resolution_clock;
  DistinctNodeIDMap g_distinct_chunks = DistinctNodeIDMap(1);
  SharedNodeIDMap g_shared_chunks = SharedNodeIDMap(1);
  SharedNodeIDMap g_identical_chunks = SharedNodeIDMap(1);

  for(uint32_t idx=0; idx<num_chkpts; idx++) {
    DEBUG_PRINT("Processing checkpoint %u\n", idx);
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    DEBUG_PRINT("Set exceptions\n");
    f.open(full_chkpt_files[idx], std::ifstream::in | std::ifstream::binary);
    DEBUG_PRINT("Opened file\n");
    f.seekg(0, f.end);
    DEBUG_PRINT("Seek end of file\n");
    size_t data_len = f.tellg();
    DEBUG_PRINT("Measure length of file\n");
    f.seekg(0, f.beg);
    DEBUG_PRINT("Seek beginning of file\n");
    DEBUG_PRINT("Length of checkpoint %u: %zd\n", idx, data_len);
    uint32_t num_chunks = data_len/chunk_size;
    if(num_chunks*chunk_size < data_len)
      num_chunks += 1;
    DEBUG_PRINT("Number of chunks: %u\n", num_chunks);
    if(idx == 0) {
      g_distinct_chunks.rehash(g_distinct_chunks.size()+num_chunks);
      g_shared_chunks.rehash(num_chunks);
      g_identical_chunks.rehash(num_chunks);
    }

    std::fstream result_data;
    result_data.open(chkpt_filenames[idx]+".chunk_size."+std::to_string(chunk_size)+".csv", std::fstream::out | std::fstream::app);

    std::fstream timing_file, size_file;
    timing_file.open(chkpt_filenames[idx]+".chunk_size."+std::to_string(chunk_size)+".timing.csv", std::fstream::out | std::fstream::app);
    size_file.open(chkpt_filenames[idx]+".chunk_size."+std::to_string(chunk_size)+".size.csv", std::fstream::out | std::fstream::app);

    Kokkos::View<uint8_t*> current("Current region", data_len);
    Kokkos::View<uint8_t*>::HostMirror current_h = Kokkos::create_mirror_view(current);
    f.read((char*)(current_h.data()), data_len);
    Kokkos::deep_copy(current, current_h);
    DEBUG_PRINT("Size of full checkpoint: %zd\n", current.size());
    f.close();

    // Hash list deduplication
    {
      DistinctNodeIDMap l_distinct_chunks = DistinctNodeIDMap(num_chunks);
      SharedNodeIDMap l_shared_chunks     = SharedNodeIDMap(num_chunks);
      SharedNodeIDMap l_identical_chunks     = SharedNodeIDMap(num_chunks);
      g_distinct_chunks.rehash(num_chunks);
      g_shared_chunks.rehash(num_chunks);

      HashList list0 = HashList(num_chunks);
      DEBUG_PRINT("initialized local maps and list\n");
      Kokkos::fence();

      uint32_t num_distinct = g_distinct_chunks.size();
      Kokkos::fence();
      Timer::time_point start_compare = Timer::now();
      Kokkos::Profiling::pushRegion((std::string("Find distinct chunks ") + std::to_string(idx)).c_str());
#ifdef GLOBAL_TABLE
//      compare_lists_global(hasher, list0, idx, current, chunk_size, l_shared_chunks, g_distinct_chunks, g_shared_chunks, g_distinct_chunks);
      compare_lists_global(hasher, list0, idx, current, chunk_size, l_identical_chunks, l_shared_chunks, g_distinct_chunks, g_identical_chunks, g_shared_chunks, g_distinct_chunks);
#else
      compare_lists_local(hasher, list0, idx, current, chunk_size, l_shared_chunks, l_distinct_chunks, g_shared_chunks, g_distinct_chunks);
#endif
      Kokkos::Profiling::popRegion();
      Timer::time_point end_compare = Timer::now();

      Kokkos::fence();
      

#ifdef GLOBAL_TABLE
      STDOUT_PRINT("Size of distinct map: %u\n", g_distinct_chunks.size()-num_distinct);
      STDOUT_PRINT("Size of shared map:   %u\n", l_shared_chunks.size());
#else
      STDOUT_PRINT("Size of distinct map: %u\n", l_distinct_chunks.size());
      STDOUT_PRINT("Size of shared map:   %u\n", l_shared_chunks.size());
#endif

      auto compare_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_compare - start_compare).count();
#ifdef GLOBAL_TABLE
      // Update global repeat map
      Kokkos::deep_copy(g_shared_chunks, l_shared_chunks);
      Kokkos::deep_copy(g_identical_chunks, l_identical_chunks);
#else
      if(idx == 0) {
        // Update global distinct map
        Kokkos::deep_copy(g_distinct_chunks, l_distinct_chunks);
        // Update global shared map
        Kokkos::deep_copy(g_shared_chunks, l_shared_chunks);
        DEBUG_PRINT("Updated global lists\n");
      }
#endif

#ifdef WRITE_CHKPT
      uint32_t prior_idx = 0;
      Kokkos::fence();
      Timer::time_point start_collect = Timer::now();
      Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
      Kokkos::View<uint8_t*> buffer_d;
      header_t header;
#ifdef GLOBAL_TABLE
      std::pair<uint64_t,uint64_t> datasizes = write_incr_chkpt_hashlist_global(current, buffer_d, chunk_size, g_distinct_chunks, l_shared_chunks, prior_idx, idx, header);
#else
      std::pair<uint64_t,uint64_t> datasizes = write_incr_chkpt_hashlist_local( current, buffer_d, chunk_size, l_distinct_chunks, l_shared_chunks, prior_idx, idx, header);
#endif
      Kokkos::Profiling::popRegion();
      Timer::time_point end_collect = Timer::now();
      auto collect_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_collect - start_collect).count();
      STDOUT_PRINT("Time spect collecting updates: %f\n", collect_time);

      Timer::time_point start_write = Timer::now();
      Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
      auto buffer_h = Kokkos::create_mirror_view(buffer_d);
      Kokkos::deep_copy(buffer_h, buffer_d);
//      memcpy(buffer_h.data(), &header, sizeof(header_t));
      Kokkos::fence();
      Kokkos::Profiling::popRegion();
      Timer::time_point end_write = Timer::now();
      auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
      STDOUT_PRINT("Time spect copying updates: %f\n", write_time);
      result_data << compare_time << ',' << collect_time << ',' << write_time << ',' << datasizes.first << ',' << datasizes.second << ',';
      timing_file << "List" << "," << idx << "," << chunk_size << "," << compare_time << "," << collect_time << "," << write_time << std::endl;
      size_file << "List" << "," << idx << "," << chunk_size << "," << datasizes.first << "," << datasizes.second << ",";
      STDOUT_PRINT("Ref ID:           %u\n" , header.ref_id);
      STDOUT_PRINT("Chkpt ID:         %u\n" , header.chkpt_id);
      STDOUT_PRINT("Data len:         %lu\n", header.datalen);
      STDOUT_PRINT("Chunk size:       %u\n" , header.chunk_size);
      STDOUT_PRINT("Window size:      %u\n" , header.window_size);
      STDOUT_PRINT("Distinct size:    %u\n" , header.distinct_size);
      STDOUT_PRINT("Curr repeat size: %u\n" , header.curr_repeat_size);
      STDOUT_PRINT("Prev repeat size: %u\n" , header.prev_repeat_size);
      STDOUT_PRINT("Num prior chkpts: %u\n" , header.num_prior_chkpts);
      write_metadata_breakdown(size_file, header, buffer_h, num_chkpts);

      std::ofstream file;
      file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
      file.open(full_chkpt_files[idx]+".hashlist.incr_chkpt", std::ofstream::out | std::ofstream::binary);
      uint64_t dlen = current.size();
      uint32_t repeatlen = l_shared_chunks.size();
      uint32_t distinctlen = l_distinct_chunks.size();
#ifdef GLOBAL_TABLE
      distinctlen = g_distinct_chunks.size();
#endif
      file.write((char*)(&header), sizeof(header_t));
      file.write((const char*)(buffer_h.data()), buffer_h.size());
      file.flush();
      file.close();
#endif
    }
    Kokkos::fence();
    DEBUG_PRINT("Closing files\n");
    result_data.close();
    timing_file.close();
    size_file.close();
    STDOUT_PRINT("------------------------------------------------------\n");
  }
  STDOUT_PRINT("------------------------------------------------------\n");
}

template<typename Hasher>
void tree_chkpt(Hasher& hasher, 
                std::vector<std::string>& full_chkpt_files, 
                std::vector<std::string>& chkpt_filenames, 
                uint32_t chunk_size, 
                uint32_t num_chkpts) {
  using Timer = std::chrono::high_resolution_clock;
  DistinctNodeIDMap g_distinct_nodes  = DistinctNodeIDMap(1);
#ifndef GLOBAL_TABLE
  SharedNodeIDMap g_shared_nodes = SharedNodeIDMap(1);
#endif
  SharedNodeIDMap g_shared_nodes = SharedNodeIDMap(1);
  SharedNodeIDMap g_identical_nodes = SharedNodeIDMap(1);
//  NodeMap g_nodes = NodeMap(1);

  std::ifstream fs;
  fs.open(full_chkpt_files[0], std::ifstream::in | std::ifstream::binary);
  fs.seekg(0, fs.end);
  size_t data_length = fs.tellg();
  fs.seekg(0, fs.beg);
  fs.close();
  uint32_t n_chunks = data_length/chunk_size;
  if(n_chunks*chunk_size < data_length)
    n_chunks += 1;
  MerkleTree tree0 = MerkleTree(n_chunks);
  HashList leaves = HashList(n_chunks);

  for(uint32_t idx=0; idx<num_chkpts; idx++) {
    DEBUG_PRINT("Processing checkpoint %u\n", idx);
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    DEBUG_PRINT("Set exceptions\n");
    f.open(full_chkpt_files[idx], std::ifstream::in | std::ifstream::binary);
    DEBUG_PRINT("Opened file\n");
    f.seekg(0, f.end);
    DEBUG_PRINT("Seek end of file\n");
    size_t data_len = f.tellg();
    DEBUG_PRINT("Measure length of file\n");
    f.seekg(0, f.beg);
    DEBUG_PRINT("Seek beginning of file\n");
    DEBUG_PRINT("Length of checkpoint %u: %zd\n", idx, data_len);
    uint32_t num_chunks = data_len/chunk_size;
    if(num_chunks*chunk_size < data_len)
      num_chunks += 1;
    DEBUG_PRINT("Number of chunks: %u\n", num_chunks);
    if(idx == 0) {
      g_distinct_nodes.rehash(g_distinct_nodes.size()+2*num_chunks - 1);
#ifndef GLOBAL_TABLE
      g_shared_nodes.rehash(2*num_chunks - 1);
#endif
      g_shared_nodes.rehash(2*num_chunks - 1);
      g_identical_nodes.rehash(2*num_chunks - 1);
//      g_nodes.rehash(2*num_chunks - 1);
//      g_nodes.rehash(num_chunks);
    }

    std::fstream result_data;
    result_data.open(chkpt_filenames[idx]+".chunk_size."+std::to_string(chunk_size)+".csv", std::fstream::out | std::fstream::app);

    std::fstream timing_file, size_file;
    timing_file.open(chkpt_filenames[idx]+".chunk_size."+std::to_string(chunk_size)+".timing.csv", std::fstream::out | std::fstream::app);
    size_file.open(chkpt_filenames[idx]+".chunk_size."+std::to_string(chunk_size)+".size.csv", std::fstream::out | std::fstream::app);

    Kokkos::View<uint8_t*> current("Current region", data_len);
    Kokkos::View<uint8_t*>::HostMirror current_h = Kokkos::create_mirror_view(current);
    f.read((char*)(current_h.data()), data_len);
    Kokkos::deep_copy(current, current_h);
    DEBUG_PRINT("Size of full checkpoint: %zd\n", current.size());
    f.close();

    // Merkle Tree deduplication
    {

//      MerkleTree tree0 = MerkleTree(num_chunks);
      DEBUG_PRINT("Allocated tree and tables\n");

      Kokkos::fence();

      if(idx == 0) {
        Timer::time_point start_create_tree0 = Timer::now();
        Kokkos::Profiling::pushRegion((std::string("Deduplicate chkpt ") + std::to_string(idx)).c_str());
        create_merkle_tree_deterministic(hasher, tree0, current, chunk_size, idx, g_distinct_nodes, g_shared_nodes);
//        create_merkle_tree(hasher, tree0, current, chunk_size, idx, g_distinct_nodes, g_shared_nodes);
//        create_merkle_tree(hasher, tree0, current, chunk_size, idx, g_distinct_nodes, g_nodes);
        Kokkos::Profiling::popRegion();
        Timer::time_point end_create_tree0 = Timer::now();
        Kokkos::deep_copy(tree0.tree_h, tree0.tree_d);

//        STDOUT_PRINT("Size of shared entries: %u\n", g_shared_nodes.size());
        STDOUT_PRINT("Size of distinct entries: %u\n", g_distinct_nodes.size());
//        STDOUT_PRINT("Size of shared updates: %u\n", shared_updates.size());
//        STDOUT_PRINT("Size of distinct updates: %u\n", distinct_updates.size());
        auto compare_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_create_tree0 - start_create_tree0).count();
#ifdef WRITE_CHKPT
        uint32_t prior_idx = 0;
        if(idx > 0)
          prior_idx = idx-1;
        Kokkos::fence();
        Kokkos::View<uint8_t*> buffer_d;
        header_t header;
        Timer::time_point start_collect = Timer::now();
        Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
        auto datasizes = write_incr_chkpt_hashtree_local_mode(current, buffer_d, chunk_size, g_distinct_nodes, g_shared_nodes, prior_idx, idx, header);
//        auto datasizes = write_incr_chkpt_hashtree_local_mode(current, buffer_d, chunk_size, g_distinct_nodes, g_nodes, prior_idx, idx, header);
        Kokkos::Profiling::popRegion();
        Timer::time_point end_collect = Timer::now();
        auto collect_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_collect - start_collect).count();

        Timer::time_point start_write = Timer::now();
        Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
        auto buffer_h = Kokkos::create_mirror_view(buffer_d);
        Kokkos::deep_copy(buffer_h, buffer_d);
        Kokkos::fence();
        Kokkos::Profiling::popRegion();
        Timer::time_point end_write = Timer::now();
        auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
        result_data << compare_time << ',' << collect_time << ',' << write_time << ',' << datasizes.first << ',' << datasizes.second << std::endl;
        timing_file << "Tree" << "," << idx << "," << chunk_size << "," << compare_time << "," << collect_time << "," << write_time << std::endl;
        size_file << "Tree" << "," << idx << "," << chunk_size << "," << datasizes.first << "," << datasizes.second << ",";
        write_metadata_breakdown(size_file, header, buffer_h, num_chkpts);
        std::ofstream file;
        file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        file.open(full_chkpt_files[idx]+".hashtree.incr_chkpt", std::ofstream::out | std::ofstream::binary);
        uint64_t dlen = current.size();
        uint32_t distinctlen = g_distinct_nodes.size();
        file.write((char*)(&header), sizeof(header_t));
        file.write((const char*)(buffer_h.data()), buffer_h.size());
        file.flush();
        file.close();
#endif
      } else {
        CompactTable shared_updates   = CompactTable(num_chunks);
        CompactTable distinct_updates = CompactTable(num_chunks);
//        NodeMap updates = NodeMap(num_chunks);
#ifndef GLOBAL_TABLE
//        CompactTable shared_updates   = CompactTable(num_chunks);
//        CompactTable distinct_updates = CompactTable(num_chunks);
        DistinctNodeIDMap l_distinct_nodes(2*num_chunks-1);
        SharedNodeIDMap l_shared_nodes = SharedNodeIDMap(2*num_chunks-1);
#endif
       SharedNodeIDMap l_shared_nodes = SharedNodeIDMap(2*num_chunks-1);
       SharedNodeIDMap l_identical_nodes     = SharedNodeIDMap(2*num_chunks-1);
//        NodeMap l_nodes = NodeMap(num_chunks);
//        NodeMap l_nodes = NodeMap(2*num_chunks-1);
        g_distinct_nodes.rehash(g_distinct_nodes.size()+2*num_chunks-1);
        DEBUG_PRINT("Allocated maps\n");

        Timer::time_point start_create_tree0 = Timer::now();
        Kokkos::Profiling::pushRegion((std::string("Deduplicate chkpt ") + std::to_string(idx)).c_str());
#ifdef GLOBAL_TABLE
//        deduplicate_data(current, chunk_size, hasher, tree0, idx, g_identical_nodes, g_shared_nodes, g_distinct_nodes, l_identical_nodes, l_shared_nodes, g_distinct_nodes, shared_updates, distinct_updates);
//        deduplicate_data(current, chunk_size, hasher, tree0, idx, g_nodes, g_distinct_nodes, l_nodes, g_distinct_nodes, updates);
        shared_updates.clear();
        distinct_updates.clear();
//        deduplicate_data_deterministic(current, chunk_size, hasher, tree0, idx, g_distinct_nodes, shared_updates, distinct_updates);
        deduplicate_data(current, chunk_size, hasher, tree0, idx, g_distinct_nodes, shared_updates, distinct_updates);
//        STDOUT_PRINT("Size of shared updates: %u\n", shared_updates.size());
//        STDOUT_PRINT("Size of distinct updates: %u\n", distinct_updates.size());
//        shared_updates.clear();
//        distinct_updates.clear();
//        num_subtree_roots(current, chunk_size, tree0, idx, g_distinct_nodes, shared_updates, distinct_updates);
        Kokkos::deep_copy(tree0.tree_h, tree0.tree_d);
#else
        deduplicate_data(current, chunk_size, hasher, tree0, idx, g_shared_nodes, g_distinct_nodes, l_shared_nodes, l_distinct_nodes, shared_updates, distinct_updates);
#endif
        Kokkos::Profiling::popRegion();
        Timer::time_point end_create_tree0 = Timer::now();

        STDOUT_PRINT("Size of shared updates: %u\n", shared_updates.size());
        STDOUT_PRINT("Size of distinct updates: %u\n", distinct_updates.size());
        STDOUT_PRINT("Size of identical updates: %u\n", l_identical_nodes.size());
#ifdef GLOBAL_TABLE
        Kokkos::deep_copy(g_shared_nodes, l_shared_nodes);
        Kokkos::deep_copy(g_identical_nodes, l_identical_nodes);
//        Kokkos::deep_copy(g_nodes, l_nodes);
#endif

        auto compare_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_create_tree0 - start_create_tree0).count();

#ifdef WRITE_CHKPT
        uint32_t prior_idx = 0;
        if(idx > 0) {
          Kokkos::fence();

          Kokkos::View<uint8_t*> buffer_d;
          header_t header;
          Timer::time_point start_collect = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
#ifdef GLOBAL_TABLE
          auto datasizes = write_incr_chkpt_hashtree_global_mode(current, buffer_d, chunk_size, distinct_updates, shared_updates, prior_idx, idx, header);
//          auto datasizes = write_incr_chkpt_hashtree_global_mode(current, buffer_d, chunk_size, distinct_updates, shared_updates, prior_idx, idx, header);
//          auto datasizes = write_incr_chkpt_hashtree_global_mode(current, buffer_d, chunk_size, updates, prior_idx, idx, header);
#else
          auto datasizes = write_incr_chkpt_hashtree_local_mode(current, buffer_d, chunk_size, distinct_updates, shared_updates, prior_idx, idx, header);
#endif
          Kokkos::Profiling::popRegion();
          Timer::time_point end_collect = Timer::now();
          auto collect_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_collect - start_collect).count();

          Timer::time_point start_write = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
          auto buffer_h = Kokkos::create_mirror_view(buffer_d);
          Kokkos::deep_copy(buffer_h, buffer_d);
          Kokkos::fence();
          Kokkos::Profiling::popRegion();
          Timer::time_point end_write = Timer::now();
          auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
          result_data << compare_time << ',' << collect_time << ',' << write_time << ',' << datasizes.first << ',' << datasizes.second << std::endl;
          timing_file << "Tree" << "," << idx << "," << chunk_size << "," << compare_time << "," << collect_time << "," << write_time << std::endl;
          size_file << "Tree" << "," << idx << "," << chunk_size << "," << datasizes.first << "," << datasizes.second << ",";
          write_metadata_breakdown(size_file, header, buffer_h, num_chkpts);

          std::ofstream file;
          file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
          file.open(full_chkpt_files[idx]+".hashtree.incr_chkpt", std::ofstream::out | std::ofstream::binary);
          uint64_t dlen = current.size();
          file.write((char*)(&header), sizeof(header_t));
          file.write((const char*)(buffer_h.data()), buffer_h.size());
          file.flush();
          file.close();
        }
#endif
      }
      Kokkos::fence();
    }
    Kokkos::fence();
    DEBUG_PRINT("Closing files\n");
    result_data.close();
    timing_file.close();
    size_file.close();
    STDOUT_PRINT("------------------------------------------------------\n");
  }
  STDOUT_PRINT("------------------------------------------------------\n");
}

template<typename Hasher>
void tree_chkpt_deduplicator(Hasher& hasher, 
                std::vector<std::string>& full_chkpt_files, 
                std::vector<std::string>& chkpt_filenames, 
                uint32_t chunk_size, 
                uint32_t num_chkpts) {
  using Timer = std::chrono::high_resolution_clock;
  DistinctNodeIDMap g_distinct_nodes  = DistinctNodeIDMap(1);
#ifndef GLOBAL_TABLE
  SharedNodeIDMap g_shared_nodes = SharedNodeIDMap(1);
#endif
  SharedNodeIDMap g_shared_nodes = SharedNodeIDMap(1);
  SharedNodeIDMap g_identical_nodes = SharedNodeIDMap(1);
//  NodeMap g_nodes = NodeMap(1);

  std::ifstream fs;
  fs.open(full_chkpt_files[0], std::ifstream::in | std::ifstream::binary);
  fs.seekg(0, fs.end);
  size_t data_length = fs.tellg();
  fs.seekg(0, fs.beg);
  fs.close();
  uint32_t n_chunks = data_length/chunk_size;
  if(n_chunks*chunk_size < data_length)
    n_chunks += 1;
  MerkleTree tree0 = MerkleTree(n_chunks);
  HashList leaves = HashList(n_chunks);

  Deduplicator<Hasher> deduplicator(chunk_size);

  for(uint32_t idx=0; idx<num_chkpts; idx++) {
    DEBUG_PRINT("Processing checkpoint %u\n", idx);
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    DEBUG_PRINT("Set exceptions\n");
    f.open(full_chkpt_files[idx], std::ifstream::in | std::ifstream::binary);
    DEBUG_PRINT("Opened file\n");
    f.seekg(0, f.end);
    DEBUG_PRINT("Seek end of file\n");
    size_t data_len = f.tellg();
    DEBUG_PRINT("Measure length of file\n");
    f.seekg(0, f.beg);
    DEBUG_PRINT("Seek beginning of file\n");
    DEBUG_PRINT("Length of checkpoint %u: %zd\n", idx, data_len);
    uint32_t num_chunks = data_len/chunk_size;
    if(num_chunks*chunk_size < data_len)
      num_chunks += 1;
    DEBUG_PRINT("Number of chunks: %u\n", num_chunks);
    if(idx == 0) {
      g_distinct_nodes.rehash(g_distinct_nodes.size()+2*num_chunks - 1);
#ifndef GLOBAL_TABLE
      g_shared_nodes.rehash(2*num_chunks - 1);
#endif
      g_shared_nodes.rehash(2*num_chunks - 1);
      g_identical_nodes.rehash(2*num_chunks - 1);
//      g_nodes.rehash(2*num_chunks - 1);
//      g_nodes.rehash(num_chunks);
    }

    std::fstream result_data;
    result_data.open(chkpt_filenames[idx]+".chunk_size."+std::to_string(chunk_size)+".csv", std::fstream::out | std::fstream::app);

    std::fstream timing_file, size_file;
    timing_file.open(chkpt_filenames[idx]+".chunk_size."+std::to_string(chunk_size)+".timing.csv", std::fstream::out | std::fstream::app);
    size_file.open(chkpt_filenames[idx]+".chunk_size."+std::to_string(chunk_size)+".size.csv", std::fstream::out | std::fstream::app);

    Kokkos::View<uint8_t*> current("Current region", data_len);
    Kokkos::View<uint8_t*>::HostMirror current_h = Kokkos::create_mirror_view(current);
    f.read((char*)(current_h.data()), data_len);
    f.close();
    Kokkos::deep_copy(current, current_h);
    DEBUG_PRINT("Size of full checkpoint: %zd\n", current.size());

    // Merkle Tree deduplication
    {

//      MerkleTree tree0 = MerkleTree(num_chunks);
      DEBUG_PRINT("Allocated tree and tables\n");

      Kokkos::fence();

//      deduplicator.dedup(current, idx==0);
//      Kokkos::fence();
//      header_t header;
//      Kokkos::View<uint8_t*> buffer_d;
//      deduplicator.create_diff(current, header, buffer_d, idx==0);
//      Kokkos::fence();
//      std::string filename = full_chkpt_files[idx] + ".hashtree.incr_chkpt";
//      deduplicator.write_diff(header, buffer_d, filename);
//      Kokkos::fence();

      std::string filename = full_chkpt_files[idx] + ".hashtree.incr_chkpt";
      std::string logname = chkpt_filenames[idx];
      deduplicator.checkpoint(current, filename, logname, idx==0);
      Kokkos::fence();

//      if(idx == 0) {
//        Timer::time_point start_create_tree0 = Timer::now();
//        Kokkos::Profiling::pushRegion((std::string("Deduplicate chkpt ") + std::to_string(idx)).c_str());
//        create_merkle_tree_deterministic(hasher, tree0, current, chunk_size, idx, g_distinct_nodes, g_shared_nodes);
////        create_merkle_tree(hasher, tree0, current, chunk_size, idx, g_distinct_nodes, g_shared_nodes);
////        create_merkle_tree(hasher, tree0, current, chunk_size, idx, g_distinct_nodes, g_nodes);
//        Kokkos::Profiling::popRegion();
//        Timer::time_point end_create_tree0 = Timer::now();
//        Kokkos::deep_copy(tree0.tree_h, tree0.tree_d);
//
////        STDOUT_PRINT("Size of shared entries: %u\n", g_shared_nodes.size());
//        STDOUT_PRINT("Size of distinct entries: %u\n", g_distinct_nodes.size());
////        STDOUT_PRINT("Size of shared updates: %u\n", shared_updates.size());
////        STDOUT_PRINT("Size of distinct updates: %u\n", distinct_updates.size());
//        auto compare_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_create_tree0 - start_create_tree0).count();
//#ifdef WRITE_CHKPT
//        uint32_t prior_idx = 0;
//        if(idx > 0)
//          prior_idx = idx-1;
//        Kokkos::fence();
//        Kokkos::View<uint8_t*> buffer_d;
//        header_t header;
//        Timer::time_point start_collect = Timer::now();
//        Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
//        auto datasizes = write_incr_chkpt_hashtree_local_mode(current, buffer_d, chunk_size, g_distinct_nodes, g_shared_nodes, prior_idx, idx, header);
////        auto datasizes = write_incr_chkpt_hashtree_local_mode(current, buffer_d, chunk_size, g_distinct_nodes, g_nodes, prior_idx, idx, header);
//        Kokkos::Profiling::popRegion();
//        Timer::time_point end_collect = Timer::now();
//        auto collect_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_collect - start_collect).count();
//
//        Timer::time_point start_write = Timer::now();
//        Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
//        auto buffer_h = Kokkos::create_mirror_view(buffer_d);
//        Kokkos::deep_copy(buffer_h, buffer_d);
//        Kokkos::fence();
//        Kokkos::Profiling::popRegion();
//        Timer::time_point end_write = Timer::now();
//        auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
//        result_data << compare_time << ',' << collect_time << ',' << write_time << ',' << datasizes.first << ',' << datasizes.second << std::endl;
//        timing_file << "Tree" << "," << idx << "," << chunk_size << "," << compare_time << "," << collect_time << "," << write_time << std::endl;
//        size_file << "Tree" << "," << idx << "," << chunk_size << "," << datasizes.first << "," << datasizes.second << ",";
//        write_metadata_breakdown(size_file, header, buffer_h, num_chkpts);
//        std::ofstream file;
//        file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
//        file.open(full_chkpt_files[idx]+".hashtree.incr_chkpt", std::ofstream::out | std::ofstream::binary);
//        uint64_t dlen = current.size();
//        uint32_t distinctlen = g_distinct_nodes.size();
//        file.write((char*)(&header), sizeof(header_t));
//        file.write((const char*)(buffer_h.data()), buffer_h.size());
//        file.flush();
//        file.close();
//#endif
//      } else {
//        CompactTable shared_updates   = CompactTable(num_chunks);
//        CompactTable distinct_updates = CompactTable(num_chunks);
////        NodeMap updates = NodeMap(num_chunks);
//#ifndef GLOBAL_TABLE
////        CompactTable shared_updates   = CompactTable(num_chunks);
////        CompactTable distinct_updates = CompactTable(num_chunks);
//        DistinctNodeIDMap l_distinct_nodes(2*num_chunks-1);
//        SharedNodeIDMap l_shared_nodes = SharedNodeIDMap(2*num_chunks-1);
//#endif
//       SharedNodeIDMap l_shared_nodes = SharedNodeIDMap(2*num_chunks-1);
//       SharedNodeIDMap l_identical_nodes     = SharedNodeIDMap(2*num_chunks-1);
////        NodeMap l_nodes = NodeMap(num_chunks);
////        NodeMap l_nodes = NodeMap(2*num_chunks-1);
//        g_distinct_nodes.rehash(g_distinct_nodes.size()+2*num_chunks-1);
//        DEBUG_PRINT("Allocated maps\n");
//
//        Timer::time_point start_create_tree0 = Timer::now();
//        Kokkos::Profiling::pushRegion((std::string("Deduplicate chkpt ") + std::to_string(idx)).c_str());
//#ifdef GLOBAL_TABLE
////        deduplicate_data(current, chunk_size, hasher, tree0, idx, g_identical_nodes, g_shared_nodes, g_distinct_nodes, l_identical_nodes, l_shared_nodes, g_distinct_nodes, shared_updates, distinct_updates);
////        deduplicate_data(current, chunk_size, hasher, tree0, idx, g_nodes, g_distinct_nodes, l_nodes, g_distinct_nodes, updates);
//        shared_updates.clear();
//        distinct_updates.clear();
////        deduplicate_data_deterministic(current, chunk_size, hasher, tree0, idx, g_distinct_nodes, shared_updates, distinct_updates);
//        deduplicate_data(current, chunk_size, hasher, tree0, idx, g_distinct_nodes, shared_updates, distinct_updates);
////        STDOUT_PRINT("Size of shared updates: %u\n", shared_updates.size());
////        STDOUT_PRINT("Size of distinct updates: %u\n", distinct_updates.size());
////        shared_updates.clear();
////        distinct_updates.clear();
////        num_subtree_roots(current, chunk_size, tree0, idx, g_distinct_nodes, shared_updates, distinct_updates);
//        Kokkos::deep_copy(tree0.tree_h, tree0.tree_d);
//#else
//        deduplicate_data(current, chunk_size, hasher, tree0, idx, g_shared_nodes, g_distinct_nodes, l_shared_nodes, l_distinct_nodes, shared_updates, distinct_updates);
//#endif
//        Kokkos::Profiling::popRegion();
//        Timer::time_point end_create_tree0 = Timer::now();
//
//        STDOUT_PRINT("Size of shared updates: %u\n", shared_updates.size());
//        STDOUT_PRINT("Size of distinct updates: %u\n", distinct_updates.size());
//        STDOUT_PRINT("Size of identical updates: %u\n", l_identical_nodes.size());
//#ifdef GLOBAL_TABLE
//        Kokkos::deep_copy(g_shared_nodes, l_shared_nodes);
//        Kokkos::deep_copy(g_identical_nodes, l_identical_nodes);
////        Kokkos::deep_copy(g_nodes, l_nodes);
//#endif
//
//        auto compare_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_create_tree0 - start_create_tree0).count();
//
//#ifdef WRITE_CHKPT
//        uint32_t prior_idx = 0;
//        if(idx > 0) {
//          Kokkos::fence();
//
//          Kokkos::View<uint8_t*> buffer_d;
//          header_t header;
//          Timer::time_point start_collect = Timer::now();
//          Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
//#ifdef GLOBAL_TABLE
//          auto datasizes = write_incr_chkpt_hashtree_global_mode(current, buffer_d, chunk_size, distinct_updates, shared_updates, prior_idx, idx, header);
////          auto datasizes = write_incr_chkpt_hashtree_global_mode(current, buffer_d, chunk_size, distinct_updates, shared_updates, prior_idx, idx, header);
////          auto datasizes = write_incr_chkpt_hashtree_global_mode(current, buffer_d, chunk_size, updates, prior_idx, idx, header);
//#else
//          auto datasizes = write_incr_chkpt_hashtree_local_mode(current, buffer_d, chunk_size, distinct_updates, shared_updates, prior_idx, idx, header);
//#endif
//          Kokkos::Profiling::popRegion();
//          Timer::time_point end_collect = Timer::now();
//          auto collect_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_collect - start_collect).count();
//
//          Timer::time_point start_write = Timer::now();
//          Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
//          auto buffer_h = Kokkos::create_mirror_view(buffer_d);
//          Kokkos::deep_copy(buffer_h, buffer_d);
//          Kokkos::fence();
//          Kokkos::Profiling::popRegion();
//          Timer::time_point end_write = Timer::now();
//          auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
//          result_data << compare_time << ',' << collect_time << ',' << write_time << ',' << datasizes.first << ',' << datasizes.second << std::endl;
//          timing_file << "Tree" << "," << idx << "," << chunk_size << "," << compare_time << "," << collect_time << "," << write_time << std::endl;
//          size_file << "Tree" << "," << idx << "," << chunk_size << "," << datasizes.first << "," << datasizes.second << ",";
//          write_metadata_breakdown(size_file, header, buffer_h, num_chkpts);
//
//          std::ofstream file;
//          file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
//          file.open(full_chkpt_files[idx]+".hashtree.incr_chkpt", std::ofstream::out | std::ofstream::binary);
//          uint64_t dlen = current.size();
//          file.write((char*)(&header), sizeof(header_t));
//          file.write((const char*)(buffer_h.data()), buffer_h.size());
//          file.flush();
//          file.close();
//        }
//#endif
//      }
      Kokkos::fence();
    }
    Kokkos::fence();
    DEBUG_PRINT("Closing files\n");
    result_data.close();
    timing_file.close();
    size_file.close();
    STDOUT_PRINT("------------------------------------------------------\n");
  }
  STDOUT_PRINT("------------------------------------------------------\n");
}

#endif //DEDUP_APPROACHES_HPP
