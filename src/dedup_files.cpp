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
#include <utility>
#include "utils.hpp"

#define WRITE_CHKPT

typedef struct veloc_header {
  size_t chkpt_size;
  size_t header_size;
  size_t num_regions;
} veloc_header_t;

struct region_t {
  void* ptr;
  size_t size;
//  ptr_type_t ptr_type;
};
typedef std::map<int, region_t> regions_t;

void write_metadata_breakdown(std::fstream& fs, header_t& header, Kokkos::View<uint8_t*>::HostMirror& buffer, uint32_t num_chkpts) {
  printf("==========Header==========\n");
  printf("Baseline chkpt          : %u\n" , header.ref_id);
  printf("Current chkpt           : %u\n" , header.chkpt_id);
  printf("Memory size             : %lu\n", header.datalen);
  printf("Chunk size              : %u\n" , header.chunk_size);
  printf("Window size             : %u\n" , header.window_size);
  printf("Num distinct            : %u\n" , header.distinct_size);
  printf("Num repeats for current : %u\n" , header.curr_repeat_size);
  printf("Num repeats for previous: %u\n" , header.prev_repeat_size);
  printf("Num prior chkpts        : %u\n" , header.num_prior_chkpts);
  printf("==========Header==========\n");
  printf("==========Repeat Map==========\n");
  for(uint32_t i=0; i<header.num_prior_chkpts; i++) {
    uint32_t chkpt = 0, num = 0;
    memcpy(&chkpt, buffer.data()+header.distinct_size*sizeof(uint32_t)+i*2*sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&num, buffer.data()+header.distinct_size*sizeof(uint32_t)+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
    printf("%u:%u\n", chkpt, num);
  }
  printf("==========Repeat Map==========\n");
  printf("Header bytes: %lu\n", 40);
  printf("Distinct bytes: %lu\n", header.distinct_size*sizeof(uint32_t));
  fs << 40 << "," << header.distinct_size*sizeof(uint32_t) << ",";
  if(header.ref_id != header.chkpt_id) {
    printf("Repeat map bytes: %lu\n", 2*sizeof(uint32_t)*header.num_prior_chkpts);
    fs << 2*sizeof(uint32_t)*header.num_prior_chkpts;
    for(uint32_t i=0; i<num_chkpts; i++) {
      if(i < header.num_prior_chkpts) {
        uint32_t chkpt = 0, num = 0;
        memcpy(&chkpt, buffer.data()+header.distinct_size*sizeof(uint32_t)+i*2*sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&num, buffer.data()+header.distinct_size*sizeof(uint32_t)+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
        printf("Repeat bytes for %u: %lu\n", chkpt, num*2*sizeof(uint32_t));
        fs << "," << num*2*sizeof(uint32_t);
      } else {
        printf("Repeat bytes for %u: %lu\n", i, 0);;
        fs << "," << 0;
      }
    }
    fs << std::endl;
  } else {
    printf("Repeat map bytes: %lu\n", 0);
    fs << 0 << ",";
    printf("Repeat bytes for %u: %lu\n", header.chkpt_id, header.curr_repeat_size*2*sizeof(uint32_t));
    fs << header.curr_repeat_size*2*sizeof(uint32_t);
    for(uint32_t i=1; i<num_chkpts; i++) {
      printf("Repeat bytes for %u: %lu\n", i, 0);;
      fs << "," << 0;
    }
    fs << std::endl;
  }
}

void full_chkpt(Hasher& hasher, 
                std::vector<std::string>& full_chkpt_files, 
                std::vector<std::string>& chkpt_filenames, 
                uint32_t chunk_size, 
                uint32_t num_chkpts) 
{
  using Timer = std::chrono::high_resolution_clock;
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

    // Full checkpoint
    {
      std::string filename = full_chkpt_files[idx] + ".full_chkpt";
      std::ofstream file;
      file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
      file.open(filename, std::ofstream::out | std::ofstream::binary);
      DEBUG_PRINT("Opened full checkpoint file\n");
      Timer::time_point start_write = Timer::now();
      Kokkos::Profiling::pushRegion((std::string("Write full checkpoint ") + std::to_string(idx)).c_str());
      Kokkos::deep_copy(current_h, current);
      DEBUG_PRINT("Wrote full checkpoint\n");
      Kokkos::Profiling::popRegion();
      Timer::time_point end_write = Timer::now();
      auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write-start_write).count();
      file.write((char*)(current_h.data()), current_h.size());
      STDOUT_PRINT("Time spent writing full checkpoint: %f\n", write_time);
      result_data << "0.0" << "," << "0.0" << "," << write_time << "," << current.size() << ',' << "0" << ',';
      timing_file << "Full" << "," << idx << "," << chunk_size << "," << "0.0" << "," << "0.0" << "," << write_time << std::endl;
      size_file << "Full" << "," << idx << "," << chunk_size << "," << current.size() << "," << "0" << ",0,0,0";
      for(uint32_t j=0; j<num_chkpts; j++) {
        size_file << ",0";
      }
      size_file << std::endl;
      file.close();
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
void naive_chkpt(Hasher& hasher, 
                std::vector<std::string>& full_chkpt_files, 
                std::vector<std::string>& chkpt_filenames, 
                uint32_t chunk_size, 
                uint32_t num_chkpts) 
{
  using Timer = std::chrono::high_resolution_clock;
  HashList prev_list(0);

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
      Kokkos::resize(prev_list.list_d, num_chunks);
      Kokkos::resize(prev_list.list_h, num_chunks);
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

    // Naive hash list deduplication
    {
      HashList list0 = HashList(num_chunks);
      Kokkos::Bitset<Kokkos::DefaultExecutionSpace> changes_bitset(num_chunks);
      DEBUG_PRINT("initialized local maps and list\n");
      Kokkos::fence();

      Kokkos::fence();
      Timer::time_point start_compare = Timer::now();
      Kokkos::Profiling::pushRegion((std::string("Find distinct chunks ") + std::to_string(idx)).c_str());
      compare_lists_naive(hasher, prev_list, list0, changes_bitset, idx, current, chunk_size);
      Kokkos::Profiling::popRegion();
      Timer::time_point end_compare = Timer::now();

      Kokkos::fence();

      auto compare_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_compare - start_compare).count();
  	  Kokkos::deep_copy(prev_list.list_d, list0.list_d);

#ifdef WRITE_CHKPT
      uint32_t prior_idx = 0;
      Kokkos::fence();
      Timer::time_point start_collect = Timer::now();
      Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
      Kokkos::View<uint8_t*> buffer_d;
      header_t header;
      std::pair<uint64_t,uint64_t> datasizes = write_incr_chkpt_hashlist_naive(full_chkpt_files[idx]+".naivehashlist.incr_chkpt", current, buffer_d, chunk_size, changes_bitset, prior_idx, idx, header);
      Kokkos::Profiling::popRegion();
      Timer::time_point end_collect = Timer::now();
      auto collect_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_collect - start_collect).count();
      STDOUT_PRINT("Time spect collecting updates: %f\n", collect_time);

      Timer::time_point start_write = Timer::now();
      Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
      auto buffer_h = Kokkos::create_mirror_view(buffer_d);
      Kokkos::deep_copy(buffer_h, buffer_d);
      Kokkos::fence();
      Kokkos::Profiling::popRegion();
      Timer::time_point end_write = Timer::now();
      auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
      STDOUT_PRINT("Time spect copying updates: %f\n", write_time);
      result_data << compare_time << ',' << collect_time << ',' << write_time << ',' << datasizes.first << ',' << datasizes.second << ',';
      timing_file << "Naive" << "," << idx << "," << chunk_size << "," << compare_time << "," << collect_time << "," << write_time << std::endl;
      size_file << "Naive" << "," << idx << "," << chunk_size << "," << datasizes.first << "," << datasizes.second << ",";

      write_metadata_breakdown(size_file, header, buffer_h, num_chkpts);

      std::ofstream file;
      file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
      file.open(full_chkpt_files[idx]+".naivehashlist.incr_chkpt", std::ofstream::out | std::ofstream::binary);
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
//        compare_lists_global(hasher, list0, idx, current, chunk_size, l_shared_chunks, g_distinct_chunks, g_shared_chunks, g_distinct_chunks);
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

//	    prior_list = current_list;
//	    current_list = list0;
//      if(idx > 0) {
//      //	  Kokkos::deep_copy(prior_list.list_h, prior_list.list_d);
//      //	  Kokkos::deep_copy(current_list.list_h, current_list.list_d);
//        std::string region_log("region-data-");
//        region_log = region_log + chkpt_files[idx] + "chunk_size." + std::to_string(chunk_size) + std::string(".log");
//        std::fstream fs(region_log, std::fstream::out|std::fstream::app);
//        uint32_t num_changed = print_changed_blocks(fs, current_list.list_d, prior_list.list_d);
//        auto contiguous_regions = print_contiguous_regions(region_log, current_list.list_d, prior_list.list_d);
//      }

#ifdef WRITE_CHKPT
        uint32_t prior_idx = 0;
        Kokkos::fence();
        Timer::time_point start_collect = Timer::now();
        Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
        Kokkos::View<uint8_t*> buffer_d;
        header_t header;
#ifdef GLOBAL_TABLE
        std::pair<uint64_t,uint64_t> datasizes = write_incr_chkpt_hashlist_global_new(full_chkpt_files[idx]+".hashlist.incr_chkpt", current, buffer_d, chunk_size, g_distinct_chunks, l_shared_chunks, prior_idx, idx, header);
#else
        std::pair<uint64_t,uint64_t> datasizes = write_incr_chkpt_hashlist_local(full_chkpt_files[idx]+".hashlist.incr_chkpt", current, buffer_d, chunk_size, l_distinct_chunks, l_shared_chunks, prior_idx, idx, header);
#endif
        Kokkos::Profiling::popRegion();
        Timer::time_point end_collect = Timer::now();
        auto collect_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_collect - start_collect).count();
        STDOUT_PRINT("Time spect collecting updates: %f\n", collect_time);

        Timer::time_point start_write = Timer::now();
        Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
        auto buffer_h = Kokkos::create_mirror_view(buffer_d);
        Kokkos::deep_copy(buffer_h, buffer_d);
        Kokkos::fence();
        Kokkos::Profiling::popRegion();
        Timer::time_point end_write = Timer::now();
        auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
        STDOUT_PRINT("Time spect copying updates: %f\n", write_time);
        result_data << compare_time << ',' << collect_time << ',' << write_time << ',' << datasizes.first << ',' << datasizes.second << ',';
        timing_file << "List" << "," << idx << "," << chunk_size << "," << compare_time << "," << collect_time << "," << write_time << std::endl;
        size_file << "List" << "," << idx << "," << chunk_size << "," << datasizes.first << "," << datasizes.second << ",";
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
                uint32_t num_chkpts) 
{
    using Timer = std::chrono::high_resolution_clock;
    DistinctNodeIDMap g_distinct_nodes  = DistinctNodeIDMap(1);
//    SharedNodeIDMap g_shared_nodes = SharedNodeIDMap(1);
//    SharedNodeIDMap g_identical_nodes = SharedNodeIDMap(1);
    NodeMap g_nodes = NodeMap(1);

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
//        g_shared_nodes.rehash(2*num_chunks - 1);
//        g_identical_nodes.rehash(2*num_chunks - 1);
        g_nodes.rehash(2*num_chunks - 1);
g_nodes.clear();
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

        MerkleTree tree0 = MerkleTree(num_chunks);
        DEBUG_PRINT("Allocated tree and tables\n");

        Kokkos::fence();

        if(idx == 0) {
          Timer::time_point start_create_tree0 = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Deduplicate chkpt ") + std::to_string(idx)).c_str());
//          create_merkle_tree(hasher, tree0, current, chunk_size, idx, g_distinct_nodes, g_shared_nodes);
          create_merkle_tree(hasher, tree0, current, chunk_size, idx, g_distinct_nodes, g_nodes);
          Kokkos::Profiling::popRegion();
          Timer::time_point end_create_tree0 = Timer::now();

//          STDOUT_PRINT("Size of shared entries: %u\n", g_shared_nodes.size());
          STDOUT_PRINT("Size of distinct entries: %u\n", g_distinct_nodes.size());
//          STDOUT_PRINT("Size of shared updates: %u\n", shared_updates.size());
//          STDOUT_PRINT("Size of distinct updates: %u\n", distinct_updates.size());
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
//          auto datasizes = write_incr_chkpt_hashtree_local_mode(full_chkpt_files[idx]+".hashtree.incr_chkpt", current, buffer_d, chunk_size, g_distinct_nodes, g_shared_nodes, prior_idx, idx, header);
          auto datasizes = write_incr_chkpt_hashtree_local_mode(full_chkpt_files[idx]+".hashtree.incr_chkpt", current, buffer_d, chunk_size, g_distinct_nodes, g_nodes, prior_idx, idx, header);
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
//          uint32_t repeatlen = g_shared_nodes.size();
          uint32_t distinctlen = g_distinct_nodes.size();
          file.write((char*)(&header), sizeof(header_t));
          file.write((const char*)(buffer_h.data()), buffer_h.size());
          file.flush();
          file.close();
#endif
        } else {
          CompactTable shared_updates   = CompactTable(num_chunks);
          CompactTable distinct_updates = CompactTable(num_chunks);
          DistinctNodeIDMap l_distinct_nodes(2*num_chunks-1);
//          SharedNodeIDMap l_shared_nodes = SharedNodeIDMap(2*num_chunks-1);
//          SharedNodeIDMap l_identical_nodes     = SharedNodeIDMap(2*num_chunks-1);
          NodeMap l_nodes = NodeMap(2*num_chunks-1);
          g_distinct_nodes.rehash(g_distinct_nodes.size()+2*num_chunks-1);
          DEBUG_PRINT("Allocated maps\n");

          Timer::time_point start_create_tree0 = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Deduplicate chkpt ") + std::to_string(idx)).c_str());
#ifdef GLOBAL_TABLE
//          deduplicate_data(current, chunk_size, hasher, tree0, idx, g_shared_nodes, g_distinct_nodes, l_shared_nodes, g_distinct_nodes, shared_updates, distinct_updates);
//          deduplicate_data(current, chunk_size, hasher, tree0, idx, g_identical_nodes, g_shared_nodes, g_distinct_nodes, l_identical_nodes, l_shared_nodes, g_distinct_nodes, shared_updates, distinct_updates);
          deduplicate_data(current, chunk_size, hasher, tree0, idx, g_nodes, g_distinct_nodes, l_nodes, g_distinct_nodes, shared_updates, distinct_updates);
#else
          deduplicate_data(current, chunk_size, hasher, tree0, idx, g_shared_nodes, g_distinct_nodes, l_shared_nodes, l_distinct_nodes, shared_updates, distinct_updates);
#endif
          Kokkos::Profiling::popRegion();
          Timer::time_point end_create_tree0 = Timer::now();

          STDOUT_PRINT("Size of distinct entries: %u\n", g_distinct_nodes.size());
//          STDOUT_PRINT("Size of shared entries: %u\n", l_shared_nodes.size());
          STDOUT_PRINT("Size of distinct entries: %u\n", l_distinct_nodes.size());
          STDOUT_PRINT("Size of shared updates: %u\n", shared_updates.size());
          STDOUT_PRINT("Size of distinct updates: %u\n", distinct_updates.size());
#ifdef GLOBAL_TABLE
//          Kokkos::deep_copy(g_shared_nodes, l_shared_nodes);
//          Kokkos::deep_copy(g_identical_nodes, l_identical_nodes);
          Kokkos::deep_copy(g_nodes, l_nodes);
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
            auto datasizes = write_incr_chkpt_hashtree_global_mode(full_chkpt_files[idx]+".hashtree.incr_chkpt", current, buffer_d, chunk_size, distinct_updates, shared_updates, prior_idx, idx, header);
#else
            auto datasizes = write_incr_chkpt_hashtree_local_mode(full_chkpt_files[idx]+".hashtree.incr_chkpt", current, buffer_d, chunk_size, distinct_updates, shared_updates, prior_idx, idx, header);
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
            uint32_t repeatlen = shared_updates.size();
            uint32_t distinctlen = distinct_updates.size();
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
    DEBUG_PRINT("Argv[1]: %s\n", argv[1]);
    uint32_t chunk_size = static_cast<uint32_t>(atoi(argv[1]));
    DEBUG_PRINT("Loaded chunk size\n");
    uint32_t num_chkpts = static_cast<uint32_t>(atoi(argv[2]));
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
    std::vector<std::string> full_chkpt_files;
    std::vector<std::string> chkpt_filenames;
    for(uint32_t i=0; i<num_chkpts; i++) {
      full_chkpt_files.push_back(std::string(argv[3+arg_offset+i]));
      chkpt_files.push_back(std::string(argv[3+arg_offset+i]));
      chkpt_filenames.push_back(std::string(argv[3+arg_offset+i]));
      size_t name_start = chkpt_filenames[i].rfind('/') + 1;
      chkpt_filenames[i].erase(chkpt_filenames[i].begin(), chkpt_filenames[i].begin()+name_start);
    }
    DEBUG_PRINT("Read checkpoint files\n");
    DEBUG_PRINT("Number of checkpoints: %u\n", num_chkpts);

//    SHA1 hasher;
//    Murmur3C hasher;
    MD5Hash hasher;

    // Full checkpoint
    if(run_full) {
      printf("====================Full Checkpoint====================\n");
      full_chkpt(hasher, full_chkpt_files, chkpt_filenames, chunk_size, num_chkpts);
      printf("====================Full Checkpoint====================\n");
      if(arg_offset > 1) {
        flush_cache();
        arg_offset -= 1;
      }
    }
    // Naive list checkpoint
    if(run_naive) {
      printf("====================Naive List Checkpoint====================\n");
      naive_chkpt(hasher, full_chkpt_files, chkpt_filenames, chunk_size, num_chkpts);
      printf("====================Naive List Checkpoint====================\n");
      if(arg_offset > 1) {
        flush_cache();
        arg_offset -= 1;
      }
    }
    // Hash list checkpoint
    if(run_list) {
      printf("====================Hash List Checkpoint====================\n");
      list_chkpt(hasher, full_chkpt_files, chkpt_filenames, chunk_size, num_chkpts);
      printf("====================Hash List Checkpoint====================\n");
      if(arg_offset > 1) {
        flush_cache();
        arg_offset -= 1;
      }
    }
    // Tree checkpoint
    if(run_tree) {
      printf("====================Hash Tree Checkpoint====================\n");
      tree_chkpt(hasher, full_chkpt_files, chkpt_filenames, chunk_size, num_chkpts);
      printf("====================Hash Tree Checkpoint====================\n");
    }
  }
  Kokkos::finalize();
}

