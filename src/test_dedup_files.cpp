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

enum DataGenerationMode {
  Random=0,
  BeginningIdentical,
  Checkered,
  Identical,
  Sparse
};

Kokkos::View<uint8_t*> generate_initial_data(uint32_t max_data_len) {
  Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
  Kokkos::View<uint8_t*> data("Data", max_data_len);
  auto policy = Kokkos::RangePolicy<>(0, max_data_len);
  Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
    auto rand_gen = rand_pool.get_state();
    data(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
    rand_pool.free_state(rand_gen);
  });
  return data;
}

void perturb_data(Kokkos::View<uint8_t*>& data0, Kokkos::View<uint8_t*>& data1, 
                  const uint32_t num_changes, DataGenerationMode mode) {
  if(mode == Random) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    auto policy = Kokkos::RangePolicy<>(0, data0.size());
    Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
      auto rand_gen = rand_pool.get_state();
      data1(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
      rand_pool.free_state(rand_gen);
    });
  } else if(mode == BeginningIdentical) {
    Kokkos::deep_copy(data1, data0);
    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    auto policy = Kokkos::RangePolicy<>(num_changes, data0.size());
    Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
      auto rand_gen = rand_pool.get_state();
      data1(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
      rand_pool.free_state(rand_gen);
    });
  } else if(mode == Checkered) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    auto policy = Kokkos::RangePolicy<>(0, data0.size());
    Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
      auto rand_gen = rand_pool.get_state();
      data1(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
      rand_pool.free_state(rand_gen);
    });
    for(uint32_t j=0; j<data0.size(); j+=2*num_changes) {
      uint32_t end = j+num_changes;
      if(end > data0.size())
        end = data0.size() - j*num_changes;
      auto base_view  = Kokkos::subview(data0, Kokkos::make_pair(j, end));
      auto chkpt_view = Kokkos::subview(data1, Kokkos::make_pair(j, end));
      Kokkos::deep_copy(chkpt_view, base_view);
    }
  } else if(mode == Identical) {
    Kokkos::deep_copy(data1, data0);
  } else if(mode == Sparse) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    Kokkos::deep_copy(data1, data0);
    Kokkos::parallel_for("Randomize data", Kokkos::RangePolicy<>(0, num_changes), KOKKOS_LAMBDA(const uint32_t j) {
      auto rand_gen = rand_pool.get_state();
      uint32_t pos = rand_gen.urand() % data0.size();
      uint8_t val = static_cast<uint8_t>(rand_gen.urand() % 256);
      while(val == data1(pos)) {
        val = static_cast<uint8_t>(rand_gen.urand() % 256);
      }
      data1(pos) = val; 
      rand_pool.free_state(rand_gen);
    });
  }
}

typedef struct header {
  size_t chkpt_size;
  size_t header_size;
  size_t num_regions;
} header_t;

struct region_t {
  void* ptr;
  size_t size;
//  ptr_type_t ptr_type;
};
typedef std::map<int, region_t> regions_t;

// Read header and region map from VeloC checkpoint
bool read_full_header(const std::string& chkpt, header_t& header, std::map<int, size_t>& region_map) {
  try {
    std::ifstream f;
    size_t expected_size = 0;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(chkpt, std::ifstream::in | std::fstream::binary);
    int id;
    size_t num_regions, region_size, header_size;
    f.read((char*)(&num_regions), sizeof(size_t));
    for(uint32_t i=0; i<num_regions; i++) {
      f.read((char*)&id, sizeof(int));
      f.read((char*)&region_size, sizeof(size_t));
      region_map.insert(std::make_pair(id, region_size));
      expected_size += region_size;
    }
    header_size = f.tellg();
    f.seekg(0, f.end);
    size_t file_size = (size_t)f.tellg() - header_size;
    if(file_size != expected_size)
      throw std::ifstream::failure("file size " + std::to_string(file_size) + " does not match expected size " + std::to_string(expected_size));
    header.chkpt_size = file_size;
    header.header_size = header_size;
    header.num_regions = num_regions;
    return true;
  } catch(std::ifstream::failure &e) {
    std::cout << "cannot validate header for checkpoint " << chkpt << ", reason: " << e.what();
    return false;
  }
}

KOKKOS_INLINE_FUNCTION
void copy_memory(void* dst, void* src, size_t len) {
#ifdef __CUDA_ARCH__
  size_t offset = 0;
  if((reinterpret_cast<uintptr_t>(dst)%16 == 0) && (reinterpret_cast<uintptr_t>(src)%16 == 0)) {
    for(size_t i=0; i<len/16; i++) {
      ((uint4*) dst)[i] = ((uint4*) src)[i];
    }
    offset = 16*(len/16);
  } else if((reinterpret_cast<uintptr_t>(dst)%8 == 0) && (reinterpret_cast<uintptr_t>(src)%8 == 0)) {
    for(size_t i=0; i<len/8; i++) {
      ((uint2*) dst)[i] = ((uint2*) src)[i];
    }
    offset = 8*(len/8);
  } else if((reinterpret_cast<uintptr_t>(dst)%4 == 0) && (reinterpret_cast<uintptr_t>(src)%4 == 0)) {
    for(size_t i=0; i<len/4; i++) {
      ((uint1*) dst)[i] = ((uint1*) src)[i];
    }
    offset = 4*(len/4);
  } else if((reinterpret_cast<uintptr_t>(dst)%2 == 0) && (reinterpret_cast<uintptr_t>(src)%2 == 0)) {
    for(size_t i=0; i<len/2; i++) {
      ((ushort1*) dst)[i] = ((ushort1*) src)[i];
    }
    offset = 2*(len/2);
  } else if((reinterpret_cast<uintptr_t>(dst)%1 == 0) && (reinterpret_cast<uintptr_t>(src)%1 == 0)) {
    for(size_t i=0; i<len; i++) {
      ((uchar1*) dst)[i] = ((uchar1*) src)[i];
    }
    offset = len;
  }
  for(size_t i=offset; i<len; i++) {
    ((uint8_t*) dst)[i] = ((uint8_t*) src)[i];
  }
#else
  for(size_t i=0; i<len/4; i++) {
    ((uint32_t*) dst)[i] = ((uint32_t*) src)[i];
  }
  size_t offset = 4*(len/4);
  if(len-offset >= 2) {
    ((uint16_t*) dst+offset)[0] = ((uint16_t*) src+offset)[0];
    offset += 2;
  }
  if(len-offset >= 1) {
    ((uint8_t*) dst+offset)[0] = ((uint8_t*) src+offset)[0];
    offset += 1;
  }
#endif
}

void dump_tree_metadata(const std::string& filename, 
                        const DistinctMap& distinct, 
                        const SharedMap& shared,
                        const uint32_t num_chunks,
                        const uint32_t output_chunks) {
  uint32_t num_nodes = 2*num_chunks - 1;
  Kokkos::View<uint32_t*> reg_size("size of region list", num_nodes);
  Kokkos::View<uint32_t*> offset("Offset list", num_nodes);
  Kokkos::View<char*>     labels("Label list", num_nodes);
  Kokkos::deep_copy(offset, 0);
  Kokkos::deep_copy(labels, 'I');
  Kokkos::deep_copy(reg_size, 0);
  DEBUG_PRINT("Setup offset and label views\n");
  DistinctHostMap distinct_h;
  SharedHostMap shared_h;
  Kokkos::deep_copy(distinct_h, distinct);
  Kokkos::deep_copy(shared_h, shared);
  DEBUG_PRINT("Copied tables to host\n");
  printf("Copied tables to host\n");
  Kokkos::fence();
  Kokkos::parallel_for("Update repeat entries", Kokkos::RangePolicy<>(0,shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      uint32_t node = shared.key_at(i);
      uint32_t prev = shared.value_at(i);
      uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
      uint32_t end = rightmost_leaf(node, num_nodes) - (num_chunks-1);
      offset(node) = prev;
      labels(node) = 'R';
      reg_size(node) = num_leaf_descendents(node, num_nodes);
//      for(uint32_t j=start; j<=end; j++) {
//        offset(j) = prev;
//        labels(j) = 'R';
//      }
    }
  });
  DEBUG_PRINT("Update repeat chunks\n");

  Kokkos::parallel_for("Update distinct entries", Kokkos::RangePolicy<>(0,distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      NodeInfo info = distinct.value_at(i);
      if(info.node >= num_chunks-1) {
        uint32_t node = info.node;
        uint32_t prev = info.src;
        uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
        uint32_t end = rightmost_leaf(node, num_nodes) - (num_chunks-1);
      offset(node) = prev;
      if(node == prev) {
        labels(node) = 'd';
      } else {
        labels(node) = 'r';
      }
      reg_size(node) = num_leaf_descendents(node, num_nodes);
//        char l = 'r';
//        if(node == prev)
//          l = 'd';
//        for(uint32_t j=start; j<=end; j++) {
//          offset(j) = prev;
//          labels(j) = l;
//        }
      }
    }
  });
  DEBUG_PRINT("Update distinct chunks\n");
  auto offset_h = Kokkos::create_mirror_view(offset);
  auto labels_h = Kokkos::create_mirror_view(labels);
  auto reg_size_h = Kokkos::create_mirror_view(reg_size);
  Kokkos::deep_copy(offset_h, offset);
  Kokkos::deep_copy(labels_h, labels);
  Kokkos::deep_copy(reg_size_h, reg_size);
  std::ofstream file;
  file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  file.open(filename, std::ofstream::out);
  std::cout << "Opened log file: " << filename << std::endl;
  file << "Position: Offset: Label\n";
  for(size_t i=0; i<num_nodes; i++) {
if(labels_h(i) != 'I') 
    file << i << "(" << i+num_chunks-1 << ")" << "\t: " << offset_h(i) << "\t: " << reg_size_h(i) << "\t: " << labels_h(i) << std::endl;
//    file << i << "(" << i+num_chunks-1 << ")" << "\t: " << offset_h(i) << "\t: " << labels_h(i) << std::endl;
  }
  file.close();
  STDOUT_PRINT("Wrote log\n");
}

void dump_tree_metadata(const std::string& filename, 
                        const CompactTable<10>& distinct, 
                        const CompactTable<10>& shared,
                        const uint32_t num_chunks,
                        const uint32_t output_chunks) {
  uint32_t num_nodes = 2*num_chunks - 1;
  Kokkos::View<uint32_t*> reg_size("size of region list", num_nodes);
  Kokkos::View<uint32_t*> offset("Offset list", num_nodes);
  Kokkos::View<char*>     labels("Label list", num_nodes);
  Kokkos::deep_copy(offset, 0);
  Kokkos::deep_copy(labels, 'I');
  Kokkos::deep_copy(reg_size, 0);
  DEBUG_PRINT("Setup offset and label views\n");
  CompactHostTable<10> distinct_h, shared_h;
  Kokkos::deep_copy(distinct_h, distinct);
  Kokkos::deep_copy(shared_h, shared);
  DEBUG_PRINT("Copied tables to host\n");
  printf("Copied tables to host\n");
  Kokkos::fence();
  Kokkos::parallel_for("Update repeat entries", Kokkos::RangePolicy<>(0,shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      CompactNodeInfo info = shared.key_at(i);
      uint32_t node = info.node;
      uint32_t prev = info.size;
      uint32_t start = leftmost_leaf(node, num_nodes);
      uint32_t end   = rightmost_leaf(node, num_nodes);
      offset(node) = prev;
      labels(node) = 'R';
      reg_size(node) = num_leaf_descendents(node, num_nodes);
//      for(uint32_t j=start; j<=end; j++) {
//        offset(j) = prev;
//        labels(j) = 'R';
//      }
    }
  });
  DEBUG_PRINT("Update repeat chunks\n");

  Kokkos::parallel_for("Update distinct entries", Kokkos::RangePolicy<>(0,distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      CompactNodeInfo info = distinct.key_at(i);
      uint32_t node = info.node;
      uint32_t prev = info.size;
      uint32_t start = leftmost_leaf(node, num_nodes);
      uint32_t end = rightmost_leaf(node, num_nodes);
      offset(node) = prev;
      if(node == prev) {
        labels(node) = 'd';
      } else {
        labels(node) = 'r';
      }
      reg_size(node) = num_leaf_descendents(node, num_nodes);
//      char l = 'r';
//      if(node == prev)
//        l = 'd';
//      for(uint32_t j=start; j<=end; j++) {
//        offset(j) = prev;
//        labels(j) = l;
//      }
    }
  });
  DEBUG_PRINT("Update distinct chunks\n");
  auto offset_h = Kokkos::create_mirror_view(offset);
  auto labels_h = Kokkos::create_mirror_view(labels);
  auto reg_size_h = Kokkos::create_mirror_view(reg_size);
  Kokkos::deep_copy(offset_h, offset);
  Kokkos::deep_copy(labels_h, labels);
  Kokkos::deep_copy(reg_size_h, reg_size);
  std::ofstream file;
  file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  file.open(filename, std::ofstream::out);
  std::cout << "Opened log file: " << filename << std::endl;
  file << "Position: Offset: Label\n";
  for(size_t i=0; i<num_nodes; i++) {
if(labels_h(i) != 'I') 
    file << i << "(" << i+num_chunks-1 << ")" << "\t: " << offset_h(i) << "\t: " << reg_size_h(i) << "\t: " << labels_h(i) << std::endl;
//    file << i << "(" << i+num_chunks-1 << ")" << "\t: " << offset_h(i) << "\t: " << labels_h(i) << std::endl;
  }
  file.close();
  STDOUT_PRINT("Wrote log\n");
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
    std::vector<std::string> chkpt_files;
    std::vector<std::string> full_chkpt_files;
    for(uint32_t i=0; i<num_chkpts; i++) {
      full_chkpt_files.push_back(std::string(argv[3+i]));
      chkpt_files.push_back(std::string(argv[3+i]));
    }
    DEBUG_PRINT("Read checkpoint files\n");
    DEBUG_PRINT("Number of checkpoints: %u\n", num_chkpts);

    DistinctMap g_distinct_chunks = DistinctMap(1);
    SharedMap g_shared_chunks = SharedMap(1);
    DistinctMap g_distinct_nodes  = DistinctMap(1);
    SharedMap g_shared_nodes = SharedMap(1);

//    HashList prior_list(0), current_list(0);

    for(uint32_t idx=0; idx<num_chkpts; idx++) {
      DEBUG_PRINT("Processing checkpoint %u\n", idx);
      std::ifstream f;
      f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
      DEBUG_PRINT("Set exceptions\n");
      f.open(chkpt_files[idx], std::ifstream::in | std::ifstream::binary);
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
//        g_distinct_chunks.rehash(g_distinct_chunks.size()+num_chunks);
//        g_shared_chunks.rehash(g_shared_chunks.size()+num_chunks);
//        g_distinct_nodes.rehash(g_distinct_nodes.size()+2*num_chunks + 1);
//        g_shared_nodes.rehash(g_shared_nodes.size()+2*num_chunks + 1);
        g_distinct_chunks.rehash(num_chunks);
        g_shared_chunks.rehash(num_chunks);
uint32_t nentries=0;
for(uint32_t sz=128; sz<8192; sz*=2) {
  uint32_t n = data_len/sz;
  if(n*sz < data_len)
    n += 1;
  nentries += n;
}
        g_distinct_nodes.rehash(nentries);
        g_shared_nodes.rehash(  nentries);
//        g_shared_nodes.rehash(2*num_chunks + 1);
      }

      size_t name_start = chkpt_files[idx].rfind('/') + 1;
      chkpt_files[idx].erase(chkpt_files[idx].begin(), chkpt_files[idx].begin()+name_start);

      std::fstream result_data;
      result_data.open(chkpt_files[idx]+".chunk_size."+std::to_string(chunk_size)+".csv", std::fstream::out | std::fstream::app);

//      Kokkos::View<uint8_t*> first("First region", data_len);
      Kokkos::View<uint8_t*> current("Current region", data_len);
      Kokkos::View<uint8_t*>::HostMirror current_h = Kokkos::create_mirror_view(current);
      f.read((char*)(current_h.data()), data_len);
      Kokkos::deep_copy(current, current_h);
      DEBUG_PRINT("Size of full checkpoint: %zd\n", current.size());

//      SHA1 hasher;
//      Murmur3C hasher;
      MD5Hash hasher;

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
//        file.write((const char*)(current_h.data()), current_h.size());
//        file.flush();
        DEBUG_PRINT("Wrote full checkpoint\n");
        Kokkos::Profiling::popRegion();
        Timer::time_point end_write = Timer::now();
        auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write-start_write).count();
        STDOUT_PRINT("Time spent writing full checkpoint: %f\n", write_time);
        result_data << "0.0" << "," << "0.0" << "," << write_time << "," << current.size() << ',' << "0" << ',';
        file.close();
      }
      // Hash list deduplication
      {
        DistinctMap l_distinct_chunks = DistinctMap(num_chunks);
        SharedMap l_shared_chunks = SharedMap(num_chunks);

        HashList list0 = HashList(num_chunks);
        DEBUG_PRINT("initialized local maps and list\n");
        Kokkos::fence();

        Kokkos::fence();
        Timer::time_point start_compare = Timer::now();
        Kokkos::Profiling::pushRegion((std::string("Find distinct chunks ") + std::to_string(idx)).c_str());
        compare_lists(hasher, list0, idx, current, chunk_size, l_shared_chunks, l_distinct_chunks, g_shared_chunks, g_distinct_chunks);
        Kokkos::Profiling::popRegion();
        Timer::time_point end_compare = Timer::now();

        Kokkos::fence();

        STDOUT_PRINT("Size of distinct map: %u\n", l_distinct_chunks.size());
        STDOUT_PRINT("Size of shared map:   %u\n", l_shared_chunks.size());

        auto compare_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_compare - start_compare).count();

        if(idx == 0) {
          // Update global distinct map
          g_distinct_chunks.rehash(g_distinct_chunks.size()+l_distinct_chunks.size());
          Kokkos::deep_copy(g_distinct_chunks, l_distinct_chunks);
          // Update global shared map
          g_shared_chunks.rehash(g_shared_chunks.size()+l_shared_chunks.size());
          Kokkos::deep_copy(g_shared_chunks, l_shared_chunks);
DEBUG_PRINT("Updated global lists\n");
        }

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
        if(idx > 0) {
          prior_idx = idx-1;
        }
        Kokkos::fence();
        Timer::time_point start_collect = Timer::now();
        Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
        Kokkos::View<uint8_t*> buffer_d;
        std::pair<uint64_t,uint64_t> datasizes = write_incr_chkpt_hashlist(full_chkpt_files[idx]+".hashlist.incr_chkpt",  current, buffer_d, chunk_size, l_distinct_chunks, l_shared_chunks, prior_idx, idx);
        Kokkos::Profiling::popRegion();
        Timer::time_point end_collect = Timer::now();
        auto collect_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_collect - start_collect).count();
printf("Time spect collecting updates: %f\n", collect_time);

        Timer::time_point start_write = Timer::now();
        Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
        auto buffer_h = Kokkos::create_mirror_view(buffer_d);
        Kokkos::deep_copy(buffer_h, buffer_d);
        Kokkos::fence();
        Kokkos::Profiling::popRegion();
        Timer::time_point end_write = Timer::now();
        auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
printf("Time spect copying updates: %f\n", write_time);
        result_data << compare_time << ',' << collect_time << ',' << write_time << ',' << datasizes.first << ',' << datasizes.second << ',';
#endif
      }
      // Merkle Tree deduplication
      {
//if(idx > 0 && idx%2 == 0)
//chunk_size *= 2;

        uint32_t n_chunks = data_len/128;
        if(n_chunks*128 < data_len)
          n_chunks += 1;
        MerkleTree tree0 = MerkleTree(n_chunks);
        CompactTable<10> shared_updates = CompactTable<10>(num_chunks);
        CompactTable<10> distinct_updates = CompactTable<10>(num_chunks);
        DEBUG_PRINT("Allocated tree and tables\n");

        Kokkos::fence();

        if(idx == 0) {
          Timer::time_point start_create_tree0 = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Deduplicate chkpt ") + std::to_string(idx)).c_str());
          create_merkle_tree(hasher, tree0, current, chunk_size, idx, g_distinct_nodes, g_shared_nodes);
          Kokkos::Profiling::popRegion();
          Timer::time_point end_create_tree0 = Timer::now();

          STDOUT_PRINT("Size of shared entries: %u\n", g_shared_nodes.size());
          STDOUT_PRINT("Size of distinct entries: %u\n", g_distinct_nodes.size());
          STDOUT_PRINT("Size of shared updates: %u\n", shared_updates.size());
          STDOUT_PRINT("Size of distinct updates: %u\n", distinct_updates.size());
          auto compare_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_create_tree0 - start_create_tree0).count();
#ifdef WRITE_CHKPT
          uint32_t prior_idx = 0;
          if(idx > 0)
            prior_idx = idx-1;
          Kokkos::fence();
//tree0.print();
//          dump_tree_metadata(chkpt_files[idx]+".chunk_size."+std::to_string(chunk_size)+".incr_chkpt.log", g_distinct_nodes, g_shared_nodes, num_chunks, num_chunks);
          Kokkos::View<uint8_t*> buffer_d;
          Timer::time_point start_collect = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
          auto datasizes = write_incr_chkpt_hashtree(full_chkpt_files[idx]+".hashtree.incr_chkpt", current, buffer_d, chunk_size, g_distinct_nodes, g_shared_nodes, prior_idx, idx);
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
#endif
        } else {
          DistinctMap l_distinct_nodes(g_distinct_nodes.capacity());
          SharedMap l_shared_nodes = SharedMap(2*num_chunks-1);
//g_distinct_nodes.rehash(g_distinct_nodes.size()+2*num_chunks-1);
//g_shared_nodes.rehash(g_shared_nodes.size()+2*num_chunks-1);
          DEBUG_PRINT("Allocated maps\n");

          Timer::time_point start_create_tree0 = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Deduplicate chkpt ") + std::to_string(idx)).c_str());
//          deduplicate_data(current, chunk_size, hasher, tree0, idx, g_shared_nodes, g_distinct_nodes, l_shared_nodes, l_distinct_nodes, distinct_updates);
          deduplicate_data(current, chunk_size, hasher, tree0, idx, g_shared_nodes, g_distinct_nodes, l_shared_nodes, l_distinct_nodes, shared_updates, distinct_updates);
//          deduplicate_data_team(current, chunk_size, hasher, 128, tree0, idx, g_shared_nodes, g_distinct_nodes, l_shared_nodes, l_distinct_nodes, updates);
          Kokkos::Profiling::popRegion();
          Timer::time_point end_create_tree0 = Timer::now();

          STDOUT_PRINT("Size of shared entries: %u\n", l_shared_nodes.size());
          STDOUT_PRINT("Size of distinct entries: %u\n", l_distinct_nodes.size());
          STDOUT_PRINT("Size of shared updates: %u\n", shared_updates.size());
          STDOUT_PRINT("Size of distinct updates: %u\n", distinct_updates.size());

//	        if(idx == 0) {
//            // Update global shared map
//            g_shared_nodes.rehash(g_shared_nodes.size()+l_shared_nodes.size());
//            Kokkos::deep_copy(g_shared_nodes, l_shared_nodes);
//          }
          auto compare_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_create_tree0 - start_create_tree0).count();

#ifdef WRITE_CHKPT
          uint32_t prior_idx = 0;
          if(idx > 0) {
            prior_idx = idx-1;
            Kokkos::fence();

//tree0.print();
//            dump_tree_metadata(chkpt_files[idx]+".chunk_size."+std::to_string(chunk_size)+".incr_chkpt.log", distinct_updates, shared_updates, num_chunks, num_chunks);

            Kokkos::View<uint8_t*> buffer_d;
            Timer::time_point start_collect = Timer::now();
            Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
            auto datasizes = write_incr_chkpt_hashtree(full_chkpt_files[idx]+".hashtree.incr_chkpt", current, buffer_d, chunk_size, distinct_updates, shared_updates, prior_idx, idx);
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
          }
#endif
        }
        Kokkos::fence();
      }
      Kokkos::fence();
      DEBUG_PRINT("Closing files\n");
      result_data.close();
      STDOUT_PRINT("------------------------------------------------------\n");
    }
    STDOUT_PRINT("------------------------------------------------------\n");
  }
  Kokkos::finalize();
}

