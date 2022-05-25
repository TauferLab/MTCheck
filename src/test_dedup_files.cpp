#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include "stdio.h"
#include <string>
#include <map>
#include <fstream>
#include "kokkos_merkle_tree.hpp"
#include "kokkos_hash_list.hpp"
#include <libgen.h>

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

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Timer = std::chrono::high_resolution_clock;
printf("------------------------------------------------------\n");

    // Process data from checkpoint files
    uint32_t chunk_size = static_cast<uint32_t>(strtoul(argv[1], NULL, 0));
    std::vector<std::string> chkpt_files;
    for(int i=2; i<argc; i++) {
      chkpt_files.push_back(std::string(argv[i]));
    }
    uint32_t num_chkpts = argc-2;

    DistinctMap g_distinct_chunks = DistinctMap(1);
    DistinctMap g_distinct_nodes  = DistinctMap(1);

    for(uint32_t idx=0; idx<num_chkpts; idx++) {
      header_t chkpt_header;
      std::map<int, size_t> region_map;
      read_full_header(chkpt_files[idx], chkpt_header, region_map);
printf("Read header for checkpoint %u\n", idx);
      g_distinct_chunks.rehash(g_distinct_chunks.size()+(chkpt_header.chkpt_size/chunk_size) + 1);
      g_distinct_nodes.rehash(g_distinct_nodes.size()+2*(chkpt_header.chkpt_size/chunk_size));
      regions_t regions;
      std::ifstream f;
      f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
      f.open(chkpt_files[idx], std::ifstream::in | std::ifstream::binary);
      f.seekg(0, f.end);
      size_t data_len = f.tellg();
      f.seekg(0, f.beg);
//      f.seekg(chkpt_header.header_size);

      size_t name_start = chkpt_files[idx].rfind('/') + 1;
      chkpt_files[idx].erase(chkpt_files[idx].begin(), chkpt_files[idx].begin()+name_start);
      std::string list_timing = "hashlist_filename_" + chkpt_files[idx] + 
                                "_chunk_size_" + std::to_string(chunk_size) + 
                                ".csv";
      std::fstream list_fs;
      list_fs.open(list_timing, std::fstream::out | std::fstream::app);
      list_fs << "CreateList, FindDstinctChunks\n";

      std::string tree_timing = "hashtree_filename_" + chkpt_files[idx] + 
                                "_chunk_size_" + std::to_string(chunk_size) + 
                                ".csv";
      std::fstream tree_fs;
      tree_fs.open(tree_timing, std::fstream::out | std::fstream::app);
      tree_fs << "CreateTree, FindDistinctSubtrees, CompareTrees\n";
//      tree_fs << "CreateTree, CompareTrees\n";

      Kokkos::View<uint8_t*> current("Current region", data_len);
      Kokkos::View<uint8_t*>::HostMirror current_h = Kokkos::create_mirror_view(current);
      f.read((char*)(current_h.data()), data_len);
      Kokkos::deep_copy(current, current_h);

//      for(auto &e : region_map) {
//        region_t region;
//        region.size = e.second;
//        size_t data_len = region.size;
//        Kokkos::View<uint8_t*> current("Current region", data_len);
//        Kokkos::View<uint8_t*>::HostMirror current_h = Kokkos::create_mirror_view(current);
//        f.read((char*)(current_h.data()), region.size);
//        Kokkos::deep_copy(current, current_h);
//printf("Read region of size %zd\n", data_len);

        SHA1 hasher;
        uint32_t num_chunks = data_len/chunk_size;
        if(num_chunks*chunk_size < data_len)
          num_chunks += 1;
        uint32_t num_nodes = 2*num_chunks-1;
        // Hash list deduplication
        {
          DistinctMap l_distinct_chunks = DistinctMap(num_chunks);
          SharedMap l_shared_chunks = SharedMap(num_chunks);

HashList list0 = HashList(num_chunks);

          Timer::time_point start_create_list0 = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Create List ") + std::to_string(idx)).c_str());
//          HashList list0 = create_hash_list(hasher, current, chunk_size);
          create_hash_list(hasher, list0, current, chunk_size);
          Kokkos::Profiling::popRegion();
          Timer::time_point end_create_list0 = Timer::now();

          Kokkos::fence();
          Timer::time_point start_find_distinct0 = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Find distinct chunks ") + std::to_string(idx)).c_str());
          find_distinct_chunks(list0, idx, l_distinct_chunks, l_shared_chunks, g_distinct_chunks);
          Kokkos::Profiling::popRegion();
          Timer::time_point end_find_distinct0 = Timer::now();

          Kokkos::fence();

          count_distinct_nodes(list0, idx, l_distinct_chunks);

          list_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_create_list0 - start_create_list0).count();
          list_fs << ",";
          list_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_find_distinct0 - start_find_distinct0).count();
          list_fs << "\n";
//          count_distinct_nodes(list0, idx, l_distinct_chunks);
//          print_distinct_nodes(list0, idx, l_distinct_chunks);
//          printf("Number of distinct chunks: %u\n", l_distinct_chunks.size());

          // Update global map
          g_distinct_chunks.rehash(g_distinct_chunks.size()+l_distinct_chunks.size());
          Kokkos::parallel_for(l_distinct_chunks.capacity(), KOKKOS_LAMBDA(const uint32_t i) {
            if(l_distinct_chunks.valid_at(i) && !g_distinct_chunks.exists(l_distinct_chunks.key_at(i))) {
              auto result = g_distinct_chunks.insert(l_distinct_chunks.key_at(i), l_distinct_chunks.value_at(i));
              if(result.existing()) {
                printf("Key already exists in global chunk map\n");
              } else if(result.failed()) {
                printf("Failed to insert local entry into global chunk map\n");
              }
            }
          });
        }
        // Merkle Tree deduplication
        {
          DistinctMap l_distinct_nodes  = DistinctMap(num_nodes*2);
          SharedMap l_shared_nodes  = SharedMap(num_nodes*2);

  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  const int32_t levels = INT_MAX;
//uint32_t num_chunks = current.size()/chunk_size;
MerkleTree tree0 = MerkleTree(num_chunks);
          Timer::time_point start_create_tree0 = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Create Tree ") + std::to_string(idx)).c_str());
//          MerkleTree tree0 = create_merkle_tree(hasher, current, chunk_size);
          create_merkle_tree(hasher, tree0, current, chunk_size, levels);
          Kokkos::Profiling::popRegion();
          Timer::time_point end_create_tree0 = Timer::now();

//          Kokkos::fence();
//
//          Timer::time_point start_find_distinct0 = Timer::now();
//          Kokkos::Profiling::pushRegion((std::string("Find distinct nodes ") + std::to_string(idx)).c_str());
//          find_distinct_subtrees(tree0, idx, l_distinct_nodes, l_shared_nodes);
//          Kokkos::Profiling::popRegion();
//          Timer::time_point end_find_distinct0 = Timer::now();

//          Timer::time_point start_create_tree0 = Timer::now();
//          Kokkos::Profiling::pushRegion("Create Tree 0");
//          MerkleTree tree0 = create_merkle_tree_find_distinct_subtrees(hasher, current, chunk_size, idx, l_distinct_nodes, l_shared_nodes);
//          Kokkos::Profiling::popRegion();
//          Timer::time_point end_create_tree0 = Timer::now();

          Kokkos::fence();

          Queue queue(num_nodes);
queue.clear();
queue.host_push(0);
//queue.fill((1 << (num_levels-levels) - 1), (1 << ((num_levels-levels) + 1)) - 1);
//printf("Filled queue\n");
          Kokkos::fence();
          Timer::time_point start_compare1 = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Compare trees ") + std::to_string(idx)).c_str());
//          compare_trees(tree0, idx, l_distinct_nodes, g_distinct_nodes, queue);
//          compare_trees(tree0, idx, l_distinct_nodes, g_distinct_nodes);
          compare_trees_fused(tree0, queue, idx, l_distinct_nodes, l_shared_nodes, g_distinct_nodes);
          Kokkos::Profiling::popRegion();
          Timer::time_point end_compare1 = Timer::now();

          Kokkos::fence();

queue.clear();
queue.host_push(0);
//queue.fill((1 << (num_levels-levels) - 1), (1 << ((num_levels-levels) + 1)) - 1);
          Kokkos::fence();
          count_distinct_nodes(tree0, queue, idx, l_distinct_nodes, l_shared_nodes);
          print_nodes(tree0, idx, l_distinct_nodes, l_shared_nodes);

          tree_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_create_tree0 - start_create_tree0).count();
          tree_fs << ",";
//          tree_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_find_distinct0 - start_find_distinct0).count();
//          tree_fs << ",";
          tree_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_compare1 - start_compare1).count();
          tree_fs << "\n";

//          print_distinct_nodes(tree0, idx, l_distinct_nodes);

          // Update global map
          g_distinct_nodes.rehash(g_distinct_nodes.size()+l_distinct_nodes.size());
          Kokkos::parallel_for(l_distinct_nodes.capacity(), KOKKOS_LAMBDA(const uint32_t i) {
            if(l_distinct_nodes.valid_at(i) && !g_distinct_nodes.exists(l_distinct_nodes.key_at(i))) {
              auto result = g_distinct_nodes.insert(l_distinct_nodes.key_at(i), l_distinct_nodes.value_at(i));
              if(result.existing()) {
                printf("Key already exists in global node map\n");
              } else if(result.failed()) {
                printf("Failed to insert local entry into global node map\n");
              }
            }
          });
        }
//      }
      list_fs.close();
      tree_fs.close();
printf("------------------------------------------------------\n");
    }
  }
printf("------------------------------------------------------\n");
  Kokkos::finalize();
}

