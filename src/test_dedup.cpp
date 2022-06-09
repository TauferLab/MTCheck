#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include "stdio.h"
#include <string>
#include <fstream>
#include "kokkos_merkle_tree.hpp"
#include "kokkos_hash_list.hpp"

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

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Timer = std::chrono::high_resolution_clock;
printf("------------------------------------------------------\n");

//    uint32_t chunk_size = static_cast<uint32_t>(strtoul(argv[1], NULL, 0));
//    uint32_t data_len   = static_cast<uint32_t>(strtoul(argv[2], NULL, 0));
//    uint32_t chance     = static_cast<uint32_t>(strtoul(argv[3], NULL, 0));
//    uint32_t num_chkpts = static_cast<uint32_t>(strtoul(argv[4], NULL, 0));
//    char generator_mode = *(argv[5]);
  
    uint32_t chunk_size = 1;
    uint32_t data_len = 53;
    const char* test_str0 = "Hello Muddah. Hello Fadduh. Here I am at camp Granada"; //53
    const char* test_str1 = "Hello Mother. Hello Father. Here I am at camp Granada"; //53
    printf("Set initial test strings\n");
    Kokkos::View<uint8_t*> str0_d("Test string 0", 53);
    Kokkos::View<uint8_t*> str1_d("Test string 1", 53);
    printf("Allocate device Views\n");
    Kokkos::View<uint8_t*>::HostMirror str0_h = Kokkos::create_mirror_view(str0_d);
    Kokkos::View<uint8_t*>::HostMirror str1_h = Kokkos::create_mirror_view(str1_d);
    printf("Initialized Views\n");
    for(uint32_t i=0; i<53; i++) {
      str0_h(i) = test_str0[i];
      str1_h(i) = test_str1[i];
    }
    Kokkos::deep_copy(str0_d, str0_h);
    Kokkos::deep_copy(str1_d, str1_h);
    printf("Copied data to Views\n");

    DistinctMap g_distinct_nodes  = DistinctMap(2*105);
    
    SHA1 hasher;
    int counter = 0;
    {
      uint32_t num_nodes = 2*53-1;

      uint32_t num_chunks = data_len/chunk_size;
      if(num_chunks*chunk_size < data_len)
        num_chunks += 1;
      MerkleTree tree0 = MerkleTree(num_chunks);

      Kokkos::fence();

      create_merkle_tree(hasher, tree0, str0_d, chunk_size);

      Kokkos::fence();

      Queue queue(num_nodes);
      queue.clear();
      queue.host_push(0);

      Kokkos::fence();

      compare_trees_fused(tree0, queue, counter, g_distinct_nodes);

      Kokkos::fence();

      queue.clear();
      queue.host_push(0);
      
      Kokkos::fence();

      print_nodes(tree0, counter, g_distinct_nodes);

      Kokkos::fence();

      counter += 1;
    }
    {
      uint32_t num_nodes = 2*53-1;

      uint32_t num_chunks = data_len/chunk_size;
      if(num_chunks*chunk_size < data_len)
        num_chunks += 1;
      MerkleTree tree0 = MerkleTree(num_chunks);

      Kokkos::fence();

      create_merkle_tree(hasher, tree0, str1_d, chunk_size);

      Kokkos::fence();

      Queue queue(num_nodes);
      queue.clear();
      queue.host_push(0);

      Kokkos::fence();

      compare_trees_fused(tree0, queue, counter, g_distinct_nodes);

      Kokkos::fence();

      queue.clear();
      queue.host_push(0);
      
      Kokkos::fence();

      print_nodes(tree0, counter, g_distinct_nodes);

      Kokkos::fence();

      counter += 1;
    }

////////
//    std::string list_timing = "hashlist_len_" + std::to_string(data_len) + 
//                              "_chunk_size_" + std::to_string(chunk_size) + 
//                              "_n_chkpts_" + std::to_string(num_chkpts) + 
//                              "_generator_mode_" + generator_mode + 
//                              "_chance_" + std::to_string(chance) + ".csv";
//    std::fstream list_fs;
//    list_fs.open(list_timing, std::fstream::out | std::fstream::app);
//    list_fs << "CreateList, FindDstinctChunks\n";
//
//    std::string tree_timing = "hashtree_len_" + std::to_string(data_len) + 
//                              "_chunk_size_" + std::to_string(chunk_size) + 
//                              "_n_chkpts_" + std::to_string(num_chkpts) + 
//                              "_generator_mode_" + generator_mode + 
//                              "_chance_" + std::to_string(chance) + ".csv";
//    std::fstream tree_fs;
//    tree_fs.open(tree_timing, std::fstream::out | std::fstream::app);
//    tree_fs << "CreateTree, FindDistinctSubtrees, CompareTrees\n";
////    tree_fs << "CreateTree, CompareTrees\n";
//
//    SHA1 hasher;
//    uint32_t num_chunks = data_len/chunk_size;
//    if(num_chunks*chunk_size < data_len)
//      num_chunks += 1;
//    uint32_t num_nodes = 2*num_chunks-1;
//    DistinctMap g_distinct_chunks = DistinctMap(num_chunks);
//    DistinctMap g_distinct_nodes  = DistinctMap(num_nodes);
//    Kokkos::View<uint8_t*> previous("Previous chkpt", data_len);
//    Kokkos::View<uint8_t*> current = generate_initial_data(data_len);
//    int counter = 0;
//    do {
//      // Hash list deduplication
//      {
//        DistinctMap l_distinct_chunks = DistinctMap(num_chunks);
//        SharedMap l_shared_chunks = SharedMap(num_chunks);
//
//uint32_t num_chunks = current.size()/chunk_size;
//if(num_chunks*chunk_size < current.size())
//  num_chunks += 1;
//HashList list0 = HashList(num_chunks);
//
//        Timer::time_point start_create_list0 = Timer::now();
//        Kokkos::Profiling::pushRegion((std::string("Create List ") + std::to_string(counter)).c_str());
////        HashList list0 = create_hash_list(hasher, current, chunk_size);
//        create_hash_list(hasher, list0, current, chunk_size);
//        Kokkos::Profiling::popRegion();
//        Timer::time_point end_create_list0 = Timer::now();
//
//        Kokkos::fence();
//        Timer::time_point start_find_distinct0 = Timer::now();
//        Kokkos::Profiling::pushRegion((std::string("Find distinct chunks ") + std::to_string(counter)).c_str());
//        find_distinct_chunks(list0, counter, l_distinct_chunks, l_shared_chunks, g_distinct_chunks);
//        Kokkos::Profiling::popRegion();
//        Timer::time_point end_find_distinct0 = Timer::now();
//
//        Kokkos::fence();
//
//        count_distinct_nodes(list0, counter, l_distinct_chunks);
//
//        list_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_create_list0 - start_create_list0).count();
//        list_fs << ",";
//        list_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_find_distinct0 - start_find_distinct0).count();
//        list_fs << "\n";
////        count_distinct_nodes(list0, counter, l_distinct_chunks);
////        print_distinct_nodes(list0, counter, l_distinct_chunks);
////        printf("Number of distinct chunks: %u\n", l_distinct_chunks.size());
//
//        // Update global map
//        g_distinct_chunks.rehash(g_distinct_chunks.size()+l_distinct_chunks.size());
//        Kokkos::parallel_for(l_distinct_chunks.capacity(), KOKKOS_LAMBDA(const uint32_t i) {
//          if(l_distinct_chunks.valid_at(i) && !g_distinct_chunks.exists(l_distinct_chunks.key_at(i))) {
//            auto result = g_distinct_chunks.insert(l_distinct_chunks.key_at(i), l_distinct_chunks.value_at(i));
//            if(result.existing()) {
//              printf("Key already exists in global chunk map\n");
//            } else if(result.failed()) {
//              printf("Failed to insert local entry into global chunk map\n");
//            }
//          }
//        });
//      }
//      // Merkle Tree deduplication
//      {
//        DistinctMap l_distinct_nodes  = DistinctMap(num_nodes*2);
//        SharedMap l_shared_nodes  = SharedMap(num_nodes*2);
//
//uint32_t num_chunks = current.size()/chunk_size;
//if(num_chunks*chunk_size < current.size())
//  num_chunks += 1;
//MerkleTree tree0 = MerkleTree(num_chunks);
//        Timer::time_point start_create_tree0 = Timer::now();
//        Kokkos::Profiling::pushRegion((std::string("Create Tree ") + std::to_string(counter)).c_str());
////        MerkleTree tree0 = create_merkle_tree(hasher, current, chunk_size);
//        create_merkle_tree(hasher, tree0, current, chunk_size);
//        Kokkos::Profiling::popRegion();
//        Timer::time_point end_create_tree0 = Timer::now();
//
////        Kokkos::fence();
////
////        Timer::time_point start_find_distinct0 = Timer::now();
////        Kokkos::Profiling::pushRegion((std::string("Find distinct nodes ") + std::to_string(counter)).c_str());
////        find_distinct_subtrees(tree0, counter, l_distinct_nodes, l_shared_nodes);
////        Kokkos::Profiling::popRegion();
////        Timer::time_point end_find_distinct0 = Timer::now();
//
////        Timer::time_point start_create_tree0 = Timer::now();
////        Kokkos::Profiling::pushRegion("Create Tree 0");
////        MerkleTree tree0 = create_merkle_tree_find_distinct_subtrees(hasher, current, chunk_size, counter, l_distinct_nodes, l_shared_nodes);
////        Kokkos::Profiling::popRegion();
////        Timer::time_point end_create_tree0 = Timer::now();
//
//        Kokkos::fence();
//
//        Queue queue(num_nodes);
//queue.clear();
//queue.host_push(0);
//        Kokkos::fence();
//        Timer::time_point start_compare1 = Timer::now();
//        Kokkos::Profiling::pushRegion((std::string("Compare trees ") + std::to_string(counter)).c_str());
////        compare_trees(tree0, counter, l_distinct_nodes, g_distinct_nodes, queue);
////        compare_trees(tree0, counter, l_distinct_nodes, g_distinct_nodes);
//        compare_trees_fused(tree0, queue, counter, l_distinct_nodes, l_shared_nodes, g_distinct_nodes);
//        Kokkos::Profiling::popRegion();
//        Timer::time_point end_compare1 = Timer::now();
//
////        Timer::time_point start_compare1 = Timer::now();
////        Kokkos::Profiling::pushRegion((std::string("Compare trees ") + std::to_string(counter)).c_str());
////        compare_trees_fused(tree0, counter, l_distinct_nodes, l_shared_nodes, g_distinct_nodes);
////        Kokkos::Profiling::popRegion();
////        Timer::time_point end_compare1 = Timer::now();
//
//        Kokkos::fence();
//
//queue.clear();
//queue.host_push(0);
//        count_distinct_nodes(tree0, counter, l_distinct_nodes);
//
//        tree_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_create_tree0 - start_create_tree0).count();
//        tree_fs << ",";
////        tree_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_find_distinct0 - start_find_distinct0).count();
////        tree_fs << ",";
//        tree_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_compare1 - start_compare1).count();
//        tree_fs << "\n";
//
////        print_distinct_nodes(tree0, counter, l_distinct_nodes);
//
//        // Update global map
//        g_distinct_nodes.rehash(g_distinct_nodes.size()+l_distinct_nodes.size());
//        Kokkos::parallel_for(l_distinct_nodes.capacity(), KOKKOS_LAMBDA(const uint32_t i) {
//          if(l_distinct_nodes.valid_at(i) && !g_distinct_nodes.exists(l_distinct_nodes.key_at(i))) {
//            auto result = g_distinct_nodes.insert(l_distinct_nodes.key_at(i), l_distinct_nodes.value_at(i));
//            if(result.existing()) {
//              printf("Key already exists in global node map\n");
//            } else if(result.failed()) {
//              printf("Failed to insert local entry into global node map\n");
//            }
//          }
//        });
//      }
//
//      // Generate next set of artificial data
//      Kokkos::deep_copy(previous, current);
//      if(generator_mode == 'R') {
//        perturb_data(previous, current, chance, Random);
//      } else if(generator_mode == 'B') {
//        perturb_data(previous, current, chance, BeginningIdentical);
//      } else if(generator_mode == 'C') {
//        perturb_data(previous, current, chance, Checkered);
//      } else if(generator_mode == 'I') {
//        perturb_data(previous, current, chance, Identical);
//      } else if(generator_mode == 'S') {
//        perturb_data(previous, current, chance, Sparse);
//      }
//      counter += 1;
//    } while(counter < num_chkpts);
//    list_fs.close();
//    tree_fs.close();
  }
printf("------------------------------------------------------\n");
  Kokkos::finalize();
}
