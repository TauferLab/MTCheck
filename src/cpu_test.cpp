#include <iostream>
#include <fstream>
#include "dedup.hpp"
#include "hash_functions.hpp"
//#include "merkle_tree.hpp"
#include <string>
#include <vector>

int main(int argc, char**argv) {
//const char* str_a = "abcd";
//const char* str_b = "abce";
//uint8_t* tree_a = create_merkle_tree((uint8_t*)str_a, 4, 20, 1);
//printf("Tree A\n");
//print_merkle_tree(tree_a, 20, 4);
//uint8_t* tree_b = create_merkle_tree((uint8_t*)str_b, 4, 20, 1);
//printf("\nTree B\n");
//print_merkle_tree(tree_b, 20, 4);
//size_t* unique_subtree_roots = new size_t[num_nodes(4)];
//int* unique_subtree_ids = new int[num_nodes(4)];
//size_t num_unique_subtrees = 0;
//size_t* shared_subtree_roots = new size_t[num_nodes(4)];
//int* shared_subtree_ids = new int[num_nodes(4)];
//size_t num_shared_subtrees = 0;
//compare_merkle_trees(tree_b, tree_a, 
//                     0, 1,
//		     20, 4,
//                     unique_subtree_roots, unique_subtree_ids,
//                     num_unique_subtrees,
//                     shared_subtree_roots, shared_subtree_ids,
//                     num_shared_subtrees);
//printf("\nNum unique subtrees: %zd\n", num_unique_subtrees);
//for(int i=0; i<num_unique_subtrees; i++) {
//  printf("Unique subtree: %zd, %d\n", unique_subtree_roots[i], unique_subtree_ids[i]);
//}
//printf("Num shared subtrees: %zd\n", num_shared_subtrees);
//for(int i=0; i<num_shared_subtrees; i++) {
//  printf("Shared subtree: %zd, %d\n", shared_subtree_roots[i], shared_subtree_ids[i]);
//}

  int chunk_size = atoi(argv[1]);
  std::string full_chkpt(argv[2]);
  std::string incr_chkpt = full_chkpt + ".cpu_test.incr_chkpt";

  SHA1 hasher;
  std::vector<std::string> prev_chkpt;
  config_t config;
  config.dedup_on_gpu = false;
  config.use_merkle_trees = true;
  config.chunk_size = chunk_size;
  config.hash_func = &hasher;

  for(int i=3; i<argc; i++) {
    prev_chkpt.push_back(std::string(argv[i]));
  }

  deduplicate_module_t module;
  
  // Read checkpoint data to memory
  header_t chkpt_header;
  std::map<int, size_t> region_map;
  try {
    std::ifstream f;
    size_t expected_size = 0;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(full_chkpt, std::ifstream::in | std::fstream::binary);
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
    chkpt_header.chkpt_size = file_size;
    chkpt_header.header_size = header_size;
    chkpt_header.num_regions = num_regions;
    f.seekg(0, f.beg);
    f.close();
  } catch(std::ifstream::failure &e) {
    std::cout << "cannot validate header for checkpoint " << full_chkpt << ", reason: " << e.what();
  }

  regions_t regions;
  std::ifstream f;
  f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  f.open(full_chkpt, std::ifstream::in | std::ifstream::binary);
  f.seekg(chkpt_header.header_size);
  for(auto &e : region_map) {
    region_t region;
    region.size = e.second;
    region.ptr = (void*) malloc(e.second);
    region.ptr_type = Host;
    f.read((char*)region.ptr, e.second);
    regions.insert(std::make_pair(e.first, region));
  }

//  std::string incr_chkpt_mem = full_chkpt + ".cpu_test.in_mem.incr_chkpt";
//  std::cout << "Starting data mode deduplication\n";
//  module.deduplicate_data(regions, incr_chkpt_mem, prev_chkpt, config);
//  std::cout << "Done with data mode\n";

  std::cout << "Deduplicating with Merkle trees\n";
  std::string incr_chkpt_merkle_tree = full_chkpt + ".cpu_test.merkle_trees.incr_chkpt";
  module.deduplicate_file(full_chkpt, incr_chkpt_merkle_tree, prev_chkpt, config);
//  std::cout << "Deduplicating with Hash lists\n";
//  std::string incr_chkpt_hash_list = full_chkpt + ".cpu_test.hash_list.incr_chkpt";
//  config.use_merkle_trees = false;
//  module.deduplicate_file(full_chkpt, incr_chkpt_hash_list, prev_chkpt, config);
}

