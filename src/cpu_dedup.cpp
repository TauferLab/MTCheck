#include "dedup.hpp"
#include "hash_functions.hpp"
#include "merkle_tree.hpp"
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <chrono>

//#define DEBUG

// Check if hashes are the same
bool identical_hashes(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, size_t len) {
  for(size_t i=0; i<a.size(); i++) {
    if(a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

// Calculate hashes
void calculate_hashes(const uint8_t* data, 
                      size_t data_len,
                      std::vector<std::vector<uint32_t>>& hashes, 
                      size_t chunk_size,
                      Hasher* hasher) {
  // Calculate the number of hashes
  size_t num_hashes = data_len / chunk_size;
  if(chunk_size*num_hashes < data_len)
    num_hashes += 1;
  // Split data into chunks and compute hashes
  for(size_t idx=0; idx<num_hashes; idx++) {
    int block_size = chunk_size;
    if(chunk_size*(idx+1) > data_len)
      block_size = (data_len)-(idx*chunk_size);
    std::vector<uint32_t> digest(hasher->digest_size(), 0);
    hasher->hash(data + (idx*chunk_size), block_size, (uint8_t*)(digest.data()));
    hashes.push_back(digest);
  }
}

void find_unique(std::vector<std::vector<uint32_t>>& curr_hashes, 
                    std::map<std::vector<uint32_t>, size_t>& changed_regions,
                    size_t& num_changes, 
                    const int hash_len) {
  // Prune current hashes to only those that are unique
  if(changed_regions.size() == 0) {
    for(size_t i=0; i<curr_hashes.size(); i++) {
      bool unique = true;
      std::vector<uint32_t> curr_digest = curr_hashes[i];
      for(size_t j=0; j<curr_hashes.size(); j++) {
        // Check if the hashes i and j are identical
        if(i != j && identical_hashes(curr_hashes[j], curr_hashes[i], hash_len)) {
          unique = false;
          // Ensure that only one of the duplicate hashes are in the map of changed hashes
          if(i < j) {
            changed_regions.insert(std::make_pair(curr_digest, i));
          }
          break;
        }
      }
      // Insert unique hash into the map
      if(unique) {
        changed_regions.insert(std::make_pair(curr_digest, i));
      }
    }
  }
}

void find_unique(const uint8_t* tree,
                 const size_t num_hashes,
                 const size_t hash_len,
                 size_t* unique_subtree_roots,
                 int* unique_subtree_ids,
                 size_t num_unique_subtrees
                )
{
  for(size_t i=0; i<num_unique_subtrees; i++) {
    for(size_t j=0; j<num_unique_subtrees; j++) {
      if(i != j && identical_hashes(tree+(num_hashes-1)*hash_len+unique_subtree_roots[i]*hash_len,
                                    tree+(num_hashes-1)*hash_len+unique_subtree_roots[j]*hash_len, hash_len)) {
        if(i < j) {
          unique_subtree_roots[i] = 2*num_hashes-1;
          unique_subtree_ids[i] = -1;
        }
      }
    }
  }
}

// Compare set of hashes with the previous set
void compare_hashes(std::vector<std::vector<uint32_t>>& curr_hashes, 
                    std::map<std::vector<uint32_t>, size_t>& prev_hashes,
                    std::map<std::vector<uint32_t>, size_t>& changed_regions,
                    size_t& num_changes, 
                    const int hash_len) {
//  // Prune current hashes to only those that are unique
//  if(changed_regions.size() == 0) {
//    for(size_t i=0; i<curr_hashes.size(); i++) {
//      bool unique = true;
//      std::vector<uint32_t> curr_digest = curr_hashes[i];
//      for(size_t j=0; j<curr_hashes.size(); j++) {
//        // Check if the hashes i and j are identical
//        if(i != j && identical_hashes(curr_hashes[j], curr_hashes[i], hash_len)) {
//          unique = false;
//          // Ensure that only one of the duplicate hashes are in the map of changed hashes
//          if(i < j) {
//            changed_regions.insert(std::make_pair(curr_digest, i));
//          }
//          break;
//        }
//      }
//      // Insert unique hash into the map
//      if(unique) {
//        changed_regions.insert(std::make_pair(curr_digest, i));
//      }
//    }
//  }

  // Compare unique hashes with prior unique hashes
  for(size_t i=0; i<curr_hashes.size(); i++) {
    std::vector<uint32_t> curr_digest = curr_hashes[i];
    // Search for hash in map
    auto iter = changed_regions.find(curr_digest);
    if(iter != changed_regions.end()) {
      // Check if hash exists in prior checkpoints
      auto unique_iter = prev_hashes.find(curr_digest);
      if(unique_iter != prev_hashes.end()) {
        // Remove hash if it exists in prior checkpoints
        changed_regions.erase(iter);
      }
    }
  }
  num_changes = changed_regions.size();
}

// Gather changes into a contiguous buffer
template<typename T>
void gather_changes(T* data,
                    size_t data_len,
                    std::map<std::vector<uint32_t>, size_t>& changed_regions,
                    int chunk_size,
                    uint8_t** buffer,
                    size_t& diff_size) {
  // Ensure that the buffer has enough space
  *buffer = (uint8_t*) malloc(changed_regions.size()*chunk_size);
  diff_size = 0;
  size_t counter=0;
  // Iterate through all changed regions
  for(auto it=changed_regions.begin(); it!=changed_regions.end(); ++it) {
    // Calculate how much to copy to buffer
    size_t num_write = chunk_size;
    if(chunk_size*(it->second+1) > data_len*sizeof(T)) 
      num_write = (data_len*sizeof(T)) - it->second*chunk_size;
    diff_size += num_write;
    size_t pos = counter++;
    // Copy data into contiguous buffer
    memcpy((uint8_t*)(*buffer)+chunk_size*pos, (uint8_t*)(data)+chunk_size*it->second, num_write);
  }
}

// Gather changes into a contiguous buffer
template<typename T>
void gather_changes(T* data,
                    size_t data_len,
                    size_t* changed_regions,
		    size_t num_changed_regions,
                    int chunk_size,
                    uint8_t** buffer,
                    size_t& diff_size) {
  // Ensure that the buffer has enough space
  *buffer = (uint8_t*) malloc(num_changed_regions*chunk_size);
  diff_size = 0;
  size_t counter=0;
  size_t num_hashes = data_len/chunk_size;
  if(num_hashes*chunk_size < data_len)
    num_hashes += 1;
  // Iterate through all changed regions
  for(size_t i=0; i<num_changed_regions; i++) {
    if(changed_regions[i] < 2*num_hashes-1) {
      // Calculate how much to copy to buffer
      size_t num_write = chunk_size;
      if(chunk_size*((changed_regions[i]-(num_hashes-1))+1) > data_len*sizeof(T)) 
        num_write = (data_len*sizeof(T)) - (changed_regions[i]-(num_hashes-1))*chunk_size;
      diff_size += num_write;
      size_t pos = counter++;
      // Copy data into contiguous buffer
      memcpy((uint8_t*)(*buffer)+chunk_size*pos, (uint8_t*)(data)+chunk_size*(changed_regions[i]-(num_hashes-1)), num_write);
    }
  }
}

void deduplicate_module_t::cpu_dedup(uint8_t* data, 
                                     size_t data_len,
                                     std::map<std::vector<uint32_t>, size_t>& prev_hashes,
                                     region_header_t& header,
                                     uint8_t** incr_data,
                                     size_t& incr_len,
                                     config_t& config) {
  using namespace std::chrono;
  int chunk_size = config.chunk_size;
  size_t num_changes = 0;
  std::vector<uint8_t> buffer;
  size_t hash_len = config.hash_func->digest_size();
  std::map<std::vector<uint32_t>, size_t> unique_hashes;
  size_t num_hashes = data_len/hash_len;
  if(num_hashes*hash_len < data_len)
    num_hashes += 1;

  high_resolution_clock::time_point calc_start, calc_end, find_start, find_end, comp_start, comp_end, gather_start, gather_end;

  calc_start = high_resolution_clock::now();
  calculate_hashes(data, data_len, header.hashes, config.chunk_size, config.hash_func);
  calc_end = high_resolution_clock::now();
  find_start = high_resolution_clock::now();
  find_unique(header.hashes, unique_hashes, num_changes, hash_len);
  find_end = high_resolution_clock::now();
  comp_start = high_resolution_clock::now();
  compare_hashes(header.hashes, prev_hashes, unique_hashes, num_changes, hash_len);
  comp_end = high_resolution_clock::now();
#ifdef DEBUG
  printf("Number of changes: %zd\n", num_changes);
#endif
  gather_start = high_resolution_clock::now();
  gather_changes(data, data_len, unique_hashes, config.chunk_size, incr_data, incr_len);
  gather_end = high_resolution_clock::now();
#ifdef DEBUG
  printf("Checkpoint size: %zd\n", incr_len);
#endif

  std::cout << "\tTime spent calculating hashes: " << std::chrono::duration_cast<std::chrono::duration<double>>(calc_end-calc_start).count() << std::endl;
  std::cout << "\tTime spent finding hashes: " << std::chrono::duration_cast<std::chrono::duration<double>>(find_end-find_start).count() << std::endl;
  std::cout << "\tTime spent comparing hashes: " << std::chrono::duration_cast<std::chrono::duration<double>>(comp_end-comp_start).count() << std::endl;
  std::cout << "\tTime spent gathering changes: " << std::chrono::duration_cast<std::chrono::duration<double>>(gather_end-gather_start).count() << std::endl;

//  for(int i=0; i<5; i++) {
//    printf("%d ", header.hashes[0][i]);
//  }
//  printf("\n");

  // Update region header
  header.hash_size = hash_len;
  header.chunk_size = config.chunk_size;
  header.num_hashes = header.hashes.size();
  header.num_unique = unique_hashes.size();
  for(auto it=unique_hashes.begin(); it!=unique_hashes.end(); it++) {
    header.unique_hashes.push_back(it->second);
  }
  header.region_size = sizeof(size_t)*5 + hash_len*header.hashes.size() + sizeof(size_t)*header.unique_hashes.size();
}

void deduplicate_module_t::cpu_dedup(uint8_t* data, 
                                     size_t data_len,
                                     int chkpt_id,
                                     std::map<int, uint8_t*> prev_trees,
                                     region_header_t& header,
                                     uint8_t** incr_data,
                                     size_t& incr_len,
                                     config_t& config) {
  using namespace std::chrono;
  int chunk_size = config.chunk_size;
  size_t num_changes = 0;
  std::vector<uint8_t> buffer;
  size_t hash_len = config.hash_func->digest_size();
  std::map<std::vector<uint32_t>, size_t> unique_hashes;
  size_t num_hashes = data_len/chunk_size;
  if(num_hashes*chunk_size < data_len)
    num_hashes += 1;

  high_resolution_clock::time_point calc_start, calc_end, find_start, find_end, comp_start, comp_end, gather_start, gather_end;
#ifdef DEBUG
  printf("%d previous trees\n", prev_trees.size());
#endif

  calc_start = high_resolution_clock::now();
  header.merkle_tree = create_merkle_tree(data, data_len, hash_len, chunk_size);
#ifdef DEBUG
  printf("Tree for checkpoint %d, region %d\n", chkpt_id, header.id);
//print_merkle_tree(header.merkle_tree, hash_len, num_hashes);
#endif
  calc_end = high_resolution_clock::now();
  comp_start = high_resolution_clock::now();
  std::vector<std::map<size_t, int>> prev_diffs;
  size_t* unique_subtree_roots = new size_t[2*num_hashes-1];
  int* unique_subtree_ids = new int[2*num_hashes-1];
  size_t num_unique_subtrees = 0;
  size_t* shared_subtree_roots = new size_t[2*num_hashes-1];
  int* shared_subtree_ids = new int[2*num_hashes-1];
  size_t num_shared_subtrees = 0;
  for(auto it=prev_trees.begin(); it!=prev_trees.end(); it++) {
    num_unique_subtrees = 0;
    num_shared_subtrees = 0;
    std::map<size_t, int> node_chkpt_map;
    compare_merkle_trees(header.merkle_tree, it->second, 
                          chkpt_id, it->first,
                          hash_len, num_hashes, 
                          unique_subtree_roots, unique_subtree_ids,
                          num_unique_subtrees,
                          shared_subtree_roots, shared_subtree_ids,
                          num_shared_subtrees
                          ); 
    for(size_t i=0; i<num_shared_subtrees; i++) {
      node_chkpt_map.insert(std::make_pair(shared_subtree_roots[i], shared_subtree_ids[i]));
    }
    prev_diffs.push_back(node_chkpt_map);
  }
#ifdef DEBUG
  printf("Scanned through %d diffs\n", prev_diffs.size());
#endif

  num_unique_subtrees = 0;
  num_shared_subtrees = 0;

  const size_t num_nodes = 2*num_hashes-1;
#ifdef DEBUG
  printf("%zd nodes, %zd hashes\n", num_nodes, num_hashes);
#endif
  size_t* queue = new size_t[num_nodes];
  queue[0] = 0;
  size_t queue_size = 1;
  size_t queue_start = 0;
  size_t queue_end = 1;
  while(queue_size > 0) {
    size_t node = queue[queue_start];
    queue_start = (queue_start + 1) % num_nodes;
//    queue_start += 1;
    queue_size -= 1;
    bool found = false;
    for(size_t i=0; i<prev_diffs.size(); i++) {
      auto it = prev_diffs[i].find(node);
      if(it != prev_diffs[i].end()) {
        shared_subtree_roots[num_shared_subtrees] = node;
        shared_subtree_ids[num_shared_subtrees] = it->second;
	num_shared_subtrees += 1;
	found = true;
	break;
      }
    }
    if(!found) {
      size_t l_child = left_child_index(node);
      size_t r_child = right_child_index(node);
      if(l_child < num_nodes) {
        queue[queue_end] = l_child;
        queue_end = (queue_end + 1) % num_nodes;
//        queue_end += 1;
        queue_size += 1;
      }

      if(r_child < num_nodes) {
        queue[queue_end] = r_child;
        queue_end = (queue_end + 1) % num_nodes;
//        queue_end += 1;
        queue_size += 1;
      }
      if((l_child >= num_nodes) && (r_child >= num_nodes)) {
        unique_subtree_roots[num_unique_subtrees] = node;
        unique_subtree_ids[num_unique_subtrees] = chkpt_id;
        num_unique_subtrees += 1;
      }
    }
  }
#ifdef DEBUG
printf("Num unique subtrees: %zd\n", num_unique_subtrees);
//for(int i=0; i<num_unique_subtrees; i++) {
//  printf("Unique subtree: %zd, %d\n", unique_subtree_roots[i], unique_subtree_ids[i]);
//}
printf("Num shared subtrees: %zd\n", num_shared_subtrees);
//for(int i=0; i<num_shared_subtrees; i++) {
//  printf("Shared subtree: %zd, %d\n", shared_subtree_roots[i], shared_subtree_ids[i]);
//}
#endif

  comp_end = high_resolution_clock::now();
  find_start = high_resolution_clock::now();
//  find_unique(header.merkle_tree,
//              num_hashes,
//              hash_len,
//              unique_subtree_roots,
//              unique_subtree_ids,
//              num_unique_subtrees);
  find_end = high_resolution_clock::now();
#ifdef DEBUG
  printf("Number of changes: %zd\n", num_changes);
#endif
  gather_start = high_resolution_clock::now();
  gather_changes(data, data_len, unique_subtree_roots, num_unique_subtrees, config.chunk_size, incr_data, incr_len);
  gather_end = high_resolution_clock::now();
#ifdef DEBUG
  printf("Checkpoint size: %zd\n", incr_len);
#endif

  std::cout << "\tTime spent calculating hashes: " << std::chrono::duration_cast<std::chrono::duration<double>>(calc_end-calc_start).count() << std::endl;
  std::cout << "\tTime spent finding hashes: " << std::chrono::duration_cast<std::chrono::duration<double>>(find_end-find_start).count() << std::endl;
  std::cout << "\tTime spent comparing hashes: " << std::chrono::duration_cast<std::chrono::duration<double>>(comp_end-comp_start).count() << std::endl;
  std::cout << "\tTime spent gathering changes: " << std::chrono::duration_cast<std::chrono::duration<double>>(gather_end-gather_start).count() << std::endl;

//  for(int i=0; i<5; i++) {
//    printf("%d ", header.hashes[0][i]);
//  }
//  printf("\n");

  // Update region header
  header.hash_size = hash_len;
  header.chunk_size = config.chunk_size;
  header.num_hashes = num_hashes;
  header.unique_subtree_indices = unique_subtree_roots;
  header.unique_subtree_ids = unique_subtree_ids;
  header.num_unique_subtrees = num_unique_subtrees;
  header.shared_subtree_indices = shared_subtree_roots;
  header.shared_subtree_ids = shared_subtree_ids;
  header.num_shared_subtrees = num_shared_subtrees;
  header.region_size = sizeof(size_t)*5 + hash_len*header.hashes.size() + sizeof(size_t)*header.unique_hashes.size();
  delete[] queue;
  //delete[] unique_subtree_roots;
  //delete[] unique_subtree_ids;
  //delete[] shared_subtree_roots;
  //delete[] shared_subtree_ids;
}
