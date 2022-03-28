#include "dedup.hpp"
#include "hash_functions.hpp"
//#include "merkle_tree.hpp"
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <chrono>

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
                    std::map<std::vector<uint32_t>, size_t>& prev_hashes,
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


void deduplicate_module_t::cpu_dedup(uint8_t* data, 
                                     size_t data_len,
                                     std::map<std::vector<uint32_t>, size_t>& prev_hashes,
//				     std::map<int, std::pair<uint8_t*,size_t>>& prev_trees,
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

//  if(config.use_merkle_trees) {
//    calc_start = high_resolution_clock::now();
//    header.merkle_tree = create_merkle_tree(data, data_len, hash_len, chunk_size);
//    calc_end = high_resolution_clock::now();
//    find_start = high_resolution_clock::now();
//    find_end = high_resolution_clock::now();
//    comp_start = high_resolution_clock::now();
//    for(auto it=prev_tress.begin(); it!=prev_trees.end(); it++) {
//      compare_merkle_trees(header.merkle_tree, it->second.first, num_hashes, it->second.second, hash_len, num_hashes, unique_hashes, num_changes);
//    }
//    comp_end = high_resolution_clock::now();
//  } else {
    calc_start = high_resolution_clock::now();
    calculate_hashes(data, data_len, header.hashes, config.chunk_size, config.hash_func);
    calc_end = high_resolution_clock::now();
    find_start = high_resolution_clock::now();
    find_unique(header.hashes, prev_hashes, unique_hashes, num_changes, hash_len);
    find_end = high_resolution_clock::now();
    comp_start = high_resolution_clock::now();
    compare_hashes(header.hashes, prev_hashes, unique_hashes, num_changes, hash_len);
    comp_end = high_resolution_clock::now();
//  }
  printf("Number of changes: %zd\n", num_changes);
  gather_start = high_resolution_clock::now();
  gather_changes(data, data_len, unique_hashes, config.chunk_size, incr_data, incr_len);
  gather_end = high_resolution_clock::now();
  printf("Checkpoint size: %zd\n", incr_len);

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
