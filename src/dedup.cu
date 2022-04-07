#include "dedup.hpp"
//#include "hash_functions.hpp"
//#include "gpu_sha1.hpp"
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <chrono>

// collect unique hashes from prior incremental checkpoints
void collect_unique_hashes(std::vector<std::pair<header_t, std::map<int, region_header_t>>> &prev_chkpt_headers, std::map<int, std::map<std::vector<uint32_t>, size_t>>& unique_hashes) {
  // go through prior checkpoint headers
  for(size_t i=0; i<prev_chkpt_headers.size(); i++) {
    std::map<int, region_header_t>& region_headers = prev_chkpt_headers[i].second;
    // iterate through region headers
    for(auto it=region_headers.begin(); it!=region_headers.end(); it++) {
      // get iterator to unique hashes
      std::map<std::vector<uint32_t>,size_t> hashes;
      auto pos = unique_hashes.find(it->first);
      auto ret = unique_hashes.insert(pos, std::make_pair(it->first, hashes));
      auto hash_it = ret->second;
      // add unique hashes to map
      for(size_t k=0; k<it->second.num_unique; k++) {
        int id = it->first;
        size_t idx = it->second.unique_hashes[k];
        ret->second.insert(std::make_pair(it->second.hashes[idx], idx));
      }
    }
  }
}

void read_merkle_trees(std::vector<std::pair<header_t, std::map<int, region_header_t>>> &prev_chkpt_headers, std::map<int, std::map<int, uint8_t*>>& prev_trees) {
  if(prev_chkpt_headers.size() > 0) {
    // go through prior checkpoint headers
    for(size_t i=0; i<prev_chkpt_headers.size(); i++) {
      // Region map <region_id, region_header>
      std::map<int, region_header_t>& region_headers = prev_chkpt_headers[i].second;
      // iterate through region headers
      for(auto it=region_headers.begin(); it!=region_headers.end(); it++) {
        // get iterator to unique hashes
        std::map<int, uint8_t*> trees;
        auto pos = prev_trees.find(it->first);
        auto ret = prev_trees.insert(pos, std::make_pair(it->first, trees));
        auto tree_it = ret->second;
        ret->second.insert(std::make_pair(i, it->second.merkle_tree));
      }
    }
  }
}

// read header and region map from veloc checkpoint
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

// read incremental checkpoint header
bool read_incremental_headers(std::string& chkpt, header_t& header, std::map<int, region_header_t>& region_headers, config_t& config) {
  try {
    // read main header
    std::ifstream f;
    f.open(chkpt, std::ifstream::in | std::ifstream::binary);
    f.read((char*)(&header.chkpt_size), sizeof(size_t));
    f.read((char*)(&header.header_size), sizeof(size_t));
    f.read((char*)(&header.num_regions), sizeof(size_t));
    for(size_t i=0; i<header.num_regions; i++) {
      int id;
      region_header_t region;
      // read region headers
      f.read((char*)&id, sizeof(int));
      f.read((char*)&(region.region_size), sizeof(size_t));
      f.read((char*)&(region.hash_size), sizeof(size_t));
      f.read((char*)&(region.chunk_size), sizeof(size_t));
      f.read((char*)&(region.num_hashes), sizeof(size_t));
      if(config.use_merkle_trees) {
        region.merkle_tree = new uint8_t[(2*region.num_hashes-1)*region.hash_size];
        f.read((char*)(region.merkle_tree), (2*region.num_hashes-1)*region.hash_size);
        f.read((char*)&(region.num_unique_subtrees), sizeof(size_t));
	region.unique_subtree_indices = new size_t[region.num_unique_subtrees];
	region.unique_subtree_ids = new int[region.num_unique_subtrees];
	f.read((char*)(region.unique_subtree_indices), sizeof(size_t)*region.num_unique_subtrees);
	f.read((char*)(region.unique_subtree_ids), sizeof(int)*region.num_unique_subtrees);
        f.read((char*)&(region.num_shared_subtrees), sizeof(size_t));
	region.shared_subtree_indices = new size_t[region.num_shared_subtrees];
	region.shared_subtree_ids = new int[region.num_shared_subtrees];
	f.read((char*)(region.shared_subtree_indices), sizeof(size_t)*region.num_shared_subtrees);
	f.read((char*)(region.shared_subtree_ids), sizeof(int)*region.num_shared_subtrees);
      } else {
        f.read((char*)&(region.num_unique), sizeof(size_t));
        for(size_t j=0; j<region.num_hashes; j++) {
          std::vector<uint32_t> hash(region.hash_size, 0);
          f.read((char*)hash.data(), region.hash_size);
          region.hashes.push_back(hash);
        }
        for(size_t j=0; j<region.num_unique; j++) {
          size_t index;
          f.read((char*)&index, sizeof(size_t));
          region.unique_hashes.push_back(index);
        }
      }
      region_headers.insert(std::make_pair(id, region));
    }
    return true;
  } catch(std::ifstream::failure &e) {
    std::cout << "error reading header for checkpoint " << chkpt << std::endl;
    return false;
  }
}

// write region header to file
void write_header(std::ofstream& f, region_header_t& header, config_t& config) {
  f.write((char*)&header.id, sizeof(int));
  f.write((char*)&header.region_size, sizeof(size_t));
  f.write((char*)&header.hash_size, sizeof(size_t));
  f.write((char*)&header.chunk_size, sizeof(size_t));
  f.write((char*)&header.num_hashes, sizeof(size_t));
//  printf("num hashes: %zd\n", header.num_hashes);
//  printf("wrote subheader sections (apply to both hash list and hash trees\n");
  if(config.use_merkle_trees) {
//printf("using merkle tree\n");
//if(header.merkle_tree == NULL)
//  printf("merkle tree null!\n");
//printf("writing merkle tree of length %zd\n", (2*header.num_hashes-1)*header.hash_size);
    f.write((char*)(header.merkle_tree), (2*header.num_hashes-1)*header.hash_size);
//    printf("wrote merkle tree\n");
    f.write((char*)&(header.num_unique_subtrees), sizeof(size_t));
    f.write((char*)(header.unique_subtree_indices), sizeof(size_t)*header.num_unique_subtrees);
    f.write((char*)(header.unique_subtree_ids), sizeof(int)*header.num_unique_subtrees);
//    printf("wrote unique sub trees\n");
    f.write((char*)&(header.num_shared_subtrees), sizeof(size_t));
    f.write((char*)(header.shared_subtree_indices), sizeof(size_t)*header.num_shared_subtrees);
    f.write((char*)(header.shared_subtree_ids), sizeof(int)*header.num_shared_subtrees);
//    printf("wrote shared sub trees\n");
//    printf("wrote merkle tree header\n");
  } else {
    f.write((char*)&header.num_unique, sizeof(size_t));
    for(size_t i=0; i<header.num_hashes; i++) {
      f.write((char*)(header.hashes[i].data()), header.hash_size);
    }
    for(size_t i=0; i<header.num_unique; i++) {
      f.write((char*)&header.unique_hashes[i], sizeof(size_t));
    }
  }
}

// check if hashes are the same
//bool identical_hashes(const uint32_t* a, const uint32_t* b, size_t len) {
//  return memcmp(a, b, len*sizeof(uint32_t)) == 0;
//}

//// check if hashes are the same
//bool identical_hashes(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, size_t len) {
//  for(size_t i=0; i<a.size(); i++) {
//    if(a[i] != b[i]) {
//      return false;
//    }
//  }
//  return true;
//}

//// calculate hashes
//void calculate_hashes(const uint8_t* data, 
//                      size_t data_len,
//                      std::vector<std::vector<uint32_t>>& hashes, 
//                      size_t chunk_size,
//                      hasher* hasher) {
//  // calculate the number of hashes
//  size_t num_hashes = data_len / chunk_size;
//  if(chunk_size*num_hashes < data_len)
//    num_hashes += 1;
//  // split data into chunks and compute hashes
//  for(size_t idx=0; idx<num_hashes; idx++) {
//    int block_size = chunk_size;
//    if(chunk_size*(idx+1) > data_len)
//      block_size = (data_len)-(idx*chunk_size);
//    std::vector<uint32_t> digest(hasher->digest_size(), 0);
//    hasher->hash(data + (idx*chunk_size), block_size, (uint8_t*)(digest.data()));
//    hashes.push_back(digest);
//  }
//}
//
//// compare set of hashes with the previous set
//void compare_hashes(std::vector<std::vector<uint32_t>>& curr_hashes, 
//                    std::map<std::vector<uint32_t>, size_t>& prev_hashes,
//                    std::map<std::vector<uint32_t>, size_t>& changed_regions,
//                    size_t& num_changes, 
//                    const int hash_len) {
//  // prune current hashes to only those that are unique
//  if(changed_regions.size() == 0) {
//    for(size_t i=0; i<curr_hashes.size(); i++) {
//      bool unique = true;
//      std::vector<uint32_t> curr_digest = curr_hashes[i];
//      for(size_t j=0; j<curr_hashes.size(); j++) {
//        // check if the hashes i and j are identical
//        if(i != j && identical_hashes(curr_hashes[j], curr_hashes[i], hash_len)) {
//          unique = false;
//          // ensure that only one of the duplicate hashes are in the map of changed hashes
//          if(i < j) {
//            changed_regions.insert(std::make_pair(curr_digest, i));
//          }
//          break;
//        }
//      }
//      // insert unique hash into the map
//      if(unique) {
//        changed_regions.insert(std::make_pair(curr_digest, i));
//      }
//    }
//  }
//
//  // compare unique hashes with prior unique hashes
//  for(size_t i=0; i<curr_hashes.size(); i++) {
//    std::vector<uint32_t> curr_digest = curr_hashes[i];
//    // search for hash in map
//    auto iter = changed_regions.find(curr_digest);
//    if(iter != changed_regions.end()) {
//      // check if hash exists in prior checkpoints
//      auto unique_iter = prev_hashes.find(curr_digest);
//      if(unique_iter != prev_hashes.end()) {
//        // remove hash if it exists in prior checkpoints
//        changed_regions.erase(iter);
//      }
//    }
//  }
//  num_changes = changed_regions.size();
//}
//
//// gather changes into a contiguous buffer
//template<typename t>
//void gather_changes(t* data,
//                    size_t data_len,
//                    std::map<std::vector<uint32_t>, size_t>& changed_regions,
//                    int chunk_size,
//                    uint8_t** buffer,
//                    size_t& diff_size) {
//  // ensure that the buffer has enough space
//  *buffer = (uint8_t*) malloc(changed_regions.size()*chunk_size);
//  diff_size = 0;
//  size_t counter=0;
//  // iterate through all changed regions
//  for(auto it=changed_regions.begin(); it!=changed_regions.end(); ++it) {
//    // calculate how much to copy to buffer
//    size_t num_write = chunk_size;
//    if(chunk_size*(it->second+1) > data_len*sizeof(t)) 
//      num_write = (data_len*sizeof(t)) - it->second*chunk_size;
//    diff_size += num_write;
//    size_t pos = counter++;
//    // copy data into contiguous buffer
//    memcpy((uint8_t*)(*buffer)+chunk_size*pos, (uint8_t*)(data)+chunk_size*it->second, num_write);
//  }
//}

//// calculate hash for each chunk
//__global__
//void calculate_hashes(const uint8_t* data,
//                      const size_t data_len,
//                      uint32_t* hashes,
//                      size_t chunk_size,
//                      const size_t num_hashes) {
//  // get index for chunk
//  int idx = blockidx.x*blockdim.x + threadidx.x;
//  if(idx < num_hashes) {
//    // calculate size of chunk, last chunk may be smaller than chunk_size
//    size_t block_size = chunk_size;
//    if(chunk_size*(idx+1) > data_len)
//      block_size = data_len - idx*chunk_size;
//    // calculate hash
//    sha1_hash(data + (idx*chunk_size), block_size, (uint8_t*)(hashes)+digest_size()*idx);
//  }
//}
//
//__global__
//void print_hashes(uint32_t* hashes) {
//  for(int i=0; i<5; i++) {
//    printf("%d ", hashes[i]);
//  }
//  printf("\n");
//}
//
//// test if 2 hashes are identical
//__device__
//bool identical_hashes(const uint32_t* a, const uint32_t* b, size_t len) {
//  for(size_t i=0; i<len; i++) {
//    if(a[i] != b[i]) {
//      return false;
//    }
//  }
//  return true;
//}
//
//// find which hashes are unique
//// total # of threads should be greater than the # of hashes
//__global__
//void find_unique_hashes(const uint32_t* hashes, 
//                        const int hash_len, 
//                        const size_t num_hashes,
//                        size_t* unique_chunks,
//                        int* num_unique) {
//  // index of hash
//  size_t hash_idx = blockdim.x*blockidx.x + threadidx.x;
//  if(hash_idx < num_hashes) {
//    bool unique = true;
//    // compare with each hash and test if duplicate
//    for(size_t i=0; i<num_hashes; i++) {
//      if(hash_idx != i && identical_hashes(hashes+(hash_len/sizeof(uint32_t))*i, 
//                                           hashes+(hash_len/sizeof(uint32_t))*hash_idx, 
//					   hash_len/sizeof(uint32_t))) {
//	unique = false;
//        // save only the first instance of a non unique hash
//        if(hash_idx < i) {
//          int offset = atomicadd(num_unique, 1);
//          unique_chunks[offset] = hash_idx;
//        }
//        break;
//      }
//    }
//    // save unique hash
//    if(unique) {
//      int offset = atomicadd(num_unique, 1);
//      unique_chunks[offset] = hash_idx;
//    }
//  }
//}
//
//__global__
//void print_changes(const size_t* unique_chunks, const int* num_unique, const size_t* diff_size) {
//  printf("\tnum unique chunks: %d\n", *num_unique);
//  printf("\tcheckpoint size: %d\n", *diff_size);
//}
//
//// compare unique hashes with prior unique hashes
//// total # of threads should be greater than the # of unique hashes
//__global__
//void compare_prior_hashes(const uint32_t* hashes,
//                          const size_t num_hashes,
//                          const uint32_t* prior_hashes,
//                          const size_t num_prior_hashes,
//                          const int hash_len, 
//			  const int num_unique_hashes,
//                          size_t* changed_regions,
//                          int* num_changes) {
//  // index of unique hash
//  size_t idx = blockdim.x*blockidx.x+threadidx.x;
//  if(idx < num_unique_hashes) {
//    size_t region_idx = changed_regions[idx];
//    // compare with prior hashes
//    for(size_t i=0; i<num_prior_hashes; i++) {
//      if(identical_hashes(hashes+(hash_len/sizeof(uint32_t))*region_idx, prior_hashes+(hash_len/sizeof(uint32_t))*i, hash_len/sizeof(uint32_t))) {
//        changed_regions[idx] = num_hashes;
//	atomicsub(num_changes, 1);
//      }
//    }
//  }
//}
//
//// gather updated chunks into a contiguous buffer
//// # of thread blocks should be the # of changed chunks
//__global__
//void gather_changes(const uint8_t* data,
//                    const size_t data_len,
//                    const size_t* changed_regions,
//		    int* num_unique,
//		    const int num_changes,
//		    const size_t num_hashes,
//                    const int chunk_size,
//                    uint8_t* buffer,
//                    size_t* diff_size) {
//
//  size_t idx = blockdim.x*blockidx.x+threadidx.x;
//  if(idx < num_changes) {
//    size_t chunk_idx = changed_regions[idx];
//    if(chunk_idx < num_hashes) {
//      size_t num_write = chunk_size;
//      if((chunk_idx+1)*chunk_size >= data_len)
//        num_write = data_len - chunk_size*chunk_idx;
//      // copy data chunk by iterating in a strided fashion for better memory access pattern
//      for(int byte=0; byte<num_write; byte++) {
//        buffer[chunk_size*idx+byte] = data[chunk_idx*chunk_size+byte];
//      }
//      atomicadd((unsigned long long*)(diff_size), num_write);
//    } 
//  }
//
////  // index of changed region
////  size_t chunk_idx = changed_regions[blockidx.x];
////  // number of byte to write
////  size_t num_write = chunk_size;
////  if(chunk_size*(chunk_idx+1) > data_len)
////    num_write = data_len - chunk_size*chunk_idx;
////  // copy data chunk by iterating in a strided fashion for better memory access pattern
////  for(int byte=threadidx.x; byte<num_write; byte+=blockdim.x) {
////    buffer[chunk_size*blockidx.x+byte] = data[chunk_idx*chunk_size+byte];
////  }
////  if(threadidx.x == 0) {
////    atomicadd((unsigned long long*)(diff_size), num_write);
////    printf("wrote %llu bytes\n", num_write);
////  }
//}

//void deduplicate_module_t::cpu_dedup(uint8_t* data, 
//                                     size_t data_len,
//                                     std::map<std::vector<uint32_t>, size_t>& prev_hashes,
//                                     region_header_t& header,
//                                     uint8_t** incr_data,
//                                     size_t& incr_len,
//                                     config_t& config) {
//  int chunk_size = config.chunk_size;
//  size_t num_changes = 0;
//  std::vector<uint8_t> buffer;
//  size_t hash_len = config.hash_func->digest_size();
//  std::map<std::vector<uint32_t>, size_t> unique_hashes;
//
//  calculate_hashes(data, data_len, header.hashes, config.chunk_size, config.hash_func);
//  compare_hashes(header.hashes, prev_hashes, unique_hashes, num_changes, hash_len);
//  printf("number of changes: %zd\n", num_changes);
//  gather_changes(data, data_len, unique_hashes, config.chunk_size, incr_data, incr_len);
//  printf("checkpoint size: %zd\n", incr_len);
//
//  for(int i=0; i<5; i++) {
//    printf("%d ", header.hashes[0][i]);
//  }
//  printf("\n");
//
//  // update region header
//  header.hash_size = hash_len;
//  header.chunk_size = config.chunk_size;
//  header.num_hashes = header.hashes.size();
//  header.num_unique = unique_hashes.size();
//  for(auto it=unique_hashes.begin(); it!=unique_hashes.end(); it++) {
//    header.unique_hashes.push_back(it->second);
//  }
//  header.region_size = sizeof(size_t)*5 + hash_len*header.hashes.size() + sizeof(size_t)*header.unique_hashes.size();
//}

//void deduplicate_module_t::gpu_dedup(uint8_t* data, 
//                                     size_t data_len,
//                                     std::map<std::vector<uint32_t>, size_t>& prev_hashes,
//                                     region_header_t& header,
//                                     uint8_t** incr_data,
//                                     size_t& incr_len,
//                                     config_t& config) {
//  int chunk_size = config.chunk_size;
//  int hash_len = digest_size();
//  int num_changes = 0;
//  size_t num_unique = 0;
//  incr_len = 0;
//  int *num_changes_d;
//  size_t *num_unique_d;
//  size_t *incr_len_d;
//  uint32_t *hashes, *hashes_d;
//  uint8_t* incr_data_d;
//  size_t *changed_regions, *changed_regions_d;
//  size_t num_hashes = data_len/chunk_size;
//  if(num_hashes*chunk_size < data_len)
//    num_hashes += 1;
//
//  hashes = (uint32_t*) malloc(num_hashes*hash_len);
//  cudamalloc(&hashes_d, num_hashes*hash_len);
//  cudamalloc(&changed_regions_d, num_hashes*sizeof(size_t));
//  cudamalloc(&num_changes_d, sizeof(int));
//  cudamalloc(&num_unique_d, sizeof(size_t));
//  cudamalloc(&incr_len_d, sizeof(size_t));
//  cudamemcpy(num_changes_d, &num_changes, sizeof(int), cudamemcpyhosttodevice);
//  cudamemcpy(num_unique_d, &num_unique, sizeof(size_t), cudamemcpyhosttodevice);
//  cudamemcpy(incr_len_d, &incr_len, sizeof(size_t), cudamemcpyhosttodevice);
//
//  uint32_t* prior_hashes = (uint32_t*) malloc(hash_len*prev_hashes.size());
//  size_t num_prior_hashes = prev_hashes.size();
//  size_t offset = 0;
//  for(auto it=prev_hashes.begin(); it!=prev_hashes.end(); it++) {
//    for(size_t i=0; i<hash_len/sizeof(uint32_t); i++) {
//      prior_hashes[(hash_len/sizeof(uint32_t))*offset+i] = it->first[i];
//    }
//    offset += 1;
//  }
//  uint32_t* prior_hashes_d;
//  cudamalloc(&prior_hashes_d, num_prior_hashes*hash_len);
//  cudamemcpy(prior_hashes_d, prior_hashes, num_prior_hashes*hash_len, cudamemcpyhosttodevice);
//
//  int num_blocks = num_hashes/32;
//  if(num_blocks*32 < num_hashes)
//    num_blocks += 1;
//  // calculate hashes
//  calculate_hashes<<<num_blocks,32>>>(data, data_len, hashes_d, chunk_size, num_hashes);
//  cudamemcpy(hashes, hashes_d, num_hashes*hash_len, cudamemcpydevicetohost);
//  // find the unique hashes
//  find_unique_hashes<<<num_blocks,32>>>(hashes_d, hash_len, num_hashes, changed_regions_d, num_changes_d);
//  cudamemcpy(&num_changes, num_changes_d, sizeof(int), cudamemcpydevicetohost);
//  // compare hashes with prior hashes
//  compare_prior_hashes<<<num_blocks,32>>>(hashes_d, num_hashes, prior_hashes_d, num_prior_hashes, hash_len, num_changes, changed_regions_d, num_changes_d);
//  // gather updated chunks into a contiguous buffer
//  cudamalloc(&incr_data_d, num_changes*chunk_size);
//  gather_changes<<<num_blocks,32>>>(data, data_len, changed_regions_d, num_changes_d, num_changes, num_hashes, chunk_size, incr_data_d, incr_len_d);
//  cudamemcpy(&num_changes, num_changes_d, sizeof(int), cudamemcpydevicetohost);
//
//  // copy buffer to host for checkpointing
//  *incr_data = (uint8_t*) malloc(num_changes*chunk_size);
//  changed_regions = (size_t*) malloc(sizeof(size_t)*num_changes);
//  cudamemcpy(*incr_data, incr_data_d, num_changes*chunk_size, cudamemcpydevicetohost);
//  cudamemcpy(changed_regions, changed_regions_d, num_changes*sizeof(size_t), cudamemcpydevicetohost);
//  cudamemcpy(&incr_len, incr_len_d, sizeof(size_t), cudamemcpydevicetohost);
//  cudadevicesynchronize();
//
//  // update region header
//  header.hash_size = hash_len;
//  header.chunk_size = config.chunk_size;
//  header.num_hashes = num_hashes;
//  header.num_unique = num_changes;
//  for(size_t i=0; i<num_hashes; i++) {
//    std::vector<uint32_t> hash_digest;
//    for(size_t j=0; j<hash_len/sizeof(uint32_t); j++) {
//      hash_digest.push_back(hashes[i*(hash_len/sizeof(uint32_t)) + j]);
//    }
//    header.hashes.push_back(hash_digest);
//  }
//  for(size_t i=0; i<num_changes; i++) {
//    header.unique_hashes.push_back(changed_regions[i]);
//  }
//  header.region_size = sizeof(size_t)*5 + hash_len*header.hashes.size() + sizeof(size_t)*header.unique_hashes.size();
//
//  cudafree(num_changes_d);
//  cudafree(num_unique_d);
//  cudafree(incr_len_d);
//  cudafree(hashes_d);
//  cudafree(prior_hashes_d);
//  cudafree(changed_regions_d);
//  cudafree(incr_data_d);
//  free(hashes);
//  free(prior_hashes);
//  free(changed_regions);
//}


void deduplicate_module_t::deduplicate_file(const std::string &full, const std::string &incr, std::vector<std::string> &prev_chkpts, config_t& config) {
  // read checkpoint header
  header_t chkpt_header;
  std::map<int, size_t> region_map;
  read_full_header(full, chkpt_header, region_map);

  regions_t regions;
  std::ifstream f;
  f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  f.open(full, std::ifstream::in | std::ifstream::binary);
  f.seekg(chkpt_header.header_size);
  for(auto &e : region_map) {
    region_t region;
    region.size = e.second;
    region.ptr = (void*) malloc(e.second);
    if(config.dedup_on_gpu) {
      region.ptr_type = Cuda;
    } else {
      region.ptr_type = Host;
    }
    f.read((char*)region.ptr, e.second);
    regions.insert(std::make_pair(e.first, region));
  }
  deduplicate_data(regions, incr, prev_chkpts, config);
}

void deduplicate_module_t::deduplicate_data(regions_t& full_regions, const std::string& incr, std::vector<std::string>& prev_chkpts, config_t& config) {
  using namespace std::chrono;
  // read checkpoint header
  header_t chkpt_header;

  printf("region map:\n");
  for(auto &e: full_regions) {
    printf("(%d,%zd)\n", e.first, e.second.size);
  }

  using HashDigest = DefaultHash::Digest;

  // read headers from prior checkpoints
  std::vector<std::pair<header_t, std::map<int, region_header_t>>> prev_chkpt_headers;
  for(size_t i=0; i<prev_chkpts.size(); i++) {
    header_t main_header;
    std::map<int, region_header_t> region_headers;
    printf("reading incremental header for chkpt %d\n", i);
    read_incremental_headers(prev_chkpts[i], main_header, region_headers, config);
    prev_chkpt_headers.push_back(std::make_pair(main_header, region_headers));
  }
  printf("Read previous headers\n");

  // Get unique hashes
  std::map<int, std::map<std::vector<uint32_t>,size_t>> unique_hashes;
  std::map<int, std::map<int, uint8_t*>> prev_trees;
  if(config.use_merkle_trees) {
printf("Reading merkle tree\n");
    read_merkle_trees(prev_chkpt_headers, prev_trees);
printf("Read merkle tree\n");
  } else {
    collect_unique_hashes(prev_chkpt_headers, unique_hashes);
  }

  int chkpt_id = prev_chkpts.size();
  printf("Found %d prior trees\n", prev_trees.size());

  // Read checkpoint data and deduplicate
  std::map<int, std::pair<uint8_t*,size_t>> region_data;
  std::map<int, region_header_t> region_headers;
  high_resolution_clock::time_point dedup_start = high_resolution_clock::now();
  for(auto &e : full_regions) {
    uint8_t* incr_data;
    size_t incr_size;
    region_header_t region_header;
    region_header.id = e.first;
    if(e.second.ptr_type == Host) {
      printf("\nRegion %d\n", e.first);
      if(config.use_merkle_trees) {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        cpu_dedup((uint8_t*)e.second.ptr, e.second.size, chkpt_id, prev_trees[e.first], region_header, &incr_data, incr_size, config);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        std::cout << "Time spent on CPU deduplication: " << duration_cast<duration<double>>(t2-t1).count() << std::endl;
      } else {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        cpu_dedup((uint8_t*)e.second.ptr, e.second.size, unique_hashes[e.first], region_header, &incr_data, incr_size, config);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        std::cout << "Time spent on CPU deduplication: " << duration_cast<duration<double>>(t2-t1).count() << std::endl;
      }
    } else if(config.dedup_on_gpu) {
      printf("Deduplicating region %d with %zd bytes, data already on GPU\n", e.first, e.second.size);
      high_resolution_clock::time_point t1 = high_resolution_clock::now();
      gpu_dedup((uint8_t*)e.second.ptr, e.second.size, unique_hashes[e.first], region_header, &incr_data, incr_size, config);
      high_resolution_clock::time_point t2 = high_resolution_clock::now();
      std::cout << "Time spent on GPU deduplication: " << duration_cast<duration<double>>(t2-t1).count() << std::endl;
    } else {
      uint8_t* data_h = (uint8_t*) malloc(e.second.size);
      high_resolution_clock::time_point t1 = high_resolution_clock::now();
      cudaMemcpy(data_h, e.second.ptr, e.second.size, cudaMemcpyDeviceToHost);
      high_resolution_clock::time_point t2 = high_resolution_clock::now();
      printf("Deduplicating region %d with %zd bytes, data copied to CPU\n", e.first, e.second.size);
      std::cout << "\tTime spent on copying data to Host: " << duration_cast<duration<double>>(t2-t1).count() << std::endl;
      t1 = high_resolution_clock::now();
      gpu_dedup((uint8_t*)e.second.ptr, e.second.size, unique_hashes[e.first], region_header, &incr_data, incr_size, config);
      t2 = high_resolution_clock::now();
      std::cout << "Time spent on GPU deduplication: " << duration_cast<duration<double>>(t2-t1).count() << std::endl;
      printf("\n");
      free(data_h);
    }
    printf("Inserting checkpoint data\n");
    region_data.insert(std::make_pair(e.first, std::make_pair(incr_data, incr_size)));
    printf("Inserted checkpoint data\n");
    region_headers.insert(std::make_pair(e.first, region_header));
    printf("Inserted header\n");
  }
  high_resolution_clock::time_point dedup_end = high_resolution_clock::now();
  std::cout << "Total time spent on deduplication: " << duration_cast<duration<double>>(dedup_end-dedup_start).count() << std::endl;

  // Create main header
  size_t chkpt_size = 0;
  for(auto &region: region_data) {
    chkpt_size += region.second.second;
  }
  chkpt_header.chkpt_size = chkpt_size;
  size_t header_size = 3*sizeof(size_t);
  for(auto &e: region_headers) {
    header_size += sizeof(size_t)*5;
    header_size += config.hash_func->digest_size()*e.second.hashes.size();
    header_size += sizeof(size_t)*e.second.unique_hashes.size();
  }
  chkpt_header.header_size = header_size;
  chkpt_header.num_regions = region_data.size();

  printf("Create main header\n");
  printf("Checkpoint size: %zd\n", chkpt_header.chkpt_size);
  printf("Header size: %zd\n", chkpt_header.header_size);
  printf("Num regions: %zd\n", chkpt_header.num_regions);

  // Write main header, incremental headers, and data
  std::ofstream os;
  os.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  os.open(incr, std::ofstream::out | std::ofstream::binary);
  os.write((char*)&(chkpt_header), sizeof(size_t)*3);
  printf("Wrote main header\n");

  for(auto &e: region_headers) {
  int id;
  size_t region_size;
  size_t hash_size;
  size_t chunk_size;
  size_t num_hashes;
  size_t num_unique;
//    printf("Subheader %d\n", e.second.id);
//    printf("Hash size: %zd\n", e.second.hash_size);
//    printf("Chunk size: %zd\n", e.second.chunk_size);
//    printf("Num hashes: %zd\n", e.second.num_hashes);
//    printf("Num unique subtrees: %zd\n", e.second.num_unique_subtrees);
//    printf("Num shared subtrees: %zd\n", e.second.num_shared_subtrees);
//    printf("Num unique: %zd\n", e.second.num_unique);
    write_header(os, e.second, config);
//    printf("Wrote subheader\n");
  }

  for(auto &e: region_data) {
    os.write((char*)e.second.first, e.second.second);
    free(e.second.first);
  }
  printf("Wrote checkpoint\n");
}
