#include "dedup.hpp"
#include "hash_functions.hpp"
#include "gpu_sha1.hpp"
#include <vector>
#include <map>
#include <fstream>
#include <iostream>

// Collect unique hashes from prior incremental checkpoints
void collect_unique_hashes(std::vector<std::pair<header_t, std::map<int, region_header_t>>> &prev_chkpt_headers, std::map<int, std::map<std::vector<uint32_t>, size_t>>& unique_hashes) {
  // Go through prior checkpoint headers
  for(size_t i=0; i<prev_chkpt_headers.size(); i++) {
    std::map<int, region_header_t>& region_headers = prev_chkpt_headers[i].second;
    // Iterate through region headers
    for(auto it=region_headers.begin(); it!=region_headers.end(); it++) {
      // Get iterator to unique hashes
      std::map<std::vector<uint32_t>,size_t> hashes;
      auto pos = unique_hashes.find(it->first);
      auto ret = unique_hashes.insert(pos, std::make_pair(it->first, hashes));
      auto hash_it = ret->second;
      // Add unique hashes to map
      for(size_t k=0; k<it->second.num_unique; k++) {
        int id = it->first;
        size_t idx = it->second.unique_hashes[k];
        ret->second.insert(std::make_pair(it->second.hashes[idx], idx));
      }
    }
  }
}

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

// Read incremental checkpoint header
bool read_incremental_headers(std::string& chkpt, header_t& header, std::map<int, region_header_t>& region_headers) {
  try {
    // Read main header
    std::ifstream f;
    f.open(chkpt, std::ifstream::in | std::ifstream::binary);
    f.read((char*)(&header.chkpt_size), sizeof(size_t));
    f.read((char*)(&header.header_size), sizeof(size_t));
    f.read((char*)(&header.num_regions), sizeof(size_t));
    for(size_t i=0; i<header.num_regions; i++) {
      int id;
      region_header_t region;
      // Read region headers
      f.read((char*)&id, sizeof(int));
      f.read((char*)&(region.region_size), sizeof(size_t));
      f.read((char*)&(region.hash_size), sizeof(size_t));
      f.read((char*)&(region.chunk_size), sizeof(size_t));
      f.read((char*)&(region.num_hashes), sizeof(size_t));
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
      region_headers.insert(std::make_pair(id, region));
    }
    return true;
  } catch(std::ifstream::failure &e) {
    std::cout << "Error reading header for checkpoint " << chkpt << std::endl;
    return false;
  }
}

// Write region header to file
void write_header(std::ofstream& f, region_header_t& header) {
  f.write((char*)&header.id, sizeof(int));
  f.write((char*)&header.region_size, sizeof(size_t));
  f.write((char*)&header.hash_size, sizeof(size_t));
  f.write((char*)&header.chunk_size, sizeof(size_t));
  f.write((char*)&header.num_hashes, sizeof(size_t));
  f.write((char*)&header.num_unique, sizeof(size_t));
  for(size_t i=0; i<header.num_hashes; i++) {
    f.write((char*)(header.hashes[i].data()), header.hash_size);
  }
  for(size_t i=0; i<header.num_unique; i++) {
    f.write((char*)&header.unique_hashes[i], sizeof(size_t));
  }
}

// Check if hashes are the same
//bool identical_hashes(const uint32_t* a, const uint32_t* b, size_t len) {
//  return memcmp(a, b, len*sizeof(uint32_t)) == 0;
//}

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

// Compare set of hashes with the previous set
void compare_hashes(std::vector<std::vector<uint32_t>>& curr_hashes, 
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

// Calculate hash for each chunk
__global__
void calculate_hashes(const uint8_t* data,
                      const size_t data_len,
                      uint32_t* hashes,
                      size_t chunk_size,
                      const size_t num_hashes) {
  // Get index for chunk
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < num_hashes) {
    // Calculate size of chunk, last chunk may be smaller than chunk_size
    size_t block_size = chunk_size;
    if(chunk_size*(idx+1) > data_len)
      block_size = data_len - idx*chunk_size;
    // Calculate hash
    sha1_hash(data + (idx*chunk_size), block_size, (uint8_t*)(hashes+digest_size()*idx));
  }
}

// Test if 2 hashes are identical
__device__
bool identical_hashes(const uint32_t* a, const uint32_t* b, size_t len) {
  for(size_t i=0; i<len; i++) {
    if(a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

// Find which hashes are unique
// Total # of threads should be greater than the # of hashes
__global__
void find_unique_hashes(uint32_t* hashes, 
                        const int hash_len, 
                        const size_t num_hashes,
                        size_t* unique_chunks,
                        int& num_unique) {
  // Index of hash
  size_t hash_idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(hash_idx < num_hashes) {
    bool unique = true;
    // Compare with each hash and test if duplicate
    for(size_t i=0; i<num_hashes; i++) {
      if(hash_idx != i && identical_hashes(hashes+hash_len*i, hashes+hash_len*hash_idx, hash_len)) {
        // Save only the first instance of a non unique hash
        if(hash_idx < i) {
          int offset = atomicAdd(&num_unique, static_cast<int>(1));
          unique_chunks[offset] = hash_idx;
          break;
        }
      }
    }
    // Save unique hash
    if(unique) {
      int offset = atomicAdd(&num_unique, static_cast<int>(1));
      unique_chunks[offset] = hash_idx;
    }
  }
}

// Compare unique hashes with prior unique hashes
// Total # of threads should be greater than the # of unique hashes
__global__
void compare_prior_hashes(uint32_t* hashes,
                          const size_t num_hashes,
                          uint32_t* prior_hashes,
                          const size_t num_prior_hashes,
                          const int hash_len, 
                          size_t* changed_regions,
                          int& num_changes) {
  // Index of unique hash
  size_t hash_idx = blockDim.x*blockIdx.x+threadIdx.x;
  if(hash_idx < num_changes) {
    // Compare with prior hashes
    for(size_t i=0; i<num_prior_hashes; i++) {
      // If hash is not unique get the index of the last 
      // hash (atomics) and swap duplicate with last chunk
      if(identical_hashes(hashes+hash_len*hash_idx, prior_hashes+hash_len*i, hash_len)) {
        int swap_idx = atomicSub(&num_changes, static_cast<int>(1));
        changed_regions[hash_idx] = changed_regions[swap_idx];
      }
    }
  }
}

// Gather updated chunks into a contiguous buffer
// # of thread blocks should be the # of changed chunks
__global__
void gather_changes(uint8_t* data,
                    size_t data_len,
                    size_t* changed_regions,
                    size_t num_regions,
                    int chunk_size,
                    uint8_t* buffer,
                    size_t& diff_size) {
  // Index of changed region
  size_t chunk_idx = changed_regions[blockIdx.x];
  // Number of byte to write
  size_t num_write = chunk_size;
  if(chunk_size*chunk_idx > data_len)
    num_write = data_len - chunk_size*chunk_idx;
  // Copy data chunk by Iterating in a strided fashion for better memory access pattern
  for(int byte=threadIdx.x; byte<num_write; byte+=blockDim.x) {
    buffer[chunk_size*blockIdx.x+byte] = data[chunk_idx*chunk_size+byte];
  }
}

void deduplicate_module_t::cpu_dedup(uint8_t* data, 
                                     size_t data_len,
                                     std::map<std::vector<uint32_t>, size_t>& prev_hashes,
                                     region_header_t& header,
                                     uint8_t** incr_data,
                                     size_t& incr_len,
                                     config_t& config) {
  int chunk_size = config.chunk_size;
  size_t num_changes = 0;
  std::vector<uint8_t> buffer;
  size_t hash_len = config.hash_func->digest_size();
  std::map<std::vector<uint32_t>, size_t> unique_hashes;

  calculate_hashes(data, data_len, header.hashes, config.chunk_size, config.hash_func);
  compare_hashes(header.hashes, prev_hashes, unique_hashes, num_changes, hash_len);
  gather_changes(data, data_len, unique_hashes, config.chunk_size, incr_data, incr_len);

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

void deduplicate_module_t::gpu_dedup(uint8_t* data, 
                                     size_t data_len,
                                     std::map<std::vector<uint32_t>, size_t>& prev_hashes,
                                     region_header_t& header,
                                     uint8_t** incr_data,
                                     size_t& incr_len,
                                     config_t& config) {
  int chunk_size = config.chunk_size;
  int hash_len = digest_size();
  int num_changes = 0;
  size_t num_unique = 0;
  incr_len = 0;
  int num_changes_d;
  size_t num_unique_d;
  size_t incr_len_d;
  cudaMemcpy(&num_changes_d, &num_changes, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&num_unique_d, &num_unique, sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(&incr_len_d, &incr_len, sizeof(size_t), cudaMemcpyHostToDevice);

  size_t num_hashes = data_len/chunk_size;
  if(num_hashes*chunk_size < data_len)
    num_hashes += 1;

  uint32_t *hashes, *hashes_d;
  size_t *changed_regions, *changed_regions_d;
  uint8_t* incr_data_d;
  hashes = (uint32_t*) malloc(num_hashes*hash_len);
  cudaMalloc(&hashes_d, num_hashes*sizeof(uint32_t));
  cudaMalloc(&changed_regions_d, num_hashes*sizeof(size_t));

  uint32_t* prior_hashes = (uint32_t*) malloc(hash_len*prev_hashes.size());
  size_t num_prior_hashes = prev_hashes.size();
  size_t offset = 0;
  for(auto it=prev_hashes.begin(); it!=prev_hashes.end(); it++) {
    for(size_t i=0; i<hash_len/sizeof(uint32_t); i++) {
      prior_hashes[offset] = it->first[i];
    }
  }

  int num_blocks = num_hashes/32;
  if(num_blocks*32 < num_hashes)
    num_blocks += 1;
  calculate_hashes<<<num_blocks,32>>>(data, data_len, hashes_d, chunk_size, num_hashes);
  cudaDeviceSynchronize();
  find_unique_hashes<<<num_blocks,32>>>(hashes_d, hash_len, num_hashes, changed_regions_d, num_changes_d);
  cudaDeviceSynchronize();
  compare_prior_hashes<<<num_blocks,32>>>(hashes, num_hashes, prior_hashes, num_prior_hashes, hash_len, changed_regions_d, num_changes_d);
  cudaDeviceSynchronize();
  cudaMemcpy(&num_changes, &num_changes_d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMalloc(&incr_data_d, num_changes*chunk_size);
  gather_changes<<<num_changes,32>>>(data, data_len, changed_regions_d, num_changes_d, chunk_size, incr_data_d, incr_len_d);
  cudaDeviceSynchronize();

  *incr_data = (uint8_t*) malloc(num_changes*chunk_size);
  changed_regions = (size_t*) malloc(sizeof(int)*num_changes);
  cudaMemcpy(*incr_data, incr_data_d, num_changes*chunk_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(changed_regions, changed_regions_d, num_changes*sizeof(size_t), cudaMemcpyDeviceToHost);

  // Update region header
  header.hash_size = hash_len;
  header.chunk_size = config.chunk_size;
  header.num_hashes = header.hashes.size();
  header.num_unique = num_changes;
  for(size_t i=0; i<num_hashes; i++) {
    std::vector<uint32_t> hash_digest;
    for(size_t j=0; j<hash_len/sizeof(uint32_t); j++) {
      hash_digest.push_back(hashes[changed_regions[i]*hash_len/sizeof(uint32_t)]);
    }
    header.hashes.push_back(hash_digest);
  }
  for(size_t i=0; i<num_changes; i++) {
    header.unique_hashes.push_back(changed_regions[i]);
  }
  header.region_size = sizeof(size_t)*5 + hash_len*header.hashes.size() + sizeof(size_t)*header.unique_hashes.size();
  cudaFree(hashes_d);
  cudaFree(changed_regions_d);
  cudaFree(incr_data_d);
  free(hashes);
  free(changed_regions);
  free(prior_hashes);
}


void deduplicate_module_t::deduplicate_file(const std::string &full, const std::string &incr, std::vector<std::string> &prev_chkpts, config_t& config) {
  // Read checkpoint header
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
    region.ptr_type = Host;
    f.read((char*)region.ptr, e.second);
    regions.insert(std::make_pair(e.first, region));
  }
  deduplicate_data(regions, incr, prev_chkpts, config);
}

void deduplicate_module_t::deduplicate_data(regions_t& full_regions, const std::string& incr, std::vector<std::string>& prev_chkpts, config_t& config) {
  // Read checkpoint header
  header_t chkpt_header;

  printf("Region map:\n");
  for(auto &e: full_regions) {
    printf("(%d,%zd)\n", e.first, e.second.size);
  }

  using HashDigest = DefaultHash::Digest;

  // Read headers from prior checkpoints
  std::vector<std::pair<header_t, std::map<int, region_header_t>>> prev_chkpt_headers;
  for(size_t i=0; i<prev_chkpts.size(); i++) {
    header_t main_header;
    std::map<int, region_header_t> region_headers;
    read_incremental_headers(prev_chkpts[i], main_header, region_headers);
    prev_chkpt_headers.push_back(std::make_pair(main_header, region_headers));
  }

  // Get unique hashes
  std::map<int, std::map<std::vector<uint32_t>,size_t>> unique_hashes;
  collect_unique_hashes(prev_chkpt_headers, unique_hashes);

  // Read checkpoint data and deduplicate
  std::map<int, std::pair<uint8_t*,size_t>> region_data;
  std::map<int, region_header_t> region_headers;
  for(auto &e : full_regions) {
    if(e.second.ptr_type == Host) {
      uint8_t* incr_data;
      size_t incr_size;
      std::map<std::vector<uint32_t>, size_t> updates;
      region_header_t region_header;
      region_header.id = e.first;
      cpu_dedup((uint8_t*)e.second.ptr, e.second.size, unique_hashes[e.first], region_header, &incr_data, incr_size, config);
      region_data.insert(std::make_pair(e.first, std::make_pair(incr_data, incr_size)));
      region_headers.insert(std::make_pair(e.first, region_header));
    } else {
      uint8_t* incr_data;
      size_t incr_size;
      std::map<std::vector<uint32_t>, size_t> updates;
      region_header_t region_header;
      region_header.id = e.first;
      gpu_dedup((uint8_t*)e.second.ptr, e.second.size, unique_hashes[e.first], region_header, &incr_data, incr_size, config);
      region_data.insert(std::make_pair(e.first, std::make_pair(incr_data, incr_size)));
      region_headers.insert(std::make_pair(e.first, region_header));
    }
  }

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
//    printf("Num unique: %zd\n", e.second.num_unique);
    write_header(os, e.second);
  }

  for(auto &e: region_data) {
    os.write((char*)e.second.first, e.second.second);
    free(e.second.first);
  }
  printf("Wrote checkpoint\n");
}
