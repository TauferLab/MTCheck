#include "dedup.hpp"
//#include "hash_functions.hpp"
#include "gpu_sha1.hpp"
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <chrono>

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
    sha1_hash(data + (idx*chunk_size), block_size, (uint8_t*)(hashes)+digest_size()*idx);
  }
}

__global__
void print_hashes(uint32_t* hashes) {
  for(int i=0; i<5; i++) {
    printf("%d ", hashes[i]);
  }
  printf("\n");
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
void find_unique_hashes(const uint32_t* hashes, 
                        const int hash_len, 
                        const size_t num_hashes,
                        size_t* unique_chunks,
                        int* num_unique) {
  // Index of hash
  size_t hash_idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(hash_idx < num_hashes) {
    bool unique = true;
    // Compare with each hash and test if duplicate
    for(size_t i=0; i<num_hashes; i++) {
      if(hash_idx != i && identical_hashes(hashes+(hash_len/sizeof(uint32_t))*i, 
                                           hashes+(hash_len/sizeof(uint32_t))*hash_idx, 
					   hash_len/sizeof(uint32_t))) {
	unique = false;
        // Save only the first instance of a non unique hash
        if(hash_idx < i) {
          int offset = atomicAdd(num_unique, 1);
          unique_chunks[offset] = hash_idx;
        }
        break;
      }
    }
    // Save unique hash
    if(unique) {
      int offset = atomicAdd(num_unique, 1);
      unique_chunks[offset] = hash_idx;
    }
  }
}

__global__
void print_changes(const size_t* unique_chunks, const int* num_unique, const size_t* diff_size) {
  printf("\tNum unique chunks: %d\n", *num_unique);
  printf("\tCheckpoint size: %d\n", *diff_size);
}

// Compare unique hashes with prior unique hashes
// Total # of threads should be greater than the # of unique hashes
__global__
void compare_prior_hashes(const uint32_t* hashes,
                          const size_t num_hashes,
                          const uint32_t* prior_hashes,
                          const size_t num_prior_hashes,
                          const int hash_len, 
			  const int num_unique_hashes,
                          size_t* changed_regions,
                          int* num_changes) {
  // Index of unique hash
  size_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  if(idx < num_unique_hashes) {
    size_t region_idx = changed_regions[idx];
    // Compare with prior hashes
    for(size_t i=0; i<num_prior_hashes; i++) {
      if(identical_hashes(hashes+(hash_len/sizeof(uint32_t))*region_idx, prior_hashes+(hash_len/sizeof(uint32_t))*i, hash_len/sizeof(uint32_t))) {
        changed_regions[idx] = num_hashes;
	atomicSub(num_changes, 1);
      }
    }
  }
}

// Gather updated chunks into a contiguous buffer
// # of thread blocks should be the # of changed chunks
__global__
void gather_changes(const uint8_t* data,
                    const size_t data_len,
                    const size_t* changed_regions,
		    int* num_unique,
		    const int num_changes,
		    const size_t num_hashes,
                    const int chunk_size,
                    uint8_t* buffer,
                    size_t* diff_size) {

  size_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  if(idx < num_changes) {
    size_t chunk_idx = changed_regions[idx];
    if(chunk_idx < num_hashes) {
      size_t num_write = chunk_size;
      if((chunk_idx+1)*chunk_size >= data_len)
        num_write = data_len - chunk_size*chunk_idx;
      // Copy data chunk by Iterating in a strided fashion for better memory access pattern
      for(int byte=0; byte<num_write; byte++) {
        buffer[chunk_size*idx+byte] = data[chunk_idx*chunk_size+byte];
      }
      atomicAdd((unsigned long long*)(diff_size), num_write);
    } 
  }

//  // Index of changed region
//  size_t chunk_idx = changed_regions[blockIdx.x];
//  // Number of byte to write
//  size_t num_write = chunk_size;
//  if(chunk_size*(chunk_idx+1) > data_len)
//    num_write = data_len - chunk_size*chunk_idx;
//  // Copy data chunk by Iterating in a strided fashion for better memory access pattern
//  for(int byte=threadIdx.x; byte<num_write; byte+=blockDim.x) {
//    buffer[chunk_size*blockIdx.x+byte] = data[chunk_idx*chunk_size+byte];
//  }
//  if(threadIdx.x == 0) {
//    atomicAdd((unsigned long long*)(diff_size), num_write);
//    printf("Wrote %llu bytes\n", num_write);
//  }
}

void deduplicate_module_t::gpu_dedup(uint8_t* data, 
                                     size_t data_len,
                                     std::map<std::vector<uint32_t>, size_t>& prev_hashes,
                                     region_header_t& header,
                                     uint8_t** incr_data,
                                     size_t& incr_len,
                                     config_t& config) {
  using namespace std::chrono;
  int chunk_size = config.chunk_size;
  int hash_len = digest_size();
  int num_changes = 0;
  size_t num_unique = 0;
  incr_len = 0;
  int *num_changes_d;
  size_t *num_unique_d;
  size_t *incr_len_d;
  uint32_t *hashes, *hashes_d;
  uint8_t* incr_data_d;
  size_t *changed_regions, *changed_regions_d;
  size_t num_hashes = data_len/chunk_size;
  if(num_hashes*chunk_size < data_len)
    num_hashes += 1;

  hashes = (uint32_t*) malloc(num_hashes*hash_len);
  cudaMalloc(&hashes_d, num_hashes*hash_len);
  cudaMalloc(&changed_regions_d, num_hashes*sizeof(size_t));
  cudaMalloc(&num_changes_d, sizeof(int));
  cudaMalloc(&num_unique_d, sizeof(size_t));
  cudaMalloc(&incr_len_d, sizeof(size_t));
  cudaMemcpy(num_changes_d, &num_changes, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(num_unique_d, &num_unique, sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(incr_len_d, &incr_len, sizeof(size_t), cudaMemcpyHostToDevice);

  uint32_t* prior_hashes = (uint32_t*) malloc(hash_len*prev_hashes.size());
  size_t num_prior_hashes = prev_hashes.size();
  size_t offset = 0;
  for(auto it=prev_hashes.begin(); it!=prev_hashes.end(); it++) {
    for(size_t i=0; i<hash_len/sizeof(uint32_t); i++) {
      prior_hashes[(hash_len/sizeof(uint32_t))*offset+i] = it->first[i];
    }
    offset += 1;
  }
  uint32_t* prior_hashes_d;
  cudaMalloc(&prior_hashes_d, num_prior_hashes*hash_len);
  cudaMemcpy(prior_hashes_d, prior_hashes, num_prior_hashes*hash_len, cudaMemcpyHostToDevice);

  int num_blocks = num_hashes/32;
  if(num_blocks*32 < num_hashes)
    num_blocks += 1;
  // Calculate hashes
  high_resolution_clock::time_point calc_start = high_resolution_clock::now();
  calculate_hashes<<<num_blocks,32>>>(data, data_len, hashes_d, chunk_size, num_hashes);
  cudaDeviceSynchronize();
  high_resolution_clock::time_point calc_end = high_resolution_clock::now();
  cudaMemcpy(hashes, hashes_d, num_hashes*hash_len, cudaMemcpyDeviceToHost);
  // Find the unique hashes
  high_resolution_clock::time_point find_start = high_resolution_clock::now();
  find_unique_hashes<<<num_blocks,32>>>(hashes_d, hash_len, num_hashes, changed_regions_d, num_changes_d);
  cudaDeviceSynchronize();
  high_resolution_clock::time_point find_end = high_resolution_clock::now();
  cudaMemcpy(&num_changes, num_changes_d, sizeof(int), cudaMemcpyDeviceToHost);
  // Compare hashes with prior hashes
  high_resolution_clock::time_point comp_start = high_resolution_clock::now();
  compare_prior_hashes<<<num_blocks,32>>>(hashes_d, num_hashes, prior_hashes_d, num_prior_hashes, hash_len, num_changes, changed_regions_d, num_changes_d);
  cudaDeviceSynchronize();
  high_resolution_clock::time_point comp_end = high_resolution_clock::now();
  // Gather updated chunks into a contiguous buffer
  cudaMalloc(&incr_data_d, num_changes*chunk_size);
  high_resolution_clock::time_point gather_start = high_resolution_clock::now();
  gather_changes<<<num_blocks,32>>>(data, data_len, changed_regions_d, num_changes_d, num_changes, num_hashes, chunk_size, incr_data_d, incr_len_d);
  cudaDeviceSynchronize();
  high_resolution_clock::time_point gather_end = high_resolution_clock::now();
  cudaMemcpy(&num_changes, num_changes_d, sizeof(int), cudaMemcpyDeviceToHost);

  // Copy buffer to host for checkpointing
  *incr_data = (uint8_t*) malloc(num_changes*chunk_size);
  changed_regions = (size_t*) malloc(sizeof(size_t)*num_changes);
  high_resolution_clock::time_point copy_diff_start = high_resolution_clock::now();
  cudaMemcpy(*incr_data, incr_data_d, num_changes*chunk_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(changed_regions, changed_regions_d, num_changes*sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&incr_len, incr_len_d, sizeof(size_t), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  high_resolution_clock::time_point copy_diff_end = high_resolution_clock::now();

  std::cout << "\tTime spent calculating hashes: " << std::chrono::duration_cast<std::chrono::duration<double>>(calc_end-calc_start).count() << std::endl;
  std::cout << "\tTime spent finding  hashes: " << std::chrono::duration_cast<std::chrono::duration<double>>(find_end-find_start).count() << std::endl;
  std::cout << "\tTime spent comparing hashes: " << std::chrono::duration_cast<std::chrono::duration<double>>(comp_end-comp_start).count() << std::endl;
  std::cout << "\tTime spent gathering changes: " << std::chrono::duration_cast<std::chrono::duration<double>>(gather_end-gather_start).count() << std::endl;
  std::cout << "\tTime spent copying diff: " << std::chrono::duration_cast<std::chrono::duration<double>>(copy_diff_end-copy_diff_start).count() << std::endl;

  // Update region header
  header.hash_size = hash_len;
  header.chunk_size = config.chunk_size;
  header.num_hashes = num_hashes;
  header.num_unique = num_changes;
  for(size_t i=0; i<num_hashes; i++) {
    std::vector<uint32_t> hash_digest;
    for(size_t j=0; j<hash_len/sizeof(uint32_t); j++) {
      hash_digest.push_back(hashes[i*(hash_len/sizeof(uint32_t)) + j]);
    }
    header.hashes.push_back(hash_digest);
  }
  for(size_t i=0; i<num_changes; i++) {
    header.unique_hashes.push_back(changed_regions[i]);
  }
  header.region_size = sizeof(size_t)*5 + hash_len*header.hashes.size() + sizeof(size_t)*header.unique_hashes.size();

  cudaFree(num_changes_d);
  cudaFree(num_unique_d);
  cudaFree(incr_len_d);
  cudaFree(hashes_d);
  cudaFree(prior_hashes_d);
  cudaFree(changed_regions_d);
  cudaFree(incr_data_d);
  free(hashes);
  free(prior_hashes);
  free(changed_regions);
}

