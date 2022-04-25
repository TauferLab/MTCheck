#include "gpu_hash_lists.cuh"
#include "gpu_sha1.hpp"

// Calculate hash for each chunk
__global__
void calculate_hashes_kernel(const uint8_t* data,
                      const size_t data_len,
                      uint32_t* hashes,
                      size_t chunk_size,
                      const size_t num_hashes) {
  // Get index for chunk
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  for(int offset=idx; offset<num_hashes; offset+=blockDim.x) {
    // Calculate size of chunk, last chunk may be smaller than chunk_size
    size_t block_size = chunk_size;
    if(chunk_size*(offset+1) > data_len)
      block_size = data_len - offset*chunk_size;
    // Calculate hash
    sha1_hash(data + (offset*chunk_size), block_size, (uint8_t*)(hashes)+digest_size()*offset);
  }
}

void calculate_hashes(const uint8_t* data,
                      const size_t data_len,
                      uint32_t* hashes,
                      size_t chunk_size,
                      const size_t num_hashes) {
  calculate_hashes_kernel<<<1,32>>>(data, data_len, hashes, chunk_size, num_hashes);
  cudaDeviceSynchronize();
}

__global__
void print_hashes_kernel(uint32_t* hashes) {
  for(int i=0; i<5; i++) {
    printf("%d ", hashes[i]);
  }
  printf("\n");
}

void print_hashes(uint32_t* hashes) {
  print_hashes_kernel<<<1,1>>>(hashes);
  cudaDeviceSynchronize();
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
void find_unique_hashes_kernel(const uint32_t* hashes, 
                        const int hash_len, 
                        const size_t num_hashes,
                        size_t* unique_chunks,
                        int* num_unique) {
  // Index of hash
  size_t hash_idx = blockDim.x*blockIdx.x + threadIdx.x;
  for(int offset=hash_idx; offset<num_hashes; offset+=blockDim.x) {
    bool unique = true;
    // Compare with each hash and test if duplicate
    for(size_t i=0; i<num_hashes; i++) {
      if(offset != i && identical_hashes(hashes+(hash_len/sizeof(uint32_t))*i, 
                                           hashes+(hash_len/sizeof(uint32_t))*offset, 
					   hash_len/sizeof(uint32_t))) {
	unique = false;
        // Save only the first instance of a non unique hash
        if(offset < i) {
          int offset = atomicAdd(num_unique, 1);
          unique_chunks[offset] = offset;
        }
        break;
      }
    }
    // Save unique hash
    if(unique) {
      int offset = atomicAdd(num_unique, 1);
      unique_chunks[offset] = offset;
    }
  }
}
void find_unique_hashes(const uint32_t* hashes, 
                        const int hash_len, 
                        const size_t num_hashes,
                        size_t* unique_chunks,
                        int* num_unique) {
  find_unique_hashes(hashes, 
                     hash_len, 
                     num_hashes,
                     unique_chunks,
                     num_unique);
  cudaDeviceSynchronize();
}

__global__
void print_changes_kernel(const size_t* unique_chunks, const int* num_unique, const size_t* diff_size) {
  printf("\tNum unique chunks: %d\n", *num_unique);
  printf("\tCheckpoint size: %d\n", *diff_size);
}

// Compare unique hashes with prior unique hashes
// Total # of threads should be greater than the # of unique hashes
__global__
void compare_prior_hashes_kernel(const uint32_t* hashes,
                          const size_t num_hashes,
                          const uint32_t* prior_hashes,
                          const size_t num_prior_hashes,
                          const int hash_len, 
                          const int num_unique_hashes,
                          size_t* changed_regions,
                          int* num_changes) {
  // Index of unique hash
  size_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  for(int offset=idx; offset<num_unique_hashes; offset+=blockDim.x) {
    size_t region_offset = changed_regions[offset];
    // Compare with prior hashes
    for(size_t i=0; i<num_prior_hashes; i++) {
      if(identical_hashes(hashes+(hash_len/sizeof(uint32_t))*region_offset, prior_hashes+(hash_len/sizeof(uint32_t))*i, hash_len/sizeof(uint32_t))) {
        changed_regions[offset] = num_hashes;
        atomicSub(num_changes, 1);
      }
    }
  }
}
void compare_prior_hashes(const uint32_t* hashes,
                          const size_t num_hashes,
                          const uint32_t* prior_hashes,
                          const size_t num_prior_hashes,
                          const int hash_len, 
                          const int num_unique_hashes,
                          size_t* changed_regions,
                          int* num_changes) {
  compare_prior_hashes(hashes,
                       num_hashes,
                       prior_hashes,
                       num_prior_hashes,
                       hash_len, 
                       num_unique_hashes,
                       changed_regions,
                       num_changes);
  cudaDeviceSynchronize();
}

// Gather updated chunks into a contiguous buffer
// # of thread blocks should be the # of changed chunks
__global__
void gather_changes_kernel(const uint8_t* data,
                    const size_t data_len,
                    const size_t* changed_regions,
                    int* num_unique,
                    const int num_changes,
                    const size_t num_hashes,
                    const int chunk_size,
                    uint8_t* buffer,
                    size_t* diff_size) {

  size_t idx = blockDim.x*blockIdx.x+threadIdx.x;
  for(int offset=idx; offset<num_changes; offset+=blockDim.x) {
    size_t chunk_offset = changed_regions[offset];
    if(chunk_offset < num_hashes) {
      size_t num_write = chunk_size;
      if((chunk_offset+1)*chunk_size >= data_len)
        num_write = data_len - chunk_size*chunk_offset;
      // Copy data chunk by Iterating in a strided fashion for better memory access pattern
      for(int byte=0; byte<num_write; byte++) {
        buffer[chunk_size*offset+byte] = data[chunk_offset*chunk_size+byte];
      }
      atomicAdd((unsigned long long*)(diff_size), num_write);
    } 
  }
}
void gather_changes(const uint8_t* data,
                    const size_t data_len,
                    const size_t* changed_regions,
                    int* num_unique,
                    const int num_changes,
                    const size_t num_hashes,
                    const int chunk_size,
                    uint8_t* buffer,
                    size_t* diff_size) {
  gather_changes(data,
                 data_len,
                 changed_regions,
                 num_unique,
                 num_changes,
                 num_hashes,
                 chunk_size,
                 buffer,
                 diff_size);
  cudaDeviceSynchronize();
}

