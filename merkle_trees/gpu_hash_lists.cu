#include "gpu_hash_lists.cuh"
#include "gpu_sha1.hpp"

// Calculate hash for each chunk
__global__
void calculate_hashes_kernel(const uint8_t* data,
                      const unsigned int data_len,
                      uint32_t* hashes,
                      unsigned int chunk_size,
                      const unsigned int num_hashes) {
  // Get index for chunk
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  for(int offset=idx; offset<num_hashes; offset+=blockDim.x) {
//unsigned int offset = idx;
//if(offset<num_hashes) {
    // Calculate size of chunk, last chunk may be smaller than chunk_size
    unsigned int block_size = chunk_size;
    if(chunk_size*(offset+1) > data_len)
      block_size = data_len - offset*chunk_size;
    // Calculate hash
    sha1_hash(data + (offset*chunk_size), block_size, (uint8_t*)(hashes)+digest_size()*offset);
  }
}

void calculate_hashes(const uint8_t* data,
                      const unsigned int data_len,
                      uint32_t* hashes,
                      unsigned int chunk_size,
                      const unsigned int num_hashes) {
  unsigned int nhashes = data_len/chunk_size;
  unsigned int nblocks = nhashes/32;
  if(nblocks*32 < nhashes)
    nblocks += 1;
//  calculate_hashes_kernel<<<nblocks,32>>>(data, data_len, hashes, chunk_size, num_hashes);
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
bool identical_hashes(const uint32_t* a, const uint32_t* b, unsigned int len) {
  for(unsigned int i=0; i<len; i++) {
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
                        const unsigned int num_hashes,
                        unsigned int* unique_chunks,
                        int* num_unique) {
  // Index of hash
  unsigned int hash_idx = blockDim.x*blockIdx.x + threadIdx.x;
  for(int offset=hash_idx; offset<num_hashes; offset+=blockDim.x) {
    bool unique = true;
    // Compare with each hash and test if duplicate
    for(unsigned int i=0; i<num_hashes; i++) {
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

__global__
void find_unique_hashes_kernel(const uint32_t* hashes, 
                        const unsigned int num_hashes,
                        const unsigned int chkpt_id,
                        stdgpu::unordered_map<HashDigest, HashListInfo, transparent_sha1_hash> distinct_map,
                        stdgpu::unordered_map<unsigned int, unsigned int> shared_map
                        ) {
  using DistinctMap = stdgpu::unordered_map<HashDigest, HashListInfo, transparent_sha1_hash>;
  using SharedMap   = stdgpu::unordered_map<unsigned int, unsigned int>;
  // Index of hash
  unsigned int hash_idx = blockDim.x*blockIdx.x + threadIdx.x;
  for(unsigned int offset=hash_idx; offset<num_hashes; offset+=blockDim.x) {
//unsigned int offset = hash_idx;
//if(offset<num_hashes) {
    HashDigest digest;
    digest.ptr = (uint8_t*)(hashes+hash_idx*5);
    HashListInfo val(offset, chkpt_id);
    thrust::pair<DistinctMap::iterator, bool> result = distinct_map.insert(thrust::make_pair(digest, val));
    if(!result.second) {
      thrust::pair<SharedMap::iterator, bool> shared_insert = shared_map.insert(thrust::make_pair(offset, result.first->second.index));
      if(!shared_insert.second)
        printf("Failed to insert %u in the distinct and shared map\n", offset);
    }
  }
}

void find_unique_hashes(const uint32_t* hashes, 
                        const unsigned int num_hashes,
                        const unsigned int chkpt_id,
                        stdgpu::unordered_map<HashDigest, HashListInfo, transparent_sha1_hash> distinct,
                        stdgpu::unordered_map<unsigned int, unsigned int> shared) {
  unsigned int nblocks = num_hashes/32;
  if(nblocks*32 < num_hashes)
    nblocks += 1;
//  find_unique_hashes_kernel<<<nblocks,32>>>(hashes, num_hashes, chkpt_id, distinct, shared);
  find_unique_hashes_kernel<<<1,32>>>(hashes, num_hashes, chkpt_id, distinct, shared);
  cudaDeviceSynchronize();
}

__global__
void print_changes_kernel(const unsigned int* unique_chunks, const int* num_unique, const unsigned int* diff_size) {
  printf("\tNum unique chunks: %d\n", *num_unique);
  printf("\tCheckpoint size: %d\n", *diff_size);
}

// Compare unique hashes with prior unique hashes
// Total # of threads should be greater than the # of unique hashes
__global__
void compare_prior_hashes_kernel(const uint32_t* hashes,
                          const unsigned int num_hashes,
                          stdgpu::unordered_map<HashDigest, HashListInfo, transparent_sha1_hash> distinct,
                          stdgpu::unordered_map<unsigned int, unsigned int> shared,
                          stdgpu::unordered_map<HashDigest, HashListInfo, transparent_sha1_hash> prior) {
  // Index of unique hash
  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  for(unsigned int offset=idx; offset<num_hashes; offset+=blockDim.x*gridDim.x) {
//unsigned int offset = idx;
//if(offset<num_hashes) {
    HashDigest digest;
    digest.ptr = (uint8_t*)(hashes+5*offset);
    auto distinct_node = distinct.find(digest);
//    if(distinct_node != distinct.end() && distinct_node->second.index == offset) {
    if(distinct_node != distinct.end()) {
      auto already_exists = prior.find(distinct_node->first);
      if(already_exists != prior.end()) {
        distinct.erase(digest);
      }
    }
//}
  }
}

void compare_prior_hashes(const uint32_t* hashes,
                          const unsigned int num_hashes,
                          stdgpu::unordered_map<HashDigest, HashListInfo, transparent_sha1_hash> distinct,
                          stdgpu::unordered_map<unsigned int, unsigned int> shared,
                          stdgpu::unordered_map<HashDigest, HashListInfo, transparent_sha1_hash> prior) {
  unsigned int nblocks = num_hashes/32;
  if(nblocks*32 < num_hashes)
    nblocks += 1;
//printf("Size of distinct map %u\n", distinct.size());
//printf("Size of prior map %u \n", prior.size());
//printf("Number of hashes: %u\n", num_hashes);
//  compare_prior_hashes_kernel<<<nblocks, 32>>>(hashes, num_hashes, distinct, shared, prior);
  compare_prior_hashes_kernel<<<1, 32>>>(hashes, num_hashes, distinct, shared, prior);
  cudaDeviceSynchronize();
}

// Gather updated chunks into a contiguous buffer
// # of thread blocks should be the # of changed chunks
__global__
void gather_changes_kernel(const uint8_t* data,
                    const unsigned int data_len,
                    const unsigned int* changed_regions,
                    int* num_unique,
                    const int num_changes,
                    const unsigned int num_hashes,
                    const int chunk_size,
                    uint8_t* buffer,
                    unsigned int* diff_size) {

  unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
  for(int offset=idx; offset<num_changes; offset+=blockDim.x) {
    unsigned int chunk_offset = changed_regions[offset];
    if(chunk_offset < num_hashes) {
      unsigned int num_write = chunk_size;
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
                    const unsigned int data_len,
                    const unsigned int* changed_regions,
                    int* num_unique,
                    const int num_changes,
                    const unsigned int num_hashes,
                    const int chunk_size,
                    uint8_t* buffer,
                    unsigned int* diff_size) {
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

