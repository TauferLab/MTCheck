#ifndef GPU_HASH_LISTS
#define GPU_HASH_LISTS

#include <cuda/std/cstdint>
#include <cstdint>
#include <limits.h>
#include <stdgpu/cstddef.h>
#include <stdgpu/platform.h>
#include <stdgpu/utility.h>
#include <stdgpu/unordered_map.cuh>
#include <stdgpu/queue.cuh>
#include "helpers.cuh"
#include "hash_table.cuh"

void calculate_hashes(const uint8_t* data,
                      const unsigned int data_len,
                      uint32_t* hashes,
                      unsigned int chunk_size,
                      const unsigned int num_hashes);

void print_hashes(uint32_t* hashes);

// Find which hashes are unique
// Total # of threads should be greater than the # of hashes
void find_unique_hashes(const uint32_t* hashes, 
                        const unsigned int num_hashes,
                        const unsigned int chkpt_id,
                        stdgpu::unordered_map<HashDigest, HashListInfo, transparent_sha1_hash> distinct,
                        stdgpu::unordered_map<unsigned int, unsigned int> shared);

void print_changes(const unsigned int* unique_chunks, const int* num_unique, const unsigned int* diff_size);

// Compare unique hashes with prior unique hashes
// Total # of threads should be greater than the # of unique hashes
void compare_prior_hashes(const uint32_t* hashes,
                          const unsigned int num_hashes,
                          stdgpu::unordered_map<HashDigest, HashListInfo, transparent_sha1_hash> distinct,
                          stdgpu::unordered_map<unsigned int, unsigned int> shared,
                          stdgpu::unordered_map<HashDigest, HashListInfo, transparent_sha1_hash> prior);

// Gather updated chunks into a contiguous buffer
// # of thread blocks should be the # of changed chunks
void gather_changes(const uint8_t* data,
                    const unsigned int data_len,
                    const unsigned int* changed_regions,
                    int* num_unique,
                    const int num_changes,
                    const unsigned int num_hashes,
                    const int chunk_size,
                    uint8_t* buffer,
                    unsigned int* diff_size);

#endif

