#ifndef GPU_HASH_LISTS
#define GPU_HASH_LISTS

#include <cuda/std/cstdint>

void calculate_hashes(const uint8_t* data,
                      const size_t data_len,
                      uint32_t* hashes,
                      size_t chunk_size,
                      const size_t num_hashes);

void print_hashes(uint32_t* hashes);

// Find which hashes are unique
// Total # of threads should be greater than the # of hashes
void find_unique_hashes(const uint32_t* hashes, 
                        const int hash_len, 
                        const size_t num_hashes,
                        size_t* unique_chunks,
                        int* num_unique);

void print_changes(const size_t* unique_chunks, const int* num_unique, const size_t* diff_size);

// Compare unique hashes with prior unique hashes
// Total # of threads should be greater than the # of unique hashes
void compare_prior_hashes(const uint32_t* hashes,
                          const size_t num_hashes,
                          const uint32_t* prior_hashes,
                          const size_t num_prior_hashes,
                          const int hash_len, 
                          const int num_unique_hashes,
                          size_t* changed_regions,
                          int* num_changes);

// Gather updated chunks into a contiguous buffer
// # of thread blocks should be the # of changed chunks
void gather_changes(const uint8_t* data,
                    const size_t data_len,
                    const size_t* changed_regions,
                    int* num_unique,
                    const int num_changes,
                    const size_t num_hashes,
                    const int chunk_size,
                    uint8_t* buffer,
                    size_t* diff_size);

#endif

