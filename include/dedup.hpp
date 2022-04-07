#ifndef __DEDUP_HPP
#define __DEDUP_HPP

#include <string>
#include <cstdint>
#include <vector>
#include <map>
#include "hash_functions.hpp"

typedef struct header {
  size_t chkpt_size;
  size_t header_size;
  size_t num_regions;
} header_t;

struct region_header {
  int id;
  size_t region_size;
  size_t hash_size;
  size_t chunk_size;
  size_t num_hashes;
  size_t num_unique;
  std::vector<std::vector<uint32_t>> hashes;
  std::vector<size_t> unique_hashes;
  uint8_t* merkle_tree;
  size_t* unique_subtree_indices;
  int* unique_subtree_ids;
  size_t num_unique_subtrees;
  size_t* shared_subtree_indices;
  int* shared_subtree_ids;
  size_t num_shared_subtrees;
};
using region_header_t = region_header;
//template <typename HashDigest>
//using region_header_t = region_header<HashDigest>;

enum ptr_type_t {
  Host=0,
  Cuda=1
};

struct region_t {
  void* ptr;
  size_t size;
  ptr_type_t ptr_type;
};
typedef std::map<int, region_t> regions_t;

typedef struct config {
  bool dedup_on_gpu;
  bool use_merkle_trees;
  int chunk_size;
  Hasher* hash_func;
} config_t;

class deduplicate_module_t {
private:
  void cpu_dedup(uint8_t* data, 
                size_t data_len,
                std::map<std::vector<uint32_t>, size_t>& prev_hashes,
                region_header_t& header,
                uint8_t** incr_data,
                size_t& incr_len,
                config_t& config);
  void cpu_dedup(uint8_t* data, 
                size_t data_len,
                int chkpt_id,
		std::map<int, uint8_t*> prev_trees,
                region_header_t& header,
                uint8_t** incr_data,
                size_t& incr_len,
                config_t& config);
  void gpu_dedup(uint8_t* data, 
                size_t data_len,
                std::map<std::vector<uint32_t>, size_t>& prev_hashes,
                region_header_t& header,
                uint8_t** incr_data,
                size_t& incr_len,
                config_t& config);
public:
  void deduplicate_data(regions_t& full_regions, const std::string& incr, std::vector<std::string>& prev_chkpts, config_t& config);
  void deduplicate_file(const std::string& full, const std::string& incr, std::vector<std::string>& prev_chkpts, config_t& config);
};

#endif // __DEDUP_HPP
