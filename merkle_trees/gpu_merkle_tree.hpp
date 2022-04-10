#ifndef __GPU_MERKLE_TREE_HPP
#define __GPU_MERKLE_TREE_HPP

#include <cstdint>
#include <stdgpu/cstddef.h>
#include <stdgpu/platform.h>
#include <stdgpu/utility.h>
#include <stdgpu/unordered_map.cuh>
//#include "gpu_sha1.hpp"

struct NodeInfo {
  uint32_t node;
  uint32_t src;
  uint32_t tree;

  inline STDGPU_HOST_DEVICE
  NodeInfo(uint32_t n, uint32_t s, uint32_t t) {
    node = n;
    src = s;
    tree = t;
  }
};

struct HashDigest {
  const uint8_t* ptr;

  inline STDGPU_HOST_DEVICE
  bool operator==(const HashDigest &other) const {
    for(int i=0; i<20; i++) {
      if(ptr[i] != other.ptr[i])
        return false;
    }
    return true;
  }
};

struct transparent_sha1_hash {
  using is_transparent = void;

//  template <typename T>
//  inline STDGPU_HOST_DEVICE std::size_t 
//  operator()(const T& key) const
//  {
//    return stdgpu::hash<T>{}(key);
//  }

  inline STDGPU_HOST_DEVICE std::size_t
  operator()(const HashDigest& key) const {
    size_t hash = 0;
    uint64_t* key_u64 = (uint64_t*)key.ptr;
    uint32_t* key_u32 = (uint32_t*)key.ptr;
    hash ^= key_u64[0];
    hash ^= key_u64[1];
    hash ^= key_u32[4];
    return hash;
  }
};

void gpu_create_merkle_tree(const uint8_t* data, const size_t len, const size_t chunk_size, uint8_t* tree);

void gpu_find_distinct_subtrees( const uint8_t* tree, 
                                 const size_t num_nodes, 
                                 const int id, 
                                 stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> distinct_map, 
                                 stdgpu::unordered_map<uint32_t, uint32_t> shared_map);

//void print_merkle_tree(uint8_t* tree, const size_t hash_len, const size_t num_leaves) {
//  char buffer[80];
//  for(size_t i=0; i<2*num_leaves-1; i++) {
//    digest_to_hex(tree+i*hash_len, buffer);
//    printf("Node: %zd: %s \n", i, buffer);
//  }
//}

#endif
