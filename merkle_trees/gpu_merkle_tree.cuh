#ifndef __GPU_MERKLE_TREE_HPP
#define __GPU_MERKLE_TREE_HPP

#include <cstdint>
#include <limits.h>
#include <stdgpu/cstddef.h>
#include <stdgpu/platform.h>
#include <stdgpu/utility.h>
#include <stdgpu/unordered_map.cuh>
#include <stdgpu/queue.cuh>
#include "hash_table.cuh"
//#include "gpu_sha1.hpp"

struct NodeInfo {
  uint32_t node;
  uint32_t src;
  uint32_t tree;

  inline STDGPU_HOST_DEVICE
  NodeInfo() {
    node = UINT_MAX;
    src = UINT_MAX;
    tree = UINT_MAX;
  }

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
  HashDigest() {
    ptr = NULL;
  }

  inline STDGPU_HOST_DEVICE
  bool operator==(const HashDigest &other) const {
    if(ptr == NULL || other.ptr == NULL) {
      return false;
    }
    for(int i=0; i<20; i++) {
      if(ptr[i] != other.ptr[i])
        return false;
    }
    return true;
  }
};

struct transparent_sha1_hash {
//  using is_transparent = void;

  inline STDGPU_HOST_DEVICE unsigned int
  operator()(const HashDigest& key) const {
    unsigned int hash = 0;
    const unsigned int* key_u32 = (const unsigned int*)(key.ptr);
    hash ^= key_u32[0];
    hash ^= key_u32[1];
    hash ^= key_u32[2];
    hash ^= key_u32[3];
    hash ^= key_u32[4];
//printf("Hash: %u\n", hash);
    return hash;
  }
};

void gpu_create_merkle_tree(const uint8_t* data, const unsigned int len, const unsigned int chunk_size, uint8_t* tree);

void gpu_find_distinct_subtrees( const uint8_t* tree, 
                                 const unsigned int num_nodes, 
                                 const int id, 
                                 HashTable<HashDigest, NodeInfo>& distinct_map);

void gpu_compare_trees( const uint8_t* tree,
                        const unsigned int num_nodes,
                        const int id,
                        HashTable<HashDigest, NodeInfo>& distinct_map,
                        HashTable<HashDigest, NodeInfo>& prior_map);

void gpu_compare_trees_parallel( const uint8_t* tree,
                        const unsigned int num_nodes,
                        const int id,
                        HashTable<HashDigest, NodeInfo>& distinct_map,
                        HashTable<HashDigest, NodeInfo>& prior_map);

void gpu_compare_trees_parallel( const uint8_t* tree,
                        const unsigned int num_nodes,
                        const int id,
                        stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> distinct_map,
                        stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> prior_map);


void gpu_find_distinct_subtrees( const uint8_t* tree, 
                                 const unsigned int num_nodes, 
                                 const int id, 
                                 stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> distinct_map, 
//                                 stdgpu::unordered_map<HashDigest, NodeInfo> distinct_map, 
                                 stdgpu::unordered_map<uint32_t, uint32_t> shared_map);

//void print_merkle_tree(uint8_t* tree, const unsigned int hash_len, const unsigned int num_leaves) {
//  char buffer[80];
//  for(unsigned int i=0; i<2*num_leaves-1; i++) {
//    digest_to_hex(tree+i*hash_len, buffer);
//    printf("Node: %zd: %s \n", i, buffer);
//  }
//}

#endif
