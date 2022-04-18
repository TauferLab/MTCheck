#ifndef __MERKLE_TREE_HPP
#define __MERKLE_TREE_HPP

#ifdef __CUDACC__
#include "gpu_merkle_tree.cuh"
#endif
#include "stdgpu/unordered_map.cuh"

enum Mode {
  CPU=0,
  GPU
};

void CreateMerkleTree(const uint8_t* data, const unsigned int len, const unsigned int chunk_size, uint8_t* tree, Mode mode) {
  if(mode == GPU) {
#ifdef __CUDACC__
    gpu_create_merkle_tree(data, len, chunk_size, tree);
#else
    printf("Error: Requested CUDA mode for CreateMerkleTree but can't find CUDACC\n");
#endif
  } else {
  }
}

void FindDistinctSubtrees(const uint8_t* tree, 
                          const unsigned int num_nodes, 
                          const int id, 
                          HashTable<HashDigest, NodeInfo> distinct_map, 
                          Mode mode) {
  if(mode == GPU) {
#ifdef __CUDACC__
    gpu_find_distinct_subtrees(tree, num_nodes, id, distinct_map);
#else
    printf("Error: Requested CUDA mode for FindDistinctSubtrees but can't find CUDACC\n");
#endif
  } else {
  }
}

void CompareTrees( const uint8_t* tree,
                   const unsigned int num_nodes,
                   const int id,
                   HashTable<HashDigest, NodeInfo>& distinct_map,
                   HashTable<HashDigest, NodeInfo>& prior_map,
                   Mode mode) {
  if(mode == GPU) {
#ifdef __CUDACC__
    gpu_compare_trees(tree, num_nodes, id, distinct_map, prior_map);
#else
    printf("Error: Requested CUDA mode for CompareSubtrees but can't find CUDACC\n");
#endif
  } else {
  }
}

void FindDistinctSubtrees(const uint8_t* tree, 
                          const size_t num_nodes, 
                          const int id, 
                          stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> distinct_map, 
//                          stdgpu::unordered_map<HashDigest, NodeInfo> distinct_map, 
                          stdgpu::unordered_map<uint32_t, uint32_t> shared_map,
                          Mode mode) {
  if(mode == GPU) {
#ifdef __CUDACC__
    gpu_find_distinct_subtrees(tree, num_nodes, id, distinct_map, shared_map);
#else
    printf("Error: Requested CUDA mode for FindDistinctSubtrees but can't find CUDACC\n");
#endif
  } else {
  }
}

#endif
