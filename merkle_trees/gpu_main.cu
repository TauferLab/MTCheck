#include "merkle_tree.hpp"
#include <cuda.h>
#include <stdio.h>
#include <stdgpu/memory.h>
#include <stdgpu/unordered_map.cuh>

void digest_to_hex_(const uint8_t digest[20], char* output) {
  int i,j;
  char* c = output;
  for(i=0; i<20/4; i++) {
    for(j=0; j<4; j++) {
      sprintf(c, "%02X", digest[i*4 + j]);
      c += 2;
    }
    sprintf(c, " ");
    c += 1;
  }
  *(c-1) = '\0';
}

void print_merkle_tree(uint8_t* tree, const unsigned int hash_len, const unsigned int num_leaves) {
  printf("============================================================\n");
  char buffer[80];
  unsigned int counter = 2;
  for(unsigned int i=0; i<2*num_leaves-1; i++) {
    digest_to_hex_(tree+i*hash_len, buffer);
    printf("Node: %u: %s \n", i, buffer);
    if(i == counter) {
      printf("\n");
      counter += 2*counter;
    }
  }
  printf("============================================================\n");
}

int main(int argc, char** argv) {
  const char* test_str0 = "Hello Muddah. Hello Fadduh. Here I am at camp Granada"; //53
  const char* test_str1 = "Hello Mother. Hello Father. Here I am at camp Granada"; //53
//  const char* test_str0 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"; //53
  unsigned int data_len = 53;
  unsigned int chunk_size = 1;
  unsigned int num_leaves = data_len/chunk_size;
  if(chunk_size*num_leaves < data_len)
    num_leaves += 1;
  unsigned int num_nodes = 2*num_leaves-1;
  printf("Data length: %d\n", data_len);
  printf("Chunk size: %d\n", chunk_size);
  printf("Num leaves: %d\n", num_leaves);
  printf("Num nodes: %d\n", num_nodes);

  uint8_t* gpu_str0;
  cudaMalloc(&gpu_str0, data_len);
  cudaMemcpy(gpu_str0, test_str0, data_len, cudaMemcpyHostToDevice);
  uint8_t* gpu_str1;
  cudaMalloc(&gpu_str1, data_len);
  cudaMemcpy(gpu_str1, test_str1, data_len, cudaMemcpyHostToDevice);
  printf("Copied data to GPU\n");
  uint8_t* tree0 = (uint8_t*)malloc(num_nodes*20);
  uint8_t* tree0_d;
  cudaMalloc(&tree0_d, num_nodes*20);
  uint8_t* tree1 = (uint8_t*)malloc(num_nodes*20);
  uint8_t* tree1_d;
  cudaMalloc(&tree1_d, num_nodes*20);
  printf("Creating Merkle Tree\n");
  cudaDeviceSynchronize();
  CreateMerkleTree(gpu_str0, data_len, chunk_size, tree0_d, GPU);
  CreateMerkleTree(gpu_str1, data_len, chunk_size, tree1_d, GPU);
  cudaDeviceSynchronize();
  printf("Created Merkle Tree\n");
  cudaMemcpy(tree0, tree0_d, num_nodes*20, cudaMemcpyDeviceToHost);
  cudaMemcpy(tree1, tree1_d, num_nodes*20, cudaMemcpyDeviceToHost);
  printf("Copied data to CPU\n");
  
  print_merkle_tree(tree0, 20, num_leaves);
  print_merkle_tree(tree1, 20, num_leaves);

  HashTable<HashDigest, NodeInfo> distinct_map0(2*num_nodes);
  HashTable<HashDigest, NodeInfo> distinct_map1(2*num_nodes);
  printf("Pointer: %p\n", tree0_d);
  FindDistinctSubtrees(tree0_d, num_nodes, 0, distinct_map0, GPU);
  printf("Num distinct: %u\n", distinct_map0.size());
//  print_hash_table(distinct_map0.m_capacity_d, distinct_map0.m_available_indices_d);

printf("Distinct map 0 capacity: %u\n", distinct_map0.capacity());
printf("Distinct map 0 size: %u\n", distinct_map0.size());
printf("Distinct map 1 capacity: %u\n", distinct_map1.capacity());
printf("Distinct map 1 size: %u\n", distinct_map1.size());

  printf("Pointer: %p\n", tree1_d);
  FindDistinctSubtrees(tree1_d, num_nodes, 1, distinct_map1, GPU);
  printf("Num distinct: %u\n", distinct_map1.size());
//  print_hash_table(distinct_map1.m_capacity_d, distinct_map1.m_available_indices_d);
  
printf("Distinct map 0 capacity: %u\n", distinct_map0.capacity());
printf("Distinct map 0 size: %u\n", distinct_map0.size());
printf("Distinct map 1 capacity: %u\n", distinct_map1.capacity());
printf("Distinct map 1 size: %u\n", distinct_map1.size());
//  print_hash_table(distinct_map1.m_capacity_d, distinct_map1.m_available_indices_d, distinct_map1.m_values_d);

  CompareTrees(tree1_d, num_nodes, 1, distinct_map1, distinct_map0, GPU);
  
printf("Distinct map 0 capacity: %u\n", distinct_map0.capacity());
printf("Distinct map 0 size: %u\n", distinct_map0.size());
printf("Distinct map 1 capacity: %u\n", distinct_map1.capacity());
printf("Distinct map 1 size: %u\n", distinct_map1.size());

  print_hash_table(distinct_map1.m_capacity_d, distinct_map1.m_available_indices_d, distinct_map1.m_keys_d, distinct_map1.m_values_d);
printf("Done printing table\n");

//  using DistinctMap = stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash>;
//  using SharedMap = stdgpu::unordered_map<uint32_t,uint32_t>;
//  DistinctMap distinct_map0 = DistinctMap::createDeviceObject(2*num_nodes);
//  DistinctMap distinct_map1 = DistinctMap::createDeviceObject(2*num_nodes);
//  SharedMap shared_map0 = SharedMap::createDeviceObject(num_nodes);
//  SharedMap shared_map1 = SharedMap::createDeviceObject(num_nodes);
//
//  FindDistinctSubtrees(tree0_d, num_nodes, 0, distinct_map0, shared_map0, GPU);
//
//  printf("Num distinct entries: %d\n", distinct_map0.size());
//
//  DistinctMap::destroyDeviceObject(distinct_map0);
//  DistinctMap::destroyDeviceObject(distinct_map1);
//  SharedMap::destroyDeviceObject(shared_map0);
//  SharedMap::destroyDeviceObject(shared_map1);

}
