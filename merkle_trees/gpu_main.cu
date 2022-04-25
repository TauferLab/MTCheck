#include "merkle_tree.hpp"
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <string>
#include <sys/stat.h>
#include <stdgpu/memory.h>
#include <stdgpu/unordered_map.cuh>
#include <stdlib.h>

#define MERKLE

long get_file_size(std::string filename) {
  struct stat stat_buf;
  int rc = stat(filename.c_str(), &stat_buf);
  return rc == 0 ? stat_buf.st_size : -1;
}

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

uint8_t* generate_data(const unsigned int len) {
  uint8_t* data = (uint8_t*)(malloc(len));
  srand(NULL);
  for(unsigned int i=0; i<len; i++) {
    data[i] = static_cast<uint8_t>(rand() % 128);
  }
  return data;
}

//uint8_t* copy_and_perturb(const uint8_t* data, const unsigned int len, const unsigned int chance_of_change) {
//  srand(NULL);
//  uint8_t* new_data = (uint8_t*)(malloc(len));
//  for(unsigned int i=0; i<len; i++) {
//    new_data[i] = data[i];
//    unsigned int roll = rand() % 100;
//    if(roll < chance_of_change)
//      new_data[i] = static_cast<uint8_t>(rand() % 128);
//  }
//  return new_data;
//}

uint8_t* copy_and_perturb(const uint8_t* data, const unsigned int len, const unsigned int chance_of_change) {
  srand(time(NULL));
  unsigned int end = static_cast<unsigned int>(len*(static_cast<float>(chance_of_change)/100.f));
  printf("Randomize data from 0 to %u\n", end);
  uint8_t* new_data = (uint8_t*)(malloc(len));
  for(unsigned int i=0; i<len; i++) {
    new_data[i] = data[i];
    if(i<end)
      new_data[i] = static_cast<uint8_t>(rand() % 128);
  }
  return new_data;
}

int main(int argc, char** argv) {
//  if(argc < 3) {
//    printf("Expected more arguments.\n");
//    printf("./gpu_test chunk_size checkpoint_file\n");
//  }
//  int chunk_size = atoi(argv[1]);
//  std::string full_chkpt(argv[2]);
//  std::string incr_chkpt = full_chkpt + ".gpu_test.incr_chkpt";
//  std::vector<std::string> prev_chkpt;
//  for(int i=3; i<argc; i++) {
//    prev_chkpt.push_back(std::string(argv[i]));
//  }

  unsigned int chunk_size = static_cast<unsigned int>(atoi(argv[1]));
  unsigned int data_len   = static_cast<unsigned int>(atoi(argv[2]));
  unsigned int chance     = static_cast<unsigned int>(atoi(argv[3]));

//  const char* test_str0 = "Hello Muddah. Hello Fadduh. Here I am at camp Granada"; //53
//  const char* test_str1 = "Hello Mother. Hello Father. Here I am at camp Granada"; //53
//  unsigned int data_len = 53;
//  unsigned int chunk_size = 1;

  unsigned int num_leaves = data_len/chunk_size;
  if(chunk_size*num_leaves < data_len)
    num_leaves += 1;
  unsigned int num_nodes = 2*num_leaves-1;
  printf("Data length: %d\n", data_len);
  printf("Chunk size: %d\n", chunk_size);
  printf("Num leaves: %d\n", num_leaves);
  printf("Num nodes: %d\n", num_nodes);

  uint8_t* test_str0 = generate_data(data_len);
  uint8_t* test_str1 = copy_and_perturb(test_str0, data_len, chance);

//printf("Test string 0: ");
//for(int i=0; i<data_len; i++) {
//  printf("%hhx", test_str0[i]);
//}
//printf("\n");
//printf("Test string 1: ");
//for(int i=0; i<data_len; i++) {
//  printf("%hhx", test_str1[i]);
//}
//printf("\n");

  uint8_t* gpu_str0;
  cudaMalloc(&gpu_str0, data_len);
  cudaMemcpy(gpu_str0, test_str0, data_len, cudaMemcpyHostToDevice);
  uint8_t* gpu_str1;
  cudaMalloc(&gpu_str1, data_len);
  cudaMemcpy(gpu_str1, test_str1, data_len, cudaMemcpyHostToDevice);
  printf("Copied data to GPU\n");
#ifdef MERKLE
  uint8_t* tree0 = (uint8_t*)malloc(num_nodes*20);
  uint8_t* tree0_d;
  cudaMalloc(&tree0_d, num_nodes*20);
  uint8_t* tree1 = (uint8_t*)malloc(num_nodes*20);
  uint8_t* tree1_d;
  cudaMalloc(&tree1_d, num_nodes*20);
  printf("Creating Merkle Tree\n");
  cudaDeviceSynchronize();
#else
  uint32_t *hashlist0_h, *hashlist0_d, *hashlist1_h, *hashlist1_d;
  uint32_t* hashlist0 = (uint32_t)(malloc(num_leaves*20));
  uint32_t* hashlist1 = (uint32_t)(malloc(num_leaves*20));
  cudaMalloc(&hashlist0_d, num_leaves*20);
  cudaMalloc(&hashlist1_d, num_leaves*20);
  printf("Allocated hash lists\n");
#endif

#ifdef MERKLE
  CreateMerkleTree(gpu_str0, data_len, chunk_size, tree0_d, GPU);
  CreateMerkleTree(gpu_str1, data_len, chunk_size, tree1_d, GPU);
  cudaDeviceSynchronize();
  printf("Created Merkle Tree\n");
  cudaMemcpy(tree0, tree0_d, num_nodes*20, cudaMemcpyDeviceToHost);
  cudaMemcpy(tree1, tree1_d, num_nodes*20, cudaMemcpyDeviceToHost);
  printf("Copied data to CPU\n");
#else
  CreateHashList(gpu_str0, data_len, hashlist0_d, chunk_size, num_leaves, GPU);
  CreateHashList(gpu_str1, data_len, hashlist1_d, chunk_size, num_leaves, GPU);
  cudaDeviceSynchronize();
  printf("Created Hash list Tree\n");
  cudaMemcpy(hashlist0_h, hashlist0_d, num_leaves*20, cudaMemcpyDeviceToHost);
  cudaMemcpy(hashlist1_h, hashlist1_d, num_leaves*20, cudaMemcpyDeviceToHost);
  printf("Copied data to CPU\n");
#endif
  
////  print_merkle_tree(tree0, 20, num_leaves);
////  print_merkle_tree(tree1, 20, num_leaves);
//
//#ifdef MERKLE
//  HashTable<HashDigest, NodeInfo> distinct_map0(2*num_nodes);
//  HashTable<HashDigest, NodeInfo> distinct_map1(2*num_nodes);
//  printf("Pointer: %p\n", tree0_d);
//  FindDistinctSubtrees(tree0_d, num_nodes, 0, distinct_map0, GPU);
//  printf("Num distinct: %u\n", distinct_map0.size());
////  print_hash_table(distinct_map0.m_capacity_d, distinct_map0.m_available_indices_d);
//
//printf("Distinct map 0 capacity: %u\n", distinct_map0.capacity());
//printf("Distinct map 0 size: %u\n", distinct_map0.size());
//printf("Distinct map 1 capacity: %u\n", distinct_map1.capacity());
//printf("Distinct map 1 size: %u\n", distinct_map1.size());
//
//  printf("Pointer: %p\n", tree1_d);
//#ifdef MERKLE
//  FindDistinctSubtrees(tree1_d, num_nodes, 1, distinct_map1, GPU);
//#else
//#endif
//  printf("Num distinct: %u\n", distinct_map1.size());
////  print_hash_table(distinct_map1.m_capacity_d, distinct_map1.m_available_indices_d, distinct_map1.m_keys_d, distinct_map1.m_values_d);
//  
//printf("Distinct map 0 capacity: %u\n", distinct_map0.capacity());
//printf("Distinct map 0 size: %u\n", distinct_map0.size());
//printf("Distinct map 1 capacity: %u\n", distinct_map1.capacity());
//printf("Distinct map 1 size: %u\n", distinct_map1.size());
//  print_hash_table(distinct_map1.m_capacity_d, distinct_map1.m_available_indices_d, distinct_map1.m_keys_d, distinct_map1.m_values_d);
//#else
//  size_t *unique_chunks0, *unique_chunks1;
//  int *num_unique0, *num_unique1;
//  cudaMalloc(&unique_chunks0, sizeof(size_t)*num_leaves);
//  cudaMalloc(&num_unique0, sizeof(int)*num_leaves);
//  cudaMalloc(&unique_chunks1, sizeof(size_t)*num_leaves);
//  cudaMalloc(&num_unique1, sizeof(int)*num_leaves);
//  FindDistinctHashes(hashlist0_d, 20,  num_leaves, unique_chunks0, num_unique0, GPU);
//  FindDistinctHashes(hashlist1_d, 20,  num_leaves, unique_chunks1, num_unique1, GPU);
//#endif
//
//#ifdef MERKLE
//  CompareTrees(tree1_d, num_nodes, 1, distinct_map1, distinct_map0, GPU);
//printf("Distinct map 0 capacity: %u\n", distinct_map0.capacity());
//printf("Distinct map 0 size: %u\n", distinct_map0.size());
//printf("Distinct map 1 capacity: %u\n", distinct_map1.capacity());
//printf("Distinct map 1 size: %u\n", distinct_map1.size());
//
//  print_hash_table(distinct_map1.m_capacity_d, distinct_map1.m_available_indices_d, distinct_map1.m_keys_d, distinct_map1.m_values_d);
//printf("Done printing table\n");
//#else
//void ComparePriorHashes(const uint32_t* hashlist1_d,
//                        const size_t num_leaves,
//                        const uint32_t* hashlist0_d,
//                        const size_t num_leaves,
//                        const int 20, 
//                        const int num_unique_hashes,
//                        size_t* changed_regions,
//                        int* num_changes,
//                        Mode mode) {
//#endif
  

  using DistinctMap = stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash>;
  using SharedMap = stdgpu::unordered_map<uint32_t,uint32_t>;
  DistinctMap distinct_map0 = DistinctMap::createDeviceObject(2*num_nodes);
  DistinctMap distinct_map1 = DistinctMap::createDeviceObject(2*num_nodes);
  SharedMap shared_map0 = SharedMap::createDeviceObject(num_nodes);
  SharedMap shared_map1 = SharedMap::createDeviceObject(num_nodes);

  FindDistinctSubtrees(tree0_d, num_nodes, 0, distinct_map0, shared_map0, GPU);
  FindDistinctSubtrees(tree1_d, num_nodes, 0, distinct_map1, shared_map1, GPU);

  printf("Num distinct entries (tree 0): %d\n", distinct_map0.size());
  printf("Num distinct entries (tree 1): %d\n", distinct_map1.size());

  CompareTrees(tree1_d, num_nodes, 1, distinct_map1, distinct_map0, GPU);

  printf("Num distinct entries (tree 0): %d\n", distinct_map0.size());
  printf("Num distinct entries (tree 1): %d\n", distinct_map1.size());

  DistinctMap::destroyDeviceObject(distinct_map0);
  DistinctMap::destroyDeviceObject(distinct_map1);
  SharedMap::destroyDeviceObject(shared_map0);
  SharedMap::destroyDeviceObject(shared_map1);

}
