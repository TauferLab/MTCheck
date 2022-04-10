#include "merkle_tree.hpp"
#include <cuda.h>
#include <stdio.h>

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

void print_merkle_tree(uint8_t* tree, const size_t hash_len, const size_t num_leaves) {
  char buffer[80];
  int counter = 2;
  for(size_t i=0; i<2*num_leaves-1; i++) {
    digest_to_hex_(tree+i*hash_len, buffer);
    printf("Node: %zd: %s \n", i, buffer);
    if(i == counter) {
      printf("\n");
      counter += 2*counter;
    }
  }
}

int main(int argc, char** argv) {
  const char* test_str0 = "Hello Muddah. Hello Fadduh. Here I am at camp Granada"; //53
  const char* test_str1 = "Hello Mother. Hello Father. Here I am at camp Granada"; //53
//  const char* test_str0 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"; //53
  uint8_t* gpu_str;
  cudaMalloc(&gpu_str, 53);
  cudaMemcpy(gpu_str, test_str0, 53, cudaMemcpyDeviceToHost);
  printf("Copied data to GPU\n");
  uint8_t* tree = (uint8_t*)malloc(53*20);
  uint8_t* tree_d;
  cudaMalloc(&tree_d, 53*20);
  size_t chunk_size = 1;
  printf("Creating Merkle Tree\n");
  cudaDeviceSynchronize();
  CreateMerkleTree((uint8_t*)test_str0, 53, chunk_size, tree, GPU);
  printf("Created Merkle Tree\n");
  cudaMemcpy(tree, tree_d, 53*20, cudaMemcpyHostToDevice);
  printf("Copied data to CPU\n");
  
  print_merkle_tree(tree, 20, 53);
  
}
