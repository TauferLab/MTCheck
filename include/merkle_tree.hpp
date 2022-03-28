#include "gpu_sha1.hpp"

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

// Calculate hashes and create Merkle tree
uint8_t* create_merkle_tree(uint8_t* data, const size_t len, const size_t hash_len, const size_t chunksize) {
  uint8_t* tree;
  // Calculate # of hashes
  uint32_t num_hashes = len/chunksize;
  if(num_hashes*chunksize < len)
    num_hashes += 1;
  // Allocate tree
  tree = (uint8_t*) malloc(hash_len*(2*num_hashes-1));
  size_t leaf_start = num_hashes-1;
  // Fill tree from leaves to root
  for(int64_t idx=2*num_hashes-2; idx>=0; idx--) {
    if(idx >= num_hashes-1) { // Leaves
      sha1_hash(data+(idx-leaf_start)*chunksize, chunksize, (uint8_t*)(tree) + idx*hash_len);
    } else { // Branches
      sha1_hash(tree+(2*idx+1)*hash_len, hash_len*2, (uint8_t*)(tree) + idx*hash_len);
    }
  }
  return tree;
}

// Test if 2 hashes are identical
bool identical_hashes(const uint8_t* a, const uint8_t* b, size_t len) {
  for(size_t i=0; i<len; i++) {
    if(a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

size_t parent_index(size_t index) {
  return (index-1)/2;
}

size_t left_child_index(size_t index) {
  return 2*index+1;
}

size_t right_child_index(size_t index) {
  return 2*index+2;
}

size_t num_nodes(size_t num_hashes) {
  return 2*num_hashes-1;
}

void compare_merkle_trees(uint8_t* tree_a, uint8_t* tree_b, const size_t hash_len, const size_t num_hashes, bool* unique_chunks, int* tree_id, size_t& num_unique) {
  const size_t num_nodes = 2*num_hashes-1;
  size_t* queue = new size_t[num_nodes];
  queue[0] = 0;
  size_t queue_size = 1;
  size_t queue_start = 0;
  size_t queue_end = 1;
  while(queue_size > 0) {
//    for(size_t i=queue_start; i != queue_end; i = (i+1)%num_hashes) {
//      printf("%zd ", queue[i]);
//    }
//    printf("\n");
    size_t node = queue[queue_start];
//    printf("Removed node %zd\n", node);
    queue_start = (queue_start + 1) % num_hashes;
    queue_size -= 1;
    if(!identical_hashes(tree_a+node*hash_len, tree_b+node*hash_len, hash_len)) {
      size_t l_child = left_child_index(node);
      size_t r_child = right_child_index(node);

      if(l_child < num_nodes) {
//printf("Inserting l child node %zd\n", l_child);
        queue[queue_end] = l_child;
        queue_end = (queue_end + 1) % num_hashes;
        queue_size += 1;
      }

      if(r_child < num_nodes) {
//printf("Inserting r child node %zd\n", r_child);
        queue[queue_end] = r_child;
        queue_end = (queue_end + 1) % num_hashes;
        queue_size += 1;
      }

      if(l_child >= num_nodes && r_child >= num_nodes) {
//printf("Found leaf %zd \n", node);
        size_t chunk_idx = node - (num_hashes-1);
//        unique_chunks[num_unique] = chunk_idx;
        unique_chunks[node] = true;
//	tree_id[node] = ;
        num_unique += 1;
      }
    }
  }
}

void digest_to_hex(const uint8_t* digest, char* output, uint32_t digest_size) {
  char* c = output;
  for(uint32_t i=0; i<digest_size/4; i++) {
    for(uint32_t j=0; j<4; j++) {
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
  for(size_t i=0; i<2*num_leaves-1; i++) {
    digest_to_hex(tree+i*hash_len, buffer, hash_len);
    printf("%s | ", buffer);
    if((i & (i-1)) == 0 && i != 1)
      printf("\n");
  }
}

//class MerkleTree {
//public:
//  uint32_t *tree;
//  uint32_t hash_len; 
//
//  MerkleTree(uint8_t* data, size_t len, bool cuda) {
//    uint32_t num_hashes = len/digest_size();
//    if(num_hashes*digest_size() < len)
//      num_hashes += 1;
//    uint32_t height = = static_cast<uint32_t>(log2(len));
//    tree = (uint32_t*) malloc(hash_len*(2*num_hashes-1));
//    for(size_t idx=num_hashes-1; idx<2*num_hashes-1; idx++) {
//    }
//  }
//
//  ~MerkleTree() {
//  }
//};

