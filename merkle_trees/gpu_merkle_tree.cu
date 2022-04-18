#include "gpu_merkle_tree.cuh"
//#include "gpu_sha1.hpp"

//namespace stdgpu {
//  template<>
//  struct hash<HashDigest> {
//    using is_transparent = void;
//    STDGPU_HOST_DEVICE std::size_t
//    operator()(const HashDigest& key) const {
//      std::size_t hash = 0;
//      uint64_t* key_u64 = key.ptr;
//      uint32_t* key_u64 = key.ptr;
//      hash ^= key_u64[0];
//      hash ^= key_u64[1];
//      hash ^= key_u32[4];
//      return hash;
//    }
//  }
//
//  template<>
//  struct equal_to<HashDigest> {
//    using is_transparent = void;
//    STDGPU_HOST_DEVICE bool
//    operator()(const HashDigest& lhs,
//               const HashDigest& rhs) const {
//      uint64_t* lhs_u64 = lhs.ptr;
//      uint64_t* rhs_u64 = rhs.ptr;
//      uint32_t* lhs_u32 = lhs.ptr;
//      uint32_t* rhs_u32 = rhs.ptr;
//      return (lhs_u64[0] == rhs_u64[0]) && (lhs_u64[1] == rhs_u64[1]) && (lhs_u32[4] == rhs_u32[4]);
//    }
//  }
//}

//struct transparent_hash {
//  using is_transparent = void;
//
//  template <typename T>
//  inline STDGPU_HOST_DEVICE std::size_t 
//  operator()(const T& key) const
//  {
//    return stdgpu::hash<T>{}(key);
//  }
//
//  template<>
//  inline STDGPU_HOST_DEVICE std::size_t
//  operator()<HashDigest>(const HashDigest& key) {
//    size_t hash = 0;
//    uint64_t* key_u64 = key.ptr;
//    uint32_t* key_u32 = key.ptr;
//    hash ^= key_u64[0];
//    hash ^= key_u64[1];
//    hash ^= key_u32[4];
//    return hash;
//  }
//};

__global__ void create_merkle_tree(const uint8_t* data, const unsigned int len, const unsigned int chunk_size, uint8_t* tree) {
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int num_leaves = len/chunk_size;
  if(num_leaves*chunk_size < len) {
    num_leaves += 1;
  }
  unsigned int num_nodes = 2*num_leaves - 1;
  unsigned int num_levels = static_cast<unsigned int>(ceil(log2(static_cast<double>(num_leaves)) + 1));
  unsigned int leaf_start = num_leaves - 1;
  if(idx == 0) {
    printf("Num leaves: %u\n", num_leaves);
    printf("Num nodes : %u\n", num_nodes);
    printf("Num levels: %u\n", num_levels);
    printf("Leaf start: %u\n", leaf_start);
  }
  for(int i=num_levels-1; i>=0; i--) {
    unsigned int start = (1 << i) - 1;
    unsigned int end = (1 << (i+1)) - 1;
    if(end > num_nodes)
      end = num_nodes;
    if(idx == 0) {
      printf("start: %d\n", start);
      printf("end  : %d\n", end);
    }
    for(unsigned int offset=idx; offset<(end-start); offset += blockDim.x) {
      if(offset+start >= leaf_start) {
        unsigned int block_size = chunk_size;
        if(chunk_size*(offset+1) > len)
          block_size = len - offset*chunk_size;
        sha1_hash(data + ((offset+start-leaf_start)*chunk_size), block_size, tree+digest_size()*(start+offset));
      } else {
        sha1_hash(tree+(2*(start+offset) + 1)*digest_size(), 2*digest_size(), tree+digest_size()*(start+offset));
      }
    }
    __syncthreads();
  }
}

void gpu_create_merkle_tree(const uint8_t* data, const unsigned int len, const unsigned int chunk_size, uint8_t* tree) {
  create_merkle_tree<<<1,32>>>(data, len, chunk_size, tree);
  cudaDeviceSynchronize();
}

__global__ void find_distinct_subtrees( const uint8_t* tree, 
                                        const unsigned int num_nodes, 
                                        const int id, 
                                        HashTable<HashDigest,NodeInfo> distinct_map) {
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
if(idx == 0)
printf("Tree pointer: %p\n", (void*)(tree));
  for(unsigned int offset=idx; offset<num_nodes; offset+=blockDim.x) {
    NodeInfo val(offset, offset, id);
    HashDigest digest;
    digest.ptr = tree+offset*digest_size();
printf("Inserting (%p, (%u,%u,%u))\n", digest.ptr, val.node, val.src, val.tree);
//printf("Inserting node info and hash digest: node %d\n", offset);
//    bool result = distinct_map.insert(tree+offset*digest_size(), val);
    bool result = distinct_map.insert(digest, val);
//printf("Created node info and hash digest: node %d\n", offset);
//if(!result) {
//  printf("Found duplicate at node %d\n", offset);
//}
  }
}

void gpu_find_distinct_subtrees( const uint8_t* tree, 
                                 const unsigned int num_nodes, 
                                 const int id, 
                                 HashTable<HashDigest, NodeInfo>& distinct_map) {
printf("Calling find_distinct_subtrees<<<1,32>>>(%p, %u, %d, distinct_map)\n", tree, num_nodes, id);
  find_distinct_subtrees<<<1,1>>>(tree, num_nodes, id, distinct_map);
  cudaDeviceSynchronize();
}

__global__ void find_distinct_subtrees( const uint8_t* tree, 
                                        const unsigned int num_nodes, 
                                        const int id, 
//                                        stdgpu::unordered_map<HashDigest, NodeInfo> distinct_map, 
                                        stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> distinct_map, 
                                        stdgpu::unordered_map<uint32_t, uint32_t> shared_map) {
  using DistinctMap = stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash>;
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  for(unsigned int offset=idx; offset<num_nodes; offset+=blockDim.x) {
    NodeInfo val(offset, offset, id);
    HashDigest digest;
    digest.ptr = tree+offset*digest_size();
printf("Created node info and hash digest: node %d\n", offset);
//    thrust::pair<DistinctMap::iterator, bool> result = distinct_map.emplace(digest, val);
    thrust::pair<DistinctMap::iterator, bool> result = distinct_map.insert(thrust::make_pair(digest, val));
if(!result.second) {
  printf("Found duplicate at node %d\n", offset);
}
  }
}

void gpu_find_distinct_subtrees( const uint8_t* tree, 
                                 const unsigned int num_nodes, 
                                 const int id, 
                                 stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> distinct_map, 
                                 stdgpu::unordered_map<uint32_t, uint32_t> shared_map) {
printf("Number of nodes: %u\n", num_nodes);
  find_distinct_subtrees<<<1,1>>>(tree, num_nodes, id, distinct_map, shared_map);
  cudaDeviceSynchronize();
}

__device__ void remove_subtree(const uint8_t* tree, const unsigned int num_nodes, const unsigned int root, unsigned int* queue, HashTable<HashDigest, NodeInfo>& distinct_map) {
  queue[0] = root;
  unsigned int queue_len = 1;
  unsigned int queue_start = 0;
  unsigned int queue_end = 1;
int num_remove = 0;
  while(queue_len > 0) {
    unsigned int node = queue[queue_start];
    queue_start = (queue_start+1) % num_nodes;
    queue_len -= 1;
    HashDigest digest;
    digest.ptr = tree+node*digest_size();
printf("Removing node %u (%p)\n", node, digest.ptr);
    NodeInfo* old_info = (distinct_map.find(digest));
printf("NodeInfo: (%u,%u,%u)\n", old_info->node, old_info->src, old_info->tree);
    if(old_info->node == node) {
      distinct_map.remove(digest);
num_remove += 1;
    }
    unsigned int child_l = 2*node+1;
    unsigned int child_r = 2*node+2;
printf("Need to remove children %u and %u\n", child_l, child_r);
    if(child_l < num_nodes) {
      queue[queue_end] = child_l;
      queue_end = (queue_end+1) % num_nodes;
      queue_len += 1;
    }
    if(child_r < num_nodes) {
      queue[queue_end] = child_r;
      queue_end = (queue_end+1) % num_nodes;
      queue_len += 1;
    }
  }
printf("Num removed: %d\n", num_remove);
}

__global__ void gpu_compare_trees(const uint8_t* tree,
                                  const unsigned int num_nodes,
                                  const int id,
                                  unsigned int* queue,
                                  HashTable<HashDigest, NodeInfo>& distinct_map,
                                  HashTable<HashDigest, NodeInfo>& prior_map) {
  queue[0] = 0;
  unsigned int queue_len = 1;
  unsigned int queue_start = 0;
  unsigned int queue_end = 1;
printf("Setup queue\n");
  while(queue_len > 0) {
    unsigned int node = queue[queue_start];
    queue_start = (queue_start+1) % num_nodes;
    queue_len -= 1;
    HashDigest digest;
    digest.ptr = tree+node*digest_size();
printf("Searching for node %u\n", node);
    NodeInfo* node_info = (distinct_map.find(digest));
//    if(node_info.node != UINT_MAX) {
    if(node_info != distinct_map.m_values_d+*(distinct_map.m_capacity_d)) {
printf("Found node in distinct map\n");
      NodeInfo* prior_info = (prior_map.find(digest));
//      if(prior_info.node != UINT_MAX) {
      if(prior_info != prior_map.m_values_d+*(prior_map.m_capacity_d)) {
printf("Found node in prior map\n");
        NodeInfo new_node;
//        new_node.node = node_info.node;
//        new_node.src = prior_info.src;
//        new_node.tree = prior_info.tree;
        new_node.node = node_info->node;
        new_node.src = prior_info->src;
        new_node.tree = prior_info->tree;
printf("Replacing subtree at %p with (%u,%u,%u)\n", digest.ptr, new_node.node, new_node.src, new_node.tree);
        remove_subtree(tree, num_nodes, node, queue+num_nodes, distinct_map);
//        distinct_map.insert(tree+node*digest_size(), new_node);
        distinct_map.insert(digest, new_node);
      } else {
printf("Node is distinct for this tree\n");
        unsigned int child_l = 2*node+1;
        unsigned int child_r = 2*node+2;
        if(child_l < num_nodes) {
          queue[queue_end] = child_l;
          queue_end = (queue_end+1) % num_nodes;
          queue_len += 1;
printf("Added child %u\n", child_l);
        }
        if(child_r < num_nodes) {
          queue[queue_end] = child_r;
          queue_end = (queue_end+1) % num_nodes;
          queue_len += 1;
printf("Added child %u\n", child_r);
        }
      }
    }
  }
}

void gpu_compare_trees( const uint8_t* tree,
                        const unsigned int num_nodes,
                        const int id,
                        HashTable<HashDigest, NodeInfo>& distinct_map,
                        HashTable<HashDigest, NodeInfo>& prior_map) {
  unsigned int* queue;
  cudaMalloc(&queue, 2*num_nodes*sizeof(unsigned int));
printf("Calling compare_subtrees<<<1,1>>>(%p, %u, %d, queue, distinct_map, prior_map)\n", tree, num_nodes, id);
  gpu_compare_trees<<<1,1>>>(tree, num_nodes, id, queue, distinct_map, prior_map);
  cudaDeviceSynchronize();
  cudaFree(queue);
}
