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
#ifdef DEBUG
  if(idx == 0) {
    printf("Num leaves: %u\n", num_leaves);
    printf("Num nodes : %u\n", num_nodes);
    printf("Num levels: %u\n", num_levels);
    printf("Leaf start: %u\n", leaf_start);
  }
#endif
  for(int i=num_levels-1; i>=0; i--) {
    unsigned int start = (1 << i) - 1;
    unsigned int end = (1 << (i+1)) - 1;
    if(end > num_nodes)
      end = num_nodes;
#ifdef DEBUG
    if(idx == 0) {
      printf("start: %d\n", start);
      printf("end  : %d\n", end);
    }
#endif
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
  unsigned int num_leaves = len/chunk_size;
  if(num_leaves*chunk_size < len) {
    num_leaves += 1;
  }
  unsigned int nblocks = num_leaves/32;
  if(nblocks*32 < num_leaves)
   nblocks += 1;
  create_merkle_tree<<<nblocks,32>>>(data, len, chunk_size, tree);
  cudaDeviceSynchronize();
}

__global__ void find_distinct_subtrees( const uint8_t* tree, 
                                        const unsigned int num_nodes, 
                                        const int id, 
                                        HashTable<HashDigest,NodeInfo> distinct_map) {
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
#ifdef DEBUG
if(idx == 0)
printf("Tree pointer: %p\n", (void*)(tree));
#endif
  for(unsigned int offset=idx; offset<num_nodes; offset+=blockDim.x) {
    NodeInfo val(offset, offset, id);
    HashDigest digest;
    digest.ptr = tree+offset*digest_size();
#ifdef DEBUG
printf("Inserting (%p, (%u,%u,%u))\n", digest.ptr, val.node, val.src, val.tree);
#endif
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
                                        stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> distinct_map, 
                                        stdgpu::unordered_map<uint32_t, uint32_t> shared_map) {
  using DistinctMap = stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash>;
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  for(unsigned int offset=idx; offset<num_nodes; offset+=blockDim.x) {
    NodeInfo val(offset, offset, id);
    HashDigest digest;
    digest.ptr = tree+offset*digest_size();
#ifdef DEBUG
printf("Created node info and hash digest: node %d\n", offset);
#endif
    thrust::pair<DistinctMap::iterator, bool> result = distinct_map.insert(thrust::make_pair(digest, val));
#ifdef DEBUG
if(!result.second) {
  printf("Found duplicate at node %d\n", offset);
}
#endif
  }
}

void gpu_find_distinct_subtrees( const uint8_t* tree, 
                                 const unsigned int num_nodes, 
                                 const int id, 
                                 stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> distinct_map, 
                                 stdgpu::unordered_map<uint32_t, uint32_t> shared_map) {
  printf("Number of nodes: %u\n", num_nodes);
  int num_blocks = num_nodes/32;
  if(num_blocks*32 < num_nodes)
    num_blocks += 1;
  find_distinct_subtrees<<<num_blocks,32>>>(tree, num_nodes, id, distinct_map, shared_map);
  cudaDeviceSynchronize();
}

__device__ void remove_subtree(const uint8_t* tree, const unsigned int num_nodes, const unsigned int root, unsigned int* queue, HashTable<HashDigest, NodeInfo>& distinct_map) {
  queue[0] = root;
  unsigned int queue_len = 1;
  unsigned int queue_start = 0;
  unsigned int queue_end = 1;
int num_remove = 0;
  while(queue_len > 0) {
#ifdef DEBUG
__syncthreads();
printf("Queue at start: ");
for(int i=queue_start; i<queue_end; i++) {
printf("%u, ", queue[i]);
}
printf("\n");
__syncthreads();
#endif
    unsigned int node = queue[queue_start];
    queue_start = (queue_start+1) % num_nodes;
    queue_len -= 1;
    HashDigest digest;
    digest.ptr = tree+node*digest_size();
#ifdef DEBUG
printf("\tRemoving node %u (%p)\n", node, digest.ptr);
#endif
    NodeInfo* old_info = (distinct_map.find(digest));
if(old_info != distinct_map.m_values_d+*(distinct_map.m_capacity_d)) {
#ifdef DEBUG
printf("\tNodeInfo: (%u,%u,%u)\n", old_info->node, old_info->src, old_info->tree);
#endif
    if(old_info->node == node) {
    distinct_map.remove(digest);
num_remove += 1;
    }
} else {
#ifdef DEBUG
  printf("\tnode %u is not in the distinct map\n", node);
#endif
}
    unsigned int child_l = 2*node+1;
    unsigned int child_r = 2*node+2;
//printf("Need to remove children %u and %u\n", child_l, child_r);
    if(child_l < num_nodes) {
#ifdef DEBUG
printf("\tNeed to remove child %u\n", child_l);
#endif
      queue[queue_end] = child_l;
      queue_end = (queue_end+1) % num_nodes;
      queue_len += 1;
    }
    if(child_r < num_nodes) {
#ifdef DEBUG
printf("\tNeed to remove child %u\n", child_r);
#endif
      queue[queue_end] = child_r;
      queue_end = (queue_end+1) % num_nodes;
      queue_len += 1;
    }
#ifdef DEBUG
__syncthreads();
printf("Queue at end  : ");
for(int i=queue_start; i<queue_end; i++) {
printf("%u, ", queue[i]);
}
printf("\n");
__syncthreads();
#endif
  }
#ifdef DEBUG
printf("Num removed: %d\n", num_remove);
#endif
}

__device__ unsigned int number_of_leaves(const unsigned int root, const unsigned num_nodes) {
  unsigned int num_leaves = 1;
  unsigned int last = root;
  unsigned int level = 1;
//  unsigned int child_l = 2*root + 1;
//  unsigned int child_r = 2*root + 2;
//  bool flag = true;
  while(last < num_nodes) {
    unsigned int n_nodes_for_level = 1 << level;
    last = 2*last+2;
    num_leaves += n_nodes_for_level;
#ifdef DEBUG
printf("Root (%u): %u nodes for level %u, %u is the last, %u leaves so far\n", root, n_nodes_for_level, level, last, num_leaves);
#endif
    level += 1;
  }
//  num_leaves -= (last-num_nodes);
  return num_leaves;
}

__device__ void remove_subtree_parallel(const uint8_t* tree, const unsigned int num_nodes, const unsigned int root, unsigned int* queue, HashTable<HashDigest, NodeInfo>& distinct_map) {

  unsigned int num_remove = 0;
  unsigned int num_leaves = number_of_leaves(root, num_nodes);
  unsigned int num_levels = static_cast<unsigned int>(ceil(log2(static_cast<double>(num_leaves)) + 1));
#ifdef DEBUG
    printf("Root (%u): Num levels: %u\n", root, num_levels);
    printf("Root (%u): Num leaves: %u\n", root, num_leaves);
#endif
unsigned int first = root;
unsigned int last = root;
  for(int i=0; i<num_levels; i++) {
    unsigned int start = first;
    unsigned int end = last+1;
    if(start > num_nodes)
      start = num_nodes;
    if(end > num_nodes)
      end = num_nodes;
#ifdef DEBUG
    printf("Root (%u): Removing subtree in parallel\n", root);
      printf("Root (%u): start: %d\n", root, start);
      printf("Root (%u): end  : %d\n", root, end);
#endif
    for(unsigned int offset=0; offset<(end-start); offset += 1) {
      unsigned int node = offset+start;
      HashDigest digest;
      digest.ptr = tree+node*digest_size();
#ifdef DEBUG
printf("\tRoot (%u): Removing node %u (%p)\n", root, node, digest.ptr);
#endif
      NodeInfo* old_info = (distinct_map.find(digest));
      if(old_info != distinct_map.m_values_d+*(distinct_map.m_capacity_d)) {
#ifdef DEBUG
printf("\tRoot (%u): NodeInfo: (%u,%u,%u)\n", root, old_info->node, old_info->src, old_info->tree);
#endif
        if(old_info->node == node) {
          distinct_map.remove(digest);
          num_remove += 1;
        }
      } else {
#ifdef DEBUG
        printf("\tRoot (%u): node %u is not in the distinct map\n", root, node);
#endif
      }
    }
    first = 2*first+1;
    last = 2*last+2;
  }
#ifdef DEBUG
  printf("Root (%u): Num removed: %d\n", root, num_remove);
#endif
}

__global__ void gpu_compare_trees_parallel(const uint8_t* tree,
                                  const unsigned int num_nodes,
                                  const int id,
                                  unsigned int* queue,
                                  HashTable<HashDigest, NodeInfo> distinct_map,
                                  HashTable<HashDigest, NodeInfo> prior_map) {
  
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  queue[0] = 0;
  unsigned int queue_len = 1;
  unsigned int queue_start = 0;
  unsigned int queue_end = 1;
#ifdef DEBUG
printf("Setup queue\n");
#endif
  while(queue_len > 0) {
    unsigned int node = queue[queue_start];
    queue_start = (queue_start+1) % num_nodes;
    queue_len -= 1;
    HashDigest digest;
    digest.ptr = tree+node*digest_size();
#ifdef DEBUG
printf("Searching for node %u\n", node);
#endif
    NodeInfo* node_info = (distinct_map.find(digest));
//    if(node_info.node != UINT_MAX) {
    if(node_info != distinct_map.m_values_d+*(distinct_map.m_capacity_d)) {
#ifdef DEBUG
printf("Found node in distinct map\n");
#endif
      NodeInfo* prior_info = (prior_map.find(digest));
//      if(prior_info.node != UINT_MAX) {
      if(prior_info != prior_map.m_values_d+*(prior_map.m_capacity_d)) {
#ifdef DEBUG
printf("Found node in prior map\n");
#endif
        NodeInfo new_node;
        new_node.node = node_info->node;
        new_node.src = prior_info->src;
        new_node.tree = prior_info->tree;
#ifdef DEBUG
printf("Replacing subtree at %p with (%u,%u,%u)\n", digest.ptr, new_node.node, new_node.src, new_node.tree);
#endif
node_info->src = prior_info->src;
node_info->tree = prior_info->tree;
        remove_subtree_parallel(tree, num_nodes, node, queue+num_nodes, distinct_map);
        distinct_map.insert(digest, new_node);
      } else {
#ifdef DEBUG
printf("Node is distinct for this tree\n");
#endif
        unsigned int child_l = 2*node+1;
        unsigned int child_r = 2*node+2;
        if(child_l < num_nodes) {
          queue[queue_end] = child_l;
          queue_end = (queue_end+1) % num_nodes;
          queue_len += 1;
#ifdef DEBUG
printf("Added child %u\n", child_l);
#endif
        }
        if(child_r < num_nodes) {
          queue[queue_end] = child_r;
          queue_end = (queue_end+1) % num_nodes;
          queue_len += 1;
#ifdef DEBUG
printf("Added child %u\n", child_r);
#endif
        }
      }
    }
  }
}

void gpu_compare_trees_parallel( const uint8_t* tree,
                        const unsigned int num_nodes,
                        const int id,
                        HashTable<HashDigest, NodeInfo>& distinct_map,
                        HashTable<HashDigest, NodeInfo>& prior_map) {
  unsigned int* queue;
  cudaMalloc(&queue, 2*num_nodes*sizeof(unsigned int));
printf("Calling compare_subtrees<<<1,1>>>(%p, %u, %d, queue, distinct_map, prior_map)\n", tree, num_nodes, id);
  gpu_compare_trees_parallel<<<1,1>>>(tree, num_nodes, id, queue, distinct_map, prior_map);
  cudaDeviceSynchronize();
  cudaFree(queue);
}

__device__ void remove_subtree_parallel(const uint8_t* tree, const unsigned int num_nodes, const unsigned int root, unsigned int* queue, stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> distinct_map) {

  unsigned int num_remove = 0;
  unsigned int num_leaves = number_of_leaves(root, num_nodes);
  unsigned int num_levels = static_cast<unsigned int>(ceil(log2(static_cast<double>(num_leaves)) + 1));
#ifdef DEBUG
    printf("Root (%u): Num levels: %u\n", root, num_levels);
    printf("Root (%u): Num leaves: %u\n", root, num_leaves);
#endif
unsigned int first = root;
unsigned int last = root;
  for(int i=0; i<num_levels; i++) {
    unsigned int start = first;
    unsigned int end = last+1;
    if(start > num_nodes)
      start = num_nodes;
    if(end > num_nodes)
      end = num_nodes;
#ifdef DEBUG
    printf("Root (%u): Removing subtree in parallel\n", root);
      printf("Root (%u): start: %d\n", root, start);
      printf("Root (%u): end  : %d\n", root, end);
#endif
    for(unsigned int offset=0; offset<(end-start); offset += 1) {
      unsigned int node = offset+start;
      HashDigest digest;
      digest.ptr = tree+node*digest_size();
#ifdef DEBUG
printf("\tRoot (%u): Removing node %u (%p)\n", root, node, digest.ptr);
#endif
      thrust::pair<const HashDigest, NodeInfo>* old_info = distinct_map.find(digest);
      if(old_info != distinct_map.end()) {
#ifdef DEBUG
printf("\tRoot (%u): NodeInfo: (%u,%u,%u)\n", root, old_info->node, old_info->src, old_info->tree);
#endif
        if(old_info->second.node == node) {
          distinct_map.erase(digest);
          num_remove += 1;
        }
      } else {
#ifdef DEBUG
        printf("\tRoot (%u): node %u is not in the distinct map\n", root, node);
#endif
      }
    }
    first = 2*first+1;
    last = 2*last+2;
  }
#ifdef DEBUG
  printf("Root (%u): Num removed: %d\n", root, num_remove);
#endif
}

__global__ void gpu_compare_trees_parallel(const uint8_t* tree,
                                  const unsigned int num_nodes,
                                  const int id,
                                  stdgpu::queue<unsigned int> queue,
                                  stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> distinct_map,
                                  stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> prior_map) {
  
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
if(idx < queue.size()) {
  thrust::pair<unsigned int, bool> result = queue.pop();
  if(!result.second) {
    printf("Failed to pop queue!\n");
    return;
  }
  unsigned int node = result.first;
  HashDigest digest;
  digest.ptr = tree+node*digest_size();
#ifdef DEBUG
printf("Searching for node %u\n", node);
#endif
  auto node_info = distinct_map.find(digest);
  if(node_info != distinct_map.end()) {
#ifdef DEBUG
printf("Found node in distinct map\n");
#endif
    auto prior_info = prior_map.find(digest);
    if(prior_info != prior_map.end()) {
#ifdef DEBUG
printf("Found node in prior map\n");
#endif
      NodeInfo new_node;
      new_node.node = node_info->second.node;
      new_node.src = prior_info->second.src;
      new_node.tree = prior_info->second.tree;
#ifdef DEBUG
printf("Replacing subtree at %p with (%u,%u,%u)\n", digest.ptr, new_node.node, new_node.src, new_node.tree);
#endif
//        node_info->second.src = prior_info->second.src;
//        node_info->second.tree = prior_info->second.tree;
      remove_subtree_parallel(tree, num_nodes, node, NULL, distinct_map);
      thrust::pair<stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash>::iterator, bool> result = distinct_map.insert(thrust::make_pair(digest, new_node));
    } else {
#ifdef DEBUG
printf("Node is distinct for this tree\n");
#endif
      unsigned int child_l = 2*node+1;
      unsigned int child_r = 2*node+2;
      if(child_l < num_nodes) {
        queue.push(child_l);
#ifdef DEBUG
printf("Added child %u\n", child_l);
#endif
      }
      if(child_r < num_nodes) {
        queue.push(child_r);
#ifdef DEBUG
printf("Added child %u\n", child_r);
#endif
      }
    }
  }
}

//  queue[0] = 0;
//  unsigned int queue_len = 1;
//  unsigned int queue_start = 0;
//  unsigned int queue_end = 1;
//#ifdef DEBUG
//printf("Setup queue\n");
//#endif
//  while(queue_len > 0) {
//    unsigned int node = queue[queue_start];
//    queue_start = (queue_start+1) % num_nodes;
//    queue_len -= 1;
//    HashDigest digest;
//    digest.ptr = tree+node*digest_size();
//#ifdef DEBUG
//printf("Searching for node %u\n", node);
//#endif
//    auto node_info = distinct_map.find(digest);
//    if(node_info != distinct_map.end()) {
//#ifdef DEBUG
//printf("Found node in distinct map\n");
//#endif
//      auto prior_info = prior_map.find(digest);
//      if(prior_info != prior_map.end()) {
//#ifdef DEBUG
//printf("Found node in prior map\n");
//#endif
//        NodeInfo new_node;
//        new_node.node = node_info->second.node;
//        new_node.src = prior_info->second.src;
//        new_node.tree = prior_info->second.tree;
//#ifdef DEBUG
//printf("Replacing subtree at %p with (%u,%u,%u)\n", digest.ptr, new_node.node, new_node.src, new_node.tree);
//#endif
////        node_info->second.src = prior_info->second.src;
////        node_info->second.tree = prior_info->second.tree;
//        remove_subtree_parallel(tree, num_nodes, node, queue+num_nodes, distinct_map);
//        thrust::pair<stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash>::iterator, bool> result = distinct_map.insert(thrust::make_pair(digest, new_node));
//      } else {
//#ifdef DEBUG
//printf("Node is distinct for this tree\n");
//#endif
//        unsigned int child_l = 2*node+1;
//        unsigned int child_r = 2*node+2;
//        if(child_l < num_nodes) {
//          queue[queue_end] = child_l;
//          queue_end = (queue_end+1) % num_nodes;
//          queue_len += 1;
//#ifdef DEBUG
//printf("Added child %u\n", child_l);
//#endif
//        }
//        if(child_r < num_nodes) {
//          queue[queue_end] = child_r;
//          queue_end = (queue_end+1) % num_nodes;
//          queue_len += 1;
//#ifdef DEBUG
//printf("Added child %u\n", child_r);
//#endif
//        }
//      }
//    }
//  }
}

__global__ void init_queue_with_root(stdgpu::queue<unsigned int> queue, const unsigned int root) {
  if(!queue.push(root))
    printf("Failed to init queue!\n");
}

void gpu_compare_trees_parallel( const uint8_t* tree,
                        const unsigned int num_nodes,
                        const int id,
                        stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> distinct_map,
                        stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> prior_map) {
//  unsigned int* queue;
//  cudaMalloc(&queue, 2*num_nodes*sizeof(unsigned int));
//printf("Calling compare_subtrees<<<1,1>>>(%p, %u, %d, queue, distinct_map, prior_map)\n", tree, num_nodes, id);
//  gpu_compare_trees_parallel<<<1,1>>>(tree, num_nodes, id, queue, distinct_map, prior_map);
//  cudaDeviceSynchronize();
//  cudaFree(queue);

  stdgpu::queue<unsigned int> queue = stdgpu::queue<unsigned int>::createDeviceObject(num_nodes);
  init_queue_with_root<<<1,1>>>(queue, 0);
  while(!queue.empty()) {
    unsigned int size = queue.size();
    unsigned int nblocks = size/32;
    if(nblocks*32 < size)
      nblocks += 1;
    gpu_compare_trees_parallel<<<nblocks,32>>>(tree, num_nodes, id, queue, distinct_map, prior_map);
  } 
  stdgpu::queue<unsigned int>::destroyDeviceObject(queue);
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
#ifdef DEBUG
printf("Setup queue\n");
#endif
  while(queue_len > 0) {
    unsigned int node = queue[queue_start];
    queue_start = (queue_start+1) % num_nodes;
    queue_len -= 1;
    HashDigest digest;
    digest.ptr = tree+node*digest_size();
#ifdef DEBUG
printf("Searching for node %u\n", node);
#endif
    NodeInfo* node_info = (distinct_map.find(digest));
//    if(node_info.node != UINT_MAX) {
    if(node_info != distinct_map.m_values_d+*(distinct_map.m_capacity_d)) {
#ifdef DEBUG
printf("Found node in distinct map\n");
#endif
      NodeInfo* prior_info = (prior_map.find(digest));
//      if(prior_info.node != UINT_MAX) {
      if(prior_info != prior_map.m_values_d+*(prior_map.m_capacity_d)) {
#ifdef DEBUG
printf("Found node in prior map\n");
#endif
        NodeInfo new_node;
//        new_node.node = node_info.node;
//        new_node.src = prior_info.src;
//        new_node.tree = prior_info.tree;
        new_node.node = node_info->node;
        new_node.src = prior_info->src;
        new_node.tree = prior_info->tree;
#ifdef DEBUG
printf("Replacing subtree at %p with (%u,%u,%u)\n", digest.ptr, new_node.node, new_node.src, new_node.tree);
#endif
node_info->src = prior_info->src;
node_info->tree = prior_info->tree;
        remove_subtree(tree, num_nodes, node, queue+num_nodes, distinct_map);
        distinct_map.insert(digest, new_node);
      } else {
#ifdef DEBUG
printf("Node is distinct for this tree\n");
#endif
        unsigned int child_l = 2*node+1;
        unsigned int child_r = 2*node+2;
        if(child_l < num_nodes) {
          queue[queue_end] = child_l;
          queue_end = (queue_end+1) % num_nodes;
          queue_len += 1;
#ifdef DEBUG
printf("Added child %u\n", child_l);
#endif
        }
        if(child_r < num_nodes) {
          queue[queue_end] = child_r;
          queue_end = (queue_end+1) % num_nodes;
          queue_len += 1;
#ifdef DEBUG
printf("Added child %u\n", child_r);
#endif
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
