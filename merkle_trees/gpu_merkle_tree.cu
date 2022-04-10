#include "gpu_merkle_tree.hpp"
#include "gpu_sha1.hpp"

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

__global__ void create_merkle_tree(const uint8_t* data, const size_t len, const size_t chunk_size, uint8_t* tree) {
  size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  size_t num_leaves = len/chunk_size;
  if(num_leaves*chunk_size < len) {
    num_leaves += 1;
  }
  size_t num_nodes = 2*num_leaves - 1;
  size_t num_levels = static_cast<size_t>(ceil(log2(static_cast<double>(num_leaves)) + 1));
  size_t leaf_start = num_leaves - 1;
  if(idx == 0) {
    printf("Num leaves: %u\n", num_leaves);
    printf("Num nodes : %u\n", num_nodes);
    printf("Num levels: %u\n", num_levels);
    printf("Leaf start: %u\n", leaf_start);
  }
  for(int i=num_levels-1; i>=0; i--) {
    size_t start = (1 << i) - 1;
    size_t end = (1 << (i+1)) - 1;
    if(end > num_nodes)
      end = num_nodes;
    if(idx == 0) {
      printf("start: %d\n", start);
      printf("end  : %d\n", end);
    }
    if(idx < end-start) {
      if(idx+start >= leaf_start) {
        size_t block_size = chunk_size;
        if(chunk_size*(idx+1) > len)
          block_size = len - idx*chunk_size;
        sha1_hash(data + ((idx+start-leaf_start)*chunk_size), block_size, tree+digest_size()*(start+idx));
      } else {
        sha1_hash(tree+(2*(start+idx) + 1)*digest_size(), 2*digest_size(), tree+digest_size()*(start+idx));
      }
    }
    __syncthreads();
  }
}

void gpu_create_merkle_tree(const uint8_t* data, const size_t len, const size_t chunk_size, uint8_t* tree) {
  create_merkle_tree<<<1,64>>>(data, len, chunk_size, tree);
  cudaDeviceSynchronize();
}

__global__ void find_distinct_subtrees( const uint8_t* tree, 
                                        const size_t num_nodes, 
                                        const int id, 
                                        stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> distinct_map, 
                                        stdgpu::unordered_map<uint32_t, uint32_t> shared_map) {
  size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  const size_t hash_size = digest_size();
  if(idx < num_nodes) {
//    uint32_t val[3] = {idx, idx, id};
    NodeInfo val(idx, idx, id);
    HashDigest digest;
    digest.ptr = tree+idx*digest_size();
    thrust::pair<stdgpu::unordered_map<HashDigest,NodeInfo,transparent_sha1_hash>::iterator, bool> result = distinct_map.emplace(digest, val);
//    thrust::pair<stdgpu::unordered_map<HashDigest,NodeInfo,transparent_sha1_hash>::iterator, bool> result = distinct_map.insert(thrust::make_pair(digest, val));
  }
}

void gpu_find_distinct_subtrees( const uint8_t* tree, 
                                 const size_t num_nodes, 
                                 const int id, 
                                 stdgpu::unordered_map<HashDigest, NodeInfo, transparent_sha1_hash> distinct_map, 
                                 stdgpu::unordered_map<uint32_t, uint32_t> shared_map) {
  find_distinct_subtrees<<<1,num_nodes>>>(tree, num_nodes, id, distinct_map, shared_map);
  cudaDeviceSynchronize();
}

//__global__ void gpu_compare_trees(const uint8_t* tree, prior) {
//}


