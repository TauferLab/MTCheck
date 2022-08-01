#ifndef KOKKOS_MAP_HELPERS_HPP
#define KOKKOS_MAP_HELPERS_HPP
#include <Kokkos_Core.hpp>
//#include <Kokkos_UnorderedMap.hpp>
#include <climits>
#include "kokkos_vector.hpp"

union Storage {
  uint8_t digest[16];
  uint32_t count[4];
};

//template<uint32_t N> 
struct alignas(16) HashDigest {
  uint8_t digest[16];
//  uint32_t digest[5];
};

struct NodeID {
  uint32_t node;
  uint32_t tree;
 
  KOKKOS_INLINE_FUNCTION
  NodeID() {
    node = UINT_MAX;
    tree = UINT_MAX;
  }
 
  KOKKOS_INLINE_FUNCTION
  NodeID(uint32_t n, uint32_t t) {
    node = n;
    tree = t;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const NodeID& other) const {
    return !(other.node != node || other.tree != tree);
  }
};

struct NodeInfo {
  uint32_t curr_node;
  uint32_t prev_node;
  uint32_t prev_tree;

  KOKKOS_INLINE_FUNCTION
  NodeInfo(uint32_t n, uint32_t s, uint32_t t) {
    curr_node = n;
    prev_node = s;
    prev_tree = t;
  }

  KOKKOS_INLINE_FUNCTION
  NodeInfo() {
    curr_node = UINT_MAX;
    prev_node = UINT_MAX;
    prev_tree = UINT_MAX;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const NodeInfo &other) const {
    if(other.curr_node != curr_node || other.prev_node != prev_node || other.prev_tree != prev_tree)
      return false;
    return true;
  }
};

struct CompactNodeInfo {
  uint32_t node;
  uint32_t size;

  KOKKOS_INLINE_FUNCTION
  CompactNodeInfo(uint32_t n, uint32_t s) {
    node = n;
    size = s;
  }

  KOKKOS_INLINE_FUNCTION
  CompactNodeInfo() {
    node = UINT_MAX;
    size = UINT_MAX;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const CompactNodeInfo &other) const {
    if(other.node != node || other.size != size)
      return false;
    return true;
  }
};

struct digest_hash {
  using argument_type        = HashDigest;
  using first_argument_type  = HashDigest;
  using second_argument_type = uint32_t;
  using result_type          = uint32_t;

  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t operator()(HashDigest const& digest) const {
    uint32_t result = 0;
    uint32_t* digest_ptr = (uint32_t*) digest.digest;
    result ^= digest_ptr[0];
    result ^= digest_ptr[1];
    result ^= digest_ptr[2];
    result ^= digest_ptr[3];
//    result ^= digest_ptr[4];
    return result;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t operator()(HashDigest const& digest, uint32_t seed) const {
    uint32_t result = 0;
    uint32_t* digest_ptr = (uint32_t*) digest.digest;
    result ^= digest_ptr[0];
    result ^= digest_ptr[1];
    result ^= digest_ptr[2];
    result ^= digest_ptr[3];
//    result ^= digest_ptr[4];
    return result;
  }
};

struct digest_equal_to {
  using first_argument_type  = HashDigest;
  using second_argument_type = HashDigest;
  using result_type          = bool;

  KOKKOS_FORCEINLINE_FUNCTION
  bool operator()(const HashDigest& a, const HashDigest& b) const {
    uint32_t* a_ptr = (uint32_t*) a.digest;
    uint32_t* b_ptr = (uint32_t*) b.digest;
    for(uint32_t i=0; i<sizeof(HashDigest)/4; i++) {
      if(a_ptr[i] != b_ptr[i]) {
        return false;
      }
    }
    return true;
  }
};

using SharedMap = Kokkos::UnorderedMap<uint32_t, uint32_t>;
using SharedHostMap = Kokkos::UnorderedMap<uint32_t, uint32_t, Kokkos::DefaultHostExecutionSpace>;
using SharedTreeMap = Kokkos::UnorderedMap<uint32_t, NodeID>;
using SharedHostTreeMap = Kokkos::UnorderedMap<uint32_t, NodeID, Kokkos::DefaultHostExecutionSpace>;
using DistinctMap = Kokkos::UnorderedMap<HashDigest, 
                                         NodeID, 
                                         Kokkos::CudaUVMSpace, 
//                                         Kokkos::DefaultExecutionSpace, 
                                         digest_hash, 
                                         digest_equal_to>;
using DistinctHostMap = Kokkos::UnorderedMap<HashDigest, 
                                             NodeID, 
                                             Kokkos::DefaultHostExecutionSpace, 
                                             digest_hash, 
                                             digest_equal_to>;
template<uint32_t N>
using CompactTable = Kokkos::UnorderedMap< CompactNodeInfo, Array<N> >;
template<uint32_t N>
using CompactHostTable = Kokkos::UnorderedMap< CompactNodeInfo, Array<N> , Kokkos::DefaultHostExecutionSpace>;

#endif

