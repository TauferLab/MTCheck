#ifndef KOKKOS_MAP_HELPERS_HPP
#define KOKKOS_MAP_HELPERS_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <climits>
#include "kokkos_vector.hpp"

//union Storage {
//  uint8_t digest[16];
//  uint32_t count[4];
//};

//template<uint32_t N> 
struct alignas(16) HashDigest {
  uint8_t digest[16];
};

KOKKOS_INLINE_FUNCTION
uint32_t digest_to_u32(HashDigest& digest) {
  uint32_t* u32_ptr = (uint32_t*)(digest.digest);
  return u32_ptr[0] ^ u32_ptr[1] ^ u32_ptr[2] ^ u32_ptr[3];
}

struct CompareHashDigest {
  bool operator() (const HashDigest& lhs, const HashDigest& rhs) const {
    for(int i=0; i<16; i++) {
      if(lhs.digest[i] != rhs.digest[i]) {
        return false;
      }
    }
    return true;
  }
};

KOKKOS_INLINE_FUNCTION
bool digests_same(const HashDigest& lhs, const HashDigest& rhs) {
  for(int i=0; i<16; i++) {
    if(lhs.digest[i] != rhs.digest[i]) {
      return false;
    }
  }
  return true;
}

//enum NodeType {
//  Distinct=0,
//  Repeat=1,
//  Identical=2,
//  Other=3
//};

const uint32_t Distinct = 0;
const uint32_t Repeat = 1;
const uint32_t Identical = 2;
const uint32_t Other = 3;

struct Node {
  uint32_t node;
  uint32_t tree;
  uint32_t nodetype;

  KOKKOS_INLINE_FUNCTION
  Node() {
    node = UINT_MAX;
    tree = UINT_MAX;
    nodetype = Other;
  }
 
  KOKKOS_INLINE_FUNCTION
  Node(uint32_t n, uint32_t t, uint32_t node_type) {
    node = n;
    tree = t;
    nodetype = node_type;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const Node& other) const {
    return !(other.node != node || other.tree != tree || other.nodetype != nodetype);
  }
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
  uint32_t curr_node;
  uint32_t prev_node;

  KOKKOS_INLINE_FUNCTION
  CompactNodeInfo(uint32_t n, uint32_t s) {
    curr_node = n;
    prev_node = s;
  }

  KOKKOS_INLINE_FUNCTION
  CompactNodeInfo() {
    curr_node = UINT_MAX;
    prev_node = UINT_MAX;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const CompactNodeInfo &other) const {
    if(other.curr_node != curr_node || other.prev_node != prev_node)
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

using SharedNodeMap = Kokkos::UnorderedMap<uint32_t, uint32_t>;
using SharedHostNodeMap = Kokkos::UnorderedMap<uint32_t, uint32_t, Kokkos::DefaultHostExecutionSpace>;
using DistinctNodeMap = Kokkos::UnorderedMap<HashDigest, 
                                         uint32_t, 
                                         Kokkos::DefaultExecutionSpace, 
                                         digest_hash, 
                                         digest_equal_to>;
using DistinctHostNodeMap = Kokkos::UnorderedMap<HashDigest, 
                                             uint32_t, 
                                             Kokkos::DefaultHostExecutionSpace, 
                                             digest_hash, 
                                             digest_equal_to>;

using SharedNodeIDMap = Kokkos::UnorderedMap<uint32_t, NodeID>;
using SharedHostNodeIDMap = Kokkos::UnorderedMap<uint32_t, NodeID, Kokkos::DefaultHostExecutionSpace>;
using DistinctNodeIDMap = Kokkos::UnorderedMap<HashDigest, 
                                               NodeID, 
                                               Kokkos::DefaultExecutionSpace, 
                                               digest_hash, 
                                               digest_equal_to>;
using DistinctHostNodeIDMap = Kokkos::UnorderedMap<HashDigest, 
                                                   NodeID, 
                                                   Kokkos::DefaultHostExecutionSpace, 
                                                   digest_hash, 
                                                   digest_equal_to>;

using CompactTable = Kokkos::UnorderedMap<uint32_t, NodeID, Kokkos::DefaultExecutionSpace>;
using CompactHostTable = Kokkos::UnorderedMap<uint32_t, NodeID, Kokkos::DefaultHostExecutionSpace>;

using NodeMap = Kokkos::UnorderedMap<uint32_t, Node, Kokkos::DefaultExecutionSpace>;

using DigestNodeIDMap = DistinctNodeIDMap;
using RootNodeIDMap = Kokkos::UnorderedMap<uint32_t, NodeID, Kokkos::DefaultExecutionSpace>;

//using DigestListMap = Kokkos::UnorderedMap<HashDigest, 
//                                           Vector<uint32_t>, 
//                                           Kokkos::DefaultExecutionSpace, 
//                                           digest_hash, 
//                                           digest_equal_to>;

using DigestListMap = Kokkos::UnorderedMap<HashDigest, 
                                           uint32_t,
                                           Kokkos::DefaultExecutionSpace, 
                                           digest_hash, 
                                           digest_equal_to>;

#endif

