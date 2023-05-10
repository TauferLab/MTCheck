#ifndef KOKKOS_MAP_HELPERS_HPP
#define KOKKOS_MAP_HELPERS_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <climits>
#include "kokkos_vector.hpp"

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
    for(size_t i=0; i<sizeof(HashDigest); i++) {
      if(lhs.digest[i] != rhs.digest[i]) {
        return false;
      }
    }
    return true;
  }
};

KOKKOS_INLINE_FUNCTION
bool digests_same(const HashDigest& lhs, const HashDigest& rhs) {
  for(size_t i=0; i<sizeof(HashDigest); i++) {
    if(lhs.digest[i] != rhs.digest[i]) {
      return false;
    }
  }
  return true;
}

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

struct digest_hash {
  using argument_type        = HashDigest;
  using first_argument_type  = HashDigest;
  using second_argument_type = uint32_t;
  using result_type          = uint32_t;

  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t operator()(HashDigest const& digest) const {
//    uint32_t result = 0;
//    uint32_t* digest_ptr = (uint32_t*) digest.digest;
//    for(uint32_t i=0; i<sizeof(HashDigest)/sizeof(uint32_t); i++) {
//      result ^= digest_ptr[i];
//    }
    return *((uint32_t*)(digest.digest));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t operator()(HashDigest const& digest, uint32_t seed) const {
//    uint32_t result = 0;
//    uint32_t* digest_ptr = (uint32_t*) digest.digest;
//    for(uint32_t i=0; i<sizeof(HashDigest)/sizeof(uint32_t); i++) {
//      result ^= digest_ptr[i];
//    }
    return *((uint32_t*)(digest.digest));
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
    for(size_t i=0; i<sizeof(HashDigest)/4; i++) {
      if(a_ptr[i] != b_ptr[i]) {
        return false;
      }
    }
    return true;
  }
};

template<class Value, class ExecSpace>
using DigestMap = Kokkos::UnorderedMap<HashDigest, Value, ExecSpace, digest_hash, digest_equal_to>;
using DigestNodeIDDeviceMap = DigestMap<NodeID, Kokkos::DefaultExecutionSpace>;
using DigestNodeIDHostMap   = DigestMap<NodeID, Kokkos::DefaultHostExecutionSpace>;
using DigestIdxDeviceMap = DigestMap<uint32_t, Kokkos::DefaultExecutionSpace>;
using DigestIdxHostMap   = DigestMap<uint32_t, Kokkos::DefaultHostExecutionSpace>;

using IdxNodeIDDeviceMap = Kokkos::UnorderedMap<uint32_t, NodeID>;
using IdxNodeIDHostMap = Kokkos::UnorderedMap<uint32_t, NodeID, Kokkos::DefaultHostExecutionSpace>;
#endif

