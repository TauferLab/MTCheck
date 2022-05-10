#ifndef KOKKOS_MAP_HELPERS_HPP
#define KOKKOS_MAP_HELPERS_HPP
#include <Kokkos_Core.hpp>
//#include <Kokkos_UnorderedMap.hpp>
#include <climits>

//template<uint32_t N> 
struct alignas(16) HashDigest {
  uint8_t digest[16];
//  uint32_t digest[5];
};

struct NodeInfo {
  uint32_t node;
  uint32_t src;
  uint32_t tree;

  NodeInfo(uint32_t n, uint32_t s, uint32_t t) {
    node = n;
    src = s;
    tree = t;
  }

  NodeInfo() {
    node = UINT_MAX;
    src = UINT_MAX;
    tree = UINT_MAX;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const NodeInfo &other) const {
    if(other.node != node || other.src != src || other.tree != tree)
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
//using DistinctMap = Kokkos::UnorderedMap<HashDigest, NodeInfo>;
//using DistinctMap = Kokkos::UnorderedMap<uint32_t, NodeInfo>;
using DistinctMap = Kokkos::UnorderedMap<HashDigest, 
                                         NodeInfo, 
                                         Kokkos::DefaultExecutionSpace, 
                                         digest_hash, 
                                         digest_equal_to>;

#endif

