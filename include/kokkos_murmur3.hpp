#ifndef __KOKKOS_MURMUR3_HPP
#define __KOKKOS_MURMUR3_HPP

#include <cstring>
#include <string>
#include "map_helpers.hpp"

namespace kokkos_murmur3 {
  // MurmurHash3 was written by Austin Appleby, and is placed in the public
  // domain. The author hereby disclaims copyright to this source code.
  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t getblock32(const uint8_t* p, int i) {
    // used to avoid aliasing error which could cause errors with
    // forced inlining
    return ((uint32_t)p[i * 4 + 0]) | ((uint32_t)p[i * 4 + 1] << 8) |
           ((uint32_t)p[i * 4 + 2] << 16) | ((uint32_t)p[i * 4 + 3] << 24);
  }
  
  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t rotl32(uint32_t x, int8_t r) { return (x << r) | (x >> (32 - r)); }
  
  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
  
    return h;
  }
  
  KOKKOS_FORCEINLINE_FUNCTION
  void MurmurHash3_x86_128(const void* key, int len, uint32_t seed, void* out) {
    const uint8_t * data = (const uint8_t*)key;
    const int nblocks = len / 16;
  
    uint32_t h1 = seed;
    uint32_t h2 = seed;
    uint32_t h3 = seed;
    uint32_t h4 = seed;
  
    const uint32_t c1 = 0x239b961b; 
    const uint32_t c2 = 0xab0e9789;
    const uint32_t c3 = 0x38b34ae5; 
    const uint32_t c4 = 0xa1e38b93;
  
    //----------
    // body
  
    const uint32_t * blocks = (const uint32_t *)(data + nblocks*16);

    for(int i = -nblocks; i; i++)
    {
      uint32_t k1 = getblock32((const uint8_t*)blocks,i*4+0);
      uint32_t k2 = getblock32((const uint8_t*)blocks,i*4+1);
      uint32_t k3 = getblock32((const uint8_t*)blocks,i*4+2);
      uint32_t k4 = getblock32((const uint8_t*)blocks,i*4+3);

      k1 *= c1; k1  = rotl32(k1,15); k1 *= c2; h1 ^= k1;

      h1 = rotl32(h1,19); h1 += h2; h1 = h1*5+0x561ccd1b;

      k2 *= c2; k2  = rotl32(k2,16); k2 *= c3; h2 ^= k2;

      h2 = rotl32(h2,17); h2 += h3; h2 = h2*5+0x0bcaa747;

      k3 *= c3; k3  = rotl32(k3,17); k3 *= c4; h3 ^= k3;

      h3 = rotl32(h3,15); h3 += h4; h3 = h3*5+0x96cd1c35;

      k4 *= c4; k4  = rotl32(k4,18); k4 *= c1; h4 ^= k4;

      h4 = rotl32(h4,13); h4 += h1; h4 = h4*5+0x32ac3b17;
    }
  
    //----------
    // tail
  
    const uint8_t * tail = (const uint8_t*)(data + nblocks*16);

    uint32_t k1 = 0;
    uint32_t k2 = 0;
    uint32_t k3 = 0;
    uint32_t k4 = 0;
  
    switch(len & 15)
    {
    case 15: k4 ^= tail[14] << 16;
    case 14: k4 ^= tail[13] << 8;
    case 13: k4 ^= tail[12] << 0;
             k4 *= c4; k4  = rotl32(k4,18); k4 *= c1; h4 ^= k4;
  
    case 12: k3 ^= tail[11] << 24;
    case 11: k3 ^= tail[10] << 16;
    case 10: k3 ^= tail[ 9] << 8;
    case  9: k3 ^= tail[ 8] << 0;
             k3 *= c3; k3  = rotl32(k3,17); k3 *= c4; h3 ^= k3;
  
    case  8: k2 ^= tail[ 7] << 24;
    case  7: k2 ^= tail[ 6] << 16;
    case  6: k2 ^= tail[ 5] << 8;
    case  5: k2 ^= tail[ 4] << 0;
             k2 *= c2; k2  = rotl32(k2,16); k2 *= c3; h2 ^= k2;
  
    case  4: k1 ^= tail[ 3] << 24;
    case  3: k1 ^= tail[ 2] << 16;
    case  2: k1 ^= tail[ 1] << 8;
    case  1: k1 ^= tail[ 0] << 0;
             k1 *= c1; k1  = rotl32(k1,15); k1 *= c2; h1 ^= k1;
    };

  
    //----------
    // finalization
  
    h1 ^= len; h2 ^= len; h3 ^= len; h4 ^= len;

    h1 += h2; h1 += h3; h1 += h4;
    h2 += h1; h3 += h1; h4 += h1;
  
    h1 = fmix32(h1);
    h2 = fmix32(h2);
    h3 = fmix32(h3);
    h4 = fmix32(h4);
  
    h1 += h2; h1 += h3; h1 += h4;
    h2 += h1; h3 += h1; h4 += h1;
  
    ((uint32_t*)out)[0] = h1;
    ((uint32_t*)out)[1] = h2;
    ((uint32_t*)out)[2] = h3;
    ((uint32_t*)out)[3] = h4;

    return;
  }
  
  #if defined(__GNUC__) /* GNU C   */ || defined(__GNUG__) /* GNU C++ */ || \
      defined(__clang__)
  
  #define KOKKOS_IMPL_MAY_ALIAS __attribute__((__may_alias__))
  
  #else
  
  #define KOKKOS_IMPL_MAY_ALIAS
  
  #endif
  
  template <typename T>
  KOKKOS_FORCEINLINE_FUNCTION
  bool bitwise_equal(T const* const a_ptr,
                                                 T const* const b_ptr) {
    typedef uint64_t KOKKOS_IMPL_MAY_ALIAS T64;  // NOLINT(modernize-use-using)
    typedef uint32_t KOKKOS_IMPL_MAY_ALIAS T32;  // NOLINT(modernize-use-using)
    typedef uint16_t KOKKOS_IMPL_MAY_ALIAS T16;  // NOLINT(modernize-use-using)
    typedef uint8_t KOKKOS_IMPL_MAY_ALIAS T8;    // NOLINT(modernize-use-using)
  
    enum {
      NUM_8  = sizeof(T),
      NUM_16 = NUM_8 / 2,
      NUM_32 = NUM_8 / 4,
      NUM_64 = NUM_8 / 8
    };
  
    union {
      T const* const ptr;
      T64 const* const ptr64;
      T32 const* const ptr32;
      T16 const* const ptr16;
      T8 const* const ptr8;
    } a = {a_ptr}, b = {b_ptr};
  
    bool result = true;
  
    for (int i = 0; i < NUM_64; ++i) {
      result = result && a.ptr64[i] == b.ptr64[i];
    }
  
    if (NUM_64 * 2 < NUM_32) {
      result = result && a.ptr32[NUM_64 * 2] == b.ptr32[NUM_64 * 2];
    }
  
    if (NUM_32 * 2 < NUM_16) {
      result = result && a.ptr16[NUM_32 * 2] == b.ptr16[NUM_32 * 2];
    }
  
    if (NUM_16 * 2 < NUM_8) {
      result = result && a.ptr8[NUM_16 * 2] == b.ptr8[NUM_16 * 2];
    }
  
    return result;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void hash(const void* data, int len, uint8_t* digest)  {
//    uint64_t temp[2];
//    MurmurHash3_x86_128(data, len, 0, temp);
//    memcpy(digest, temp, 16);
    MurmurHash3_x86_128(data, len, 0, digest);
  }
}

#endif // KOKKOS_MURMUR3

