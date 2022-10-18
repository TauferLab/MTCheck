#ifndef __HASH_FUNCTIONS_HPP
#define __HASH_FUNCTIONS_HPP

#include <cstring>
#include <string>
#include <openssl/md5.h>
#include "map_helpers.hpp"

void calc_and_print_md5(Kokkos::View<uint8_t*>& data_d);
//void calc_and_print_md5(Kokkos::View<uint8_t*>& data_d) {
//  HashDigest correct;
//  auto data_h = Kokkos::create_mirror_view(data_d);
//  Kokkos::deep_copy(data_h, data_d);
//  MD5((uint8_t*)(data_h.data()), data_d.size(), correct.digest);
//  static const char hexchars[] = "0123456789ABCDEF";
//  std::string ref_digest;
//  for(int k=0; k<16; k++) {
//    unsigned char b = correct.digest[k];
//    char hex[3];
//    hex[0] = hexchars[b >> 4];
//    hex[1] = hexchars[b & 0xF];
//    hex[2] = 0;
//    ref_digest.append(hex);
//    if(k%4 == 3)
//      ref_digest.append(" ");
//  }
//  std::cout << "Reference digest:  " << ref_digest << std::endl;
//}

enum HashFunc {
  SHA1Hash=0,
  Murmur3Hash
};

class Hasher {
  public:
  
  KOKKOS_FORCEINLINE_FUNCTION
  Hasher() {}

  virtual std::string hash_name() = 0;

  KOKKOS_FORCEINLINE_FUNCTION
  virtual void hash(const void* data, int len, uint8_t* digest) const = 0;

  KOKKOS_FORCEINLINE_FUNCTION
  virtual uint32_t digest_size() const = 0;

  KOKKOS_FORCEINLINE_FUNCTION
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
};

class SHA1: public Hasher {
public:
  using DIGEST_TYPE = uint8_t;
  static constexpr uint32_t DIGEST_SIZE = 20;

  KOKKOS_FORCEINLINE_FUNCTION
  SHA1() {}

  std::string hash_name() {
    return std::string("SHA1");
  }

  struct SHA1_CTX {
    uint32_t state[5];
    uint32_t count[2];
    uint8_t buffer[64];
  };

  #define SHA1_DIGEST_SIZE 20

  #define rol(value, bits) (((value) << (bits)) | ((value) >> (32 - (bits))))

  /* blk0() and blk() perform the initial expand. */
  /* I got the idea of expanding during the round function from SSleay */
  #define blk0(i)                                         \
    (block->l[i] = (rol(block->l[i], 24) & 0xFF00FF00) |  \
                   (rol(block->l[i], 8) & 0x00FF00FF))    

  #define blk(i)                                                            \
    (block->l[i & 15] = rol(block->l[(i+13) & 15] ^ block->l[(i+8) & 15] ^  \
                        block->l[(i+2) & 15] ^ block->l[i & 15], 1))

  /* (R0+R1), R2, R3, R4 are the different operations used in SHA1 */
  #define R0(v, w, x, y, z, i)                                                   \
    z += ((w & (x ^ y)) ^ y) + blk0(i) + 0x5A827999 + rol(v, 5);                 \
    w = rol(w, 30);
  #define R1(v, w, x, y, z, i)                                                   \
    z += ((w & (x ^ y)) ^ y) + blk(i) + 0x5A827999 + rol(v, 5);                  \
    w = rol(w, 30);
  #define R2(v, w, x, y, z, i)                                                   \
    z += (w ^ x ^ y) + blk(i) + 0x6ED9EBA1 + rol(v, 5);                          \
    w = rol(w, 30);
  #define R3(v, w, x, y, z, i)                                                   \
    z += (((w | x) & y) | (w & x)) + blk(i) + 0x8F1BBCDC + rol(v, 5);            \
    w = rol(w, 30);
  #define R4(v, w, x, y, z, i)                                                   \
    z += (w ^ x ^ y) + blk(i) + 0xCA62C1D6 + rol(v, 5);                          \
    w = rol(w, 30);

  KOKKOS_FORCEINLINE_FUNCTION
  void SHA1_Transform(uint32_t state[5], const uint8_t buffer[64]) const {
    uint32_t a, b, c, d, e;

    typedef union {
      uint8_t c[64];
      uint32_t l[16];
    } CHAR64LONG16;
    CHAR64LONG16 block[1];
    memcpy(block, buffer, 64);
    
    /* Copy context state to working vars */
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];

    /* 4 rounds of 20 operations each. Loop unrolled. */
    R0(a, b, c, d, e, 0);
    R0(e, a, b, c, d, 1);
    R0(d, e, a, b, c, 2);
    R0(c, d, e, a, b, 3);
    R0(b, c, d, e, a, 4);
    R0(a, b, c, d, e, 5);
    R0(e, a, b, c, d, 6);
    R0(d, e, a, b, c, 7);
    R0(c, d, e, a, b, 8);
    R0(b, c, d, e, a, 9);
    R0(a, b, c, d, e, 10);
    R0(e, a, b, c, d, 11);
    R0(d, e, a, b, c, 12);
    R0(c, d, e, a, b, 13);
    R0(b, c, d, e, a, 14);
    R0(a, b, c, d, e, 15);
    R1(e, a, b, c, d, 16);
    R1(d, e, a, b, c, 17);
    R1(c, d, e, a, b, 18);
    R1(b, c, d, e, a, 19);
    R2(a, b, c, d, e, 20);
    R2(e, a, b, c, d, 21);
    R2(d, e, a, b, c, 22);
    R2(c, d, e, a, b, 23);
    R2(b, c, d, e, a, 24);
    R2(a, b, c, d, e, 25);
    R2(e, a, b, c, d, 26);
    R2(d, e, a, b, c, 27);
    R2(c, d, e, a, b, 28);
    R2(b, c, d, e, a, 29);
    R2(a, b, c, d, e, 30);
    R2(e, a, b, c, d, 31);
    R2(d, e, a, b, c, 32);
    R2(c, d, e, a, b, 33);
    R2(b, c, d, e, a, 34);
    R2(a, b, c, d, e, 35);
    R2(e, a, b, c, d, 36);
    R2(d, e, a, b, c, 37);
    R2(c, d, e, a, b, 38);
    R2(b, c, d, e, a, 39);
    R3(a, b, c, d, e, 40);
    R3(e, a, b, c, d, 41);
    R3(d, e, a, b, c, 42);
    R3(c, d, e, a, b, 43);
    R3(b, c, d, e, a, 44);
    R3(a, b, c, d, e, 45);
    R3(e, a, b, c, d, 46);
    R3(d, e, a, b, c, 47);
    R3(c, d, e, a, b, 48);
    R3(b, c, d, e, a, 49);
    R3(a, b, c, d, e, 50);
    R3(e, a, b, c, d, 51);
    R3(d, e, a, b, c, 52);
    R3(c, d, e, a, b, 53);
    R3(b, c, d, e, a, 54);
    R3(a, b, c, d, e, 55);
    R3(e, a, b, c, d, 56);
    R3(d, e, a, b, c, 57);
    R3(c, d, e, a, b, 58);
    R3(b, c, d, e, a, 59);
    R4(a, b, c, d, e, 60);
    R4(e, a, b, c, d, 61);
    R4(d, e, a, b, c, 62);
    R4(c, d, e, a, b, 63);
    R4(b, c, d, e, a, 64);
    R4(a, b, c, d, e, 65);
    R4(e, a, b, c, d, 66);
    R4(d, e, a, b, c, 67);
    R4(c, d, e, a, b, 68);
    R4(b, c, d, e, a, 69);
    R4(a, b, c, d, e, 70);
    R4(e, a, b, c, d, 71);
    R4(d, e, a, b, c, 72);
    R4(c, d, e, a, b, 73);
    R4(b, c, d, e, a, 74);
    R4(a, b, c, d, e, 75);
    R4(e, a, b, c, d, 76);
    R4(d, e, a, b, c, 77);
    R4(c, d, e, a, b, 78);
    R4(b, c, d, e, a, 79);

    /* Add the working vars back into the context.state[] */
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    /* Wipe variables */
    a = b = c = d = e = 0;
    memset(block, '\0', sizeof(block));
  }
    
  /* SHA1_Init - Initialize new context */
  KOKKOS_FORCEINLINE_FUNCTION
  void SHA1_Init(SHA1_CTX *context) const {
    /* SHA1 initialization constants */
    context->state[0] = 0x67452301;
    context->state[1] = 0xEFCDAB89;
    context->state[2] = 0x98BADCFE;
    context->state[3] = 0x10325476;
    context->state[4] = 0xC3D2E1F0;
    context->count[0] = context->count[1] = 0;
  }

  /* Run your data through this. */
  KOKKOS_FORCEINLINE_FUNCTION
  void SHA1_Update(SHA1_CTX *context, const uint8_t *data, const size_t len) const {
    size_t i, j;
  
    j = context->count[0];
    if ((context->count[0] += len << 3) < j)
      context->count[1]++;
    context->count[1] += (len >> 29);
    j = (j >> 3) & 63;
    if ((j + len) > 63) {
      memcpy(&context->buffer[j], data, (i = 64 - j));
      SHA1_Transform(context->state, context->buffer);
      for (; i + 63 < len; i += 64) {
        SHA1_Transform(context->state, &data[i]);
      }
      j = 0;
    } else
      i = 0;
    memcpy(&context->buffer[j], &data[i], len - i);
  }

  /* Add padding and return the message digest. */
  KOKKOS_FORCEINLINE_FUNCTION
  void SHA1_Final(SHA1_CTX *context, uint8_t digest[SHA1_DIGEST_SIZE]) const {
    unsigned i;
    uint8_t finalcount[8];
    uint8_t c;
  
    for (i = 0; i < 8; i++) {
      finalcount[i] =
          /* Endian independent */
          (uint8_t)(context->count[(i >= 4 ? 0 : 1)] >> ((3 - (i & 3)) * 8));
    }
    c = 0200;
    SHA1_Update(context, &c, 1);
    while ((context->count[0] & 504) != 448) {
      c = 0000;
      SHA1_Update(context, &c, 1);
    }
    SHA1_Update(context, finalcount, 8); /* Should cause a SHA1_Transform() */
    for (i = 0; i < 20; i++) {
      digest[i] = (uint8_t)(context->state[i >> 2] >> ((3 - (i & 3)) * 8));
    }
    /* Wipe variables */
    memset(context, '\0', sizeof(*context));
    memset(&finalcount, '\0', sizeof(finalcount));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void digest_to_hex(const uint8_t digest[SHA1_DIGEST_SIZE], char* output) {
    int i,j;
    char* c = output;
    for(i=0; i<SHA1_DIGEST_SIZE/4; i++) {
      for(j=0; j<4; j++) {
        sprintf(c, "%02X", digest[i*4 + j]);
        c += 2;
      }
      sprintf(c, " ");
      c += 1;
    }
    *(c-1) = '\0';
  }

  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t full_digest_size() const {
    return SHA1_DIGEST_SIZE;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t digest_size() const {
    return SHA1_DIGEST_SIZE-4;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void full_hash(const void* data, int len, uint8_t* digest) const {
    SHA1_CTX context;
    SHA1_Init(&context);
    SHA1_Update(&context, (const uint8_t*)(data), len);
    SHA1_Final(&context, digest);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void hash(const void* data, int len, uint8_t* digest) const {
    uint8_t temp_digest[20];
    SHA1_CTX context;
    SHA1_Init(&context);
    SHA1_Update(&context, (const uint8_t*)(data), len);
    SHA1_Final(&context, temp_digest);
    memcpy(digest, temp_digest, 16);
  }
};

class Murmur3A : Hasher {
public:
  using DIGEST_TYPE = uint32_t;

  std::string hash_name() {
    return std::string("Murmur3");
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Murmur3A() {}

  // MurmurHash3 was written by Austin Appleby, and is placed in the public
  // domain. The author hereby disclaims copyright to this source code.
  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t getblock32(const uint8_t* p, int i) const {
    // used to avoid aliasing error which could cause errors with
    // forced inlining
    return ((uint32_t)p[i * 4 + 0]) | ((uint32_t)p[i * 4 + 1] << 8) |
           ((uint32_t)p[i * 4 + 2] << 16) | ((uint32_t)p[i * 4 + 3] << 24);
  }
  
  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t rotl32(uint32_t x, int8_t r) const { return (x << r) | (x >> (32 - r)); }
  
  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t fmix32(uint32_t h) const {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
  
    return h;
  }
  
  KOKKOS_FORCEINLINE_FUNCTION
  void MurmurHash3_x86_32(const void* key, int len, uint32_t seed, void* out) const {
    const uint8_t* data = static_cast<const uint8_t*>(key);
    const int nblocks   = len / 4;
  
    uint32_t h1 = seed;
  
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
  
    //----------
    // body
  
    for (int i = 0; i < nblocks; ++i) {
      uint32_t k1 = getblock32(data, i);
  
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
  
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }
  
    //----------
    // tail
  
    const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
  
    uint32_t k1 = 0;
  
    switch (len & 3) {
      case 3: k1 ^= tail[2] << 16;
      case 2: k1 ^= tail[1] << 8;
      case 1:
        k1 ^= tail[0];
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
    };
  
    //----------
    // finalization
  
    h1 ^= len;
  
    h1 = fmix32(h1);
  
    *(uint32_t*)out = h1;
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
                                                 T const* const b_ptr) const {
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
  void hash(const void* data, int len, uint8_t* digest) const {
    MurmurHash3_x86_32(data, len, 0, digest);
  }

  /* Size of hash digest in bytes */
  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t digest_size() const {
    return 16;
  }
};

class Murmur3C : Hasher {
public:
  using DIGEST_TYPE = uint32_t;

  KOKKOS_FORCEINLINE_FUNCTION
  Murmur3C() {}

  std::string hash_name() {
    return std::string("Murmur3");
  }

  // MurmurHash3 was written by Austin Appleby, and is placed in the public
  // domain. The author hereby disclaims copyright to this source code.
  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t getblock32(const uint8_t* p, int i) const {
    // used to avoid aliasing error which could cause errors with
    // forced inlining
    return ((uint32_t)p[i * 4 + 0]) | ((uint32_t)p[i * 4 + 1] << 8) |
           ((uint32_t)p[i * 4 + 2] << 16) | ((uint32_t)p[i * 4 + 3] << 24);
  }
  
  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t rotl32(uint32_t x, int8_t r) const { return (x << r) | (x >> (32 - r)); }
  
  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t fmix32(uint32_t h) const {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
  
    return h;
  }
  
  KOKKOS_FORCEINLINE_FUNCTION
  void MurmurHash3_x86_128(const void* key, int len, uint32_t seed, void* out) const {
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
                                                 T const* const b_ptr) const {
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
  void hash(const void* data, int len, uint8_t* digest) const {
    MurmurHash3_x86_128(data, len, 0, digest);
  }

  /* Size of hash digest in bytes */
  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t digest_size() const {
    return 16;
  }
};

class Murmur3F : Hasher {
public:
  using DIGEST_TYPE = uint32_t;

  std::string hash_name() {
    return std::string("Murmur3F");
  }

  // MurmurHash3 was written by Austin Appleby, and is placed in the public
  // domain. The author hereby disclaims copyright to this source code.

  KOKKOS_FORCEINLINE_FUNCTION
  uint64_t getblock64(const uint8_t* p, int i) const {
    // used to avoid aliasing error which could cause errors with
    // forced inlining
    return ((uint64_t*)p)[i];
//    return ((uint64_t)p[i*8 + 0]) | ((uint64_t)p[i * 8 + 1] << 8) | ((uint64_t)p[i * 8 + 2] << 16) | ((uint64_t)p[i * 8 + 3] << 24) |
//           ((uint64_t)p[i*8 + 4] << 32) | ((uint64_t)p[i * 8 + 5] << 40) | ((uint64_t)p[i * 8 + 6] << 48) | ((uint64_t)p[i * 8 + 7] << 56);
  }
  
  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t rotl64(uint64_t x, int8_t r) const { return (x << r) | (x >> (64 - r)); }

  KOKKOS_FORCEINLINE_FUNCTION
  uint64_t fmix64( uint64_t k ) const {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdLLU;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53LLU;
    k ^= k >> 33;
    return k;
  }
  
  KOKKOS_FORCEINLINE_FUNCTION
  void MurmurHash3_x64_128(const void* key, int len, uint32_t seed, void* out) const {
    const uint8_t * data = (const uint8_t*)key;
    const int nblocks = len / 16;
  
    uint32_t h1 = seed;
    uint32_t h2 = seed;
  
    const uint64_t c1 = 0x87c37b91114253d5LLU;
    const uint64_t c2 = 0x4cf5ad432745937fLLU;
  
    //----------
    // body
  
    const uint64_t * blocks = (const uint64_t *)(data);

    for(int i = 0; i < nblocks; i++)
    {
      uint64_t k1 = getblock64((const uint8_t*)blocks,i*2+0);
      uint64_t k2 = getblock64((const uint8_t*)blocks,i*2+1);

      k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;

      h1 = rotl64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;

      k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;

      h2 = rotl64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
    }
  
    //----------
    // tail
  
    const uint8_t * tail = (const uint8_t*)(data + nblocks*16);
  
    uint64_t k1 = 0;
    uint64_t k2 = 0;
  
    switch(len & 15)
    {
      case 15: k2 ^= ((uint64_t)tail[14]) << 48;
      case 14: k2 ^= ((uint64_t)tail[13]) << 40;
      case 13: k2 ^= ((uint64_t)tail[12]) << 32;
      case 12: k2 ^= ((uint64_t)tail[11]) << 24;
      case 11: k2 ^= ((uint64_t)tail[10]) << 16;
      case 10: k2 ^= ((uint64_t)tail[ 9]) << 8;
      case  9: k2 ^= ((uint64_t)tail[ 8]) << 0;
               k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;
  
      case  8: k1 ^= ((uint64_t)tail[ 7]) << 56;
      case  7: k1 ^= ((uint64_t)tail[ 6]) << 48;
      case  6: k1 ^= ((uint64_t)tail[ 5]) << 40;
      case  5: k1 ^= ((uint64_t)tail[ 4]) << 32;
      case  4: k1 ^= ((uint64_t)tail[ 3]) << 24;
      case  3: k1 ^= ((uint64_t)tail[ 2]) << 16;
      case  2: k1 ^= ((uint64_t)tail[ 1]) << 8;
      case  1: k1 ^= ((uint64_t)tail[ 0]) << 0;
               k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;
    };
  
    //----------
    // finalization
  
    h1 ^= len; h2 ^= len;

    h1 += h2;
    h2 += h1;
  
    h1 = fmix64(h1);
    h2 = fmix64(h2);
  
    h1 += h2;
    h2 += h1;
  
    ((uint64_t*)out)[0] = h1;
    ((uint64_t*)out)[1] = h2;

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
                                                 T const* const b_ptr) const {
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
  void hash(const void* data, int len, uint8_t* digest) const {
    uint64_t temp[2];
    MurmurHash3_x64_128(data, len, 0, temp);
    memcpy(digest, temp, 16);
  }

  /* Size of hash digest in bytes */
  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t digest_size() const {
    return 16;
  }
};

class MD5Hash: public Hasher {
public:
  using DIGEST_TYPE = uint8_t;
  static constexpr uint32_t DIGEST_SIZE = 16;

  KOKKOS_FORCEINLINE_FUNCTION
  MD5Hash() {}

  std::string hash_name() {
    return std::string("MD5");
  }

  struct md5_context {
    uint32_t total[2];
    uint32_t state[4];
    uint8_t buffer[64];
    uint8_t ipad[64];
    uint8_t opad[64];
  };

  #define MD5_DIGEST_SIZE 16

  /*
   * 32-bit integer manipulation macros (little endian)
   */
  #ifndef GET_ULONG_LE
  #define GET_ULONG_LE(n,b,i)                             \
  {                                                       \
      (n) = ( (unsigned long) (b)[(i)    ]       )        \
          | ( (unsigned long) (b)[(i) + 1] <<  8 )        \
          | ( (unsigned long) (b)[(i) + 2] << 16 )        \
          | ( (unsigned long) (b)[(i) + 3] << 24 );       \
  }
  #endif
  
  #ifndef PUT_ULONG_LE
  #define PUT_ULONG_LE(n,b,i)                             \
  {                                                       \
      (b)[(i)    ] = (unsigned char) ( (n)       );       \
      (b)[(i) + 1] = (unsigned char) ( (n) >>  8 );       \
      (b)[(i) + 2] = (unsigned char) ( (n) >> 16 );       \
      (b)[(i) + 3] = (unsigned char) ( (n) >> 24 );       \
  }
  #endif

  /*
   * MD5 context setup
   */
  KOKKOS_FORCEINLINE_FUNCTION
  void md5_starts( md5_context *ctx ) const
  {
      ctx->total[0] = 0;
      ctx->total[1] = 0;
  
      ctx->state[0] = 0x67452301;
      ctx->state[1] = 0xEFCDAB89;
      ctx->state[2] = 0x98BADCFE;
      ctx->state[3] = 0x10325476;
  }
  
  KOKKOS_FORCEINLINE_FUNCTION
  void md5_process( md5_context *ctx, unsigned char data[64] ) const
  {
      unsigned long X[16], A, B, C, D;
  
      GET_ULONG_LE( X[ 0], data,  0 );
      GET_ULONG_LE( X[ 1], data,  4 );
      GET_ULONG_LE( X[ 2], data,  8 );
      GET_ULONG_LE( X[ 3], data, 12 );
      GET_ULONG_LE( X[ 4], data, 16 );
      GET_ULONG_LE( X[ 5], data, 20 );
      GET_ULONG_LE( X[ 6], data, 24 );
      GET_ULONG_LE( X[ 7], data, 28 );
      GET_ULONG_LE( X[ 8], data, 32 );
      GET_ULONG_LE( X[ 9], data, 36 );
      GET_ULONG_LE( X[10], data, 40 );
      GET_ULONG_LE( X[11], data, 44 );
      GET_ULONG_LE( X[12], data, 48 );
      GET_ULONG_LE( X[13], data, 52 );
      GET_ULONG_LE( X[14], data, 56 );
      GET_ULONG_LE( X[15], data, 60 );
  
  #define S(x,n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))
  
  #define P(a,b,c,d,k,s,t)                                \
  {                                                       \
      a += F(b,c,d) + X[k] + t; a = S(a,s) + b;           \
  }
  
      A = ctx->state[0];
      B = ctx->state[1];
      C = ctx->state[2];
      D = ctx->state[3];
  
  #define F(x,y,z) (z ^ (x & (y ^ z)))
  
      P( A, B, C, D,  0,  7, 0xD76AA478 );
      P( D, A, B, C,  1, 12, 0xE8C7B756 );
      P( C, D, A, B,  2, 17, 0x242070DB );
      P( B, C, D, A,  3, 22, 0xC1BDCEEE );
      P( A, B, C, D,  4,  7, 0xF57C0FAF );
      P( D, A, B, C,  5, 12, 0x4787C62A );
      P( C, D, A, B,  6, 17, 0xA8304613 );
      P( B, C, D, A,  7, 22, 0xFD469501 );
      P( A, B, C, D,  8,  7, 0x698098D8 );
      P( D, A, B, C,  9, 12, 0x8B44F7AF );
      P( C, D, A, B, 10, 17, 0xFFFF5BB1 );
      P( B, C, D, A, 11, 22, 0x895CD7BE );
      P( A, B, C, D, 12,  7, 0x6B901122 );
      P( D, A, B, C, 13, 12, 0xFD987193 );
      P( C, D, A, B, 14, 17, 0xA679438E );
      P( B, C, D, A, 15, 22, 0x49B40821 );
  
  #undef F
  
  #define F(x,y,z) (y ^ (z & (x ^ y)))
  
      P( A, B, C, D,  1,  5, 0xF61E2562 );
      P( D, A, B, C,  6,  9, 0xC040B340 );
      P( C, D, A, B, 11, 14, 0x265E5A51 );
      P( B, C, D, A,  0, 20, 0xE9B6C7AA );
      P( A, B, C, D,  5,  5, 0xD62F105D );
      P( D, A, B, C, 10,  9, 0x02441453 );
      P( C, D, A, B, 15, 14, 0xD8A1E681 );
      P( B, C, D, A,  4, 20, 0xE7D3FBC8 );
      P( A, B, C, D,  9,  5, 0x21E1CDE6 );
      P( D, A, B, C, 14,  9, 0xC33707D6 );
      P( C, D, A, B,  3, 14, 0xF4D50D87 );
      P( B, C, D, A,  8, 20, 0x455A14ED );
      P( A, B, C, D, 13,  5, 0xA9E3E905 );
      P( D, A, B, C,  2,  9, 0xFCEFA3F8 );
      P( C, D, A, B,  7, 14, 0x676F02D9 );
      P( B, C, D, A, 12, 20, 0x8D2A4C8A );
  
  #undef F
      
  #define F(x,y,z) (x ^ y ^ z)
  
      P( A, B, C, D,  5,  4, 0xFFFA3942 );
      P( D, A, B, C,  8, 11, 0x8771F681 );
      P( C, D, A, B, 11, 16, 0x6D9D6122 );
      P( B, C, D, A, 14, 23, 0xFDE5380C );
      P( A, B, C, D,  1,  4, 0xA4BEEA44 );
      P( D, A, B, C,  4, 11, 0x4BDECFA9 );
      P( C, D, A, B,  7, 16, 0xF6BB4B60 );
      P( B, C, D, A, 10, 23, 0xBEBFBC70 );
      P( A, B, C, D, 13,  4, 0x289B7EC6 );
      P( D, A, B, C,  0, 11, 0xEAA127FA );
      P( C, D, A, B,  3, 16, 0xD4EF3085 );
      P( B, C, D, A,  6, 23, 0x04881D05 );
      P( A, B, C, D,  9,  4, 0xD9D4D039 );
      P( D, A, B, C, 12, 11, 0xE6DB99E5 );
      P( C, D, A, B, 15, 16, 0x1FA27CF8 );
      P( B, C, D, A,  2, 23, 0xC4AC5665 );
  
  #undef F
  
  #define F(x,y,z) (y ^ (x | ~z))
  
      P( A, B, C, D,  0,  6, 0xF4292244 );
      P( D, A, B, C,  7, 10, 0x432AFF97 );
      P( C, D, A, B, 14, 15, 0xAB9423A7 );
      P( B, C, D, A,  5, 21, 0xFC93A039 );
      P( A, B, C, D, 12,  6, 0x655B59C3 );
      P( D, A, B, C,  3, 10, 0x8F0CCC92 );
      P( C, D, A, B, 10, 15, 0xFFEFF47D );
      P( B, C, D, A,  1, 21, 0x85845DD1 );
      P( A, B, C, D,  8,  6, 0x6FA87E4F );
      P( D, A, B, C, 15, 10, 0xFE2CE6E0 );
      P( C, D, A, B,  6, 15, 0xA3014314 );
      P( B, C, D, A, 13, 21, 0x4E0811A1 );
      P( A, B, C, D,  4,  6, 0xF7537E82 );
      P( D, A, B, C, 11, 10, 0xBD3AF235 );
      P( C, D, A, B,  2, 15, 0x2AD7D2BB );
      P( B, C, D, A,  9, 21, 0xEB86D391 );
  
  #undef F
  
      ctx->state[0] += A;
      ctx->state[1] += B;
      ctx->state[2] += C;
      ctx->state[3] += D;
  }
  
  /*
   * MD5 process buffer
   */
  KOKKOS_FORCEINLINE_FUNCTION
  void md5_update( md5_context *ctx, unsigned char *input, int ilen ) const
  {
      int fill;
      unsigned long left;
  
      if( ilen <= 0 )
          return;
  
      left = ctx->total[0] & 0x3F;
      fill = 64 - left;
  
      ctx->total[0] += ilen;
      ctx->total[0] &= 0xFFFFFFFF;
  
      if( ctx->total[0] < (unsigned long) ilen )
          ctx->total[1]++;
  
      if( left && ilen >= fill )
      {
          memcpy( (void *) (ctx->buffer + left),
                  (void *) input, fill );
          md5_process( ctx, ctx->buffer );
          input += fill;
          ilen  -= fill;
          left = 0;
      }
  
      while( ilen >= 64 )
      {
          md5_process( ctx, input );
          input += 64;
          ilen  -= 64;
      }
  
      if( ilen > 0 )
      {
          memcpy( (void *) (ctx->buffer + left),
                  (void *) input, ilen );
      }
  }
  
  /*
   * MD5 final digest
   */
  KOKKOS_FORCEINLINE_FUNCTION
  void md5_finish( md5_context *ctx, unsigned char output[16] ) const
  {
      unsigned long last, padn;
      unsigned long high, low;
      unsigned char msglen[8];
  
      high = ( ctx->total[0] >> 29 )
           | ( ctx->total[1] <<  3 );
      low  = ( ctx->total[0] <<  3 );
  
      PUT_ULONG_LE( low,  msglen, 0 );
      PUT_ULONG_LE( high, msglen, 4 );
  
      last = ctx->total[0] & 0x3F;
      padn = ( last < 56 ) ? ( 56 - last ) : ( 120 - last );

      const unsigned char md5_padding[64] =
      {
       0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
      };
  
  
      md5_update( ctx, (unsigned char *) md5_padding, padn );
      md5_update( ctx, msglen, 8 );
  
      PUT_ULONG_LE( ctx->state[0], output,  0 );
      PUT_ULONG_LE( ctx->state[1], output,  4 );
      PUT_ULONG_LE( ctx->state[2], output,  8 );
      PUT_ULONG_LE( ctx->state[3], output, 12 );
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void digest_to_hex(const uint8_t digest[MD5_DIGEST_SIZE], char* output) const {
    int i,j;
    char* c = output;
    for(i=0; i<MD5_DIGEST_SIZE/4; i++) {
      for(j=0; j<4; j++) {
        sprintf(c, "%02X", digest[i*4 + j]);
        c += 2;
      }
      sprintf(c, " ");
      c += 1;
    }
    *(c-1) = '\0';
  }

  KOKKOS_FORCEINLINE_FUNCTION
  uint32_t digest_size() const {
    return 16;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void hash(const void* data, int len, uint8_t* digest) const {
    #ifdef __CUDA_ARCH__
      md5_context md5_ctx;
      md5_starts(&md5_ctx);
      md5_ctx.state[0] ^= 0;
      md5_update( &md5_ctx, (uint8_t*)(data), len);
      md5_finish( &md5_ctx, digest);
    #else
      MD5((uint8_t*)(data), len, digest);
    #endif
  }
};


using DefaultHash = SHA1;


#endif // __HASH_FUNCTIONS_HPP
