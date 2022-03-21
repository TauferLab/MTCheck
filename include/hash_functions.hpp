#ifndef __HASH_FUNCTIONS_HPP
#define __HASH_FUNCTIONS_HPP

#include <cstring>
#include <cuda.h>
#include <string>

enum HashFunc {
  SHA1Hash=0,
  Murmur3Hash
};

class Hasher {
  public:
  
  Hasher() {}

  virtual std::string hash_name() = 0;

  virtual void hash(const void* data, int len, uint8_t* digest) = 0;

  virtual uint32_t digest_size() = 0;

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

  SHA1() {}

  class Digest {
  public:
    uint8_t digest[20];
    friend bool operator<(const Digest& l, const Digest& r) {
      int result = memcmp(l.digest, r.digest, 20);
      if(result < 0) {
        return true;
      } else {
        return false;
      }
    }
  };

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

  void SHA1_Transform(uint32_t state[5], const uint8_t buffer[64]) {
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
  void SHA1_Init(SHA1_CTX *context) {
    /* SHA1 initialization constants */
    context->state[0] = 0x67452301;
    context->state[1] = 0xEFCDAB89;
    context->state[2] = 0x98BADCFE;
    context->state[3] = 0x10325476;
    context->state[4] = 0xC3D2E1F0;
    context->count[0] = context->count[1] = 0;
  }

  /* Run your data through this. */
  void SHA1_Update(SHA1_CTX *context, const uint8_t *data, const size_t len) {
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
  void SHA1_Final(SHA1_CTX *context, uint8_t digest[SHA1_DIGEST_SIZE]) {
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

  uint32_t digest_size() {
    return SHA1_DIGEST_SIZE;
  }

  void hash(const void* data, int len, uint8_t* digest) {
    SHA1_CTX context;
    SHA1_Init(&context);
    SHA1_Update(&context, (const uint8_t*)(data), len);
    SHA1_Final(&context, digest);
  }
};

class Murmur3 : Hasher {
public:
  using DIGEST_TYPE = uint32_t;

  class Digest {
  public:
    uint32_t digest[1];
    friend bool operator<(const Digest& l, const Digest& r) {
      if(*l.digest < *r.digest) {
        return true;
      } else {
        return false;
      }
    }
  };

  std::string hash_name() {
    return std::string("Murmur3");
  }

  // MurmurHash3 was written by Austin Appleby, and is placed in the public
  // domain. The author hereby disclaims copyright to this source code.
  uint32_t getblock32(const uint8_t* p, int i) {
    // used to avoid aliasing error which could cause errors with
    // forced inlining
    return ((uint32_t)p[i * 4 + 0]) | ((uint32_t)p[i * 4 + 1] << 8) |
           ((uint32_t)p[i * 4 + 2] << 16) | ((uint32_t)p[i * 4 + 3] << 24);
  }
  
  uint32_t rotl32(uint32_t x, int8_t r) { return (x << r) | (x >> (32 - r)); }
  
  uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
  
    return h;
  }
  
  uint32_t MurmurHash3_x86_32(const void* key, int len, uint32_t seed) {
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
  
    return h1;
  }
  
  #if defined(__GNUC__) /* GNU C   */ || defined(__GNUG__) /* GNU C++ */ || \
      defined(__clang__)
  
  #define KOKKOS_IMPL_MAY_ALIAS __attribute__((__may_alias__))
  
  #else
  
  #define KOKKOS_IMPL_MAY_ALIAS
  
  #endif
  
  template <typename T>
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

  void hash(const void* data, int len, uint8_t* digest) {
    uint32_t hash = MurmurHash3_x86_32(data, len, 0);
    memcpy(digest, &hash, 4);
  }

  /* Size of hash digest in bytes */
  uint32_t digest_size() {
    return 4;
  }
};

using DefaultHash = SHA1;

//struct ChunkHasher {
//  size_t operator()(const uint8_t* data) const {
//    size_t hash = 0;
//    Murmur3 murmur3;
//    murmur3.hash(data, 
//  }
//};

#endif // __HASH_FUNCTIONS_HPP
