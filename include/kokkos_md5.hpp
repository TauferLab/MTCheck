#ifndef __KOKKOS_MD5_HPP
#define __KOKKOS_MD5_HPP

#include <cstring>
#include <string>
#include <openssl/md5.h>
#include "map_helpers.hpp"

namespace kokkos_md5 {
  // "Derived from the RSA Data Security, Inc. MD5 Message Digest Algorithm"
  
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
  void md5_starts( md5_context *ctx ) 
  {
      ctx->total[0] = 0;
      ctx->total[1] = 0;
  
      ctx->state[0] = 0x67452301;
      ctx->state[1] = 0xEFCDAB89;
      ctx->state[2] = 0x98BADCFE;
      ctx->state[3] = 0x10325476;
  }
  
  KOKKOS_FORCEINLINE_FUNCTION
  void md5_process( md5_context *ctx, unsigned char data[64] ) 
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
  
  #define P_func(a,b,c,d,k,s,t)                                \
  {                                                       \
      a += F(b,c,d) + X[k] + t; a = S(a,s) + b;           \
  }
  
      A = ctx->state[0];
      B = ctx->state[1];
      C = ctx->state[2];
      D = ctx->state[3];
  
  #define F(x,y,z) (z ^ (x & (y ^ z)))
  
      P_func( A, B, C, D,  0,  7, 0xD76AA478 );
      P_func( D, A, B, C,  1, 12, 0xE8C7B756 );
      P_func( C, D, A, B,  2, 17, 0x242070DB );
      P_func( B, C, D, A,  3, 22, 0xC1BDCEEE );
      P_func( A, B, C, D,  4,  7, 0xF57C0FAF );
      P_func( D, A, B, C,  5, 12, 0x4787C62A );
      P_func( C, D, A, B,  6, 17, 0xA8304613 );
      P_func( B, C, D, A,  7, 22, 0xFD469501 );
      P_func( A, B, C, D,  8,  7, 0x698098D8 );
      P_func( D, A, B, C,  9, 12, 0x8B44F7AF );
      P_func( C, D, A, B, 10, 17, 0xFFFF5BB1 );
      P_func( B, C, D, A, 11, 22, 0x895CD7BE );
      P_func( A, B, C, D, 12,  7, 0x6B901122 );
      P_func( D, A, B, C, 13, 12, 0xFD987193 );
      P_func( C, D, A, B, 14, 17, 0xA679438E );
      P_func( B, C, D, A, 15, 22, 0x49B40821 );
  
  #undef F
  
  #define F(x,y,z) (y ^ (z & (x ^ y)))
  
      P_func( A, B, C, D,  1,  5, 0xF61E2562 );
      P_func( D, A, B, C,  6,  9, 0xC040B340 );
      P_func( C, D, A, B, 11, 14, 0x265E5A51 );
      P_func( B, C, D, A,  0, 20, 0xE9B6C7AA );
      P_func( A, B, C, D,  5,  5, 0xD62F105D );
      P_func( D, A, B, C, 10,  9, 0x02441453 );
      P_func( C, D, A, B, 15, 14, 0xD8A1E681 );
      P_func( B, C, D, A,  4, 20, 0xE7D3FBC8 );
      P_func( A, B, C, D,  9,  5, 0x21E1CDE6 );
      P_func( D, A, B, C, 14,  9, 0xC33707D6 );
      P_func( C, D, A, B,  3, 14, 0xF4D50D87 );
      P_func( B, C, D, A,  8, 20, 0x455A14ED );
      P_func( A, B, C, D, 13,  5, 0xA9E3E905 );
      P_func( D, A, B, C,  2,  9, 0xFCEFA3F8 );
      P_func( C, D, A, B,  7, 14, 0x676F02D9 );
      P_func( B, C, D, A, 12, 20, 0x8D2A4C8A );
  
  #undef F
      
  #define F(x,y,z) (x ^ y ^ z)
  
      P_func( A, B, C, D,  5,  4, 0xFFFA3942 );
      P_func( D, A, B, C,  8, 11, 0x8771F681 );
      P_func( C, D, A, B, 11, 16, 0x6D9D6122 );
      P_func( B, C, D, A, 14, 23, 0xFDE5380C );
      P_func( A, B, C, D,  1,  4, 0xA4BEEA44 );
      P_func( D, A, B, C,  4, 11, 0x4BDECFA9 );
      P_func( C, D, A, B,  7, 16, 0xF6BB4B60 );
      P_func( B, C, D, A, 10, 23, 0xBEBFBC70 );
      P_func( A, B, C, D, 13,  4, 0x289B7EC6 );
      P_func( D, A, B, C,  0, 11, 0xEAA127FA );
      P_func( C, D, A, B,  3, 16, 0xD4EF3085 );
      P_func( B, C, D, A,  6, 23, 0x04881D05 );
      P_func( A, B, C, D,  9,  4, 0xD9D4D039 );
      P_func( D, A, B, C, 12, 11, 0xE6DB99E5 );
      P_func( C, D, A, B, 15, 16, 0x1FA27CF8 );
      P_func( B, C, D, A,  2, 23, 0xC4AC5665 );
  
  #undef F
  
  #define F(x,y,z) (y ^ (x | ~z))
  
      P_func( A, B, C, D,  0,  6, 0xF4292244 );
      P_func( D, A, B, C,  7, 10, 0x432AFF97 );
      P_func( C, D, A, B, 14, 15, 0xAB9423A7 );
      P_func( B, C, D, A,  5, 21, 0xFC93A039 );
      P_func( A, B, C, D, 12,  6, 0x655B59C3 );
      P_func( D, A, B, C,  3, 10, 0x8F0CCC92 );
      P_func( C, D, A, B, 10, 15, 0xFFEFF47D );
      P_func( B, C, D, A,  1, 21, 0x85845DD1 );
      P_func( A, B, C, D,  8,  6, 0x6FA87E4F );
      P_func( D, A, B, C, 15, 10, 0xFE2CE6E0 );
      P_func( C, D, A, B,  6, 15, 0xA3014314 );
      P_func( B, C, D, A, 13, 21, 0x4E0811A1 );
      P_func( A, B, C, D,  4,  6, 0xF7537E82 );
      P_func( D, A, B, C, 11, 10, 0xBD3AF235 );
      P_func( C, D, A, B,  2, 15, 0x2AD7D2BB );
      P_func( B, C, D, A,  9, 21, 0xEB86D391 );
  
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
  void md5_update( md5_context *ctx, unsigned char *input, int ilen ) 
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
  void md5_finish( md5_context *ctx, unsigned char output[16] ) 
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
  void digest_to_hex(const uint8_t digest[MD5_DIGEST_SIZE], char* output)  {
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
  uint32_t digest_size()  {
    return 16;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void hash(const void* data, uint64_t len, uint8_t* digest) {
//    #ifdef __CUDA_ARCH__
      md5_context md5_ctx;
      md5_starts(&md5_ctx);
      md5_ctx.state[0] ^= 0;
      if(len > INT_MAX) {
        int chunk_size = 128;
        uint64_t max = len / static_cast<uint64_t>(chunk_size);
        for(uint64_t i=0; i<max; i++) {
          int num_bytes = 128;
          if(i == max-1)
            num_bytes = len % 128;
          md5_update( &md5_ctx, (uint8_t*)(data)+(i*128), num_bytes);
        }
      } else {
        md5_update( &md5_ctx, (uint8_t*)(data), static_cast<int32_t>(len));
      }
      md5_finish( &md5_ctx, digest);
//    #else
//      MD5_CTX ctx;
//      MD5_Init(&ctx);
//      if(len > static_cast<uint64_t>(INT_MAX)) {
//        int chunk_size = 128;
//        uint64_t max = len / static_cast<uint64_t>(chunk_size);
//        for(uint64_t i=0; i<max; i++) {
//          int num_bytes = chunk_size;
//          if(i == max-1)
//            num_bytes = len % 128;
//          MD5_Update(&ctx, (uint8_t*)(data)+static_cast<uint64_t>(i)*128, num_bytes);
//        }
//      } else {
//        MD5_Update(&ctx, data, static_cast<int32_t>(len));
//      }
//      MD5_Final(digest, &ctx);
//    #endif
  }
}

#endif // KOKKOS_MD5
