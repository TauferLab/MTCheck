#ifndef __HASH_FUNCTIONS_HPP
#define __HASH_FUNCTIONS_HPP

#include <iostream>
#include <cstring>
#include <string>
#include <openssl/md5.h>
#include "map_helpers.hpp"
#include "kokkos_md5.hpp"
#include "kokkos_murmur3.hpp"

void calc_and_print_md5(Kokkos::View<uint8_t*>& data_d);

std::string digest_to_str(HashDigest& dig); 

KOKKOS_FORCEINLINE_FUNCTION
void hash(const void* data, uint64_t len, uint8_t* digest) {
//  kokkos_md5::hash(data, len, digest);
  kokkos_murmur3::hash(data, len, digest);
}

template<typename KView>
std::string calculate_digest_device(KView& data, uint64_t len) {
  Kokkos::View<uint8_t[16]> digest("Hash digest");
  Kokkos::deep_copy(digest, 0);
  Kokkos::parallel_for(1, KOKKOS_LAMBDA(const uint32_t i) {
    hash((uint8_t*)(data.data()), len, digest.data());
  });
  auto digest_h = Kokkos::create_mirror_view(digest);
  Kokkos::deep_copy(digest_h, digest);
  static const char hexchars[] = "0123456789ABCDEF";
  std::string digest_str;
  for(int k=0; k<16; k++) {
    unsigned char b = digest_h(k);
    char hex[3];
    hex[0] = hexchars[b >> 4];
    hex[1] = hexchars[b & 0xF];
    hex[2] = 0;
    digest_str.append(hex);
    if(k%4 == 3)
      digest_str.append(" ");
  }
  return digest_str;
}

template<typename KView>
std::string calculate_digest_host(KView& data_h, uint64_t len) {
  HashDigest dig;
  hash((uint8_t*)(data_h.data()), len, dig.digest);
  static const char hexchars[] = "0123456789ABCDEF";
  std::string digest_str;
  for(int k=0; k<16; k++) {
    unsigned char b = dig.digest[k];
    char hex[3];
    hex[0] = hexchars[b >> 4];
    hex[1] = hexchars[b & 0xF];
    hex[2] = 0;
    digest_str.append(hex);
    if(k%4 == 3)
      digest_str.append(" ");
  }
  return digest_str;
}

template<typename KView>
std::string calculate_digest_host(KView& data_h) {
  HashDigest dig;
  hash((uint8_t*)(data_h.data()), data_h.size(), dig.digest);
  static const char hexchars[] = "0123456789ABCDEF";
  std::string digest_str;
  for(int k=0; k<16; k++) {
    unsigned char b = dig.digest[k];
    char hex[3];
    hex[0] = hexchars[b >> 4];
    hex[1] = hexchars[b & 0xF];
    hex[2] = 0;
    digest_str.append(hex);
    if(k%4 == 3)
      digest_str.append(" ");
  }
  return digest_str;
}


#endif // __HASH_FUNCTIONS_HPP
