#include "hash_functions.hpp"

void calc_and_print_md5(Kokkos::View<uint8_t*>& data_d) {
  HashDigest correct;
  auto data_h = Kokkos::create_mirror_view(data_d);
  Kokkos::deep_copy(data_h, data_d);
  MD5((uint8_t*)(data_h.data()), data_d.size(), correct.digest);
  static const char hexchars[] = "0123456789ABCDEF";
  std::string ref_digest;
  for(unsigned int k=0; k<sizeof(HashDigest); k++) {
    unsigned char b = correct.digest[k];
    char hex[3];
    hex[0] = hexchars[b >> 4];
    hex[1] = hexchars[b & 0xF];
    hex[2] = 0;
    ref_digest.append(hex);
    if(k%4 == 3)
      ref_digest.append(" ");
  }
  std::cout << "Reference digest:  " << ref_digest << std::endl;
}

std::string digest_to_str(HashDigest& dig) {
  static const char hexchars[] = "0123456789ABCDEF";
  std::string ref_digest;
  for(unsigned int k=0; k<sizeof(HashDigest); k++) {
    unsigned char b = dig.digest[k];
    char hex[3];
    hex[0] = hexchars[b >> 4];
    hex[1] = hexchars[b & 0xF];
    hex[2] = 0;
    ref_digest.append(hex);
    if(k%4 == 3)
      ref_digest.append(" ");
  }
  return ref_digest;
}
