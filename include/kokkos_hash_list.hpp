#ifndef KOKKOS_HASH_LIST_HPP
#define KOKKOS_HASH_LIST_HPP
#include <Kokkos_Core.hpp>
#include "map_helpers.hpp"

/** \class HashList
 *  \brief Class for list of hash digests
 *
 *  Simple structure for holding a list of hash digests using Kokkos Views.
 */
class HashList {
public:
  /// List of Hash digests on device
  Kokkos::View<HashDigest*> list_d;
  /// List of Hash digests. Host mirror
  Kokkos::View<HashDigest*>::HostMirror list_h;

  /// empty constructor
  HashList() {}

  /**
   * Allocate space for list of hashes on device and host
   *
   * \param num_leaves Maximum number of hashes contained in the list
   */
  HashList(const uint32_t num_leaves) {
    list_d = Kokkos::View<HashDigest*>("Hash list", num_leaves);
    list_h = Kokkos::create_mirror_view(list_d);
  }

  /**
   * Access hash digest in list
   *
   * \param i Index of hash digest
   *
   * \return Reference to hash digest at index i
   */
  KOKKOS_INLINE_FUNCTION HashDigest& operator()(int32_t i) const {
    return list_d(i);
  }
 
  /**
   * Helper function for converting a hash digest to hex
   */
  void digest_to_hex_(const uint8_t* digest, char* output) {
    int i,j;
    char* c = output;
    for(i=0; i<static_cast<int>(sizeof(HashDigest)/4); i++) {
      for(j=0; j<4; j++) {
        sprintf(c, "%02X", digest[i*4 + j]);
        c += 2;
      }
      sprintf(c, " ");
      c += 1;
    }
    *(c-1) = '\0';
  }

  /**
   * Print list of digests using ascii hex
   */
  void print() {
    Kokkos::deep_copy(list_h, list_d);
    uint32_t num_leaves = (list_h.extent(0));
    printf("============================================================\n");
    char buffer[sizeof(HashDigest)*4];
    for(unsigned int i=0; i<num_leaves; i++) {
      digest_to_hex_((uint8_t*)(list_h(i).digest), buffer);
      printf("Node: %u: %s \n", i, buffer);
    }
    printf("============================================================\n");
  }
};

#endif

