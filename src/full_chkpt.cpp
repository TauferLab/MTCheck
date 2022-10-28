#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <openssl/md5.h>
#include "stdio.h"
#include "map_helpers.hpp"
#include "utils.hpp"
//#include "dedup_approaches.hpp"
//#include "data_generation.hpp"

template<typename KView>
std::string calculate_digest(KView& data_h) {
  HashDigest hash;
  MD5((uint8_t*)(data_h.data()), data_h.size(), hash.digest);
  static const char hexchars[] = "0123456789ABCDEF";
  std::string digest_str;
  for(int k=0; k<16; k++) {
    unsigned char b = hash.digest[k];
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

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using Timer = std::chrono::high_resolution_clock;
    STDOUT_PRINT("------------------------------------------------------\n");

    // Process data from checkpoint files
    DEBUG_PRINT("Argv[1]: %s\n", argv[1]);
    uint32_t chunk_size = static_cast<uint32_t>(atoi(argv[1]));
    DEBUG_PRINT("Loaded chunk size\n");
    uint32_t num_chkpts = static_cast<uint32_t>(atoi(argv[2]));

//    SHA1 hasher;
//    Murmur3C hasher;
//    MD5Hash hasher;

    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    std::default_random_engine generator(1931);

    uint64_t data_len = 1024*1024;
    Kokkos::View<uint8_t**, Kokkos::LayoutLeft> data_views_d("Data", data_len, num_chkpts);
    Kokkos::View<uint8_t**, Kokkos::LayoutLeft>::HostMirror data_views_h = Kokkos::create_mirror_view(data_views_d);
    for(uint32_t i=0; i<num_chkpts; i++) {
      auto subview_d = Kokkos::subview(data_views_d, Kokkos::ALL, i);
      auto subview_h = Kokkos::subview(data_views_h, Kokkos::ALL, i);

      // Generate next random data
      auto policy = Kokkos::RangePolicy<>(0, data_len);
      Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
        auto rand_gen = rand_pool.get_state();
        subview_d(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
        rand_pool.free_state(rand_gen);
      });
    
      // Calculate correct digest
      Kokkos::deep_copy(subview_h, subview_d);
      std::string correct = calculate_digest(subview_h);

      // Perform chkpt
      Kokkos::deep_copy(subview_h, subview_d);
      // Restart chkpt
      Kokkos::View<uint8_t*> restart_buf_d("Restart buffer", data_len);
      Kokkos::deep_copy(restart_buf_d, subview_d);

      // Calculate digest of full checkpoint
      Kokkos::deep_copy(subview_h, restart_buf_d);
      std::string full_digest = calculate_digest(subview_h);

      // Compare digests
      int res = correct.compare(full_digest);

      // Print digest
      std::cout << "Checkpoint " << i << std::endl;
      if(res == 0) {
        std::cout << "Hashes match!\n";
      } else {
        std::cout << "Hashes don't match!\n";
      }
      std::cout << "Correct:    " << correct << std::endl;
      std::cout << "Full chkpt: " << full_digest << std::endl;
    }
  }
  Kokkos::finalize();
}

