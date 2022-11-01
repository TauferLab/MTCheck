#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include "stdio.h"
#include <string>
#include <map>
#include <fstream>
#include "kokkos_hash_list.hpp"
#include "dedup_approaches.hpp"
#include "restart_approaches.hpp"
#include <libgen.h>
#include <iostream>
#include <utility>
#include "utils.hpp"


int main(int argc, char** argv) {
  int res = 0;
  Kokkos::initialize(argc, argv);
  {
    using Timer = std::chrono::high_resolution_clock;
    STDOUT_PRINT("------------------------------------------------------\n");

    // Process data from checkpoint files
    STDOUT_PRINT("Argv[1]: %s\n", argv[1]);
    uint32_t chunk_size = static_cast<uint32_t>(atoi(argv[1]));
    STDOUT_PRINT("Loaded chunk size\n");
    uint32_t num_chkpts = static_cast<uint32_t>(atoi(argv[2]));

//    SHA1 hasher;
//    Murmur3C hasher;
    MD5Hash hasher;

    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);

    uint64_t data_len = 1024*1024;
    Kokkos::View<uint8_t**, 
                 Kokkos::LayoutLeft, 
                 Kokkos::DefaultHostExecutionSpace> data_views_h("Host views", data_len, num_chkpts);
    std::vector< std::vector<uint8_t> > incr_chkpts;

    for(uint32_t i=0; i<num_chkpts; i++) {
      Kokkos::View<uint8_t*> data_d("Device data", data_len);
      Kokkos::deep_copy(data_d, 0);
      auto data_h = Kokkos::subview(data_views_h, Kokkos::ALL(), i);

      // Generate next random data
      auto policy = Kokkos::RangePolicy<>(0, data_len);
      Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
        auto rand_gen = rand_pool.get_state();
        data_d(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
        rand_pool.free_state(rand_gen);
      });
      Kokkos::deep_copy(data_h, data_d);
    
      // Calculate correct digest
      std::string correct = calculate_digest_host(data_h);

      // Perform chkpt
      std::string log("");
      int lineage_mode = 0;
      list_chkpt(hasher, data_views_h, incr_chkpts, log, chunk_size, i+1, lineage_mode);

      // Restart chkpt
      Kokkos::View<uint8_t*> restart_buf_d("Restart buffer", data_len);
      Kokkos::View<uint8_t*>::HostMirror restart_buf_h = Kokkos::create_mirror_view(restart_buf_d);
      Kokkos::deep_copy(restart_buf_h, restart_buf_d);
      auto times = restart_list_chkpt(incr_chkpts, restart_buf_d, chunk_size, i, lineage_mode);

      // Calculate digest of full checkpoint
      Kokkos::deep_copy(restart_buf_h, restart_buf_d);
      std::string full_digest = calculate_digest_host(restart_buf_h);

      // Compare digests
      res = correct.compare(full_digest);

      // Print digest
      std::cout << "Checkpoint " << i << std::endl;
      if(res == 0) {
        std::cout << "Hashes match!\n";
      } else {
        std::cout << "Hashes don't match!\n";
      }
      std::cout << "Correct:     " << correct << std::endl;
      std::cout << "List chkpt: " << full_digest << std::endl;

      if(res != 0) {
        break;
      }
    }
  }
  Kokkos::finalize();
  return res;
}


