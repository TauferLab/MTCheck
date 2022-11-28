#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include "stdio.h"
#include <string>
#include "deduplicator.hpp"
#include <libgen.h>
#include <utility>
#include "utils.hpp"

int main(int argc, char** argv) {
  int res = 0;
  Kokkos::initialize(argc, argv);
  {
    using Timer = std::chrono::high_resolution_clock;
    STDOUT_PRINT("------------------------------------------------------\n");

    Kokkos::View<uint8_t*> step0_d("Step 0", 8), step1_d("Step 1", 8);
    auto step0_h = Kokkos::create_mirror_view(step0_d);
    auto step1_h = Kokkos::create_mirror_view(step1_d);
    step0_h(0) = static_cast<uint8_t>('M');
    step0_h(1) = static_cast<uint8_t>('N');
    step0_h(2) = static_cast<uint8_t>('X');
    step0_h(3) = static_cast<uint8_t>('Y');
    step0_h(4) = static_cast<uint8_t>('O');
    step0_h(5) = static_cast<uint8_t>('P');
    step0_h(6) = static_cast<uint8_t>('Q');
    step0_h(7) = static_cast<uint8_t>('R');

    step1_h(0) = static_cast<uint8_t>('I');
    step1_h(1) = static_cast<uint8_t>('J');
    step1_h(2) = static_cast<uint8_t>('X');
    step1_h(3) = static_cast<uint8_t>('Y');
    step1_h(4) = static_cast<uint8_t>('I');
    step1_h(5) = static_cast<uint8_t>('J');
    step1_h(6) = static_cast<uint8_t>('I');
    step1_h(7) = static_cast<uint8_t>('B');

    Kokkos::deep_copy(step0_d, step0_h);
    Kokkos::deep_copy(step1_d, step1_h);
    
    Deduplicator<MD5Hash> deduplicator(1);

    deduplicator.dedup(step0_d, true);

    Kokkos::fence();

    deduplicator.dedup(step1_d, false);

    Kokkos::fence();

    printf("Expected 2 first occurrence region and 2 shifted duplicate regions\n");
    if(deduplicator.first_ocur_updates_d.size() != 2 || deduplicator.shift_dupl_updates_d.size() != 2) {
      res = -1;
    }
  }
  Kokkos::finalize();
  return res;
}



