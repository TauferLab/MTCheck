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
    STDOUT_PRINT("------------------------------------------------------\n");

    Kokkos::View<uint8_t*> step0_d("Step 0", 8), step1_d("Step 1", 8);
    auto step0_h = Kokkos::create_mirror_view(step0_d);
    auto step1_h = Kokkos::create_mirror_view(step1_d);
    step0_h(0) = static_cast<uint8_t>('A');
    step0_h(1) = static_cast<uint8_t>('B');
    step0_h(2) = static_cast<uint8_t>('C');
    step0_h(3) = static_cast<uint8_t>('D');
    step0_h(4) = static_cast<uint8_t>('E');
    step0_h(5) = static_cast<uint8_t>('F');
    step0_h(6) = static_cast<uint8_t>('G');
    step0_h(7) = static_cast<uint8_t>('H');

    step1_h(0) = static_cast<uint8_t>('A');
    step1_h(1) = static_cast<uint8_t>('B');
    step1_h(2) = static_cast<uint8_t>('C');
    step1_h(3) = static_cast<uint8_t>('D');
    step1_h(4) = static_cast<uint8_t>('E');
    step1_h(5) = static_cast<uint8_t>('I');
    step1_h(6) = static_cast<uint8_t>('J');
    step1_h(7) = static_cast<uint8_t>('K');

    Kokkos::deep_copy(step0_d, step0_h);
    Kokkos::deep_copy(step1_d, step1_h);
    
    Deduplicator deduplicator(1);
    std::vector< Kokkos::View<uint8_t*>::HostMirror > incr_chkpts;

    Kokkos::View<uint8_t*>::HostMirror diff_h("Buffer", 1);

    std::string correct = calculate_digest_host(step0_h);

    deduplicator.checkpoint(Tree, (uint8_t*)(step0_d.data()), step0_d.size(), diff_h, true);

    Kokkos::fence();

    incr_chkpts.push_back(diff_h);
    Kokkos::View<uint8_t*> restart_buf_d("Restart buffer", 8);
    Kokkos::View<uint8_t*>::HostMirror restart_buf_h = Kokkos::create_mirror_view(restart_buf_d);
    std::string null("/dev/null/");
    deduplicator.restart(Tree, restart_buf_d, incr_chkpts, null, 0);
    Kokkos::deep_copy(restart_buf_h, restart_buf_d);
    std::string full_digest = calculate_digest_host(restart_buf_h);
    res = correct.compare(full_digest);
    // Print digest
    std::cout << "Checkpoint " << 0 << std::endl;
    if(res == 0) {
      std::cout << "Hashes match!\n";
    } else {
      std::cout << "Hashes don't match!\n";
    }
    std::cout << "Correct:    " << correct << std::endl;
    std::cout << "Tree chkpt: " << full_digest << std::endl;

    if(res != 0) 
      return res;

    Kokkos::fence();

    deduplicator.checkpoint(Tree, (uint8_t*)(step1_d.data()), step1_d.size(), diff_h, false);

    Kokkos::fence();

    correct = calculate_digest_host(step1_h);
    incr_chkpts.push_back(diff_h);
    deduplicator.restart(Tree, restart_buf_d, incr_chkpts, null, 1);
    Kokkos::deep_copy(restart_buf_h, restart_buf_d);
    full_digest = calculate_digest_host(restart_buf_h);
    res = correct.compare(full_digest);
    // Print digest
    std::cout << "Checkpoint " << 1 << std::endl;
    if(res == 0) {
      std::cout << "Hashes match!\n";
    } else {
      std::cout << "Hashes don't match!\n";
    }
    std::cout << "Correct:    " << correct << std::endl;
    std::cout << "Tree chkpt: " << full_digest << std::endl;

    if(res != 0) 
      return res;

    printf("Expected 2 first occurrence region and 0 shifted duplicate regions\n");
    printf("Found %u first occurrence region and %u shifted duplicate regions\n", deduplicator.first_ocur_vec.size(), deduplicator.shift_dupl_vec.size());
    if(deduplicator.first_ocur_vec.size() != 2 || deduplicator.shift_dupl_vec.size() != 0) {
      res = -1;
    }
  }
  Kokkos::finalize();
  return res;
}




