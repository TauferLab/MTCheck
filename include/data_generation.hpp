#ifndef DATA_GENERATION_HPP
#define DATA_GENERATION_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include "stdio.h"
#include <string>
#include <fstream>
#include <libgen.h>
#include <random>

enum DataGenerationMode {
  Random=0,
  BeginningIdentical,
  Chunk,
  Identical,
  Sparse,
  Swap,
  Zero
};

Kokkos::View<uint8_t*> generate_initial_data(uint32_t max_data_len) {
  Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
  Kokkos::View<uint8_t*> data("Data", max_data_len);
  auto policy = Kokkos::RangePolicy<>(0, max_data_len);
  Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
    auto rand_gen = rand_pool.get_state();
    data(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
    rand_pool.free_state(rand_gen);
  });
  return data;
}

void perturb_data(Kokkos::View<uint8_t*>& data0, 
                  const uint32_t num_changes, DataGenerationMode mode, Kokkos::Random_XorShift64_Pool<>& rand_pool, std::default_random_engine& generator) {
  if(mode == Random) {
    auto policy = Kokkos::RangePolicy<>(0, data0.size());
    Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
      auto rand_gen = rand_pool.get_state();
      data0(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
      rand_pool.free_state(rand_gen);
    });
  } else if(mode == BeginningIdentical) {
    auto policy = Kokkos::RangePolicy<>(num_changes, data0.size());
    Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
      auto rand_gen = rand_pool.get_state();
      data0(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
      rand_pool.free_state(rand_gen);
    });
  } else if(mode == Zero) {
    std::uniform_int_distribution<uint64_t> distribution(0, data0.size()-num_changes);
    uint64_t offset = distribution(generator);
    Kokkos::parallel_for("Randomize chunk", Kokkos::RangePolicy<>(offset, offset+num_changes), KOKKOS_LAMBDA(const uint64_t j) {
      data0(j) = static_cast<uint8_t>(0);
    });
  } else if(mode == Identical) {
  } else if(mode == Chunk) {
    std::uniform_int_distribution<uint64_t> distribution(0, data0.size()-num_changes);
    uint64_t offset = distribution(generator);
    Kokkos::parallel_for("Randomize chunk", Kokkos::RangePolicy<>(offset, offset+num_changes), KOKKOS_LAMBDA(const uint64_t j) {
      auto rand_gen = rand_pool.get_state();
      data0(j) = static_cast<uint8_t>(rand_gen.urand() % 256);
      rand_pool.free_state(rand_gen);
    });
  } else if(mode == Identical) {
  } else if(mode == Sparse) {
    Kokkos::parallel_for("Randomize data", Kokkos::RangePolicy<>(0, num_changes), KOKKOS_LAMBDA(const uint32_t j) {
      auto rand_gen = rand_pool.get_state();
      uint32_t pos = rand_gen.urand() % data0.size();
      uint8_t val = static_cast<uint8_t>(rand_gen.urand() % 256);
      while(val == data0(pos)) {
        val = static_cast<uint8_t>(rand_gen.urand() % 256);
      }
      data0(pos) = val; 
      rand_pool.free_state(rand_gen);
    });
  } else if(mode == Swap) {
    Kokkos::View<uint8_t*> chunkA("ChunkA", num_changes);
    Kokkos::View<uint8_t*> chunkB("ChunkB", num_changes);
    std::uniform_int_distribution<uint64_t> distribution(0, data0.size()-num_changes);
    uint64_t A_offset = 0;
    uint64_t B_offset = distribution(generator);
    while((B_offset < A_offset + num_changes) || (B_offset + num_changes > data0.size()) || (((B_offset / 128)*128) != B_offset)) {
      B_offset = distribution(generator);
    }
    printf("B offset: %lu\n", B_offset);
    auto A_subview = Kokkos::subview(data0, std::pair<uint64_t, uint64_t>(A_offset, A_offset+num_changes));
    auto B_subview = Kokkos::subview(data0, std::pair<uint64_t, uint64_t>(B_offset, B_offset+num_changes));
    Kokkos::deep_copy(chunkA, A_subview);
    Kokkos::deep_copy(chunkB, B_subview);
    Kokkos::deep_copy(A_subview, chunkB);
    Kokkos::deep_copy(B_subview, chunkA);
  }
}

void write_chkpt(const std::string& filename, Kokkos::View<uint8_t*> data) {
  Kokkos::View<uint8_t*>::HostMirror data_h = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(data_h, data);
  FILE *chkpt;
  chkpt = fopen(filename.c_str(), "wb");
  if(chkpt == NULL) {
    printf("Failed to open checkpoint file %s\n", filename.c_str());
  } else {
    fwrite(data_h.data(), sizeof(uint8_t), data_h.size(), chkpt);
    fflush(chkpt);
    fclose(chkpt);
  }
}

#endif // DATA_GENERATION_HPP
