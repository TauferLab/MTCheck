#include "stdio.h"
#include <string>
#include <fstream>
#include <libgen.h>
#include <random>
#include "data_generation.hpp"

//enum DataGenerationMode {
//  Random=0,
//  BeginningIdentical,
//  Chunk,
//  Identical,
//  Sparse,
//  Swap,
//  Zero
//};
//
//Kokkos::View<uint8_t*> generate_initial_data(uint32_t max_data_len) {
//  Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
//  Kokkos::View<uint8_t*> data("Data", max_data_len);
//  auto policy = Kokkos::RangePolicy<>(0, max_data_len);
//  Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
//    auto rand_gen = rand_pool.get_state();
//    data(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
//    rand_pool.free_state(rand_gen);
//  });
//  return data;
//}
//
//void perturb_data(Kokkos::View<uint8_t*>& data0, 
//                  const uint32_t num_changes, DataGenerationMode mode, Kokkos::Random_XorShift64_Pool<>& rand_pool, std::default_random_engine& generator) {
//  if(mode == Random) {
//    auto policy = Kokkos::RangePolicy<>(0, data0.size());
//    Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
//      auto rand_gen = rand_pool.get_state();
//      data0(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
//      rand_pool.free_state(rand_gen);
//    });
//  } else if(mode == BeginningIdentical) {
//    auto policy = Kokkos::RangePolicy<>(num_changes, data0.size());
//    Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
//      auto rand_gen = rand_pool.get_state();
//      data0(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
//      rand_pool.free_state(rand_gen);
//    });
//  } else if(mode == Zero) {
//    std::uniform_int_distribution<uint64_t> distribution(0, data0.size()-num_changes);
//    uint64_t offset = distribution(generator);
//    Kokkos::parallel_for("Randomize chunk", Kokkos::RangePolicy<>(offset, offset+num_changes), KOKKOS_LAMBDA(const uint64_t j) {
//      data0(j) = static_cast<uint8_t>(0);
//    });
//  } else if(mode == Identical) {
//  } else if(mode == Chunk) {
//    std::uniform_int_distribution<uint64_t> distribution(0, data0.size()-num_changes);
//    uint64_t offset = distribution(generator);
//    Kokkos::parallel_for("Randomize chunk", Kokkos::RangePolicy<>(offset, offset+num_changes), KOKKOS_LAMBDA(const uint64_t j) {
//      auto rand_gen = rand_pool.get_state();
//      data0(j) = static_cast<uint8_t>(rand_gen.urand() % 256);
//      rand_pool.free_state(rand_gen);
//    });
//  } else if(mode == Identical) {
//  } else if(mode == Sparse) {
//    Kokkos::parallel_for("Randomize data", Kokkos::RangePolicy<>(0, num_changes), KOKKOS_LAMBDA(const uint32_t j) {
//      auto rand_gen = rand_pool.get_state();
//      uint32_t pos = rand_gen.urand() % data0.size();
//      uint8_t val = static_cast<uint8_t>(rand_gen.urand() % 256);
//      while(val == data0(pos)) {
//        val = static_cast<uint8_t>(rand_gen.urand() % 256);
//      }
//      data0(pos) = val; 
//      rand_pool.free_state(rand_gen);
//    });
//  } else if(mode == Swap) {
//    Kokkos::View<uint8_t*> chunkA("ChunkA", num_changes);
//    Kokkos::View<uint8_t*> chunkB("ChunkB", num_changes);
//    std::uniform_int_distribution<uint64_t> distribution(0, data0.size()-num_changes);
//    uint64_t A_offset = 0;
//    uint64_t B_offset = distribution(generator);
//    while((B_offset < A_offset + num_changes) || (B_offset + num_changes > data0.size()) || (((B_offset / 128)*128) != B_offset)) {
//      B_offset = distribution(generator);
//    }
//    printf("B offset: %lu\n", B_offset);
//    auto A_subview = Kokkos::subview(data0, std::pair<uint64_t, uint64_t>(A_offset, A_offset+num_changes));
//    auto B_subview = Kokkos::subview(data0, std::pair<uint64_t, uint64_t>(B_offset, B_offset+num_changes));
//    Kokkos::deep_copy(chunkA, A_subview);
//    Kokkos::deep_copy(chunkB, B_subview);
//    Kokkos::deep_copy(A_subview, chunkB);
//    Kokkos::deep_copy(B_subview, chunkA);
//  }
//}
//
//void write_chkpt(const std::string& filename, Kokkos::View<uint8_t*> data) {
//  Kokkos::View<uint8_t*>::HostMirror data_h = Kokkos::create_mirror_view(data);
//  Kokkos::deep_copy(data_h, data);
//  FILE *chkpt;
//  chkpt = fopen(filename.c_str(), "wb");
//  if(chkpt == NULL) {
//    printf("Failed to open checkpoint file %s\n", filename.c_str());
//  } else {
//    fwrite(data_h.data(), sizeof(uint8_t), data_h.size(), chkpt);
//    fflush(chkpt);
//    fclose(chkpt);
//  }
////  std::ofstream chkpt(filename.c_str(), std::ofstream::binary);
////  chkpt.write((const char*)(data.data()), data.span());
////  printf("Wrote %zd bytes\n", data.span());
////  chkpt.flush();
////  chkpt.close();
//}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    uint64_t data_len = strtoull(argv[1], NULL, 0);
    uint32_t num_chkpts = static_cast<uint32_t>(atoi(argv[2]));
    char generator_mode = *(argv[3]);
    DataGenerationMode mode = Random;
    if(generator_mode == 'R') {
      mode = Random;
    } else if(generator_mode == 'B') {
      mode = BeginningIdentical;
    } else if(generator_mode == 'C') {
      mode = Chunk;
    } else if(generator_mode == 'I') {
      mode = Identical;
    } else if(generator_mode == 'S') {
      mode = Sparse;
    } else if(generator_mode == 'Z') {
      mode = Zero;
    } else if(generator_mode == 'W') {
      mode = Swap;
    }
    uint64_t num_changes = strtoull(argv[4], NULL, 0);
    std::string chkpt_filename(argv[5]);
    chkpt_filename = chkpt_filename + std::string(".");
    printf("Data length: %lu\n", data_len);
    printf("Number of checkpoints: %d\n", num_chkpts);
    printf("Mode: %c\n", generator_mode);
    printf("Num changes: %lu\n", num_changes);
    printf("File name: %s\n", chkpt_filename.c_str());

    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    std::default_random_engine generator(1931);

    Kokkos::View<uint8_t*> data = generate_initial_data(data_len);
    Kokkos::fence();
    write_chkpt(chkpt_filename + std::to_string(0) + std::string(".chkpt"), data);

    for(uint32_t i=1; i<num_chkpts; i++) {
      perturb_data(data, num_changes, mode, rand_pool, generator);
      Kokkos::fence();
      write_chkpt(chkpt_filename + std::to_string(i) + std::string(".chkpt"), data);
    }
  }
  Kokkos::finalize();
}
