#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include "stdio.h"
#include <string>
#include <map>
#include <fstream>
#include "kokkos_merkle_tree.hpp"
#include "kokkos_hash_list.hpp"
#include "update_pattern_analysis.hpp"
#include <libgen.h>
#include <iostream>
#include "utils.hpp"

#define WRITE_CHKPT

enum DataGenerationMode {
  Random=0,
  BeginningIdentical,
  Checkered,
  Identical,
  Sparse
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

void perturb_data(Kokkos::View<uint8_t*>& data0, Kokkos::View<uint8_t*>& data1, 
                  const uint32_t num_changes, DataGenerationMode mode) {
  if(mode == Random) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    auto policy = Kokkos::RangePolicy<>(0, data0.size());
    Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
      auto rand_gen = rand_pool.get_state();
      data1(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
      rand_pool.free_state(rand_gen);
    });
  } else if(mode == BeginningIdentical) {
    Kokkos::deep_copy(data1, data0);
    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    auto policy = Kokkos::RangePolicy<>(num_changes, data0.size());
    Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
      auto rand_gen = rand_pool.get_state();
      data1(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
      rand_pool.free_state(rand_gen);
    });
  } else if(mode == Checkered) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    auto policy = Kokkos::RangePolicy<>(0, data0.size());
    Kokkos::parallel_for("Fill random", policy, KOKKOS_LAMBDA(const uint32_t i) {
      auto rand_gen = rand_pool.get_state();
      data1(i) = static_cast<uint8_t>(rand_gen.urand() % 256);
      rand_pool.free_state(rand_gen);
    });
    for(uint32_t j=0; j<data0.size(); j+=2*num_changes) {
      uint32_t end = j+num_changes;
      if(end > data0.size())
        end = data0.size() - j*num_changes;
      auto base_view  = Kokkos::subview(data0, Kokkos::make_pair(j, end));
      auto chkpt_view = Kokkos::subview(data1, Kokkos::make_pair(j, end));
      Kokkos::deep_copy(chkpt_view, base_view);
    }
  } else if(mode == Identical) {
    Kokkos::deep_copy(data1, data0);
  } else if(mode == Sparse) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    Kokkos::deep_copy(data1, data0);
    Kokkos::parallel_for("Randomize data", Kokkos::RangePolicy<>(0, num_changes), KOKKOS_LAMBDA(const uint32_t j) {
      auto rand_gen = rand_pool.get_state();
      uint32_t pos = rand_gen.urand() % data0.size();
      uint8_t val = static_cast<uint8_t>(rand_gen.urand() % 256);
      while(val == data1(pos)) {
        val = static_cast<uint8_t>(rand_gen.urand() % 256);
      }
      data1(pos) = val; 
      rand_pool.free_state(rand_gen);
    });
  }
}

typedef struct header {
  size_t chkpt_size;
  size_t header_size;
  size_t num_regions;
} header_t;

struct region_t {
  void* ptr;
  size_t size;
//  ptr_type_t ptr_type;
};
typedef std::map<int, region_t> regions_t;

// Read header and region map from VeloC checkpoint
bool read_full_header(const std::string& chkpt, header_t& header, std::map<int, size_t>& region_map) {
  try {
    std::ifstream f;
    size_t expected_size = 0;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(chkpt, std::ifstream::in | std::fstream::binary);
    int id;
    size_t num_regions, region_size, header_size;
    f.read((char*)(&num_regions), sizeof(size_t));
    for(uint32_t i=0; i<num_regions; i++) {
      f.read((char*)&id, sizeof(int));
      f.read((char*)&region_size, sizeof(size_t));
      region_map.insert(std::make_pair(id, region_size));
      expected_size += region_size;
    }
    header_size = f.tellg();
    f.seekg(0, f.end);
    size_t file_size = (size_t)f.tellg() - header_size;
    if(file_size != expected_size)
      throw std::ifstream::failure("file size " + std::to_string(file_size) + " does not match expected size " + std::to_string(expected_size));
    header.chkpt_size = file_size;
    header.header_size = header_size;
    header.num_regions = num_regions;
    return true;
  } catch(std::ifstream::failure &e) {
    std::cout << "cannot validate header for checkpoint " << chkpt << ", reason: " << e.what();
    return false;
  }
}

void write_incr_chkpt_hashlist( const std::string& filename, 
                                const Kokkos::View<uint8_t*>& data, 
                                uint32_t chunk_size, 
                                const DistinctMap& distinct, 
                                const SharedMap& shared,
                                uint32_t prior_chkpt_id,
                                uint32_t chkpt_id) {
  std::ofstream file;
  file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  file.open(filename, std::ofstream::out | std::ofstream::binary);

  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size()) {
    num_chunks += 1;
  }

  // Write whether we are storing the hashes, length full checkpoint, chunk size, number of repeat chunks, number of distinct chunks
  file << prior_chkpt_id << chkpt_id << data.size() << chunk_size << shared.size() << distinct.size();
  if(prior_chkpt_id == chkpt_id) {
    uint64_t buffer_size = 0;
    buffer_size += sizeof(uint32_t)*2*shared.size();
    buffer_size += distinct.size()*(sizeof(uint32_t) + sizeof(HashDigest) + chunk_size);
    Kokkos::View<uint8_t*> buffer_d("Buffer", buffer_size);
    Kokkos::View<uint8_t*>::HostMirror buffer_h = Kokkos::create_mirror_view(buffer_d);
    Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
    Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
    Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
    Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
    Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
    Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
    Kokkos::deep_copy(num_bytes_d, 0);
    Kokkos::deep_copy(num_bytes_data_d, 0);
    Kokkos::deep_copy(num_bytes_metadata_d, 0);
    Kokkos::parallel_for("Count shared updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
      if(shared.valid_at(i)) {
        uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
        Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
        uint32_t* buffer32 = (uint32_t*)(buffer_d.data()+pos);
        buffer32[0] = shared.key_at(i);
        buffer32[1] = shared.value_at(i);
      }
    });
    Kokkos::parallel_for("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
      if(distinct.valid_at(i)) {
        auto info = distinct.value_at(i);
        auto digest = distinct.key_at(i);
        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + sizeof(HashDigest) + chunk_size);
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t) + sizeof(HashDigest));
        Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
        uint32_t* buffer32 = (uint32_t*)(buffer_d.data()+pos);
        buffer32[0] = info.node;
        for(size_t j=0; j<sizeof(HashDigest); j++) {
          buffer_d(pos+sizeof(uint32_t)+j) = digest.digest[j];
        }
        uint32_t writesize = chunk_size;
        if(info.node == num_chunks-1) {
          writesize = data.size()-info.node*chunk_size;
        }
        for(size_t j=0; j<writesize; j++) {
          buffer_d(pos+sizeof(uint32_t)+sizeof(HashDigest)+j) = data(chunk_size*(info.node)+j);
        }
      }
    });
    Kokkos::deep_copy(num_bytes_h, num_bytes_d);
    Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
    Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
    Kokkos::deep_copy(buffer_h, buffer_d);
    file.write((const char*)(buffer_h.data()), num_bytes_h(0));
//    uint32_t bytes_written = 7*sizeof(uint32_t) + num_bytes_h(0);
//    std::cout << "Number of bytes written for incremental checkpoint: " << bytes_written << std::endl;
//    std::cout << "Number of bytes written for data: " << num_bytes_data_h(0) << std::endl;;
//    std::cout << "Number of bytes written for metadata: " << 7*sizeof(uint32_t) + num_bytes_metadata_h(0) << std::endl;
    printf("Number of bytes written for incremental checkpoint: %lu\n", 7*sizeof(uint32_t) + num_bytes_h(0));
    printf("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
    printf("Number of bytes written for metadata: %lu\n", 7*sizeof(uint32_t) + num_bytes_metadata_h(0));
  } else {
    uint32_t buffer_size = 0;
    buffer_size += sizeof(uint32_t)*2*shared.size();
    buffer_size += distinct.size()*(sizeof(uint32_t) + chunk_size);
DEBUG_PRINT("Buffer size: %u\n", buffer_size);
    Kokkos::View<uint8_t*> buffer_d("Buffer", buffer_size);
    Kokkos::View<uint8_t*>::HostMirror buffer_h = Kokkos::create_mirror_view(buffer_d);
    Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
    Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
    Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
    Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
    Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
    Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
    Kokkos::deep_copy(num_bytes_d, 0);
    Kokkos::deep_copy(num_bytes_data_d, 0);
    Kokkos::deep_copy(num_bytes_metadata_d, 0);
    Kokkos::parallel_for("Count shared updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
      if(shared.valid_at(i)) {
        Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
        uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
        uint32_t* buffer32 = (uint32_t*)(buffer_d.data()+pos);
        buffer32[0] = shared.key_at(i);
        buffer32[1] = shared.value_at(i);
      }
    });
    Kokkos::parallel_for("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
      if(distinct.valid_at(i)) {
        auto info = distinct.value_at(i);
//        auto digest = distinct.key_at(i);
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
        Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + chunk_size);
        uint32_t* buffer32 = (uint32_t*)(buffer_d.data()+pos);
        buffer32[0] = info.node;
        uint32_t writesize = chunk_size;
        if(info.node == num_chunks-1) {
          writesize = data.size()-info.node*chunk_size;
        }
        for(size_t j=0; j<writesize; j++) {
          buffer_d(pos+sizeof(uint32_t)+j) = data(chunk_size*(info.node)+j);
        }
      }
    });
    Kokkos::deep_copy(num_bytes_h, num_bytes_d);
    Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
    Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
    Kokkos::deep_copy(buffer_h, buffer_d);
    file.write((const char*)(buffer_h.data()), num_bytes_h(0));
  file.flush();
//    std::cout << "Number of bytes written for incremental checkpoint: " << 7*sizeof(uint32_t) + num_bytes_h(0) << std::endl;
//    std::cout << "Number of bytes written for data: " << num_bytes_data_h(0) << std::endl;;
//    std::cout << "Number of bytes written for metadata: " << 7*sizeof(uint32_t) + num_bytes_metadata_h(0) << std::endl;
    printf("Number of bytes written for incremental checkpoint: %lu\n", 7*sizeof(uint32_t) + num_bytes_h(0));
    printf("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
    printf("Number of bytes written for metadata: %lu\n", 7*sizeof(uint32_t) + num_bytes_metadata_h(0));
  }
DEBUG_PRINT("Trying to close file\n");
  file.flush();
  file.close();
DEBUG_PRINT("Closed file\n");
}

void write_incr_chkpt_hashtree( const std::string& filename, 
                                const Kokkos::View<uint8_t*>& data, 
                                uint32_t chunk_size, 
                                const DistinctMap& distinct, 
                                const SharedMap& shared,
                                uint32_t prior_chkpt_id,
                                uint32_t chkpt_id) {
  std::ofstream file;
  file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  file.open(filename, std::ofstream::out | std::ofstream::binary);

  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size()) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;

  // Write whether we are storing the hashes, length full checkpoint, chunk size, number of repeat chunks, number of distinct chunks
  file << prior_chkpt_id << chkpt_id << data.size() << chunk_size << shared.size() << distinct.size();
  Kokkos::View<uint8_t*> buffer_d("Buffer", shared.size()*2*sizeof(uint32_t) + distinct.size()*(sizeof(uint32_t)+sizeof(HashDigest)+chunk_size));
  Kokkos::View<uint8_t*>::HostMirror buffer_h = Kokkos::create_mirror_view(buffer_d);
  Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
  Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
  Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
  Kokkos::deep_copy(num_bytes_d, 0);
  Kokkos::deep_copy(num_bytes_data_d, 0);
  Kokkos::deep_copy(num_bytes_metadata_d, 0);

  Kokkos::parallel_for("Count shared updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
      uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
      uint32_t* buffer32 = (uint32_t*)(buffer_d.data()+pos);
      buffer32[0] = shared.key_at(i);
      buffer32[1] = shared.value_at(i);
    }
  });
  Kokkos::parallel_for("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      auto info = distinct.value_at(i);
      if(info.node >= num_chunks-1) {
        auto digest = distinct.key_at(i);
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(HashDigest));
        Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + sizeof(HashDigest) + chunk_size);
        uint32_t* buffer32 = (uint32_t*)(buffer_d.data()+pos);
        buffer32[0] = info.node;
        for(size_t j=0; j<sizeof(HashDigest); j++) {
          buffer_d(pos+sizeof(uint32_t)+j) = digest.digest[j];
        }
        uint32_t writesize = chunk_size;
        if(info.node == num_nodes-1) {
          writesize = data.size()-(info.node-num_chunks+1)*chunk_size;
        }
        for(size_t j=0; j<writesize; j++) {
          buffer_d(pos+sizeof(uint32_t)+sizeof(HashDigest)+j) = data(chunk_size*(info.node-num_chunks+1)+j);
        }
      } else {
        auto digest = distinct.key_at(i);
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(HashDigest));
        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + sizeof(HashDigest));
        uint32_t* buffer32 = (uint32_t*)(buffer_d.data()+pos);
        buffer32[0] = info.node;
        for(size_t j=0; j<sizeof(HashDigest); j++) {
          buffer_d(pos+sizeof(uint32_t)+j) = digest.digest[j];
        }
      }
    }
  });
  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
  Kokkos::deep_copy(buffer_h, buffer_d);
  file.write((const char*)(buffer_h.data()), num_bytes_h(0));
  printf("Number of bytes written for compact incremental checkpoint: %lu\n", 7*sizeof(uint32_t) + num_bytes_h(0));
  printf("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
  printf("Number of bytes written for metadata: %lu\n", 7*sizeof(uint32_t) + num_bytes_metadata_h(0));
  file.close();
}

//template<uint32_t N>
void write_incr_chkpt_hashtree( const std::string& filename, 
                                const Kokkos::View<uint8_t*>& data, 
                                uint32_t chunk_size, 
                                const CompactTable<31>& distinct, 
                                const CompactTable<31>& shared,
                                uint32_t prior_chkpt_id,
                                uint32_t chkpt_id) {
  std::ofstream file;
  file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  file.open(filename, std::ofstream::out | std::ofstream::binary);
DEBUG_PRINT("File: %s\n", filename.c_str());

//printf("Setup files.\n");

//  Kokkos::View<uint8_t*>::HostMirror data_h = Kokkos::create_mirror_view(data);
//  Kokkos::deep_copy(data_h, data);

//printf("Copied data to host\n");
  
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size()) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;

  // Write whether we are storing the hashes, length full checkpoint, chunk size, number of repeat chunks, number of distinct chunks
  file << prior_chkpt_id << chkpt_id << data.size() << chunk_size << shared.size() << distinct.size();
//printf("Wrote header: %u, %u, %u, %u, %u, %u\n", prior_chkpt_id, chkpt_id, data.size(), chunk_size, shared.size(), distinct.size());

  Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
  Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
  Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
  Kokkos::deep_copy(num_bytes_d, 0);
  Kokkos::deep_copy(num_bytes_data_d, 0);
  Kokkos::deep_copy(num_bytes_metadata_d, 0);
  DEBUG_PRINT("Setup Views\n");

  Kokkos::parallel_for("Count shared bytes", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      auto entry = shared.key_at(i);
      auto hist = shared.value_at(i);
      for(uint32_t j=0; j<hist.size(); j++) {
        if(hist(j) == chkpt_id) {
          Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t)*2);
          break;
        }
      }
    }
  });
  DEBUG_PRINT("Wrote shared metadata\n");
  Kokkos::parallel_for("Count distinct bytes", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      auto entry = distinct.key_at(i);
      auto hist = distinct.value_at(i);
      for(uint32_t j=0; j<hist.size(); j++) {
        if(hist(j) == chkpt_id) {
          if(entry.node == entry.size) {
            uint32_t size = num_leaf_descendents(entry.node, num_nodes);
            uint32_t start = leftmost_leaf(entry.node, num_nodes) - (num_chunks-1);
            Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t)*2 + size*chunk_size);
          } else {
            Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t)*2);
          }
          break;
        }
      }
    }
  });

  Kokkos::deep_copy(num_bytes_h, num_bytes_d);

  Kokkos::View<uint8_t*> buffer_d("Buffer", num_bytes_h(0));
DEBUG_PRINT("Length of buffer: %lu\n", num_bytes_h(0));
  Kokkos::View<uint8_t*>::HostMirror buffer_h = Kokkos::create_mirror_view(buffer_d);

  Kokkos::deep_copy(num_bytes_d, 0);

  DEBUG_PRINT("Start writing shared metadata\n");
  Kokkos::parallel_for("Count shared bytes", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      auto entry = shared.key_at(i);
      auto hist = shared.value_at(i);
      for(uint32_t j=0; j<hist.size(); j++) {
        if(hist(j) == chkpt_id) {
          Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)*2);
          size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)*2);
          uint32_t* buffer32 = (uint32_t*)(buffer_d.data() + pos);
          buffer32[0] = entry.node;
          buffer32[1] = entry.size;
          break;
        }
      }
    }
  });
  DEBUG_PRINT("Wrote shared metadata\n");
  Kokkos::parallel_for("Count distinct bytes", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      auto entry = distinct.key_at(i);
      auto hist = distinct.value_at(i);
      for(uint32_t j=0; j<hist.size(); j++) {
        if(hist(j) == chkpt_id) {
          if(entry.node == entry.size) {
            uint32_t size = num_leaf_descendents(entry.node, num_nodes);
            uint32_t start = leftmost_leaf(entry.node, num_nodes) - (num_chunks-1);
            Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)*2);
            Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(size*chunk_size));
            size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)*2 + size*chunk_size);
            uint32_t* buffer32 = (uint32_t*)(buffer_d.data() + pos);
            buffer32[0] = entry.node;
            buffer32[1] = entry.size;
            for(uint32_t j=0; j<size; j++) {
              uint32_t writesize = chunk_size;
              if(start+j == num_chunks-1) {
                writesize = data.size()-(start+j)*chunk_size;
              }
              memcpy((buffer_d.data()+pos+2*sizeof(uint32_t)+j*chunk_size), data.data() + (start + j)*chunk_size, writesize);
            }
          } else {
            Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)*2);
            size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)*2);
            uint32_t* buffer32 = (uint32_t*)(buffer_d.data() + pos);
            buffer32[0] = entry.node;
            buffer32[1] = entry.size;
          }
          break;
        }
      }
    }
  });
  Kokkos::fence();
  DEBUG_PRINT("Finished collecting data\n");
  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
  Kokkos::deep_copy(buffer_h, buffer_d);
  Kokkos::fence();
//  file.write((const char*)(buffer_h.data()), num_bytes_h(0));
  printf("Number of bytes written for compact incremental checkpoint: %lu\n", 7*sizeof(uint32_t) + num_bytes_h(0));
  printf("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
  printf("Number of bytes written for metadata: %lu\n", 7*sizeof(uint32_t) + num_bytes_metadata_h(0));
  Kokkos::fence();
  file.close();
  DEBUG_PRINT("Closed file\n");
  Kokkos::fence();
}

int main(int argc, char** argv) {
DEBUG_PRINT("Sanity check\n");
  Kokkos::initialize(argc, argv);
  {
    using Timer = std::chrono::high_resolution_clock;
printf("------------------------------------------------------\n");

    // Process data from checkpoint files
DEBUG_PRINT("Argv[1]: %s\n", argv[1]);
    uint32_t chunk_size = static_cast<uint32_t>(atoi(argv[1]));
DEBUG_PRINT("Loaded chunk size\n");
    std::vector<std::string> chkpt_files;
    std::vector<std::string> full_chkpt_files;
    for(int i=2; i<argc; i++) {
      full_chkpt_files.push_back(std::string(argv[i]));
      chkpt_files.push_back(std::string(argv[i]));
    }
DEBUG_PRINT("Read checkpoint files\n");
    uint32_t num_chkpts = argc-2;
DEBUG_PRINT("Number of checkpoints: %u\n", num_chkpts);

    DistinctMap g_distinct_chunks = DistinctMap(1);
    SharedMap g_shared_chunks = SharedMap(1);
    DistinctMap g_distinct_nodes  = DistinctMap(1);
    SharedMap g_shared_nodes = SharedMap(1);

//    CompactTable<31> updates = CompactTable<31>(1);

    HashList prior_list(0), current_list(0);

    for(uint32_t idx=0; idx<num_chkpts; idx++) {
//      header_t chkpt_header;
//      std::map<int, size_t> region_map;
//      read_full_header(chkpt_files[idx], chkpt_header, region_map);
//DEBUG_PRINT("Read header for checkpoint %u\n", idx);
//      regions_t regions;
DEBUG_PRINT("Processing checkpoint %u\n", idx);
      std::ifstream f;
      f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
DEBUG_PRINT("Set exceptions\n");
      f.open(chkpt_files[idx], std::ifstream::in | std::ifstream::binary);
DEBUG_PRINT("Opened file\n");
      f.seekg(0, f.end);
DEBUG_PRINT("Seek end of file\n");
      size_t data_len = f.tellg();
DEBUG_PRINT("Measure length of file\n");
      f.seekg(0, f.beg);
DEBUG_PRINT("Seek beginning of file\n");
//      f.seekg(chkpt_header.header_size);
DEBUG_PRINT("Length of checkpoint %u: %zd\n", idx, data_len);
      uint32_t num_chunks = data_len/chunk_size;
      if(num_chunks*chunk_size < data_len)
        num_chunks += 1;
      g_distinct_chunks.rehash(g_distinct_chunks.size()+num_chunks);
      g_shared_chunks.rehash(g_shared_chunks.size()+num_chunks);
      g_distinct_nodes.rehash(g_distinct_nodes.size()+2*num_chunks + 1);
      g_shared_nodes.rehash(g_shared_nodes.size()+2*num_chunks + 1);

//      updates.rehash(updates.size() + 2*(data_len/chunk_size)+1);

      size_t name_start = chkpt_files[idx].rfind('/') + 1;
      chkpt_files[idx].erase(chkpt_files[idx].begin(), chkpt_files[idx].begin()+name_start);
      std::string list_timing = "hashstructure.list.filename." + chkpt_files[idx] + 
                                ".chunk_size." + std::to_string(chunk_size) + 
                                ".csv";
      std::string list_metadata = "hashstructure.list.filename." + chkpt_files[idx] + 
                                  ".chunk_size." + std::to_string(chunk_size) + 
                                  ".metadata.csv";
      std::fstream list_fs, list_meta;
      list_fs.open(list_timing, std::fstream::out | std::fstream::app);
      list_meta.open(list_metadata, std::fstream::out | std::fstream::app);
//      list_fs << "CreateList, CompareLists\n";
DEBUG_PRINT("Opened list csv files\n");

      std::string tree_timing = "hashstructure.tree.filename." + chkpt_files[idx] + 
                                ".chunk_size." + std::to_string(chunk_size) + 
                                ".csv";
      std::string tree_metadata = "hashstructure.tree.filename." + chkpt_files[idx] + 
                                  ".chunk_size." + std::to_string(chunk_size) + 
                                  ".metadata.csv";
      std::fstream tree_fs, tree_meta;
      tree_fs.open(tree_timing, std::fstream::out | std::fstream::app);
      tree_meta.open(tree_metadata, std::fstream::out | std::fstream::app);
//      tree_fs << "CreateTree, CompareTrees\n";
DEBUG_PRINT("Opened tree csv files\n");

      Kokkos::View<uint8_t*> first("First region", data_len);
      Kokkos::View<uint8_t*> current("Current region", data_len);
      Kokkos::View<uint8_t*>::HostMirror current_h = Kokkos::create_mirror_view(current);
      f.read((char*)(current_h.data()), data_len);
      Kokkos::deep_copy(current, current_h);
DEBUG_PRINT("Size of full checkpoint: %zd\n", current.size());

//      for(auto &e : region_map) {
//        region_t region;
//        region.size = e.second;
//        size_t data_len = region.size;
//        Kokkos::View<uint8_t*> current("Current region", data_len);
//        Kokkos::View<uint8_t*>::HostMirror current_h = Kokkos::create_mirror_view(current);
//        f.read((char*)(current_h.data()), region.size);
//        Kokkos::deep_copy(current, current_h);
//DEBUG_PRINT("Read region of size %zd\n", data_len);

//        SHA1 hasher;
//        Murmur3C hasher;
        MD5Hash hasher;

//        uint32_t num_chunks = data_len/chunk_size;
//        if(num_chunks*chunk_size < data_len)
//          num_chunks += 1;
//        uint32_t num_nodes = 2*num_chunks-1;
        // Hash list deduplication
        {
          DistinctMap l_distinct_chunks = DistinctMap(num_chunks);
          SharedMap l_shared_chunks = SharedMap(num_chunks);

HashList list0 = HashList(num_chunks);
DEBUG_PRINT("initialized local maps and list\n");
Kokkos::fence();

//          Timer::time_point start_create_list0 = Timer::now();
//          Kokkos::Profiling::pushRegion((std::string("Create List ") + std::to_string(idx)).c_str());
////          HashList list0 = create_hash_list(hasher, current, chunk_size);
//          create_hash_list(hasher, list0, current, chunk_size);
//          Kokkos::Profiling::popRegion();
//          Timer::time_point end_create_list0 = Timer::now();
//
//          Kokkos::fence();
//          Timer::time_point start_find_distinct0 = Timer::now();
//          Kokkos::Profiling::pushRegion((std::string("Find distinct chunks ") + std::to_string(idx)).c_str());
//          find_distinct_chunks(list0, idx, g_distinct_chunks, l_shared_chunks, g_distinct_chunks);
//          Kokkos::Profiling::popRegion();
//          Timer::time_point end_find_distinct0 = Timer::now();
//
//          Kokkos::fence();
//
//          count_distinct_nodes(list0, idx, g_distinct_chunks);
//
//          list_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_create_list0 - start_create_list0).count();
//          list_fs << ",";
//          list_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_find_distinct0 - start_find_distinct0).count();
//          list_fs << "\n";
//
//          // Update global map
//          g_distinct_chunks.rehash(g_distinct_chunks.size()+l_distinct_chunks.size());
//          Kokkos::parallel_for(l_distinct_chunks.capacity(), KOKKOS_LAMBDA(const uint32_t i) {
//            if(l_distinct_chunks.valid_at(i) && !g_distinct_chunks.exists(l_distinct_chunks.key_at(i))) {
//              auto result = g_distinct_chunks.insert(l_distinct_chunks.key_at(i), l_distinct_chunks.value_at(i));
//              if(result.existing()) {
//                printf("Key already exists in global chunk map\n");
//              } else if(result.failed()) {
//                printf("Failed to insert local entry into global chunk map\n");
//              }
//            }
//          });

          Kokkos::fence();
          Timer::time_point start_compare = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Find distinct chunks ") + std::to_string(idx)).c_str());
          compare_lists(hasher, list0, idx, current, chunk_size, l_shared_chunks, l_distinct_chunks, g_shared_chunks, g_distinct_chunks);
          Kokkos::Profiling::popRegion();
          Timer::time_point end_compare = Timer::now();

          Kokkos::fence();

//          count_distinct_nodes(list0, idx, l_distinct_chunks, g_distinct_chunks);
          printf("Size of distinct map: %u\n", l_distinct_chunks.size());
          printf("Size of shared map:   %u\n", l_shared_chunks.size());
          list_meta << l_distinct_chunks.size() << "," << l_shared_chunks.size() << std::endl;

          list_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_compare - start_compare).count();
          list_fs << "\n";

          if(idx == 0) {
            // Update global distinct map
            g_distinct_chunks.rehash(g_distinct_chunks.size()+l_distinct_chunks.size());
            Kokkos::deep_copy(g_distinct_chunks, l_distinct_chunks);
//            Kokkos::parallel_for(l_distinct_chunks.capacity(), KOKKOS_LAMBDA(const uint32_t i) {
//              if(l_distinct_chunks.valid_at(i) && !g_distinct_chunks.exists(l_distinct_chunks.key_at(i))) {
//                auto result = g_distinct_chunks.insert(l_distinct_chunks.key_at(i), l_distinct_chunks.value_at(i));
//                if(result.existing()) {
//                  printf("Key already exists in global chunk map\n");
//                } else if(result.failed()) {
//                  printf("Failed to insert local entry into global chunk map\n");
//                }
//              }
//            });
            // Update global shared map
            g_shared_chunks.rehash(g_shared_chunks.size()+l_shared_chunks.size());
            Kokkos::deep_copy(g_shared_chunks, l_shared_chunks);
//            Kokkos::parallel_for(l_shared_chunks.capacity(), KOKKOS_LAMBDA(const uint32_t i) {
//              if(l_shared_chunks.valid_at(i) && !g_shared_chunks.exists(l_shared_chunks.key_at(i))) {
//                auto result = g_shared_chunks.insert(l_shared_chunks.key_at(i), l_shared_chunks.value_at(i));
//                if(result.existing()) {
//                  printf("Key already exists in global chunk map\n");
//                } else if(result.failed()) {
//                  printf("Failed to insert local entry into global chunk map\n");
//                }
//              }
//            });
	  }

	  prior_list = current_list;
	  current_list = list0;
//if(idx > 0) {
////	  Kokkos::deep_copy(prior_list.list_h, prior_list.list_d);
////	  Kokkos::deep_copy(current_list.list_h, current_list.list_d);
//std::string region_log("region-data-");
//region_log = region_log + chkpt_files[idx] + std::string(".log");
//std::fstream fs(region_log, std::fstream::out|std::fstream::app);
//uint32_t num_changed = print_changed_blocks(fs, current_list.list_d, prior_list.list_d);
//auto contiguous_regions = print_contiguous_regions(region_log, current_list.list_d, prior_list.list_d);
//}

#ifdef WRITE_CHKPT
          uint32_t prior_idx = 0;
          if(idx > 0) {
            prior_idx = idx-1;
          }
          write_incr_chkpt_hashlist(full_chkpt_files[idx]+".hashlist.incr_chkpt",  current, chunk_size, l_distinct_chunks, l_shared_chunks, prior_idx, idx);
#endif
        }
        // Merkle Tree deduplication
        {

//          const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
//          const int32_t levels = INT_MAX;
          MerkleTree tree0 = MerkleTree(2*num_chunks-1);
          CompactTable<31> shared_updates = CompactTable<31>(2*num_chunks - 1);
          CompactTable<31> distinct_updates = CompactTable<31>(2*num_chunks - 1);
          DEBUG_PRINT("Allocated tree and tables\n");

          Kokkos::fence();

          if(idx == 0) {
            Timer::time_point start_create_tree0 = Timer::now();
            Kokkos::Profiling::pushRegion((std::string("Deduplicate chkpt ") + std::to_string(idx)).c_str());
            create_merkle_tree(hasher, tree0, current, chunk_size, idx, g_distinct_nodes, g_shared_nodes);
            Kokkos::Profiling::popRegion();
            Timer::time_point end_create_tree0 = Timer::now();

            printf("Size of shared entries: %u\n", g_shared_nodes.size());
            printf("Size of distinct entries: %u\n", g_distinct_nodes.size());
            printf("Size of shared updates: %u\n", shared_updates.size());
            printf("Size of distinct updates: %u\n", distinct_updates.size());
            tree_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_create_tree0 - start_create_tree0).count();
            tree_fs << "\n";
//            Kokkos::deep_copy(first, current);
#ifdef WRITE_CHKPT
            uint32_t prior_idx = 0;
            if(idx > 0)
              prior_idx = idx-1;
            write_incr_chkpt_hashtree(full_chkpt_files[idx]+".hashtree.incr_chkpt", current, chunk_size, g_distinct_nodes, g_shared_nodes, prior_idx, idx);
#endif
          } else {
            DistinctMap l_distinct_nodes(g_distinct_nodes.capacity());
            SharedMap l_shared_nodes = SharedMap(2*num_chunks-1);
            DEBUG_PRINT("Allocated maps\n");

            Timer::time_point start_create_tree0 = Timer::now();
            Kokkos::Profiling::pushRegion((std::string("Deduplicate chkpt ") + std::to_string(idx)).c_str());
//            deduplicate_data(current, chunk_size, hasher, tree0, idx, g_shared_nodes, g_distinct_nodes, l_shared_nodes, l_distinct_nodes, distinct_updates);
            deduplicate_data(current, chunk_size, hasher, tree0, idx, g_shared_nodes, g_distinct_nodes, l_shared_nodes, l_distinct_nodes, shared_updates, distinct_updates);
//            deduplicate_data_team(current, chunk_size, hasher, 128, tree0, idx, g_shared_nodes, g_distinct_nodes, l_shared_nodes, l_distinct_nodes, updates);
            Kokkos::Profiling::popRegion();
            Timer::time_point end_create_tree0 = Timer::now();

            printf("Size of shared entries: %u\n", l_shared_nodes.size());
            printf("Size of distinct entries: %u\n", l_distinct_nodes.size());
            printf("Size of shared updates: %u\n", shared_updates.size());
            printf("Size of distinct updates: %u\n", distinct_updates.size());
            tree_meta << distinct_updates.size() << "," << shared_updates.size() << std::endl;

//	          if(idx == 0) {
//              // Update global shared map
//              g_shared_nodes.rehash(g_shared_nodes.size()+l_shared_nodes.size());
//              Kokkos::deep_copy(g_shared_nodes, l_shared_nodes);
//            }
            tree_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_create_tree0 - start_create_tree0).count();
//            tree_fs << ",";
//            tree_fs << std::chrono::duration_cast<std::chrono::duration<double>>(end_compare1 - start_compare1).count();
            tree_fs << "\n";

#ifdef WRITE_CHKPT
            uint32_t prior_idx = 0;
            if(idx > 0) {
              prior_idx = idx-1;
              write_incr_chkpt_hashtree(full_chkpt_files[idx]+".hashtree.incr_chkpt", current, chunk_size, distinct_updates, shared_updates, prior_idx, idx);
            }
#endif
          }
          Kokkos::fence();
        }
Kokkos::fence();
DEBUG_PRINT("Closing files\n");
      list_fs.close();
      tree_fs.close();
printf("------------------------------------------------------\n");
    }
  }
printf("------------------------------------------------------\n");
  Kokkos::finalize();
}

