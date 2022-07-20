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
#include <utility>
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

KOKKOS_INLINE_FUNCTION
void copy_memory(void* dst, void* src, size_t len) {
#ifdef __CUDA_ARCH__
  size_t offset = 0;
  if((reinterpret_cast<uintptr_t>(dst)%16 == 0) && (reinterpret_cast<uintptr_t>(src)%16 == 0)) {
    for(size_t i=0; i<len/16; i++) {
      ((uint4*) dst)[i] = ((uint4*) src)[i];
    }
    offset = 16*(len/16);
  } else if((reinterpret_cast<uintptr_t>(dst)%8 == 0) && (reinterpret_cast<uintptr_t>(src)%8 == 0)) {
    for(size_t i=0; i<len/8; i++) {
      ((uint2*) dst)[i] = ((uint2*) src)[i];
    }
    offset = 8*(len/8);
  } else if((reinterpret_cast<uintptr_t>(dst)%4 == 0) && (reinterpret_cast<uintptr_t>(src)%4 == 0)) {
    for(size_t i=0; i<len/4; i++) {
      ((uint1*) dst)[i] = ((uint1*) src)[i];
    }
    offset = 4*(len/4);
  } else if((reinterpret_cast<uintptr_t>(dst)%2 == 0) && (reinterpret_cast<uintptr_t>(src)%2 == 0)) {
    for(size_t i=0; i<len/2; i++) {
      ((ushort1*) dst)[i] = ((ushort1*) src)[i];
    }
    offset = 2*(len/2);
  } else if((reinterpret_cast<uintptr_t>(dst)%1 == 0) && (reinterpret_cast<uintptr_t>(src)%1 == 0)) {
    for(size_t i=0; i<len; i++) {
      ((uchar1*) dst)[i] = ((uchar1*) src)[i];
    }
    offset = len;
  }
  for(size_t i=offset; i<len; i++) {
    ((uint8_t*) dst)[i] = ((uint8_t*) src)[i];
  }
#else
  for(size_t i=0; i<len/4; i++) {
    ((uint32_t*) dst)[i] = ((uint32_t*) src)[i];
  }
  size_t offset = 4*(len/4);
  if(len-offset >= 2) {
    ((uint16_t*) dst+offset)[0] = ((uint16_t*) src+offset)[0];
    offset += 2;
  }
  if(len-offset >= 1) {
    ((uint8_t*) dst+offset)[0] = ((uint8_t*) src+offset)[0];
    offset += 1;
  }
#endif
}

std::pair<uint64_t,uint64_t> write_incr_chkpt_hashlist( const std::string& filename, 
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
  Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
  Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
  Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
  Kokkos::deep_copy(num_bytes_d, 0);
  Kokkos::deep_copy(num_bytes_data_d, 0);
  Kokkos::deep_copy(num_bytes_metadata_d, 0);
  if(prior_chkpt_id == chkpt_id) {
    uint64_t buffer_size = 0;
    buffer_size += sizeof(uint32_t)*2*shared.size();
    buffer_size += distinct.size()*(sizeof(uint32_t) + sizeof(HashDigest) + chunk_size);
    Kokkos::View<uint8_t*> buffer_d("Buffer", buffer_size);
    Kokkos::View<uint8_t*>::HostMirror buffer_h = Kokkos::create_mirror_view(buffer_d);
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
//        for(size_t j=0; j<sizeof(HashDigest); j++) {
//          buffer_d(pos+sizeof(uint32_t)+j) = digest.digest[j];
//        }
        memcpy(buffer_d.data()+pos+sizeof(uint32_t), digest.digest, sizeof(HashDigest));
//        copy_memory(buffer_d.data()+pos+sizeof(uint32_t), digest.digest, sizeof(HashDigest));
        uint32_t writesize = chunk_size;
        if(info.node == num_chunks-1) {
          writesize = data.size()-info.node*chunk_size;
        }
//        for(size_t j=0; j<writesize; j++) {
//          buffer_d(pos+sizeof(uint32_t)+sizeof(HashDigest)+j) = data(chunk_size*(info.node)+j);
//        }
        memcpy(buffer_d.data()+pos+sizeof(uint32_t)+sizeof(HashDigest), data.data()+chunk_size*info.node, writesize);
//        copy_memory(buffer_d.data()+pos+sizeof(uint32_t)+sizeof(HashDigest), data.data()+chunk_size*info.node, writesize);
      }
    });
    Kokkos::fence();
    Kokkos::deep_copy(num_bytes_h, num_bytes_d);
    Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
    Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
    Kokkos::deep_copy(buffer_h, buffer_d);
    Kokkos::fence();
    file.write((const char*)(buffer_h.data()), num_bytes_h(0));
    STDOUT_PRINT("Number of bytes written for incremental checkpoint: %lu\n", 7*sizeof(uint32_t) + num_bytes_h(0));
    STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
    STDOUT_PRINT("Number of bytes written for metadata: %lu\n", 7*sizeof(uint32_t) + num_bytes_metadata_h(0));
  } else {
    uint32_t buffer_size = 0;
    buffer_size += sizeof(uint32_t)*2*shared.size();
    buffer_size += distinct.size()*(sizeof(uint32_t) + chunk_size);
    DEBUG_PRINT("Buffer size: %u\n", buffer_size);
    Kokkos::View<uint8_t*> buffer_d("Buffer", buffer_size);
    Kokkos::View<uint8_t*>::HostMirror buffer_h = Kokkos::create_mirror_view(buffer_d);
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
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
        Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + chunk_size);
        uint32_t* buffer32 = (uint32_t*)(buffer_d.data()+pos);
        buffer32[0] = info.node;
        uint32_t writesize = chunk_size;
        if(info.node == num_chunks-1) {
          writesize = data.size()-info.node*chunk_size;
        }
//        for(size_t j=0; j<writesize; j++) {
//          buffer_d(pos+sizeof(uint32_t)+j) = data(chunk_size*(info.node)+j);
//        }
        memcpy(buffer_d.data()+pos+sizeof(uint32_t), data.data()+chunk_size*(info.node), writesize);
//        copy_memory(buffer_d.data()+pos+sizeof(uint32_t), data.data()+chunk_size*(info.node), writesize);
      }
    });
    Kokkos::fence();
    Kokkos::deep_copy(num_bytes_h, num_bytes_d);
    Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
    Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
    Kokkos::deep_copy(buffer_h, buffer_d);
    Kokkos::fence();
    file.write((const char*)(buffer_h.data()), num_bytes_h(0));
    file.flush();
    STDOUT_PRINT("Number of bytes written for incremental checkpoint: %lu\n", 7*sizeof(uint32_t) + num_bytes_h(0));
    STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
    STDOUT_PRINT("Number of bytes written for metadata: %lu\n", 7*sizeof(uint32_t) + num_bytes_metadata_h(0));
  }
  DEBUG_PRINT("Trying to close file\n");
  file.flush();
  file.close();
  DEBUG_PRINT("Closed file\n");
  return std::make_pair(num_bytes_data_h(0), 7*sizeof(uint32_t) + num_bytes_metadata_h(0));
}

std::pair<uint64_t,uint64_t> write_incr_chkpt_hashtree( const std::string& filename, 
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
  DEBUG_PRINT("Wrote header\n");
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
  DEBUG_PRINT("Setup counters and buffers\n");

  Kokkos::parallel_for("Count shared updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
      uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
      uint32_t* buffer32 = (uint32_t*)(buffer_d.data()+pos);
      buffer32[0] = shared.key_at(i);
      buffer32[1] = shared.value_at(i);
    }
  });
  DEBUG_PRINT("Copied metadata for repeats\n");
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
//        for(size_t j=0; j<sizeof(HashDigest); j++) {
//          buffer_d(pos+sizeof(uint32_t)+j) = digest.digest[j];
//        }
        memcpy(buffer_d.data() + (pos+sizeof(uint32_t)), digest.digest, sizeof(HashDigest));
//        copy_memory(buffer_d.data() + (pos+sizeof(uint32_t)), digest.digest, sizeof(HashDigest));
        uint32_t writesize = chunk_size;
        if(info.node == num_nodes-1) {
          writesize = data.size()-(info.node-num_chunks+1)*chunk_size;
        }
//        for(size_t j=0; j<writesize; j++) {
//          buffer_d(pos+sizeof(uint32_t)+sizeof(HashDigest)+j) = data(chunk_size*(info.node-num_chunks+1)+j);
//        }
        memcpy(buffer_d.data()+pos+sizeof(uint32_t)+sizeof(HashDigest), data.data()+chunk_size*(info.node-num_chunks+1), writesize);
//        copy_memory(buffer_d.data()+pos+sizeof(uint32_t)+sizeof(HashDigest), data.data()+chunk_size*(info.node-num_chunks+1), writesize);
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
  Kokkos::fence();
  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
  Kokkos::deep_copy(buffer_h, buffer_d);
  Kokkos::fence();
  DEBUG_PRINT("Copied data to host\n");
  file.write((const char*)(buffer_h.data()), num_bytes_h(0));
  STDOUT_PRINT("Number of bytes written for compact incremental checkpoint: %lu\n", 7*sizeof(uint32_t) + num_bytes_h(0));
  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", 7*sizeof(uint32_t) + num_bytes_metadata_h(0));
  file.close();
  return std::make_pair(num_bytes_data_h(0), 7*sizeof(uint32_t) + num_bytes_metadata_h(0));
}

//template<uint32_t N>
std::pair<uint64_t,uint64_t> write_incr_chkpt_hashtree( const std::string& filename, 
                                const Kokkos::View<uint8_t*>& data, 
                                uint32_t chunk_size, 
                                const CompactTable<10>& distinct, 
                                const CompactTable<10>& shared,
                                uint32_t prior_chkpt_id,
                                uint32_t chkpt_id) {
  std::ofstream file;
  file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  file.open(filename, std::ofstream::out | std::ofstream::binary);
  DEBUG_PRINT("File: %s\n", filename.c_str());
  
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size()) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;

  // Write whether we are storing the hashes, length full checkpoint, chunk size, number of repeat chunks, number of distinct chunks
  file << prior_chkpt_id << chkpt_id << data.size() << chunk_size << shared.size() << distinct.size();

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
            uint32_t writesize = chunk_size*size;
            if(start*chunk_size+writesize > data.size())
              writesize = data.size()-start*chunk_size;
            memcpy(buffer_d.data()+pos+2*sizeof(uint32_t), data.data()+start*chunk_size, writesize);
//            copy_memory(buffer_d.data()+pos+2*sizeof(uint32_t), data.data()+start*chunk_size, writesize);
//            for(uint32_t j=0; j<size; j++) {
//              uint32_t writesize = chunk_size;
//              if(start+j == num_chunks-1) {
//                writesize = data.size()-(start+j)*chunk_size;
//              }
//              memcpy((buffer_d.data()+pos+2*sizeof(uint32_t)+j*chunk_size), data.data() + (start + j)*chunk_size, writesize);
//            }
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
  file.write((const char*)(buffer_h.data()), num_bytes_h(0));
  STDOUT_PRINT("Number of bytes written for compact incremental checkpoint: %lu\n", 7*sizeof(uint32_t) + num_bytes_h(0));
  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", 7*sizeof(uint32_t) + num_bytes_metadata_h(0));
  file.close();
  DEBUG_PRINT("Closed file\n");
  return std::make_pair(num_bytes_data_h(0), 7*sizeof(uint32_t) + num_bytes_metadata_h(0));
}

int main(int argc, char** argv) {
  DEBUG_PRINT("Sanity check\n");
  Kokkos::initialize(argc, argv);
  {
    using Timer = std::chrono::high_resolution_clock;
    STDOUT_PRINT("------------------------------------------------------\n");

    // Process data from checkpoint files
    DEBUG_PRINT("Argv[1]: %s\n", argv[1]);
    uint32_t chunk_size = static_cast<uint32_t>(atoi(argv[1]));
    DEBUG_PRINT("Loaded chunk size\n");
    uint32_t num_chkpts = static_cast<uint32_t>(atoi(argv[2]));
    std::vector<std::string> chkpt_files;
    std::vector<std::string> full_chkpt_files;
    for(int i=0; i<num_chkpts; i++) {
      full_chkpt_files.push_back(std::string(argv[3+i]));
      chkpt_files.push_back(std::string(argv[3+i]));
    }
    DEBUG_PRINT("Read checkpoint files\n");
    DEBUG_PRINT("Number of checkpoints: %u\n", num_chkpts);

    DistinctMap g_distinct_chunks = DistinctMap(1);
    SharedMap g_shared_chunks = SharedMap(1);
    DistinctMap g_distinct_nodes  = DistinctMap(1);
    SharedMap g_shared_nodes = SharedMap(1);

    HashList prior_list(0), current_list(0);

    for(uint32_t idx=0; idx<num_chkpts; idx++) {
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
      DEBUG_PRINT("Length of checkpoint %u: %zd\n", idx, data_len);
      uint32_t num_chunks = data_len/chunk_size;
      if(num_chunks*chunk_size < data_len)
        num_chunks += 1;
      if(idx == 0) {
        g_distinct_chunks.rehash(g_distinct_chunks.size()+num_chunks);
        g_shared_chunks.rehash(g_shared_chunks.size()+num_chunks);
        g_distinct_nodes.rehash(g_distinct_nodes.size()+2*num_chunks + 1);
        g_shared_nodes.rehash(g_shared_nodes.size()+2*num_chunks + 1);
      }

      size_t name_start = chkpt_files[idx].rfind('/') + 1;
      chkpt_files[idx].erase(chkpt_files[idx].begin(), chkpt_files[idx].begin()+name_start);

      std::fstream result_data;
      result_data.open(chkpt_files[idx]+".chunk_size."+std::to_string(chunk_size)+".csv", std::fstream::out | std::fstream::app);

      Kokkos::View<uint8_t*> first("First region", data_len);
      Kokkos::View<uint8_t*> current("Current region", data_len);
      Kokkos::View<uint8_t*>::HostMirror current_h = Kokkos::create_mirror_view(current);
      f.read((char*)(current_h.data()), data_len);
      Kokkos::deep_copy(current, current_h);
      DEBUG_PRINT("Size of full checkpoint: %zd\n", current.size());

//      SHA1 hasher;
//      Murmur3C hasher;
      MD5Hash hasher;

      // Full checkpoint
      {
        std::string filename = full_chkpt_files[idx] + ".full_chkpt";
        std::ofstream file;
        file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        file.open(filename, std::ofstream::out | std::ofstream::binary);
        DEBUG_PRINT("Opened full checkpoint file\n");
        Timer::time_point start_write = Timer::now();
        Kokkos::Profiling::pushRegion((std::string("Write full checkpoint ") + std::to_string(idx)).c_str());
        Kokkos::deep_copy(current_h, current);
//        file.write((const char*)(current_h.data()), current_h.size());
//        file.flush();
        DEBUG_PRINT("Wrote full checkpoint\n");
        Kokkos::Profiling::popRegion();
        Timer::time_point end_write = Timer::now();
        auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write-start_write).count();
        STDOUT_PRINT("Time spent writing full checkpoint: %f\n", write_time);
        result_data << "0.0" << "," << write_time << "," << current.size() << ',' << "0" << ',';
        file.close();
      }
      // Hash list deduplication
      {
        DistinctMap l_distinct_chunks = DistinctMap(num_chunks);
        SharedMap l_shared_chunks = SharedMap(num_chunks);

        HashList list0 = HashList(num_chunks);
        DEBUG_PRINT("initialized local maps and list\n");
        Kokkos::fence();

        Kokkos::fence();
        Timer::time_point start_compare = Timer::now();
        Kokkos::Profiling::pushRegion((std::string("Find distinct chunks ") + std::to_string(idx)).c_str());
        compare_lists(hasher, list0, idx, current, chunk_size, l_shared_chunks, l_distinct_chunks, g_shared_chunks, g_distinct_chunks);
        Kokkos::Profiling::popRegion();
        Timer::time_point end_compare = Timer::now();

        Kokkos::fence();

//        count_distinct_nodes(list0, idx, l_distinct_chunks, g_distinct_chunks);
        STDOUT_PRINT("Size of distinct map: %u\n", l_distinct_chunks.size());
        STDOUT_PRINT("Size of shared map:   %u\n", l_shared_chunks.size());

        auto compare_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_compare - start_compare).count();

        if(idx == 0) {
          // Update global distinct map
          g_distinct_chunks.rehash(g_distinct_chunks.size()+l_distinct_chunks.size());
          Kokkos::deep_copy(g_distinct_chunks, l_distinct_chunks);
          // Update global shared map
          g_shared_chunks.rehash(g_shared_chunks.size()+l_shared_chunks.size());
          Kokkos::deep_copy(g_shared_chunks, l_shared_chunks);
        }

	    prior_list = current_list;
	    current_list = list0;
//      if(idx > 0) {
//      //	  Kokkos::deep_copy(prior_list.list_h, prior_list.list_d);
//      //	  Kokkos::deep_copy(current_list.list_h, current_list.list_d);
//        std::string region_log("region-data-");
//        region_log = region_log + chkpt_files[idx] + "chunk_size." + std::to_string(chunk_size) + std::string(".log");
//        std::fstream fs(region_log, std::fstream::out|std::fstream::app);
//        uint32_t num_changed = print_changed_blocks(fs, current_list.list_d, prior_list.list_d);
//        auto contiguous_regions = print_contiguous_regions(region_log, current_list.list_d, prior_list.list_d);
//      }

#ifdef WRITE_CHKPT
        uint32_t prior_idx = 0;
        if(idx > 0) {
          prior_idx = idx-1;
        }
        Kokkos::fence();
        Timer::time_point start_write = Timer::now();
        Kokkos::Profiling::pushRegion((std::string("Start writing incremental checkpoint ") + std::to_string(idx)).c_str());
        std::pair<uint64_t,uint64_t> datasizes = write_incr_chkpt_hashlist(full_chkpt_files[idx]+".hashlist.incr_chkpt",  current, chunk_size, l_distinct_chunks, l_shared_chunks, prior_idx, idx);
        Kokkos::Profiling::popRegion();
        Timer::time_point end_write = Timer::now();
        auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
        result_data << compare_time << ',' << write_time << ',' << datasizes.first << ',' << datasizes.second << ',';
#endif
      }
      // Merkle Tree deduplication
      {

        MerkleTree tree0 = MerkleTree(2*num_chunks-1);
        CompactTable<10> shared_updates = CompactTable<10>(num_chunks);
        CompactTable<10> distinct_updates = CompactTable<10>(num_chunks);
        DEBUG_PRINT("Allocated tree and tables\n");

        Kokkos::fence();

        if(idx == 0) {
          Timer::time_point start_create_tree0 = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Deduplicate chkpt ") + std::to_string(idx)).c_str());
          create_merkle_tree(hasher, tree0, current, chunk_size, idx, g_distinct_nodes, g_shared_nodes);
          Kokkos::Profiling::popRegion();
          Timer::time_point end_create_tree0 = Timer::now();

          STDOUT_PRINT("Size of shared entries: %u\n", g_shared_nodes.size());
          STDOUT_PRINT("Size of distinct entries: %u\n", g_distinct_nodes.size());
          STDOUT_PRINT("Size of shared updates: %u\n", shared_updates.size());
          STDOUT_PRINT("Size of distinct updates: %u\n", distinct_updates.size());
          auto compare_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_create_tree0 - start_create_tree0).count();
#ifdef WRITE_CHKPT
          uint32_t prior_idx = 0;
          if(idx > 0)
            prior_idx = idx-1;
          Timer::time_point start_write = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Start writing compact incremental checkpoint ") + std::to_string(idx)).c_str());
          auto datasizes = write_incr_chkpt_hashtree(full_chkpt_files[idx]+".hashtree.incr_chkpt", current, chunk_size, g_distinct_nodes, g_shared_nodes, prior_idx, idx);
          Kokkos::Profiling::popRegion();
          Timer::time_point end_write = Timer::now();
          DEBUG_PRINT("Wrote incremental checkpoint\n");
          auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
          result_data << compare_time << ',' << write_time << ',' << datasizes.first << ',' << datasizes.second << std::endl;
#endif
        } else {
          DistinctMap l_distinct_nodes(g_distinct_nodes.capacity());
          SharedMap l_shared_nodes = SharedMap(2*num_chunks-1);
          DEBUG_PRINT("Allocated maps\n");

          Timer::time_point start_create_tree0 = Timer::now();
          Kokkos::Profiling::pushRegion((std::string("Deduplicate chkpt ") + std::to_string(idx)).c_str());
//          deduplicate_data(current, chunk_size, hasher, tree0, idx, g_shared_nodes, g_distinct_nodes, l_shared_nodes, l_distinct_nodes, distinct_updates);
          deduplicate_data(current, chunk_size, hasher, tree0, idx, g_shared_nodes, g_distinct_nodes, l_shared_nodes, l_distinct_nodes, shared_updates, distinct_updates);
//          deduplicate_data_team(current, chunk_size, hasher, 128, tree0, idx, g_shared_nodes, g_distinct_nodes, l_shared_nodes, l_distinct_nodes, updates);
          Kokkos::Profiling::popRegion();
          Timer::time_point end_create_tree0 = Timer::now();

          STDOUT_PRINT("Size of shared entries: %u\n", l_shared_nodes.size());
          STDOUT_PRINT("Size of distinct entries: %u\n", l_distinct_nodes.size());
          STDOUT_PRINT("Size of shared updates: %u\n", shared_updates.size());
          STDOUT_PRINT("Size of distinct updates: %u\n", distinct_updates.size());

//	        if(idx == 0) {
//            // Update global shared map
//            g_shared_nodes.rehash(g_shared_nodes.size()+l_shared_nodes.size());
//            Kokkos::deep_copy(g_shared_nodes, l_shared_nodes);
//          }
          auto compare_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_create_tree0 - start_create_tree0).count();

#ifdef WRITE_CHKPT
          uint32_t prior_idx = 0;
          if(idx > 0) {
            prior_idx = idx-1;
            Timer::time_point start_write = Timer::now();
            Kokkos::Profiling::pushRegion((std::string("Start writing compact incremental checkpoint ") + std::to_string(idx)).c_str());
            auto datasizes = write_incr_chkpt_hashtree(full_chkpt_files[idx]+".hashtree.incr_chkpt", current, chunk_size, distinct_updates, shared_updates, prior_idx, idx);
            Kokkos::Profiling::popRegion();
            Timer::time_point end_write = Timer::now();
            auto write_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_write - start_write).count();
            result_data << compare_time << ',' << write_time << ',' << datasizes.first << ',' << datasizes.second << std::endl;
          }
#endif
        }
        Kokkos::fence();
      }
      Kokkos::fence();
      DEBUG_PRINT("Closing files\n");
      result_data.close();
      STDOUT_PRINT("------------------------------------------------------\n");
    }
    STDOUT_PRINT("------------------------------------------------------\n");
  }
  Kokkos::finalize();
}

