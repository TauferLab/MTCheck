#include "dedup.hpp"
#include "hash_functions.hpp"
#include <vector>
#include <map>
#include <fstream>
#include <iostream>

template <typename HashDigest>
void collect_unique_hashes(std::vector<std::pair<header_t, std::map<int, region_header_t<HashDigest>>>> &prev_chkpt_headers, std::map<int, std::map<HashDigest, size_t>>& unique_hashes) {
std::cout << "Number of prior checkpoints: " << prev_chkpt_headers.size() << std::endl;
  for(size_t i=0; i<prev_chkpt_headers.size(); i++) {
    header_t main_header = prev_chkpt_headers[i].first;
    std::map<int, region_header_t<HashDigest>>& region_headers = prev_chkpt_headers[i].second;
std::cout << "Number of regions: " << region_headers.size() << std::endl;
    for(auto it=region_headers.begin(); it!=region_headers.end(); it++) {
      auto hash_it = unique_hashes.find(it->first);
      std::map<HashDigest,size_t> hashes;
      if(hash_it != unique_hashes.end()) {
        hashes = hash_it->second;
      }
      for(size_t k=0; k<it->second.num_unique; k++) {
        int id = it->first;
        size_t idx = it->second.unique_hashes[k];
        hashes.insert(std::make_pair(it->second.hashes[idx], idx));
      }
      unique_hashes.insert(std::make_pair(it->first, hashes));
    }
  }
}

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

template <typename HashDigest>
bool read_incremental_headers(std::string& chkpt, header_t& header, std::map<int, region_header_t<HashDigest>>& region_headers) {
  try {
    std::ifstream f;
    f.open(chkpt, std::ifstream::in | std::ifstream::binary);
//    f.open(chkpt, std::ifstream::in);
//    f.read((char*)&header, 3*sizeof(size_t));
//    f >> header.chkpt_size;
//    f >> header.header_size;
//    f >> header.num_regions;
    f.read((char*)(&header.chkpt_size), sizeof(size_t));
    f.read((char*)(&header.header_size), sizeof(size_t));
    f.read((char*)(&header.num_regions), sizeof(size_t));
std::cout << "Header: " << header.chkpt_size << std::endl;
std::cout << "Header: " << header.header_size << std::endl;
std::cout << "Header: " << header.num_regions << std::endl;
    for(size_t i=0; i<header.num_regions; i++) {
      int id = i;
      region_header_t<HashDigest> region;
//      f.read((char*)&id, sizeof(int));
      f.read((char*)&(region.region_size), sizeof(size_t));
      f.read((char*)&(region.hash_size), sizeof(size_t));
      f.read((char*)&(region.chunk_size), sizeof(size_t));
      f.read((char*)&(region.num_hashes), sizeof(size_t));
      f.read((char*)&(region.num_unique), sizeof(size_t));
//f >> region.region_size;
//f >> region.hash_size;
//f >> region.chunk_size;
//f >> region.num_hashes;
//f >> region.num_unique;
//std::cout << "Incremental header id: " << id << std::endl;
//std::cout << "Incremental header region size: " << region.region_size << std::endl;
//std::cout << "Incremental header hash size: " << region.hash_size << std::endl;
//std::cout << "Incremental header chunk size: " << region.chunk_size << std::endl;
//std::cout << "Incremental header num hashes: " << region.num_hashes << std::endl;
//std::cout << "Incremental header num unique: " << region.num_unique << std::endl;
      for(size_t j=0; j<region.num_hashes; j++) {
        HashDigest hash;
//        f.read((char*)&hash, sizeof(region.hash_size));
        f.read((char*)&hash, 160);
//        for(size_t k=0; k<160; k++) {
//          f >> hash.digest[k];
//        }
        region.hashes.push_back(hash);
      }
//std::cout << "Incremental header unique hashes: ";
      for(size_t j=0; j<region.num_unique; j++) {
        size_t index;
        f.read((char*)&index, sizeof(size_t));
//        f >> index;
//std::cout << index << " ";
        region.unique_hashes.push_back(index);
      }
//std::cout << std::endl;
      region_headers.insert(std::make_pair(id, region));
    }
    return true;
  } catch(std::ifstream::failure &e) {
    std::cout << "Error reading header for checkpoint " << chkpt << std::endl;
    return false;
  }
}

template<typename HashDigest>
void write_header(std::ofstream& f, region_header_t<HashDigest>& header) {
//  std::cout << "Region size: " << header.region_size << std::endl;
//  std::cout << "Hash size  : " << header.hash_size << std::endl;
//  std::cout << "Chunk size : " << header.chunk_size << std::endl;
//  std::cout << "Num hashes : " << header.num_hashes << std::endl;
//  std::cout << "Num unique : " << header.num_unique << std::endl;
//  std::cout << "Hashes: ";
//  for(size_t i=0; i<header.num_hashes; i++) {
//    HashDigest digest;
//    SHA1 sha;
//    sha.digest_to_hex(header.hashes[i].digest, (char*)(digest.digest));
//    std::cout << std::string((char*)digest.digest) << " ";
//  }
//  std::cout << std::endl;
//  std::cout << "Unique hashes: ";
//  for(size_t i=0; i<header.num_hashes; i++) {
//    std::cout << header.unique_hashes[i] << " ";
//  }
//  std::cout << std::endl;

//  f << header.region_size;
//  f << header.hash_size;
//  f << header.chunk_size;
//  f << header.num_hashes;
//  f << header.num_unique;
  f.write((char*)&header.region_size, sizeof(size_t));
  f.write((char*)&header.hash_size, sizeof(size_t));
  f.write((char*)&header.chunk_size, sizeof(size_t));
  f.write((char*)&header.num_hashes, sizeof(size_t));
  f.write((char*)&header.num_unique, sizeof(size_t));
  for(size_t i=0; i<header.num_hashes; i++) {
    HashDigest digest;
    SHA1 sha;
    sha.digest_to_hex(header.hashes[i].digest, (char*)(digest.digest));
//    f.write((char*)(header.hashes[i].digest), sizeof(HashDigest));
    f.write((char*)(header.hashes[i].digest), 160);
  }
  for(size_t i=0; i<header.num_unique; i++) {
//    f << header.unique_hashes[i];
    f.write((char*)&header.unique_hashes[i], sizeof(size_t));
  }
}

// Check if hashes are the same
bool identical_hashes(const uint32_t* a, const uint32_t* b, size_t len) {
  return memcmp(a, b, len*sizeof(uint32_t)) == 0;
}

template<typename HashDigest>
bool identical_hashes(const HashDigest& a, HashDigest& b, size_t len) {
  for(size_t i=0; i<len; i++) {
    if(a.digest[i] != b.digest[i]) {
      return false;
    }
  }
  return true;
}

// Calculate hashes
template<typename T, typename HashDigest>
void calculate_hashes(std::vector<T>& data, 
                      std::vector<HashDigest>& hashes, 
                      const int chunk_size) {
  SHA1 hasher;
  const size_t data_len = data.size();
  size_t num_hashes = data_len*sizeof(T) / chunk_size;
  if(chunk_size*num_hashes < data_len*sizeof(T))
    num_hashes += 1;
  for(size_t idx=0; idx<num_hashes; idx++) {
    int block_size = chunk_size;
    if(chunk_size*(idx+1) > data_len*sizeof(T))
      block_size = (data_len*sizeof(T))-(idx*chunk_size);
    HashDigest digest;
    hasher.hash((uint8_t*)(data.data()) + (idx*chunk_size), block_size, (uint8_t*)(digest.digest));
    hashes.push_back(digest);
  }
}

// Compare set of hashes with the previous set
template<typename HashDigest>
void compare_hashes(std::vector<HashDigest>& curr_hashes, 
                    std::map<HashDigest, size_t>& prev_hashes,
                    std::map<HashDigest, size_t>& changed_regions,
                    size_t& num_changes, 
                    const int hash_len) {
  if(changed_regions.size() == 0) {
    for(size_t i=0; i<curr_hashes.size(); i++) {
      bool unique = true;
      HashDigest curr_digest = curr_hashes[i];
      for(size_t j=0; j<curr_hashes.size(); j++) {
        if(i != j && identical_hashes(curr_hashes[j], curr_hashes[i], hash_len)) {
          unique = false;
          if(i < j) {
            changed_regions.insert(std::make_pair(curr_digest, i));
          }
          break;
        }
      }
      if(unique) {
        changed_regions.insert(std::make_pair(curr_digest, i));
      }
    }
  }

  for(size_t i=0; i<curr_hashes.size(); i++) {
    HashDigest curr_digest = curr_hashes[i];
    auto iter = changed_regions.find(curr_digest);
    if(iter != changed_regions.end()) {
      auto unique_iter = prev_hashes.find(curr_digest);
      if(unique_iter != prev_hashes.end()) {
//std::cout << "Found prior hash\n";
        changed_regions.erase(iter);
      }
    }
  }

  num_changes = changed_regions.size();
}

// Gather changes into a contiguous buffer
template <typename T, typename HashDigest>
void gather_changes(T* data,
                    size_t data_len,
                    std::map<HashDigest, size_t>& changed_regions,
                    size_t& diff_size,
                    int chunk_size,
                    std::vector<uint8_t>& buffer) {
  buffer.resize(changed_regions.size()*chunk_size, 0);
  diff_size = 0;
  size_t counter=0;
  for(auto it=changed_regions.begin(); it!=changed_regions.end(); ++it) {
    size_t num_write = chunk_size;
    if(chunk_size*(it->second+1) > data_len*sizeof(T)) 
      num_write = (data_len*sizeof(T)) - it->second*chunk_size;
    diff_size += num_write;
    size_t pos = counter++;
    memcpy((uint8_t*)(buffer.data())+chunk_size*pos, (uint8_t*)(data)+chunk_size*it->second, num_write);
  }
}

template<typename HashDigest>
void deduplicate_module_t::cpu_dedup(std::vector<uint8_t> &data, 
                                    std::map<HashDigest, size_t>& prev_hashes,
                                    region_header_t<HashDigest>& header,
                                    std::vector<uint8_t> &incr_data) {
  int chunk_size = 1024;
  size_t diff_size = 0;
  size_t num_changes = 0;
  std::vector<uint8_t> buffer;
  size_t hash_len = sizeof(HashDigest);
  std::map<HashDigest, size_t> unique_hashes;

  calculate_hashes(data, header.hashes, chunk_size);
  compare_hashes(header.hashes, prev_hashes, unique_hashes, num_changes, hash_len);
  gather_changes(data.data(), data.size(), unique_hashes, diff_size, chunk_size, incr_data);

  header.hash_size = sizeof(HashDigest);
  header.chunk_size = chunk_size;
  header.num_hashes = header.hashes.size();
  header.num_unique = unique_hashes.size();
  for(auto it=unique_hashes.begin(); it!=unique_hashes.end(); it++) {
    header.unique_hashes.push_back(it->second);
  }
  header.region_size = sizeof(size_t)*5 + sizeof(HashDigest)*header.hashes.size() + sizeof(size_t)*header.unique_hashes.size();
}

//int deduplicate_module_t::gpu_dedup(uint8_t* data, size_t len) {
//}

void deduplicate_module_t::deduplicate_file(const std::string &full, const std::string &incr, std::vector<std::string> &prev_chkpts) {
for(int i=0; i<prev_chkpts.size(); i++) {
  std::cout << prev_chkpts[i] << std::endl;
}
  // Read checkpoint header
  header_t chkpt_header;
  std::map<int, size_t> region_map;
  read_full_header(full, chkpt_header, region_map);
printf("Checkpoint size: %zd\n", chkpt_header.chkpt_size);
printf("Header size: %zd\n", chkpt_header.header_size);
printf("Num regions: %zd\n", chkpt_header.num_regions);

  using HashDigest = DefaultHash::Digest;

  // Read headers from prior checkpoints
  std::vector<std::pair<header_t, std::map<int, region_header_t<HashDigest>>>> prev_chkpt_headers;
  for(size_t i=0; i<prev_chkpts.size(); i++) {
std::cout << "Reading checkpoint " << prev_chkpts[i] << std::endl;
    header_t main_header;
    std::map<int, region_header_t<HashDigest>> region_headers;
    read_incremental_headers(prev_chkpts[i], main_header, region_headers);
    prev_chkpt_headers.push_back(std::make_pair(main_header, region_headers));
  }
printf("Read headers\n");
printf("Collecting unique hashes\n");

  // Get unique hashes
  std::map<int, std::map<HashDigest,size_t>> unique_hashes;
  collect_unique_hashes(prev_chkpt_headers, unique_hashes);

printf("Found unique hashes\n");
std::cout << unique_hashes.size() << std::endl;
for(auto &e: unique_hashes) {
  printf("Region %d has %zd unique hashes\n", e.first, e.second.size());
}

  // Read checkpoint data and deduplicate
  std::map<int, std::vector<uint8_t>> region_data;
  std::map<int, region_header_t<HashDigest>> region_headers;
  std::ifstream f;
  f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  f.open(full, std::ifstream::in | std::ifstream::binary);
  f.seekg(chkpt_header.header_size);
  for(auto &e : region_map) {
    std::vector<uint8_t> data(e.second, 0);
    f.read((char*)data.data(), e.second);
    std::vector<uint8_t> incr_data;
    std::map<HashDigest, size_t> updates;
    region_header_t<HashDigest> region_header;
    cpu_dedup(data, unique_hashes[e.first], region_header, incr_data);
//printf("Deduplicated region %d\n", e.first);
//  std::cout << "Region size: " << region_header.region_size << std::endl;
//  std::cout << "Hash size  : " << region_header.hash_size << std::endl;
//  std::cout << "Chunk size : " << region_header.chunk_size << std::endl;
//  std::cout << "Num hashes : " << region_header.num_hashes << std::endl;
//  std::cout << "Num unique : " << region_header.num_unique << std::endl;
    region_data.insert(std::make_pair(e.first, incr_data));
    region_headers.insert(std::make_pair(e.first, region_header));
  }

printf("Deduplicated checkpoints\n");

  // Create main header
  size_t chkpt_size = 0;
  for(auto &region: region_data) {
    chkpt_size += region.second.size();
  }
  chkpt_header.chkpt_size = chkpt_size;
  size_t header_size = 3*sizeof(size_t);
  for(auto &e: region_headers) {
    header_size += sizeof(size_t)*5;
    header_size += sizeof(HashDigest)*e.second.hashes.size();
    header_size += sizeof(size_t)*e.second.unique_hashes.size();
printf("Size of header %d is %zd\n", e.first, header_size);
  }
  chkpt_header.header_size = header_size;
  chkpt_header.num_regions = region_data.size();

printf("Create main header\n");
printf("Checkpoint size: %zd\n", chkpt_header.chkpt_size);
printf("Header size: %zd\n", chkpt_header.header_size);
printf("Num regions: %zd\n", chkpt_header.num_regions);

  // Write main header, incremental headers, and data
std::cout << incr << std::endl;
  std::ofstream os;
  os.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  os.open(incr, std::ifstream::out | std::ifstream::binary);
  os.write((char*)&(chkpt_header), sizeof(size_t)*3);

  for(auto &e: region_headers) {
    write_header<HashDigest>(os, e.second);
  }

  for(auto &e: region_data) {
    os.write((char*)e.second.data(), e.second.size());
  }
}
