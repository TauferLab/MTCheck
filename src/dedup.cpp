#include "dedup.hpp"
#include "hash_functions.hpp"
#include <vector>
#include <map>
#include <fstream>
#include <iostream>

// Collect unique hashes from prior incremental checkpoints
template <typename HashDigest>
void collect_unique_hashes(std::vector<std::pair<header_t, std::map<int, region_header_t<HashDigest>>>> &prev_chkpt_headers, std::map<int, std::map<HashDigest, size_t>>& unique_hashes) {
//std::cout << "Number of prior checkpoints: " << prev_chkpt_headers.size() << std::endl;
  // Go through prior checkpoint headers
  for(size_t i=0; i<prev_chkpt_headers.size(); i++) {
    header_t main_header = prev_chkpt_headers[i].first;
    std::map<int, region_header_t<HashDigest>>& region_headers = prev_chkpt_headers[i].second;
//std::cout << "Number of regions: " << region_headers.size() << std::endl;
    for(auto it=region_headers.begin(); it!=region_headers.end(); it++) {
//printf("Reading region %d\n", it->first);
      std::map<HashDigest,size_t> hashes;
      auto pos = unique_hashes.find(it->first);
      auto ret = unique_hashes.insert(pos, std::make_pair(it->first, hashes));
      auto hash_it = ret->second;
      for(size_t k=0; k<it->second.num_unique; k++) {
        int id = it->first;
        size_t idx = it->second.unique_hashes[k];
        ret->second.insert(std::make_pair(it->second.hashes[idx], idx));
      }
//      std::cout << "Region has " << ret->second.size() << " unique hashes\n";
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
    f.read((char*)(&header.chkpt_size), sizeof(size_t));
    f.read((char*)(&header.header_size), sizeof(size_t));
    f.read((char*)(&header.num_regions), sizeof(size_t));
std::cout << "Header: " << header.chkpt_size << std::endl;
std::cout << "Header: " << header.header_size << std::endl;
std::cout << "Header: " << header.num_regions << std::endl;
    for(size_t i=0; i<header.num_regions; i++) {
      int id;
      region_header_t<HashDigest> region;
      f.read((char*)&id, sizeof(int));
      f.read((char*)&(region.region_size), sizeof(size_t));
      f.read((char*)&(region.hash_size), sizeof(size_t));
      f.read((char*)&(region.chunk_size), sizeof(size_t));
      f.read((char*)&(region.num_hashes), sizeof(size_t));
      f.read((char*)&(region.num_unique), sizeof(size_t));
std::cout << "Number of unique hashes: " << region.num_unique << std::endl;
      for(size_t j=0; j<region.num_hashes; j++) {
        HashDigest hash;
        f.read((char*)hash.digest, sizeof(HashDigest));
        region.hashes.push_back(hash);
      }
      for(size_t j=0; j<region.num_unique; j++) {
        size_t index;
        f.read((char*)&index, sizeof(size_t));
        region.unique_hashes.push_back(index);
      }
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
  f.write((char*)&header.id, sizeof(int));
  f.write((char*)&header.region_size, sizeof(size_t));
  f.write((char*)&header.hash_size, sizeof(size_t));
  f.write((char*)&header.chunk_size, sizeof(size_t));
  f.write((char*)&header.num_hashes, sizeof(size_t));
  f.write((char*)&header.num_unique, sizeof(size_t));
  for(size_t i=0; i<header.num_hashes; i++) {
    f.write((char*)(header.hashes[i].digest), sizeof(HashDigest));
  }
  for(size_t i=0; i<header.num_unique; i++) {
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
std::cout << "Comparing hashes\n";
std::cout << "Region has " << prev_hashes.size() << " prior existing hashes\n";
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

size_t duplicates = 0;
  for(size_t i=0; i<curr_hashes.size(); i++) {
    HashDigest curr_digest = curr_hashes[i];
    auto iter = changed_regions.find(curr_digest);
    if(iter != changed_regions.end()) {
      auto unique_iter = prev_hashes.find(curr_digest);
      if(unique_iter != prev_hashes.end()) {
duplicates += 1;
        changed_regions.erase(iter);
      }
    }
  }
printf("Found %zd duplicates from prior hashes\n", duplicates);
std::cout << "Region has " << changed_regions.size() << " unique hashes\n";

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

printf("Region map:\n");
for(auto &e: region_map) {
  printf("(%d,%zd)\n", e.first, e.second);
}

  using HashDigest = DefaultHash::Digest;
printf("Size of HashDigest: %d\n", sizeof(HashDigest));

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

//printf("Found unique hashes\n");
//std::cout << unique_hashes.size() << std::endl;
//for(auto &e: unique_hashes) {
//  printf("Region %d has %zd unique hashes\n", e.first, e.second.size());
//  char hex[80];
//  SHA1 sha1;
//  for(auto &hash: e.second) {
//    sha1.digest_to_hex(hash.first.digest, hex);
//    printf("%s\n", hex);
//  }
//}

  // Read checkpoint data and deduplicate
  std::map<int, std::vector<uint8_t>> region_data;
  std::map<int, region_header_t<HashDigest>> region_headers;
  std::ifstream f;
  f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  f.open(full, std::ifstream::in | std::ifstream::binary);
  f.seekg(chkpt_header.header_size);
  for(auto &e : region_map) {
printf("Region: %d\n", e.first);
    std::vector<uint8_t> data(e.second, 0);
    f.read((char*)data.data(), e.second);
    std::vector<uint8_t> incr_data;
    std::map<HashDigest, size_t> updates;
    region_header_t<HashDigest> region_header;
    region_header.id = e.first;
    cpu_dedup(data, unique_hashes[e.first], region_header, incr_data);
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
  os.open(incr, std::ofstream::out | std::ofstream::binary);
  os.write((char*)&(chkpt_header), sizeof(size_t)*3);
printf("Wrote main header \n");

  for(auto &e: region_headers) {
    write_header<HashDigest>(os, e.second);
  }
printf("Wrote region headers\n");

  for(auto &e: region_data) {
    os.write((char*)e.second.data(), e.second.size());
  }

printf("Wrote region data\n");
}
