#include <iostream>
#include <fstream>
#include "dedup.hpp"
#include <string>
#include <vector>

int main(int argc, char**argv) {
  std::string full_chkpt(argv[1]);
  std::string incr_chkpt = full_chkpt + ".incr_chkpt";

  std::vector<std::string> prev_chkpt;

  for(int i=2; i<argc; i++) {
    prev_chkpt.push_back(std::string(argv[i]));
  }

  deduplicate_module_t module;
  
  // Read checkpoint data to memory
  header_t chkpt_header;
  std::map<int, size_t> region_map;
  try {
    std::ifstream f;
    size_t expected_size = 0;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(full_chkpt, std::ifstream::in | std::fstream::binary);
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
    chkpt_header.chkpt_size = file_size;
    chkpt_header.header_size = header_size;
    chkpt_header.num_regions = num_regions;
    f.seekg(0, f.beg);
    f.close();
  } catch(std::ifstream::failure &e) {
    std::cout << "cannot validate header for checkpoint " << full_chkpt << ", reason: " << e.what();
  }

  regions_t regions;
  std::ifstream f;
  f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  f.open(full_chkpt, std::ifstream::in | std::ifstream::binary);
  f.seekg(chkpt_header.header_size);
  for(auto &e : region_map) {
    region_t region;
    region.size = e.second;
    region.ptr = (void*) malloc(e.second);
    f.read((char*)region.ptr, e.second);
    regions.insert(std::make_pair(e.first, region));
  }

  std::string incr_chkpt_mem = full_chkpt + ".in_mem.incr_chkpt";
std::cout << "Starting data mode deduplication\n";
  module.deduplicate_data(regions, incr_chkpt_mem, prev_chkpt);
std::cout << "Done with data mode\n";

std::cout << "Deduplicating file\n";
  module.deduplicate_file(full_chkpt, incr_chkpt, prev_chkpt);
std::cout << "Done deduplicating file\n";
  
}

