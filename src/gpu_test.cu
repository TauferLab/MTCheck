#include <iostream>
#include <fstream>
#include "dedup.hpp"
#include "hash_functions.hpp"
//#include "merkle_tree.hpp"
#include <string>
#include <vector>
#include <cuda.h>

int main(int argc, char**argv) {
  int chunk_size = atoi(argv[1]);
  std::string full_chkpt(argv[2]);
  std::string incr_chkpt = full_chkpt + ".gpu_test.incr_chkpt";

  SHA1 hasher;
  std::vector<std::string> prev_chkpt;
  config_t config;
  config.dedup_on_gpu = false;
  config.chunk_size = chunk_size;
  config.hash_func = &hasher;

  for(int i=3; i<argc; i++) {
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
    region.ptr_type = Host;
    f.read((char*)region.ptr, e.second);
    regions.insert(std::make_pair(e.first, region));
  }

//  std::string incr_chkpt_mem = full_chkpt + ".gpu_test.in_mem.incr_chkpt";
//std::cout << "Starting data mode deduplication\n";
//  module.deduplicate_data(regions, incr_chkpt_mem, prev_chkpt, config);
//std::cout << "Done with data mode\n";

//std::cout << "Deduplicating file\n";
//  module.deduplicate_file(full_chkpt, incr_chkpt, prev_chkpt, config);
//printf("Done deduplicating file\n");

  for(auto &e : regions) {
    void* gpu_ptr;
    cudaMalloc(&gpu_ptr, e.second.size);
    cudaMemcpy(gpu_ptr, e.second.ptr, e.second.size, cudaMemcpyHostToDevice);
    e.second.ptr = gpu_ptr;
    e.second.ptr_type = Cuda;
  }
  printf("Dedup on GPU\n");
  config.dedup_on_gpu = true;
  config.use_merkle_trees = false;
  std::string incr_chkpt_gpu = full_chkpt + ".gpu_test.gpu.incr_chkpt";
  std::cout << "Starting gpu deduplication\n";
  module.deduplicate_data(regions, incr_chkpt_gpu, prev_chkpt, config);
  std::cout << "Done with gpu deduplication\n";

//  printf("Dedup on CPU\n");
//  config.dedup_on_gpu = false;
//  std::string incr_chkpt_cpu = full_chkpt + ".gpu_test.cpu.incr_chkpt";
//  std::cout << "Starting gpu deduplication\n";
//  module.deduplicate_data(regions, incr_chkpt_cpu, prev_chkpt, config);
//  std::cout << "Done with gpu deduplication\n";
}

