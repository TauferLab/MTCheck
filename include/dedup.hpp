#ifndef __DEDUP_HPP
#define __DEDUP_HPP

#include <string>
#include <cstdint>
#include <vector>
#include <map>

typedef struct header {
  size_t chkpt_size;
  size_t header_size;
  size_t num_regions;
} header_t;

template<typename HashDigest>
struct region_header {
  size_t region_size;
  size_t hash_size;
  size_t chunk_size;
  size_t num_hashes;
  size_t num_unique;
  std::vector<HashDigest> hashes;
  std::vector<size_t> unique_hashes;
};
template <typename HashDigest>
using region_header_t = region_header<HashDigest>;

class deduplicate_module_t {
private:
  template <typename HashDigest>
  void cpu_dedup(std::vector<uint8_t> &data, 
                std::map<HashDigest, size_t>& prev_hashes,
                region_header_t<HashDigest>& header,
                std::vector<uint8_t> &incr_data);
  int gpu_dedup(uint8_t* data, size_t len);
public:
  void deduplicate_file(const std::string &full, const std::string &incr, std::vector<std::string> &prev_chkpts);
};

#endif // __DEDUP_HPP
