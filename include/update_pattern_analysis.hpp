#ifndef UPDATE_PATTERN_ANALYSIS_HPP
#define UPDATE_PATTERN_ANALYSIS_HPP

#include <Kokkos_Core.hpp>
#include <fstream>

template<typename ViewType>
uint32_t print_changed_blocks(std::fstream& fs, ViewType new_data_d, ViewType old_data_d) {
  auto new_data = Kokkos::create_mirror_view(new_data_d);
  auto old_data = Kokkos::create_mirror_view(old_data_d);
  Kokkos::deep_copy(new_data, new_data_d);
  Kokkos::deep_copy(old_data, old_data_d);
  uint32_t num_changed = 0;
  fs << "Number of blocks: " << new_data.size() << std::endl;
  fs << "Changed blocks: ";
  bool flag = false;
  for(int idx = 0; idx<new_data.size(); idx++) {
    bool same = true;
    if( (old_data(idx).digest[0] != new_data(idx).digest[0]) ||
        (old_data(idx).digest[1] != new_data(idx).digest[1]) ||
        (old_data(idx).digest[2] != new_data(idx).digest[2]) ||
        (old_data(idx).digest[3] != new_data(idx).digest[3]) ||
        (old_data(idx).digest[4] != new_data(idx).digest[4]) )
      same = false;
//    uint32_t old_hash = old_data(idx).digest[0] ^ old_data(idx).digest[1] ^ old_data(idx).digest[2] ^ old_data(idx).digest[3] ^ old_data(idx).digest[4];
//    uint32_t new_hash = new_data(idx).digest[0] ^ new_data(idx).digest[1] ^ new_data(idx).digest[2] ^ new_data(idx).digest[3] ^ new_data(idx).digest[4];
//    if(old_data.data()[idx] != new_data.data()[idx])
    if(!same)
    {
      num_changed += 1;
      if(!flag) {
        fs << "[" << idx << ",";
        flag = true;
      }
    } else {
      if(flag) {
        fs << idx << ") ";
        flag = false;
      }
    }
  }
  if(flag)
    fs << new_data.size() << ")";
  fs << std::endl;
  fs << num_changed << "/" << new_data.size() << " blocks changed " << std::endl;
  return num_changed;
}

template<typename ViewType>
std::map<int, int> print_contiguous_regions(std::fstream& fs, ViewType new_data_d, ViewType old_data_d) {
  auto new_data = Kokkos::create_mirror_view(new_data_d);
  auto old_data = Kokkos::create_mirror_view(old_data_d);
  Kokkos::deep_copy(new_data, new_data_d);
  Kokkos::deep_copy(old_data, old_data_d);
  std::map<int, int> contiguous_regions;
  int largest_region = 1;
  int region_size_counter = 0;
  for(int idx=0; idx<new_data.size(); idx++) {
    bool same = true;
    if( (old_data(idx).digest[0] != new_data(idx).digest[0]) ||
        (old_data(idx).digest[1] != new_data(idx).digest[1]) ||
        (old_data(idx).digest[2] != new_data(idx).digest[2]) ||
        (old_data(idx).digest[3] != new_data(idx).digest[3]) ||
        (old_data(idx).digest[4] != new_data(idx).digest[4]) )
      same = false;
    if(!same)
//    if(old_data.data()[idx] != new_data.data()[idx])
    {
      region_size_counter += 1;
    }
    else 
    {
      if(region_size_counter > largest_region)
      {
        largest_region = region_size_counter;
      }
      if(region_size_counter > 0) {
        auto pos = contiguous_regions.find(region_size_counter);
        if(pos != contiguous_regions.end())
        {
          pos->second = pos->second + 1;
        }
        else
        {
          contiguous_regions.insert(std::pair<int,int>(region_size_counter, 1));
        }
      }
      region_size_counter = 0;
    }
  }
  if(region_size_counter > largest_region)
  {
    largest_region = region_size_counter;
  }
  if(region_size_counter > 0) {
    auto pos = contiguous_regions.find(region_size_counter);
    if(pos != contiguous_regions.end())
    {
      pos->second = pos->second + 1;
    }
    else
    {
      contiguous_regions.insert(std::pair<int,int>(region_size_counter, 1));
    }
  }
  fs << "Largest region: " << largest_region << std::endl;
  fs << "Region map\n";
  for(auto itr = contiguous_regions.begin(); itr != contiguous_regions.end(); ++itr)
  {
    fs << itr->second << " regions of size " << itr->first << std::endl;
  }
  return contiguous_regions;
}

//template<typename ViewType>
//uint32_t print_changed_blocks(std::fstream& fs, ViewType new_data_d, ViewType old_data_d);

//template<typename ViewType>
//std::map<int, int> find_contiguous_regions(ViewType new_data, ViewType old_data);

#endif

