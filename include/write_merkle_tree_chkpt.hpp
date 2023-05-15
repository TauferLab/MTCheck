#ifndef WRITE_MERKLE_TREE_CHKPT_HPP
#define WRITE_MERKLE_TREE_CHKPT_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_ScatterView.hpp>
#include <climits>
#include "hash_functions.hpp"
#include "map_helpers.hpp"
#include "kokkos_merkle_tree.hpp"
#include <iostream>
#include "utils.hpp"

template<typename DataView>
std::pair<uint64_t,uint64_t> 
write_incr_chkpt_hashtree_global_mode( 
                                const DataView& data, 
                                Kokkos::View<uint8_t*>& buffer_d, 
                                uint32_t chunk_size, 
                                MerkleTree& curr_tree, 
                                DigestNodeIDDeviceMap& first_occur_d, 
                                const Vector<uint32_t>& first_ocur_vec, 
                                const Vector<uint32_t>& shift_dupl_vec,
                                uint32_t prior_chkpt_id,
                                uint32_t chkpt_id,
                                header_t& header) {
  std::string setup_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Setup");
  Kokkos::Profiling::pushRegion(setup_label);

  uint32_t num_chunks = data.size()/chunk_size;
  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(chunk_size) < data.size()) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;

  Kokkos::View<uint32_t*> region_leaves("Region leaves", num_chunks);
  Kokkos::View<uint32_t*> region_nodes("Region Nodes", first_ocur_vec.size());
  Kokkos::View<uint32_t*> region_len("Region lengths", first_ocur_vec.size());
  Kokkos::View<uint32_t[1]> counter_d("Counter");
  Kokkos::View<uint32_t[1]>::HostMirror counter_h = Kokkos::create_mirror_view(counter_d);
  Kokkos::deep_copy(counter_d, 0);
  Kokkos::View<uint32_t[1]> chunk_counter_d("Counter");
  Kokkos::View<uint32_t[1]>::HostMirror chunk_counter_h = Kokkos::create_mirror_view(chunk_counter_d);
  Kokkos::deep_copy(chunk_counter_d, 0);
  Kokkos::View<uint64_t*> prior_counter_d("Counter for prior repeats", chkpt_id+1);
  Kokkos::View<uint64_t*>::HostMirror prior_counter_h = Kokkos::create_mirror_view(prior_counter_d);
  Kokkos::deep_copy(prior_counter_d, 0);
  Kokkos::Experimental::ScatterView<uint64_t*> prior_counter_sv(prior_counter_d);

  DEBUG_PRINT("Setup counters\n");

  Kokkos::Profiling::popRegion();

  // Filter and count space used for distinct entries
  // Calculate number of chunks each entry maps to
  std::string count_first_ocur_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Count first ocur bytes");
  Kokkos::parallel_for(count_first_ocur_label, Kokkos::RangePolicy<>(0, first_ocur_vec.size()), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node = first_ocur_vec(i);
      NodeID prev = first_occur_d.value_at(first_occur_d.find(curr_tree(node)));
      if(node == prev.node && chkpt_id == prev.tree) {
        uint32_t size = num_leaf_descendents(node, num_nodes);
        uint32_t idx = Kokkos::atomic_fetch_add(&counter_d(0), 1);
        Kokkos::atomic_add(&chunk_counter_d(0), size);
        region_nodes(idx) = node;
        region_len(idx) = size;
      } else {
        printf("Distinct node with different node/tree. Shouldn't happen.\n");
      }
  });
  std::string alloc_bitset_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Allocate bitset");
  Kokkos::Profiling::pushRegion(alloc_bitset_label);

  DEBUG_PRINT("Count distinct bytes\n");

  // Small bitset to record which checkpoints are necessary for restart
  Kokkos::Bitset<Kokkos::DefaultExecutionSpace> chkpts_needed(chkpt_id+1);
  chkpts_needed.reset();
  
  DEBUG_PRINT("Setup chkpt bitset\n");
  Kokkos::Profiling::popRegion();

  // Calculate space needed for repeat entries and number of entries per checkpoint
  Kokkos::RangePolicy<> shared_range_policy(0, shift_dupl_vec.size());
  std::string count_shift_dupl_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Count shift dupl bytes");
  Kokkos::parallel_for(count_shift_dupl_label, shared_range_policy, KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node = shift_dupl_vec(i);
      NodeID prev = first_occur_d.value_at(first_occur_d.find(curr_tree(node)));
      auto prior_counter_sa = prior_counter_sv.access();
      prior_counter_sa(prev.tree) += 1;
      chkpts_needed.set(prev.tree);
  });
  std::string contrib_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Contribute shift dupl");
  Kokkos::Profiling::pushRegion(contrib_label);
  DEBUG_PRINT("Count repeat bytes\n");
  Kokkos::Experimental::contribute(prior_counter_d, prior_counter_sv);
  prior_counter_sv.reset_except(prior_counter_d);

  DEBUG_PRINT("Collect prior counter\n");

  uint32_t num_prior_chkpts = chkpts_needed.count();

  DEBUG_PRINT("Number of checkpoints needed: %u\n", num_prior_chkpts);

  size_t data_offset = first_ocur_vec.size()*sizeof(uint32_t) + shift_dupl_vec.size()*2*sizeof(uint32_t) + static_cast<uint64_t>(num_prior_chkpts)*2*sizeof(uint32_t);
  DEBUG_PRINT("Offset for data: %lu\n", data_offset);
  Kokkos::deep_copy(counter_h, counter_d);
  uint32_t num_distinct = counter_h(0);
  STDOUT_PRINT("Number of distinct regions: %u\n", num_distinct);
  Kokkos::Profiling::popRegion();
  // Dividers for distinct chunks. Number of chunks per region varies.
  // Need offsets for each region so that writes can be done in parallel
  std::string calc_offsets_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Calculate offsets");
  Kokkos::parallel_scan(calc_offsets_label, num_distinct, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
    const uint32_t len = region_len(i);
    if(is_final) region_len(i) = partial_sum;
    partial_sum += len;
  });

  std::string find_region_leaves_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Find region leaves");
  Kokkos::parallel_for(find_region_leaves_label, Kokkos::RangePolicy<>(0,num_distinct), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t offset = region_len(i);
    uint32_t node = region_nodes(i);
    uint32_t size = num_leaf_descendents(node, num_nodes);
    uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
    for(uint32_t j=0; j<size; j++) {
      region_leaves(offset+j) = start+j;
    }
  });

  std::string alloc_buffer_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Allocate buffer");
  Kokkos::Profiling::pushRegion(alloc_buffer_label);
  Kokkos::deep_copy(chunk_counter_h, chunk_counter_d);
  uint64_t buffer_len = sizeof(header_t)+first_ocur_vec.size()*sizeof(uint32_t)+2*sizeof(uint32_t)*static_cast<uint64_t>(chkpts_needed.count())+shift_dupl_vec.size()*2*sizeof(uint32_t)+chunk_counter_h(0)*static_cast<uint64_t>(chunk_size);
  Kokkos::resize(buffer_d, buffer_len);

  Kokkos::deep_copy(counter_d, sizeof(uint32_t)*num_distinct);

  Kokkos::Profiling::popRegion();

  std::string copy_fo_metadata_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Copy first ocur metadata");
  Kokkos::parallel_for(copy_fo_metadata_label, Kokkos::RangePolicy<>(0,num_distinct), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t node = region_nodes(i);
    memcpy(buffer_d.data()+sizeof(header_t)+static_cast<uint64_t>(i)*sizeof(uint32_t), &node, sizeof(uint32_t));
  });

  std::string copy_data_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Copy data");
  Kokkos::parallel_for(copy_data_label, Kokkos::TeamPolicy<>(chunk_counter_h(0), Kokkos::AUTO), 
                         KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
    uint32_t i = team_member.league_rank();
    uint32_t chunk = region_leaves(i);
    uint32_t writesize = chunk_size;
    uint64_t dst_offset = sizeof(header_t)+data_offset+static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);
    uint64_t src_offset = static_cast<uint64_t>(chunk)*static_cast<uint64_t>(chunk_size);
    if(chunk == num_chunks-1) {
      writesize = data.size()-src_offset;
    }

    uint8_t* dst = (uint8_t*)(buffer_d.data()+dst_offset);
    uint8_t* src = (uint8_t*)(data.data()+src_offset);
    team_memcpy(dst, src, writesize, team_member);
  });

  uint32_t num_prior = chkpts_needed.count();

  // Write Repeat map for recording how many entries per checkpoint
  // (Checkpoint ID, # of entries)
  std::string write_repeat_count_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Write repeat count");
  Kokkos::parallel_for(write_repeat_count_label, prior_counter_d.size(), KOKKOS_LAMBDA(const uint32_t i) {
    if(prior_counter_d(i) > 0) {
      uint32_t num_repeats_i = static_cast<uint32_t>(prior_counter_d(i));
      size_t pos = Kokkos::atomic_fetch_add(&counter_d(0), 2*sizeof(uint32_t));
      memcpy(buffer_d.data()+sizeof(header_t)+pos, &i, sizeof(uint32_t));
      memcpy(buffer_d.data()+sizeof(header_t)+pos+sizeof(uint32_t), &num_repeats_i, sizeof(uint32_t));
      DEBUG_PRINT("Wrote table entry (%u,%u) at offset %lu\n", i, num_repeats_i, pos);
    }
  });

  size_t prior_start = static_cast<uint64_t>(num_distinct)*sizeof(uint32_t)+static_cast<uint64_t>(num_prior)*2*sizeof(uint32_t);
  DEBUG_PRINT("Prior start offset: %lu\n", prior_start);

  Kokkos::View<uint32_t*> chkpt_id_keys("Source checkpoint IDs", shift_dupl_vec.size());
  std::string create_keys_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Create chkpt_id_keys");
  Kokkos::parallel_for(create_keys_label, Kokkos::RangePolicy<>(0,shift_dupl_vec.size()), KOKKOS_LAMBDA(const uint32_t i) {
    NodeID info = first_occur_d.value_at(first_occur_d.find(curr_tree(shift_dupl_vec(i))));
    chkpt_id_keys(i) = info.tree;
  });

  std::string sort_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Sort");
  Kokkos::Profiling::pushRegion(sort_label);
  auto keys = chkpt_id_keys;
  using key_type = decltype(keys);
  using Comparator = Kokkos::BinOp1D<key_type>;
  Comparator comp(chkpt_id, 0, chkpt_id);
  Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, shift_dupl_vec.size(), comp, 0);
  bin_sort.create_permute_vector();
  bin_sort.sort(shift_dupl_vec.vector_d);
  bin_sort.sort(chkpt_id_keys);
  Kokkos::Profiling::popRegion();

  // Write repeat entries
  std::string copy_metadata_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Write repeat metadata");
  Kokkos::parallel_for(copy_metadata_label, Kokkos::RangePolicy<>(0, shift_dupl_vec.size()), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t node = shift_dupl_vec(i);
    NodeID prev = first_occur_d.value_at(first_occur_d.find(curr_tree(shift_dupl_vec(i))));
    memcpy(buffer_d.data()+sizeof(header_t)+prior_start+static_cast<uint64_t>(i)*2*sizeof(uint32_t), &node, sizeof(uint32_t));
    memcpy(buffer_d.data()+sizeof(header_t)+prior_start+static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
  });

  DEBUG_PRINT("Wrote shared metadata\n");
  DEBUG_PRINT("Finished collecting data\n");
  header.ref_id = prior_chkpt_id;
  header.chkpt_id = chkpt_id;
  header.datalen = data.size();
  header.chunk_size = chunk_size;
  header.num_first_ocur = first_ocur_vec.size();
  header.num_shift_dupl = shift_dupl_vec.size();
  header.num_prior_chkpts = chkpts_needed.count();
  uint64_t size_metadata = first_ocur_vec.size()*sizeof(uint32_t)+static_cast<uint64_t>(num_prior)*2*sizeof(uint32_t)+shift_dupl_vec.size()*2*sizeof(uint32_t);
  uint64_t size_data = buffer_len - size_metadata;
  return std::make_pair(size_data, size_metadata);
}

#endif // WRITE_MERKLE_TREE_CHKPT_HPP


