#ifndef DEDUP_MERKLE_TREE_HPP
#define DEDUP_MERKLE_TREE_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_Sort.hpp>
#include <queue>
#include <iostream>
#include <climits>
#include "hash_functions.hpp"
#include "map_helpers.hpp"
#include "kokkos_merkle_tree.hpp"
#include "kokkos_hash_list.hpp"
#include "reference_impl.hpp"
#include "utils.hpp"

//template<class Hasher, typename DataView>
template<typename DataView>
void deduplicate_data_deterministic_baseline(DataView& data, 
                      const uint32_t chunk_size, 
//                      const Hasher hasher, 
                      MerkleTree& curr_tree, 
                      const uint32_t chkpt_id, 
                      DigestNodeIDDeviceMap& first_occur_d, 
                      Vector<uint32_t>& shift_dupl_updates,
                      Vector<uint32_t>& first_ocur_updates) {
  uint32_t num_chunks = (curr_tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = curr_tree.tree_h.extent(0);
  STDOUT_PRINT("Num chunks: %u\n", num_chunks);
  STDOUT_PRINT("Num nodes: %u\n", num_nodes);

#ifdef STATS
  Kokkos::View<uint64_t[3]> chunk_counters("Chunk counters");
  Kokkos::View<uint64_t[3]> region_counters("Region counters");
  Kokkos::View<uint64_t*> first_region_sizes("Num first regions per size", num_chunks+1);
  Kokkos::View<uint64_t*> shift_region_sizes("Num shift regions per size", num_chunks+1);
  Kokkos::deep_copy(chunk_counters, 0);
  Kokkos::deep_copy(region_counters, 0);
  Kokkos::deep_copy(first_region_sizes, 0);
  Kokkos::deep_copy(shift_region_sizes, 0);
  auto chunk_counters_h  = Kokkos::create_mirror_view(chunk_counters);
  auto region_counters_h = Kokkos::create_mirror_view(region_counters);
  auto first_region_sizes_h = Kokkos::create_mirror_view(first_region_sizes);
  auto shift_region_sizes_h = Kokkos::create_mirror_view(shift_region_sizes);
  Kokkos::Experimental::ScatterView<uint64_t[3]> chunk_counters_sv(chunk_counters);
  Kokkos::Experimental::ScatterView<uint64_t[3]> region_counters_sv(region_counters);
  Kokkos::Experimental::ScatterView<uint64_t*> first_region_sizes_sv(first_region_sizes);
  Kokkos::Experimental::ScatterView<uint64_t*> shift_region_sizes_sv(shift_region_sizes);
#endif

  uint32_t level_beg = 0;
  uint32_t level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  Kokkos::View<char*> labels("Labels", num_nodes);
  Kokkos::deep_copy(labels, DONE);
  Vector<uint32_t> tree_roots(num_chunks);

  // Process leaves first
  using member_type = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type;
  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>(((num_nodes-num_chunks+1)/TEAM_SIZE)+1, TEAM_SIZE);
  Kokkos::parallel_for("Baseline: Leaves", team_policy, 
  KOKKOS_LAMBDA(member_type team_member) {
    uint64_t i=team_member.league_rank();
    uint64_t j=team_member.team_rank();
    uint64_t leaf = num_chunks-1+i*team_member.team_size()+j;
    if(leaf < num_nodes) {
#ifdef STATS
      auto chunk_counters_sa = chunk_counters_sv.access();
      auto region_counters_sa = region_counters_sv.access();
#endif
      uint32_t num_bytes = chunk_size;
      uint64_t offset = static_cast<uint64_t>(leaf-num_chunks+1)*static_cast<uint64_t>(chunk_size);
      if(leaf == num_nodes-1) // Calculate how much data to hash
        num_bytes = data.size()-offset;
      // Hash chunk
      HashDigest digest;
      hash(data.data()+offset, num_bytes, digest.digest);
      // Insert into table
      auto result = first_occur_d.insert(digest, NodeID(leaf, chkpt_id)); 
      if(digests_same(digest, curr_tree(leaf))) { // Fixed duplicate chunk
        labels(leaf) = FIXED_DUPL;
      } else if(result.success()) { // First occurrence chunk
        labels(leaf) = FIRST_OCUR;
      } else if(result.existing()) { // Shifted duplicate chunk
        auto& info = first_occur_d.value_at(result.index());
        if(info.tree == chkpt_id) {
          Kokkos::atomic_min(&info.node, leaf);
          labels(leaf) = FIRST_OCUR;
        } else {
          labels(leaf) = SHIFT_DUPL;
        }
      }
      curr_tree(leaf) = digest;
#ifdef STATS
      chunk_counters_sa(labels(leaf)) += 1;
#endif
    }
  });
  Kokkos::parallel_for("Baseline: Leaves: Choose first occurrences", Kokkos::RangePolicy<>(num_chunks-1, num_nodes), KOKKOS_LAMBDA(const uint32_t leaf) {
#ifdef STATS
    auto chunk_counters_sa = chunk_counters_sv.access();
    auto region_counters_sa = region_counters_sv.access();
#endif
    if(labels(leaf) == FIRST_OCUR) {
      auto info = first_occur_d.value_at(first_occur_d.find(curr_tree(leaf))); 
      if((info.tree == chkpt_id) && (leaf != info.node)) {
        labels(leaf) = SHIFT_DUPL;
#ifdef STATS
        chunk_counters_sa(labels(leaf)) += 1;
#endif
      }
    }
  });

  // Build up forest of Merkle Trees
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  while(level_beg <= num_nodes) { // Intentional unsigned integer underflow
    Kokkos::parallel_for("Baseline: Build First Occurrence Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) == FIRST_OCUR && labels(child_r) == FIRST_OCUR) {
          labels(node) = FIRST_OCUR;
          hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
          first_occur_d.insert(curr_tree(node), NodeID(node, chkpt_id));
        }
        if(node == 0 && labels(0) == FIRST_OCUR) {
          first_ocur_updates.push(node);
#ifdef STATS
          auto first_region_sizes_sa = first_region_sizes_sv.access();
          first_region_sizes_sa(num_chunks) += 1;
#endif
        }
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  while(level_beg <= num_nodes) { // unsigned integer underflow
    Kokkos::parallel_for("Baseline: Build Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
#ifdef STATS
      auto region_counters_sa = region_counters_sv.access();
      auto first_region_sizes_sa = first_region_sizes_sv.access();
      auto shift_region_sizes_sa = shift_region_sizes_sv.access();
#endif
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) != labels(child_r)) { // Children have different labels
          labels(node) = DONE;
          if((labels(child_l) != FIXED_DUPL) && (labels(child_l) != DONE)) {
            if(labels(child_l) == SHIFT_DUPL) {
              shift_dupl_updates.push(child_l);
#ifdef STATS
              shift_region_sizes_sa(num_leaf_descendents(child_l, num_nodes)) += 1;
#endif
            } else {
              first_ocur_updates.push(child_l);
#ifdef STATS
              first_region_sizes_sa(num_leaf_descendents(child_l, num_nodes)) += 1;
#endif
            }
#ifdef STATS
            region_counters_sa(labels(child_l)) += 1;
#endif
          }
          if((labels(child_r) != FIXED_DUPL) && (labels(child_r) != DONE)) {
            if(labels(child_r) == SHIFT_DUPL) {
              shift_dupl_updates.push(child_r);
#ifdef STATS
              shift_region_sizes_sa(num_leaf_descendents(child_r, num_nodes)) += 1;
#endif
            } else {
              first_ocur_updates.push(child_r);
#ifdef STATS
              first_region_sizes_sa(num_leaf_descendents(child_r, num_nodes)) += 1;
#endif
            }
#ifdef STATS
            region_counters_sa(labels(child_r)) += 1;
#endif
          }
        } else if(labels(child_l) == FIXED_DUPL) { // Children are both fixed duplicates
          labels(node) = FIXED_DUPL;
        } else if(labels(child_l) == SHIFT_DUPL) { // Children are both shifted duplicates
          if(first_occur_d.exists(curr_tree(node))) { // This node is also a shifted duplicate
            labels(node) = SHIFT_DUPL;
          } else { // Node is not a shifted duplicate. Save child trees
            labels(node) = DONE; // Add children to tree root maps
            shift_dupl_updates.push(child_l);
            shift_dupl_updates.push(child_r);
#ifdef STATS
            region_counters_sa(SHIFT_DUPL) += 2;
            shift_region_sizes_sa(num_leaf_descendents(child_l, num_nodes)) += 1;
            shift_region_sizes_sa(num_leaf_descendents(child_r, num_nodes)) += 1;
#endif
          }
        }
      }
    });
    Kokkos::parallel_for("Baseline: Build Forest: Insert entries", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
        first_occur_d.insert(curr_tree(node), NodeID(node, chkpt_id));
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }

#ifdef STATS
  Kokkos::Experimental::contribute(region_counters, region_counters_sv);
  Kokkos::Experimental::contribute(first_region_sizes, first_region_sizes_sv);
  Kokkos::Experimental::contribute(shift_region_sizes, shift_region_sizes_sv);
  Kokkos::fence();
  Kokkos::deep_copy(chunk_counters_h, chunk_counters);
  Kokkos::deep_copy(region_counters_h, region_counters);
  Kokkos::deep_copy(first_region_sizes_h, first_region_sizes);
  Kokkos::deep_copy(shift_region_sizes_h, shift_region_sizes);

  STDOUT_PRINT("Checkpoint %u\n", chkpt_id);
  STDOUT_PRINT("Number of first occurrence chunks:  %lu\n", chunk_counters_h(FIRST_OCUR));
  STDOUT_PRINT("Number of fixed duplicate chunks:   %lu\n", chunk_counters_h(FIXED_DUPL));
  STDOUT_PRINT("Number of shifted duplicate chunks: %lu\n", chunk_counters_h(SHIFT_DUPL));
  STDOUT_PRINT("Number of first occurrence regions:  %lu\n", region_counters_h(FIRST_OCUR));
  STDOUT_PRINT("Number of fixed duplicate regions:   %lu\n", region_counters_h(FIXED_DUPL));
  STDOUT_PRINT("Number of shifted duplicate regions: %lu\n", region_counters_h(SHIFT_DUPL));
  for(uint32_t i=0; i<num_chunks+1; i++) {
    if(first_region_sizes_h(i) > 0) {
      printf("First Occurrence: Num regions of size %u: %lu\n", i, first_region_sizes_h(i));
    }
  }
  for(uint32_t i=0; i<num_chunks+1; i++) {
    if(shift_region_sizes_h(i) > 0) {
      printf("Shift Occurrence: Num regions of size %u: %lu\n", i, shift_region_sizes_h(i));
    }
  }
#endif
  Kokkos::fence();
  return;
}

//template<class Hasher, typename DataView>
template<typename DataView>
void deduplicate_data_deterministic(DataView& data, 
                      const uint32_t chunk_size, 
//                      const Hasher hasher, 
                      MerkleTree& curr_tree, 
                      Kokkos::View<char*>& labels,
                      const uint32_t chkpt_id, 
                      DigestNodeIDDeviceMap& first_occur_d, 
                      Vector<uint32_t>& shift_dupl_vec,
                      Vector<uint32_t>& first_ocur_vec) {
  std::string setup_label = std::string("Deduplicate Checkpoint ") + std::to_string(chkpt_id) + std::string(": Setup");
  Kokkos::Profiling::pushRegion(setup_label);
  uint32_t num_chunks = (curr_tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = curr_tree.tree_h.extent(0);
  STDOUT_PRINT("Num chunks: %u\n", num_chunks);
  STDOUT_PRINT("Num nodes: %u\n", num_nodes);

#ifdef STATS
  Kokkos::View<uint64_t[3]> chunk_counters("Chunk counters");
  Kokkos::View<uint64_t[3]> region_counters("Region counters");
  Kokkos::View<uint64_t*> first_region_sizes("Num first regions per size", num_chunks+1);
  Kokkos::View<uint64_t*> shift_region_sizes("Num shift regions per size", num_chunks+1);
  Kokkos::deep_copy(chunk_counters, 0);
  Kokkos::deep_copy(region_counters, 0);
  Kokkos::deep_copy(first_region_sizes, 0);
  Kokkos::deep_copy(shift_region_sizes, 0);
  auto chunk_counters_h  = Kokkos::create_mirror_view(chunk_counters);
  auto region_counters_h = Kokkos::create_mirror_view(region_counters);
  auto first_region_sizes_h = Kokkos::create_mirror_view(first_region_sizes);
  auto shift_region_sizes_h = Kokkos::create_mirror_view(shift_region_sizes);
  Kokkos::Experimental::ScatterView<uint64_t[3]> chunk_counters_sv(chunk_counters);
  Kokkos::Experimental::ScatterView<uint64_t[3]> region_counters_sv(region_counters);
  Kokkos::Experimental::ScatterView<uint64_t*> first_region_sizes_sv(first_region_sizes);
  Kokkos::Experimental::ScatterView<uint64_t*> shift_region_sizes_sv(shift_region_sizes);
#endif

  uint32_t level_beg = 0;
  uint32_t level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  Kokkos::deep_copy(labels, DONE);
  Kokkos::Profiling::popRegion();

  // Process leaves first
  std::string leaves_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Leaves");
  using member_type = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type;
  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>(((num_nodes-num_chunks+1)/TEAM_SIZE)+1, TEAM_SIZE);
  Kokkos::parallel_for(leaves_label, team_policy, 
  KOKKOS_LAMBDA(member_type team_member) {
    uint64_t i=team_member.league_rank();
    uint64_t j=team_member.team_rank();
#ifdef STATS
    auto chunk_counters_sa = chunk_counters_sv.access();
#endif
    uint64_t leaf = num_chunks-1+i*team_member.team_size()+j;
    if(leaf < num_nodes) {
      uint32_t num_bytes = chunk_size;
      uint64_t offset = static_cast<uint64_t>(leaf-(num_chunks-1))*static_cast<uint64_t>(chunk_size);
      if(leaf == num_nodes-1) // Calculate how much data to hash
        num_bytes = data.size()-offset;
      // Hash chunk
      HashDigest digest;
      hash(data.data()+offset, num_bytes, digest.digest);
      // Insert into table
      auto result = first_occur_d.insert(digest, NodeID(leaf, chkpt_id)); 
      if(digests_same(digest, curr_tree(leaf))) { // Fixed duplicate chunk
        labels(leaf) = FIXED_DUPL;
#ifdef STATS
      chunk_counters_sa(labels(leaf)) += 1;
#endif
      } else if(result.success()) { // First occurrence chunk
        labels(leaf) = FIRST_OCUR;
        curr_tree(leaf) = digest;
#ifdef STATS
      chunk_counters_sa(labels(leaf)) += 1;
#endif
      } else if(result.existing()) { // Shifted duplicate chunk
        auto& info = first_occur_d.value_at(result.index());
        if(info.tree == chkpt_id) {
          Kokkos::atomic_min(&info.node, leaf);
          labels(leaf) = FIRST_OCUR;
        } else {
          labels(leaf) = SHIFT_DUPL;
#ifdef STATS
      chunk_counters_sa(labels(leaf)) += 1;
#endif
        }
        curr_tree(leaf) = digest;
      }
    }
  });
  std::string leaves_first_ocur_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Leaves: Choose first occurrences");
  Kokkos::parallel_for(leaves_first_ocur_label, Kokkos::RangePolicy<>(num_chunks-1, num_nodes), KOKKOS_LAMBDA(const uint32_t leaf) {
    if(labels(leaf) == FIRST_OCUR) {
#ifdef STATS
      auto chunk_counters_sa = chunk_counters_sv.access();
#endif
      auto info = first_occur_d.value_at(first_occur_d.find(curr_tree(leaf))); 
      if((info.tree == chkpt_id) && (leaf != info.node)) {
        labels(leaf) = SHIFT_DUPL;
      }
#ifdef STATS
      chunk_counters_sa(labels(leaf)) += 1;
#endif
    }
  });

  // Build up forest of Merkle Trees
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  while(level_beg <= num_nodes) { // Intensional unsigned integer underflow
    std::string first_ocur_forest_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Build First Occurrence Forest");
    Kokkos::parallel_for(first_ocur_forest_label, Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
#ifdef STATS
      auto region_counters_sa = region_counters_sv.access();
#endif
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) == FIRST_OCUR && labels(child_r) == FIRST_OCUR) {
          labels(node) = FIRST_OCUR;
          hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
          first_occur_d.insert(curr_tree(node), NodeID(node, chkpt_id));
        }
        if(node == 0 && labels(0) == FIRST_OCUR) {
          first_ocur_vec.push(node);
#ifdef STATS
          auto first_region_sizes_sa = first_region_sizes_sv.access();
          first_region_sizes_sa(num_chunks) += 1;
          region_counters_sa(FIRST_OCUR) += 1;
#endif
        }
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  while(level_beg <= num_nodes) { // unsigned integer underflow
    std::string forest_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Build Forest");
    Kokkos::parallel_for(forest_label, Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
#ifdef STATS
      auto region_counters_sa = region_counters_sv.access();
      auto first_region_sizes_sa = first_region_sizes_sv.access();
      auto shift_region_sizes_sa = shift_region_sizes_sv.access();
#endif
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) != labels(child_r)) { // Children have different labels
          labels(node) = DONE;
          if((labels(child_l) != FIXED_DUPL) && (labels(child_l) != DONE)) {
            if(labels(child_l) == SHIFT_DUPL) {
              shift_dupl_vec.push(child_l);
#ifdef STATS
              region_counters_sa(SHIFT_DUPL) += 1;
              shift_region_sizes_sa(num_leaf_descendents(child_l, num_nodes)) += 1;
#endif
            } else {
              first_ocur_vec.push(child_l);
#ifdef STATS
              region_counters_sa(FIRST_OCUR) += 1;
              first_region_sizes_sa(num_leaf_descendents(child_l, num_nodes)) += 1;
#endif
            }
          }
          if((labels(child_r) != FIXED_DUPL) && (labels(child_r) != DONE)) {
            if(labels(child_r) == SHIFT_DUPL) {
              shift_dupl_vec.push(child_r);
#ifdef STATS
              region_counters_sa(SHIFT_DUPL) += 1;
              shift_region_sizes_sa(num_leaf_descendents(child_r, num_nodes)) += 1;
#endif
            } else {
              first_ocur_vec.push(child_r);
#ifdef STATS
              region_counters_sa(FIRST_OCUR) += 1;
              first_region_sizes_sa(num_leaf_descendents(child_r, num_nodes)) += 1;
#endif
            }
          }
        } else if(labels(child_l) == FIXED_DUPL) { // Children are both fixed duplicates
          labels(node) = FIXED_DUPL;
        } else if(labels(child_l) == SHIFT_DUPL) { // Children are both shifted duplicates
          hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
          if(first_occur_d.exists(curr_tree(node))) { // This node is also a shifted duplicate
            labels(node) = SHIFT_DUPL;
          } else { // Node is not a shifted duplicate. Save child trees
            labels(node) = DONE; // Add children to tree root maps
            shift_dupl_vec.push(child_l);
            shift_dupl_vec.push(child_r);
#ifdef STATS
            region_counters_sa(SHIFT_DUPL) += 2;
            shift_region_sizes_sa(num_leaf_descendents(child_l, num_nodes)) += 1;
            shift_region_sizes_sa(num_leaf_descendents(child_r, num_nodes)) += 1;
#endif
          }
        }
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }

#ifdef STATS
  Kokkos::Experimental::contribute(region_counters, region_counters_sv);
  Kokkos::Experimental::contribute(first_region_sizes, first_region_sizes_sv);
  Kokkos::Experimental::contribute(shift_region_sizes, shift_region_sizes_sv);
  Kokkos::fence();
  Kokkos::deep_copy(chunk_counters_h, chunk_counters);
  Kokkos::deep_copy(region_counters_h, region_counters);
  Kokkos::deep_copy(first_region_sizes_h, first_region_sizes);
  Kokkos::deep_copy(shift_region_sizes_h, shift_region_sizes);

  STDOUT_PRINT("Checkpoint %u\n", chkpt_id);
  STDOUT_PRINT("Number of first occurrence chunks:  %lu\n", chunk_counters_h(FIRST_OCUR));
  STDOUT_PRINT("Number of fixed duplicate chunks:   %lu\n", chunk_counters_h(FIXED_DUPL));
  STDOUT_PRINT("Number of shifted duplicate chunks: %lu\n", chunk_counters_h(SHIFT_DUPL));
  STDOUT_PRINT("Number of first occurrence regions:  %lu\n", region_counters_h(FIRST_OCUR));
  STDOUT_PRINT("Number of fixed duplicate regions:   %lu\n", region_counters_h(FIXED_DUPL));
  STDOUT_PRINT("Number of shifted duplicate regions: %lu\n", region_counters_h(SHIFT_DUPL));
  for(uint32_t i=0; i<num_chunks+1; i++) {
    if(first_region_sizes_h(i) > 0) {
      printf("First Occurrence: Num regions of size %u: %lu\n", i, first_region_sizes_h(i));
    }
  }
  for(uint32_t i=0; i<num_chunks+1; i++) {
    if(shift_region_sizes_h(i) > 0) {
      printf("Shift Occurrence: Num regions of size %u: %lu\n", i, shift_region_sizes_h(i));
    }
  }
#endif
  Kokkos::fence();
  return;
}

//template<class Hasher, typename DataView>
template<typename DataView>
void dedup_low_root(DataView& data,
                    const uint32_t chunk_size, 
//                    const Hasher hasher, 
                    MerkleTree& curr_tree, 
                    const uint32_t chkpt_id, 
                    DigestNodeIDDeviceMap& first_occur_d, 
                    Vector<uint32_t>& shift_dupl_vec,
                    Vector<uint32_t>& first_ocur_vec) {
  Kokkos::View<uint64_t[4]> chunk_counters("Chunk counters");
  Kokkos::View<uint64_t[4]> region_counters("Region counters");
  Kokkos::deep_copy(chunk_counters, 0);
  Kokkos::deep_copy(region_counters, 0);
  auto chunk_counters_h  = Kokkos::create_mirror_view(chunk_counters);
  auto region_counters_h = Kokkos::create_mirror_view(region_counters);
  Kokkos::Experimental::ScatterView<uint64_t[4]> chunk_counters_sv(chunk_counters);
  Kokkos::Experimental::ScatterView<uint64_t[4]> region_counters_sv(region_counters);

  uint32_t num_chunks = (curr_tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = curr_tree.tree_h.extent(0);
  STDOUT_PRINT("Num chunks: %u\n", num_chunks);
  STDOUT_PRINT("Num nodes: %u\n", num_nodes);

  DigestIdxDeviceMap first_occurrences(num_nodes);
  Kokkos::View<uint32_t*> duplicates("Duplicate nodes", num_nodes);
  Kokkos::View<uint32_t*> dupl_keys("Duplicate keys", num_nodes);
  Kokkos::View<uint32_t[1]> num_dupl_hash("Num duplicates");
  Kokkos::View<uint32_t[1]> hash_id_counter("Counter");
  Kokkos::UnorderedMap<uint32_t,uint32_t> id_map(num_nodes);
  Kokkos::deep_copy(duplicates, UINT_MAX);
  Kokkos::deep_copy(dupl_keys, UINT_MAX);
  Kokkos::deep_copy(num_dupl_hash, 0);
  Kokkos::deep_copy(hash_id_counter, 0);

  uint32_t level_beg = 0;
  uint32_t level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  Kokkos::View<char*> labels("Labels", num_nodes);
  Kokkos::deep_copy(labels, DONE);
  Vector<uint32_t> tree_roots(num_chunks);

  // Process leaves first
  Kokkos::parallel_for("Leaves", Kokkos::RangePolicy<>(num_chunks-1, num_nodes), KOKKOS_LAMBDA(const uint32_t leaf) {
    auto chunk_counters_sa = chunk_counters_sv.access();
    uint32_t num_bytes = chunk_size;
    uint64_t offset = static_cast<uint64_t>(leaf-(num_chunks-1))*static_cast<uint64_t>(chunk_size);
    if(leaf == num_nodes-1) // Calculate how much data to hash
      num_bytes = data.size()-offset;
    // Hash chunk
    HashDigest digest;
    hash(data.data()+offset, num_bytes, digest.digest);
    if(digests_same(curr_tree(leaf), digest)) {
      labels(leaf) = FIXED_DUPL;
      chunk_counters_sa(labels(leaf)) += 1;
    } else if(first_occur_d.exists(digest)) {
      labels(leaf) = SHIFT_DUPL;
      chunk_counters_sa(labels(leaf)) += 1;
    } else {
      labels(leaf) = FIRST_DUPL;
      chunk_counters_sa(labels(leaf)) += 1;

      uint32_t id = leaf+num_nodes;
      auto result = first_occurrences.insert(digest, id);
      id = first_occurrences.value_at(result.index());
      uint32_t offset = Kokkos::atomic_fetch_add(&num_dupl_hash(0), 1);
      duplicates(offset) = leaf;
      dupl_keys(offset) = id;
    }
    curr_tree(leaf) = digest;
  });

  // Build up forest of Merkle Trees
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  while(level_beg <= num_nodes) { // Intensional unsigned integer underflow
    Kokkos::parallel_for("Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) == FIRST_DUPL && labels(child_r) == FIRST_DUPL) {
          hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);

          labels(node) = FIRST_DUPL;

          uint32_t id = node+num_nodes;
          auto result = first_occurrences.insert(curr_tree(node), id);
          if(result.existing()) {
            id = first_occurrences.value_at(result.index());
          }
          uint32_t offset = Kokkos::atomic_fetch_add(&num_dupl_hash(0), 1);
          duplicates(offset) = node;
          dupl_keys(offset) = id;
        }
        if(node == 0 && labels(0) == FIRST_DUPL) {
          tree_roots.push(0);
        }
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }
  DEBUG_PRINT("Processed leaves and trees\n");

  auto hash_id_counter_h = Kokkos::create_mirror_view(hash_id_counter);
  auto num_dupl_hash_h = Kokkos::create_mirror_view(num_dupl_hash);
  Kokkos::deep_copy(num_dupl_hash_h, num_dupl_hash);
  uint32_t num_first_occur = num_dupl_hash_h(0);

  Kokkos::View<uint32_t*> num_duplicates("Number of duplicates", first_occurrences.size()+1);
  Kokkos::deep_copy(num_duplicates, 0);
  Kokkos::deep_copy(hash_id_counter, 0);
  Kokkos::parallel_for("Create id map", Kokkos::RangePolicy<>(0, first_occurrences.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(first_occurrences.valid_at(i)) {
      uint32_t& old_id = first_occurrences.value_at(i);
      uint32_t new_id = Kokkos::atomic_fetch_add(&hash_id_counter(0), static_cast<uint32_t>(1));
      id_map.insert(old_id, new_id);
      old_id = new_id;
    }
  });
  Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<>(0, num_first_occur), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t old_id = dupl_keys(i);
    uint32_t new_id = id_map.value_at(id_map.find(old_id));
    dupl_keys(i) = new_id;
    Kokkos::atomic_add(&num_duplicates(dupl_keys(i)), 1);
  });

  Kokkos::deep_copy(hash_id_counter_h, hash_id_counter);

  auto keys = dupl_keys;
  using key_type = decltype(keys);
  using Comparator = Kokkos::BinOp1D<key_type>;
  Comparator comp(hash_id_counter_h(0), 0, hash_id_counter_h(0));
  Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, num_dupl_hash_h(0), comp, 0);
  bin_sort.create_permute_vector();
  bin_sort.sort(duplicates);

  uint32_t total_duplicates = 0;
  Kokkos::parallel_scan("Find vector offsets", Kokkos::RangePolicy<>(0,num_duplicates.size()), KOKKOS_LAMBDA(uint32_t i, uint32_t& partial_sum, bool is_final) {
    uint32_t num = num_duplicates(i);
    if(is_final) num_duplicates(i) = partial_sum;
    partial_sum += num;
  }, total_duplicates);

  Kokkos::parallel_for("Remove roots with duplicate leaves", Kokkos::RangePolicy<>(0, first_occurrences.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(first_occurrences.valid_at(i)) {
      uint32_t id = first_occurrences.value_at(i);
      if(num_duplicates(id+1)-num_duplicates(id) > 1) {
        uint32_t root = num_nodes;
        bool found_dup = true;
        while(found_dup) {
          found_dup = false;
          root = num_nodes;
          for(uint32_t idx=0; idx<num_duplicates(id+1)-num_duplicates(id); idx++) {
            uint32_t u = duplicates(num_duplicates(id)+idx);
            uint32_t possible_root = u;
            while((num_nodes < possible_root) && (possible_root > 0) && first_occurrences.exists(curr_tree((possible_root-1)/2))) {
              possible_root = (possible_root-1)/2;
            }
            if(possible_root < root) {
              root = possible_root;
            } else if(possible_root == root) {
              first_occurrences.erase(curr_tree(root));
              found_dup = true;
              break;
            }
          }
        }
      }
    }
  });


  Kokkos::parallel_for("Select first occurrence leaves", Kokkos::RangePolicy<>(num_chunks-1, num_nodes), KOKKOS_LAMBDA(const uint32_t node) {
    if(labels(node) == FIRST_DUPL) {
      auto chunk_counters_sa = chunk_counters_sv.access();
      uint32_t id = first_occurrences.value_at(first_occurrences.find(curr_tree(node)));
      uint32_t select = duplicates(num_duplicates(id));
      uint32_t root = select;
      for(uint32_t idx=0; idx<num_duplicates(id+1)-num_duplicates(id); idx++) {
        uint32_t u = duplicates(num_duplicates(id)+idx);
        uint32_t possible_root = u;
        while(possible_root > 0 && first_occurrences.exists(curr_tree((possible_root-1)/2))) {
          possible_root = (possible_root-1)/2;
        }
        if(possible_root < root) {
          root = possible_root;
          select = u;
        }
      }
      for(uint32_t idx=0; idx<num_duplicates(id+1)-num_duplicates(id); idx++) {
        uint32_t u = duplicates(num_duplicates(id)+idx);
        labels(u) = SHIFT_DUPL;
        chunk_counters_sa(labels(u)) += 1;
      }
      labels(select) = FIRST_OCUR;
      chunk_counters_sa(FIRST_OCUR) += 1;
      first_occur_d.insert(curr_tree(select), NodeID(select, chkpt_id));
    }
  });

  // Build up forest of Merkle Trees
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  while(level_beg <= num_nodes) { // Intensional unsigned integer underflow
    Kokkos::parallel_for("Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) == FIRST_OCUR && labels(child_r) == FIRST_OCUR) {
          labels(node) = FIRST_OCUR;
          hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
          first_occur_d.insert(curr_tree(node), NodeID(node, chkpt_id));
        }
        if(node == 0 && labels(0) == FIRST_OCUR)
          tree_roots.push(0);
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  while(level_beg <= num_nodes) { // unsigned integer underflow
    Kokkos::parallel_for("Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
      auto region_counters_sa = region_counters_sv.access();
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) != labels(child_r)) { // Children have different labels
          labels(node) = DONE;
          if((labels(child_l) != FIXED_DUPL) && (labels(child_l) != DONE)) {
            tree_roots.push(child_l);
            region_counters_sa(labels(child_l)) += 1;
          }
          if((labels(child_r) != FIXED_DUPL) && (labels(child_r) != DONE)) {
            tree_roots.push(child_r);
            region_counters_sa(labels(child_r)) += 1;
          }
        } else if(labels(child_l) == FIXED_DUPL) { // Children are both fixed duplicates
          labels(node) = FIXED_DUPL;
        } else if(labels(child_l) == SHIFT_DUPL) { // Children are both shifted duplicates
          hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
          if(first_occur_d.exists(curr_tree(node))) { // This node is also a shifted duplicate
            labels(node) = SHIFT_DUPL;
          } else { // Node is not a shifted duplicate. Save child trees
            labels(node) = DONE; // Add children to tree root maps
            tree_roots.push(child_l);
            tree_roots.push(child_r);
            region_counters_sa(SHIFT_DUPL) += 2;
          }
        }
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }

  // Count regions
  Kokkos::parallel_for("Count regions", Kokkos::RangePolicy<>(0,tree_roots.size()), KOKKOS_LAMBDA(const uint32_t i) {
    auto chunk_counters_sa = chunk_counters_sv.access();
    auto region_counters_sa = region_counters_sv.access();
    uint32_t root = tree_roots.vector_d(i);
    if(labels(root) != DONE) {
      if(labels(root) == FIRST_OCUR) {
        first_ocur_vec.push(root);
      } else if(labels(root) == SHIFT_DUPL) {
        shift_dupl_vec.push(root);
      }
    }
  });

  Kokkos::Experimental::contribute(region_counters, region_counters_sv);
  Kokkos::deep_copy(chunk_counters_h, chunk_counters);
  Kokkos::deep_copy(region_counters_h, region_counters);

  STDOUT_PRINT("Checkpoint %u\n", chkpt_id);
  STDOUT_PRINT("Number of first occurrence chunks:  %lu\n", chunk_counters_h(FIRST_OCUR));
  STDOUT_PRINT("Number of fixed duplicate chunks:   %lu\n", chunk_counters_h(FIXED_DUPL));
  STDOUT_PRINT("Number of shifted duplicate chunks: %lu\n", chunk_counters_h(SHIFT_DUPL));
  STDOUT_PRINT("Number of first occurrence regions:  %lu\n", region_counters_h(FIRST_OCUR));
  STDOUT_PRINT("Number of fixed duplicate regions:   %lu\n", region_counters_h(FIXED_DUPL));
  STDOUT_PRINT("Number of shifted duplicate regions: %lu\n", region_counters_h(SHIFT_DUPL));
  return;
}

#endif // DEDUP_MERKLE_TREE_HPP

