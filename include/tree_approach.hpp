#ifndef TREE_APPROACH_HPP
#define TREE_APPROACH_HPP
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

/**
 * Deduplicate provided data view using the tree incremental checkpoint approach. 
 * Split data into chunks and compute hashes for each chunk. Compare each hash with 
 * the hash at the same offset. If the hash has never been seen, save the chunk.
 * If the hash has been seen before, mark it as a shifted duplicate and save metadata.
 * Compact metadata by building up a forest of Merkle trees and saving only the roots.
 * The baseline version of the function builds up the full tree and inserts all possible
 * hashes so that subsequent checkpoints have more information to work with.
 *
 * \param data          View of data for deduplicating
 * \param chunk_size    Size in bytes for splitting data into chunks
 * \param curr_tree     Tree of hashes for identifying differences
 * \param chkpt_id      ID of current checkpoint
 * \param first_occur_d Map for tracking first occurrence hashes
 * \param first_ocur    Vector of first occurrence chunks
 * \param shift_dupl    Vector of shifted duplicate chunks
 */
template<typename DataView>
void dedup_data_tree_baseline(DataView& data, 
                      const uint32_t chunk_size, 
                      MerkleTree& curr_tree, 
                      const uint32_t chkpt_id, 
                      DigestNodeIDDeviceMap& first_occur_d, 
                      Vector<uint32_t>& shift_dupl_updates,
                      Vector<uint32_t>& first_ocur_updates) {
  // Get number of chunks and nodes
  uint32_t num_chunks = (curr_tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = curr_tree.tree_h.extent(0);
  STDOUT_PRINT("Num chunks: %u\n", num_chunks);
  STDOUT_PRINT("Num nodes: %u\n", num_nodes);

  // Stats for debugging or analysis
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

  // Setup markers for beginning and end of tree level
  uint32_t level_beg = 0;
  uint32_t level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }

  // Create labels
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
      curr_tree(leaf) = digest; /// Update tree
#ifdef STATS
      chunk_counters_sa(labels(leaf)) += 1;
#endif
    }
  });

  /**
   * Identify first occurrences for leaves. In the case of duplicate hash, 
   * select the chunk with the lowest index.
   */
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
  // Iterate through each level of tree and build First occurrence trees
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

  // Build up forest of trees
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  // Iterate through each level of tree and build shifted duplicate trees
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
    // Insert digests into map
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

/**
 * Deduplicate provided data view using the tree incremental checkpoint approach. 
 * Split data into chunks and compute hashes for each chunk. Compare each hash with 
 * the hash at the same offset. If the hash has never been seen, save the chunk.
 * If the hash has been seen before, mark it as a shifted duplicate and save metadata.
 * Compact metadata by building up a forest of Merkle trees and saving only the roots.
 *
 * \param data          View of data for deduplicating
 * \param chunk_size    Size in bytes for splitting data into chunks
 * \param curr_tree     Tree of hashes for identifying differences
 * \param chkpt_id      ID of current checkpoint
 * \param first_occur_d Map for tracking first occurrence hashes
 * \param first_ocur    Vector of first occurrence chunks
 * \param shift_dupl    Vector of shifted duplicate chunks
 */
template<typename DataView>
void dedup_data_tree_low_offset(DataView& data, 
                                const uint32_t chunk_size, 
                                MerkleTree& curr_tree, 
                                const uint32_t chkpt_id, 
                                DigestNodeIDDeviceMap& first_occur_d, 
                                Vector<uint32_t>& shift_dupl_vec,
                                Vector<uint32_t>& first_ocur_vec) {
  // Get number of chunks and nodes
  std::string setup_label = std::string("Deduplicate Checkpoint ") + std::to_string(chkpt_id) + std::string(": Setup");
  Kokkos::Profiling::pushRegion(setup_label);
  uint32_t num_chunks = (curr_tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = curr_tree.tree_h.extent(0);
  STDOUT_PRINT("Num chunks: %u\n", num_chunks);
  STDOUT_PRINT("Num nodes: %u\n", num_nodes);

  // Stats for debugging or analysis
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

  // Setup markers for beginning and end of tree level
  uint32_t level_beg = 0;
  uint32_t level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }

  // Create labels
  Kokkos::View<char*> labels("Labels", num_nodes);
  Kokkos::deep_copy(labels, DONE);
  Kokkos::Profiling::popRegion();

  // Process leaves first
  std::string leaves_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Leaves");
  using member_type = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type;
  auto team_policy = Kokkos::TeamPolicy<>(((num_nodes-num_chunks+1)/TEAM_SIZE)+1, TEAM_SIZE);
  Kokkos::parallel_for(leaves_label, team_policy, KOKKOS_LAMBDA(member_type team_member) {
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

  // TODO May not be necessary
  // Ensure any duplicate first occurrences are labeled correctly
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

  // Build up forest of Merkle Trees for First occurrences
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
        if(node == 0 && labels(0) == FIRST_OCUR) { // Handle case where all chunks are new
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

  // Build up forest of Merkle trees for duplicates
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

/**
 * Deduplicate provided data view using the tree incremental checkpoint approach. 
 * Split data into chunks and compute hashes for each chunk. Compare each hash with 
 * the hash at the same offset. If the hash has never been seen, save the chunk.
 * If the hash has been seen before, mark it as a shifted duplicate and save metadata.
 * Compact metadata by building up a forest of Merkle trees and saving only the roots.
 * In the case when there are multiple possible first occurrence chunks, choose the node
 * such that the largest possible tree is built. 
 *
 * \param data          View of data for deduplicating
 * \param chunk_size    Size in bytes for splitting data into chunks
 * \param curr_tree     Tree of hashes for identifying differences
 * \param chkpt_id      ID of current checkpoint
 * \param first_occur_d Map for tracking first occurrence hashes
 * \param first_ocur    Vector of first occurrence chunks
 * \param shift_dupl    Vector of shifted duplicate chunks
 */
template<typename DataView>
void dedup_data_tree_low_root(DataView& data,
                    const uint32_t chunk_size, 
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

/**
 * Gather the scattered chunks for the diff and write the checkpoint to a contiguous buffer.
 *
 * \param data           Data View containing data to deduplicate
 * \param buffer_d       View to store the diff in
 * \param chunk_size     Size of chunks in bytes
 * \param curr_tree      Tree of hash digests
 * \param first_occur_d  Map for tracking first occurrence hashes
 * \param first_ocur     Vector of first occurrence chunks
 * \param shift_dupl     Vector of shifted duplicate chunks
 * \param prior_chkpt_id ID of the last checkpoint
 * \param chkpt_id       ID for the current checkpoint
 * \param header         Incremental checkpoint header
 *
 * \return Pair containing amount of data and metadata in the checkpoint
 */
template<typename DataView>
std::pair<uint64_t,uint64_t> 
write_diff_tree(const DataView& data, 
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

/**
 * Restart data from incremental checkpoints stored in Kokkos Views on the host.
 *
 * \param incr_chkpts Vector of Host Views containing the diffs
 * \param chkpt_idx   ID of which checkpoint to restart
 * \param data        View for restarting the checkpoint to
 *
 * \return Time spent copying incremental checkpoints from host to device and restarting data
 */
std::pair<double,double> 
restart_chkpt_tree(std::vector<Kokkos::View<uint8_t*>::HostMirror>& incr_chkpts,
                   const int chkpt_idx, 
                   Kokkos::View<uint8_t*>& data) {
  // Read main incremental checkpoint header
  size_t size = incr_chkpts[chkpt_idx].size();

  header_t header;
  memcpy(&header, incr_chkpts[chkpt_idx].data(), sizeof(header_t));
  STDOUT_PRINT("Ref ID: %u\n",               header.ref_id);
  STDOUT_PRINT("Chkpt ID: %u\n",             header.chkpt_id);
  STDOUT_PRINT("Data len: %lu\n",            header.datalen);
  STDOUT_PRINT("Chunk size: %u\n",           header.chunk_size);
  STDOUT_PRINT("Num first ocur: %u\n",        header.num_first_ocur);
  STDOUT_PRINT("Num prior chkpts: %u\n",      header.num_prior_chkpts);
  STDOUT_PRINT("Num shift dupl: %u\n",        header.num_shift_dupl);

  Kokkos::View<uint8_t*> buffer_d("Buffer", size);
  Kokkos::deep_copy(buffer_d, 0);
//  auto buffer_h = Kokkos::create_mirror_view(buffer_d);
//  Kokkos::deep_copy(buffer_h, 0);

  uint32_t num_chunks = header.datalen / header.chunk_size;
  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(header.chunk_size) < header.datalen) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;
  Kokkos::resize(data, header.datalen);

  std::pair<double,double> times;

    // Main checkpoint
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(chkpt_idx));
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(chkpt_idx)+":Read checkpoint");
    DEBUG_PRINT("Global checkpoint\n");
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    Kokkos::resize(buffer_d, size);
    auto& buffer_h = incr_chkpts[chkpt_idx];
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    Kokkos::fence();
Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(chkpt_idx)+" Setup");
    std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();
    Kokkos::deep_copy(buffer_d, buffer_h);
    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c2 = std::chrono::high_resolution_clock::now();
    Kokkos::View<NodeID*> node_list("List of NodeIDs", num_chunks);
    Kokkos::deep_copy(node_list, NodeID());
    uint32_t ref_id = header.ref_id;
    uint32_t cur_id = header.chkpt_id;
    size_t datalen = header.datalen;
    uint32_t chunk_size = header.chunk_size;
    uint32_t num_first_ocur = header.num_first_ocur;
    uint32_t num_prior_chkpts = header.num_prior_chkpts;
    uint32_t num_shift_dupl = header.num_shift_dupl;

    STDOUT_PRINT("Ref ID:           %u\n",  header.ref_id);
    STDOUT_PRINT("Chkpt ID:         %u\n",  header.chkpt_id);
    STDOUT_PRINT("Datalen:          %lu\n", header.datalen);
    STDOUT_PRINT("Chunk size:       %u\n",  header.chunk_size);
    STDOUT_PRINT("Distinct size:    %u\n",  header.num_first_ocur);
    STDOUT_PRINT("Num prior chkpts: %u\n",  header.num_prior_chkpts);
    STDOUT_PRINT("Num shift dupl:   %u\n",  header. num_shift_dupl);

    size_t first_ocur_offset = sizeof(header_t);
    size_t dupl_count_offset = first_ocur_offset + static_cast<uint64_t>(num_first_ocur)*sizeof(uint32_t);
    size_t dupl_map_offset   = dupl_count_offset + static_cast<uint64_t>(num_prior_chkpts)*2*sizeof(uint32_t);
    size_t data_offset       = dupl_map_offset   + static_cast<uint64_t>(num_shift_dupl)*2*sizeof(uint32_t);
    auto first_ocur_subview    = Kokkos::subview(buffer_d, std::make_pair(first_ocur_offset, dupl_count_offset));
    auto dupl_count_subview    = Kokkos::subview(buffer_d, std::make_pair(dupl_count_offset, dupl_map_offset));
    auto shift_dupl_subview    = Kokkos::subview(buffer_d, std::make_pair(dupl_map_offset, data_offset));
    auto data_subview  = Kokkos::subview(buffer_d, std::make_pair(data_offset, size));
    STDOUT_PRINT("Checkpoint %u\n", header.chkpt_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", size);
    STDOUT_PRINT("First ocur offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Dupl count offset: %lu\n", dupl_count_offset);
    STDOUT_PRINT("Dupl map offset: %lu\n", dupl_map_offset);
    STDOUT_PRINT("Data offset: %lu\n", data_offset);

    Kokkos::View<uint64_t[1]> counter_d("Write counter");
    auto counter_h = Kokkos::create_mirror_view(counter_d);
    Kokkos::deep_copy(counter_d, 0);

    Kokkos::UnorderedMap<NodeID, size_t> distinct_map(num_nodes);
    Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(2*num_nodes-1);
    Kokkos::View<uint32_t*> distinct_nodes("Nodes", num_first_ocur);
    Kokkos::View<uint32_t*> chunk_len("Num chunks for node", num_first_ocur);
Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(chkpt_idx)+" Restart distinct");
    // Calculate sizes of each distinct region
    Kokkos::parallel_for("Tree:Main:Calculate num chunks", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, first_ocur_subview.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      uint32_t len = num_leaf_descendents(node, num_nodes);
      distinct_nodes(i) = node;
      chunk_len(i) = len;
    });

    // Perform exclusive prefix scan to determine where to write chunks for each region
    Kokkos::parallel_scan("Tree:Main:Calc offsets", num_first_ocur, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      const uint32_t len = chunk_len(i);
      if(is_final) chunk_len(i) = partial_sum;
      partial_sum += len;
    });

    Kokkos::View<uint32_t[1]> total_region_size("Total region size");
    Kokkos::View<uint32_t[1]>::HostMirror total_region_size_h = Kokkos::create_mirror_view(total_region_size);
    Kokkos::deep_copy(total_region_size, 0);

STDOUT_PRINT("Calculated offsets\n");

    // Restart distinct entries by reading and inserting full tree into distinct map
    Kokkos::parallel_for("Tree:Main:Restart Hash tree repeats main checkpoint", Kokkos::TeamPolicy<>(num_first_ocur, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
      const uint32_t i = team_member.league_rank();
      uint32_t node = distinct_nodes(i);
      if(team_member.team_rank() == 0)
        distinct_map.insert(NodeID(node, cur_id), static_cast<uint64_t>(chunk_len(i))*static_cast<uint64_t>(chunk_size));
      uint32_t start = leftmost_leaf(node, num_nodes);
      uint32_t len = num_leaf_descendents(node, num_nodes);
      uint32_t end = start+len-1;
      uint32_t left = 2*node+1;
      uint32_t right = 2*node+2;
      while(left < num_nodes) {
        if(right >= num_nodes)
          right = num_nodes;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, right-left+1), [&] (const uint32_t j) {
          uint32_t u = left+j;
          uint32_t leaf = leftmost_leaf(u, num_nodes);
          auto result = distinct_map.insert(NodeID(u, cur_id), static_cast<uint64_t>(chunk_len(i)+leaf-start)*static_cast<uint64_t>(chunk_size));
          if(result.failed())
            printf("Failed to insert (%u,%u): %lu\n", u, cur_id, static_cast<uint64_t>(chunk_len(i)+(leaf-start))*static_cast<uint64_t>(chunk_size));
        });
        team_member.team_barrier();
        left = 2*left+1;
        right = 2*right+2;
      }
      // Update chunk metadata list
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, len), [&] (const uint32_t j) {
        node_list(start-num_chunks+1+j) = NodeID(start-num_chunks+1+j, cur_id);
      });
if(team_member.team_rank() == 0) {
Kokkos::atomic_add(&total_region_size(0), len);
}
      uint64_t src_offset = static_cast<uint64_t>(chunk_len(i))*static_cast<uint64_t>(chunk_size);
      uint64_t dst_offset = static_cast<uint64_t>(start-num_chunks+1)*static_cast<uint64_t>(chunk_size);
      uint64_t datasize = static_cast<uint64_t>(len)*static_cast<uint64_t>(chunk_size);
      if(end == num_nodes-1)
        datasize = datalen - dst_offset;

      uint8_t* dst = (uint8_t*)(data.data()+dst_offset);
      uint8_t* src = (uint8_t*)(data_subview.data()+src_offset);
      team_memcpy(dst, src, datasize, team_member);
    });
Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(chkpt_idx)+" Restart repeats");
    Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
    auto repeat_region_sizes_h = Kokkos::create_mirror_view(repeat_region_sizes);
    Kokkos::deep_copy(repeat_region_sizes, 0);
    // Read map of repeats for each checkpoint
    Kokkos::parallel_for("Tree:Main:Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t chkpt;
      memcpy(&chkpt, dupl_count_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&repeat_region_sizes(chkpt), dupl_count_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
      DEBUG_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
    });
    Kokkos::deep_copy(repeat_region_sizes_h, repeat_region_sizes);
    // Perform exclusive scan to determine where regions start/stop
    Kokkos::parallel_scan("Tree:Main:Repeat offsets", cur_id+1, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      partial_sum += repeat_region_sizes(i);
      if(is_final) repeat_region_sizes(i) = partial_sum;
    });

    STDOUT_PRINT("Num repeats: %u\n", num_shift_dupl);
    // Load repeat entries and fill in metadata for chunks
    Kokkos::parallel_for("Tree:Main:Restart Hash tree repeats main checkpoint", Kokkos::TeamPolicy<>(num_shift_dupl, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
      uint32_t i = team_member.league_rank();
      uint32_t node=0, prev=0, tree=0;
      size_t offset = 0;
      if(team_member.team_rank() == 0) {
        memcpy(&node, shift_dupl_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&prev, shift_dupl_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
        for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
          if(i < repeat_region_sizes(j)) {
            tree = j;
          }
        }
        uint32_t idx = distinct_map.find(NodeID(prev, tree));
        if(distinct_map.valid_at(idx)) {
          offset = distinct_map.value_at(idx);
        }
      }
      team_member.team_broadcast(node, 0);
      team_member.team_broadcast(prev, 0);
      team_member.team_broadcast(tree, 0);
      team_member.team_broadcast(offset, 0);
      uint32_t node_start = leftmost_leaf(node, num_nodes);
      uint32_t prev_start = leftmost_leaf(prev, num_nodes);
      uint32_t len = num_leaf_descendents(prev, num_nodes);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, len), [&] (const uint64_t j) {
        node_list(node_start+j-num_chunks+1) = NodeID(prev_start+j, tree);
      });
      if(tree == cur_id) {
Kokkos::atomic_add(&total_region_size(0), len);
        uint64_t dst_offset = static_cast<uint64_t>(chunk_size)*static_cast<uint64_t>(node_start-num_chunks+1);
        uint64_t datasize = static_cast<uint64_t>(chunk_size)*static_cast<uint64_t>(len);
        if(node_start+len-1 == num_nodes-1)
          datasize = data.size() - dst_offset;

        uint8_t* dst = (uint8_t*)(data.data()+dst_offset);
        uint8_t* src = (uint8_t*)(data_subview.data()+offset);
        team_memcpy(dst, src, datasize, team_member);
      }
    });
Kokkos::deep_copy(total_region_size_h, total_region_size);
DEBUG_PRINT("Chkpt %u: total region size: %u\n", cur_id, total_region_size_h(0));

Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(chkpt_idx)+" Fill repeats");
    // All remaining entries are identical 
    Kokkos::parallel_for("Tree:Main:Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      NodeID entry = node_list(i);
      if(entry.node == UINT_MAX) {
        node_list(i) = NodeID(i+num_chunks-1, cur_id-1);
      }
    });
    Kokkos::fence();
Kokkos::Profiling::popRegion();
Kokkos::Profiling::popRegion();

DEBUG_PRINT("Start: %u, end: %u\n", chkpt_idx-1, ref_id);
    for(int idx=static_cast<int>(chkpt_idx)-1; idx>=static_cast<int>(ref_id); idx--) {
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx));
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+":Read checkpoint");
      DEBUG_PRINT("Processing checkpoint %u\n", idx);
      t1 = std::chrono::high_resolution_clock::now();
      size_t chkpt_size = incr_chkpts[idx].size();
      auto chkpt_buffer_d = buffer_d;
      auto chkpt_buffer_h = buffer_h;
      Kokkos::resize(chkpt_buffer_d, chkpt_size);
      Kokkos::resize(chkpt_buffer_h, chkpt_size);
      chkpt_buffer_h = incr_chkpts[idx];
      t2 = std::chrono::high_resolution_clock::now();
      STDOUT_PRINT("Time spent reading checkpoint %d from file: %f\n", idx, (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()));
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+" Setup");
      header_t chkpt_header;
      memcpy(&chkpt_header, chkpt_buffer_h.data(), sizeof(header_t));
      datalen = chkpt_header.datalen;
      chunk_size = chkpt_header.chunk_size;
      Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
      ref_id = chkpt_header.ref_id;
      cur_id = chkpt_header.chkpt_id;
      datalen = chkpt_header.datalen;
      chunk_size = chkpt_header.chunk_size;
      num_first_ocur = chkpt_header.num_first_ocur;
      num_prior_chkpts = chkpt_header.num_prior_chkpts;
      num_shift_dupl = chkpt_header.num_shift_dupl;

      STDOUT_PRINT("Ref ID:           %u\n",  chkpt_header.ref_id);
      STDOUT_PRINT("Chkpt ID:         %u\n",  chkpt_header.chkpt_id);
      STDOUT_PRINT("Datalen:          %lu\n", chkpt_header.datalen);
      STDOUT_PRINT("Chunk size:       %u\n",  chkpt_header.chunk_size);
      STDOUT_PRINT("Distinct size:    %u\n",  chkpt_header.num_first_ocur);
      STDOUT_PRINT("Num prior chkpts: %u\n",  chkpt_header.num_prior_chkpts);
      STDOUT_PRINT("Num shift dupl:   %u\n",  chkpt_header. num_shift_dupl);

      first_ocur_offset = sizeof(header_t);
      dupl_count_offset = first_ocur_offset + static_cast<uint64_t>(num_first_ocur)*sizeof(uint32_t);
      dupl_map_offset   = dupl_count_offset + static_cast<uint64_t>(num_prior_chkpts)*2*sizeof(uint32_t);
      data_offset       = dupl_map_offset   + static_cast<uint64_t>(num_shift_dupl)*2*sizeof(uint32_t);
      first_ocur_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(first_ocur_offset, dupl_count_offset));
      dupl_count_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(dupl_count_offset, dupl_map_offset));
      shift_dupl_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(dupl_map_offset, data_offset));
      data_subview       = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, size));
      STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
      STDOUT_PRINT("Checkpoint size: %lu\n", size);
      STDOUT_PRINT("First ocur offset: %lu\n", sizeof(header_t));
      STDOUT_PRINT("Dupl count offset: %lu\n", dupl_count_offset);
      STDOUT_PRINT("Dupl map offset: %lu\n", dupl_map_offset);
      STDOUT_PRINT("Data offset: %lu\n", data_offset);

      distinct_map.clear();
      repeat_map.clear();
      
      Kokkos::View<uint64_t[1]> counter_d("Write counter");
      auto counter_h = Kokkos::create_mirror_view(counter_d);
      Kokkos::deep_copy(counter_d, 0);
  
      Kokkos::resize(distinct_nodes, num_first_ocur);
      Kokkos::resize(chunk_len, num_first_ocur);
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+" Load maps");
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Calculate num chunks", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
        memcpy(&node, first_ocur_subview.data()+static_cast<uint64_t>(i)*sizeof(uint32_t), sizeof(uint32_t));
        uint32_t len = num_leaf_descendents(node, num_nodes);
        distinct_nodes(i) = node;
        chunk_len(i) = len;
      });
      Kokkos::parallel_scan("Tree:"+std::to_string(idx)+":Calc offsets", num_first_ocur, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
        const uint32_t len = chunk_len(i);
        if(is_final) chunk_len(i) = partial_sum;
        partial_sum += len;
      });
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Restart Hashtree distinct", Kokkos::TeamPolicy<>(num_first_ocur, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
        uint32_t i = team_member.league_rank();
        uint32_t node = distinct_nodes(i);
        uint64_t offset = static_cast<uint64_t>(chunk_len(i)) * static_cast<uint64_t>(chunk_size);
        if(team_member.team_rank() == 0)
          distinct_map.insert(NodeID(node, cur_id), offset);
        uint32_t start = leftmost_leaf(node, num_nodes);
        uint32_t left = 2*node+1;
        uint32_t right = 2*node+2;
        while(left < num_nodes) {
          if(right >= num_nodes)
            right = num_nodes;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, right-left+1), [&] (const uint64_t j) {
            uint32_t u=left+j;
            uint32_t leaf = leftmost_leaf(u, num_nodes);
            uint64_t leaf_offset = static_cast<uint64_t>(leaf-start)*static_cast<uint64_t>(chunk_size);
            auto result = distinct_map.insert(NodeID(u, cur_id), offset + leaf_offset);
            if(result.failed())
              printf("Failed to insert (%u,%u): %lu\n", u, cur_id, offset+leaf_offset);
          });
          team_member.team_barrier();
          left = 2*left+1;
          right = 2*right+2;
        }
      });
  
      Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t chkpt;
        memcpy(&chkpt, dupl_count_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&repeat_region_sizes(chkpt), dupl_count_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
        DEBUG_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
      });
      Kokkos::parallel_scan("Tree:"+std::to_string(idx)+":Repeat offsets", cur_id+1, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
        partial_sum += repeat_region_sizes(i);
        if(is_final) repeat_region_sizes(i) = partial_sum;
      });

      DEBUG_PRINT("Num repeats: %u\n", num_shift_dupl);
  
      Kokkos::TeamPolicy<> repeat_policy(num_shift_dupl, Kokkos::AUTO);
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Restart Hash tree repeats middle chkpts", repeat_policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
        uint32_t i = team_member.league_rank();
        uint32_t node, prev, tree=0;
        memcpy(&node, shift_dupl_subview.data()+static_cast<uint64_t>(i)*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
        memcpy(&prev, shift_dupl_subview.data()+static_cast<uint64_t>(i)*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
        for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
          if(i < repeat_region_sizes(j)) {
            tree = j;
          }
        }
        if(team_member.team_rank() == 0) {
          auto result = repeat_map.insert(node, NodeID(prev,tree));
          if(result.failed())
            STDOUT_PRINT("Failed to insert previous repeat %u: (%u,%u) into repeat map\n", node, prev, tree);
        }
        uint32_t curr_start = leftmost_leaf(node, num_nodes);
        uint32_t prev_start = leftmost_leaf(prev, num_nodes);
        uint32_t len = num_leaf_descendents(node, num_nodes);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, len), [&] (const uint32_t u) {
          repeat_map.insert(curr_start+u, NodeID(prev_start+u, tree));
        });
      });
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+" Fill chunks");

      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Fill data middle chkpts", Kokkos::TeamPolicy<>(num_chunks, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
        uint32_t i = team_member.league_rank();
        if(node_list(i).tree == cur_id) {
          NodeID id = node_list(i);
          if(distinct_map.exists(id)) {
            size_t src_offset = distinct_map.value_at(distinct_map.find(id));
            size_t dst_offset = static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);
            uint32_t writesize = chunk_size;
            if(dst_offset+writesize > datalen) 
              writesize = datalen-dst_offset;

            uint8_t* dst = (uint8_t*)(data.data()+dst_offset);
            uint8_t* src = (uint8_t*)(data_subview.data()+src_offset);
            team_memcpy(dst, src, writesize, team_member);
          } else if(repeat_map.exists(id.node)) {
            NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
            if(prev.tree == cur_id) {
              if(!repeat_map.exists(id.node))
                printf("Failed to find repeat chunk %u\n", id.node);
              size_t src_offset = distinct_map.value_at(distinct_map.find(prev));
              size_t dst_offset = static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);
              uint32_t writesize = chunk_size;
              if(dst_offset+writesize > datalen) 
                writesize = datalen-dst_offset;

              uint8_t* dst = (uint8_t*)(data.data()+dst_offset);
              uint8_t* src = (uint8_t*)(data_subview.data()+src_offset);
              team_memcpy(dst, src, writesize, team_member);
            } else {
              node_list(i) = prev;
            }
          } else {
            node_list(i) = NodeID(node_list(i).node, cur_id-1);
          }
        }
      });
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::popRegion();
    }

    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c3 = std::chrono::high_resolution_clock::now();
    double copy_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
    double restart_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c3-t0).count());
    return std::make_pair(copy_time, restart_time);
}

/**
 * Restart data from incremental checkpoints stored in files.
 *
 * \param incr_chkpts Vector of Host Views containing the diffs
 * \param chkpt_idx   ID of which checkpoint to restart
 * \param data        View for restarting the checkpoint to
 *
 * \return Time spent copying incremental checkpoints from host to device and restarting data
 */
std::pair<double,double> 
restart_chkpt_tree(std::vector<std::string>& chkpt_files,
                   const int file_idx, 
                   Kokkos::View<uint8_t*>& data) {
  // Read main incremental checkpoint header
  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  file.open(chkpt_files[file_idx], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
  size_t filesize = file.tellg();
  file.seekg(0);

  header_t header;
  file.read((char*)&header, sizeof(header_t));
  file.close();
  STDOUT_PRINT("Ref ID: %u\n",               header.ref_id);
  STDOUT_PRINT("Chkpt ID: %u\n",             header.chkpt_id);
  STDOUT_PRINT("Data len: %lu\n",            header.datalen);
  STDOUT_PRINT("Chunk size: %u\n",           header.chunk_size);
  STDOUT_PRINT("Num first ocur: %u\n",        header.num_first_ocur);
  STDOUT_PRINT("Num prior chkpts: %u\n",      header.num_prior_chkpts);
  STDOUT_PRINT("Num shift dupl: %u\n",        header.num_shift_dupl);

  std::vector<Kokkos::View<uint8_t*>> chkpts_d;
  std::vector<Kokkos::View<uint8_t*>::HostMirror> chkpts_h;
  for(uint32_t i=0; i<chkpt_files.size(); i++) {
    file.open(chkpt_files[i], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
    size_t filesize = file.tellg();
    file.seekg(0);
    Kokkos::View<uint8_t*> chkpt_d("Checkpoint", filesize);
    auto chkpt_h = Kokkos::create_mirror_view(chkpt_d);;
    file.read((char*)(chkpt_h.data()), filesize);
    file.close();
    chkpts_d.push_back(chkpt_d);
    chkpts_h.push_back(chkpt_h);
  }

  Kokkos::View<uint8_t*> buffer_d("Buffer", filesize);
  Kokkos::deep_copy(buffer_d, 0);
  auto buffer_h = Kokkos::create_mirror_view(buffer_d);
  Kokkos::deep_copy(buffer_h, 0);

  uint32_t num_chunks = header.datalen / header.chunk_size;
  if(num_chunks*header.chunk_size < header.datalen) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;
  Kokkos::resize(data, header.datalen);

  std::pair<double,double> times;
    
    // Main checkpoint
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(file_idx));
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(file_idx)+":Read checkpoint");
    DEBUG_PRINT("Global checkpoint\n");
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    Kokkos::resize(buffer_d, filesize);
    Kokkos::resize(buffer_h, filesize);
    file.open(chkpt_files[file_idx], std::ifstream::in | std::ifstream::binary);
    file.read((char*)(buffer_h.data()), filesize);
    file.close();
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    STDOUT_PRINT("Time spent reading checkpoint %u from file: %f\n", file_idx, (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()));
    Kokkos::fence();
Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(file_idx)+" Setup");
    std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();
    Kokkos::deep_copy(buffer_d, buffer_h);
    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c2 = std::chrono::high_resolution_clock::now();
    Kokkos::View<NodeID*> node_list("List of NodeIDs", num_chunks);
    Kokkos::deep_copy(node_list, NodeID());
    uint32_t ref_id = header.ref_id;
    uint32_t cur_id = header.chkpt_id;
    size_t datalen = header.datalen;
    uint32_t chunk_size = header.chunk_size;
    uint32_t num_first_ocur = header.num_first_ocur;
    uint32_t num_prior_chkpts = header.num_prior_chkpts;
    uint32_t num_shift_dupl = header.num_shift_dupl;

    STDOUT_PRINT("Ref ID:           %u\n",  header.ref_id);
    STDOUT_PRINT("Chkpt ID:         %u\n",  header.chkpt_id);
    STDOUT_PRINT("Datalen:          %lu\n", header.datalen);
    STDOUT_PRINT("Chunk size:       %u\n",  header.chunk_size);
    STDOUT_PRINT("Distinct size:    %u\n",  header.num_first_ocur);
    STDOUT_PRINT("Num prior chkpts: %u\n",  header.num_prior_chkpts);
    STDOUT_PRINT("Num shift dupl:   %u\n",  header. num_shift_dupl);

    size_t first_ocur_offset = sizeof(header_t);
    size_t dupl_count_offset = first_ocur_offset + num_first_ocur*sizeof(uint32_t);
    size_t dupl_map_offset = dupl_count_offset + num_prior_chkpts*2*sizeof(uint32_t);
    size_t data_offset = dupl_map_offset + num_shift_dupl*2*sizeof(uint32_t);
    auto first_ocur_subview    = Kokkos::subview(buffer_d, std::make_pair(first_ocur_offset, dupl_count_offset));
    auto dupl_count_subview    = Kokkos::subview(buffer_d, std::make_pair(dupl_count_offset, dupl_map_offset));
    auto shift_dupl_subview    = Kokkos::subview(buffer_d, std::make_pair(dupl_map_offset, data_offset));
    auto data_subview  = Kokkos::subview(buffer_d, std::make_pair(data_offset, filesize));
    STDOUT_PRINT("Checkpoint %u\n", header.chkpt_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", filesize);
    STDOUT_PRINT("First ocur offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Dupl count offset: %lu\n", dupl_count_offset);
    STDOUT_PRINT("Dupl map offset: %lu\n", dupl_map_offset);
    STDOUT_PRINT("Data offset: %lu\n", data_offset);

    Kokkos::View<uint64_t[1]> counter_d("Write counter");
    auto counter_h = Kokkos::create_mirror_view(counter_d);
    Kokkos::deep_copy(counter_d, 0);

    Kokkos::UnorderedMap<NodeID, size_t> distinct_map(num_nodes);
    Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(2*num_nodes-1);
    Kokkos::View<uint32_t*> distinct_nodes("Nodes", num_first_ocur);
    Kokkos::View<uint32_t*> chunk_len("Num chunks for node", num_first_ocur);
Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(file_idx)+" Restart distinct");
    // Calculate sizes of each distinct region
    Kokkos::parallel_for("Tree:Main:Calculate num chunks", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, first_ocur_subview.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      uint32_t len = num_leaf_descendents(node, num_nodes);
      distinct_nodes(i) = node;
      chunk_len(i) = len;
    });
    // Perform exclusive prefix scan to determine where to write chunks for each region
    Kokkos::parallel_scan("Tree:Main:Calc offsets", num_first_ocur, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      const uint32_t len = chunk_len(i);
      if(is_final) chunk_len(i) = partial_sum;
      partial_sum += len;
    });

    Kokkos::View<uint32_t[1]> total_region_size("TOtal region size");
    Kokkos::View<uint32_t[1]>::HostMirror total_region_size_h = Kokkos::create_mirror_view(total_region_size);
    Kokkos::deep_copy(total_region_size, 0);

    // Restart distinct entries by reading and inserting full tree into distinct map
    Kokkos::parallel_for("Tree:Main:Restart Hashtree distinct", Kokkos::RangePolicy<>(0,num_first_ocur), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node = distinct_nodes(i);
      distinct_map.insert(NodeID(node, cur_id), chunk_len(i)*chunk_size);
      uint32_t start = leftmost_leaf(node, num_nodes);
      uint32_t len = num_leaf_descendents(node, num_nodes);
      uint32_t end = start+len-1;
      uint32_t left = 2*node+1;
      uint32_t right = 2*node+2;
      while(left < num_nodes) {
        if(right >= num_nodes)
          right = num_nodes;
        for(uint32_t u=left; u<=right; u++) {
          uint32_t leaf = leftmost_leaf(u, num_nodes);
          auto result = distinct_map.insert(NodeID(u, cur_id), chunk_len(i)*chunk_size + (leaf-start)*chunk_size);
          if(result.failed())
            printf("Failed to insert (%u,%u): %u\n", u, cur_id, chunk_len(i)*chunk_size+(leaf-start)*chunk_size);
        }
        left = 2*left+1;
        right = 2*right+2;
      }
      // Update chunk metadata list
      for(uint32_t j=0; j<len; j++) {
        node_list(start-num_chunks+1+j) = NodeID(start-num_chunks+1+j, cur_id);
      }
      uint32_t datasize = len*chunk_size;
      if(end == num_nodes-1)
        datasize = datalen - (start-num_chunks+1)*chunk_size;
      memcpy(data.data()+chunk_size*(start-num_chunks+1), data_subview.data()+chunk_len(i)*chunk_size, datasize);
Kokkos::atomic_add(&total_region_size(0), len);
    });

Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(file_idx)+" Restart repeats");
    Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
    auto repeat_region_sizes_h = Kokkos::create_mirror_view(repeat_region_sizes);
    Kokkos::deep_copy(repeat_region_sizes, 0);
    // Read map of repeats for each checkpoint
    Kokkos::parallel_for("Tree:Main:Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t chkpt;
      memcpy(&chkpt, dupl_count_subview.data()+i*2*sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&repeat_region_sizes(chkpt), dupl_count_subview.data()+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
      DEBUG_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
    });
    Kokkos::deep_copy(repeat_region_sizes_h, repeat_region_sizes);
    // Perform exclusive scan to determine where regions start/stop
    Kokkos::parallel_scan("Tree:Main:Repeat offsets", cur_id+1, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      partial_sum += repeat_region_sizes(i);
      if(is_final) repeat_region_sizes(i) = partial_sum;
    });

    DEBUG_PRINT("Num repeats: %u\n", num_shift_dupl);
    // Load repeat entries and fill in metadata for chunks
    Kokkos::parallel_for("Tree:Main:Restart Hash tree repeats main checkpoint", Kokkos::RangePolicy<>(0, num_shift_dupl), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      uint32_t tree = 0;
      memcpy(&node, shift_dupl_subview.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, shift_dupl_subview.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
      // Determine ID 
      for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
        if(i < repeat_region_sizes(j)) {
          tree = j;
        }
      }
      size_t offset = distinct_map.value_at(distinct_map.find(NodeID(prev, tree)));
      uint32_t node_start = leftmost_leaf(node, num_nodes);
      uint32_t prev_start = leftmost_leaf(prev, num_nodes);
      uint32_t len = num_leaf_descendents(prev, num_nodes);
      for(uint32_t j=0; j<len; j++) {
        node_list(node_start+j-num_chunks+1) = NodeID(prev_start+j, tree);
      }
      if(tree == cur_id) {
Kokkos::atomic_add(&total_region_size(0), len);
        uint32_t copysize = chunk_size*len;
        if(node_start+len-1 == num_nodes-1)
          copysize = data.size() - chunk_size*(node_start-num_chunks+1);
        memcpy(data.data()+chunk_size*(node_start-num_chunks+1), data_subview.data()+offset, copysize);
      }
    });
Kokkos::deep_copy(total_region_size_h, total_region_size);
DEBUG_PRINT("Chkpt %u: total region size: %u\n", cur_id, total_region_size_h(0));

Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(file_idx)+" Fill repeats");
    // All remaining entries are identical 
    Kokkos::parallel_for("Tree:Main:Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      NodeID entry = node_list(i);
      if(entry.node == UINT_MAX) {
        node_list(i) = NodeID(i+num_chunks-1, cur_id-1);
      }
    });
    Kokkos::fence();
Kokkos::Profiling::popRegion();
Kokkos::Profiling::popRegion();

    for(int idx=static_cast<int>(file_idx)-1; idx>=static_cast<int>(ref_id); idx--) {
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx));
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+":Read checkpoint");
      DEBUG_PRINT("Processing checkpoint %u\n", idx);
      t1 = std::chrono::high_resolution_clock::now();
      file.open(chkpt_files[idx], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
      size_t chkpt_size = file.tellg();
      file.seekg(0);
      auto chkpt_buffer_d = buffer_d;
      auto chkpt_buffer_h = buffer_h;
      Kokkos::resize(chkpt_buffer_d, chkpt_size);
      Kokkos::resize(chkpt_buffer_h, chkpt_size);
      file.read((char*)(chkpt_buffer_h.data()), chkpt_size);
      file.close();
      t2 = std::chrono::high_resolution_clock::now();
      STDOUT_PRINT("Time spent reading checkpoint %d from file: %f\n", idx, (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()));
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+" Setup");
      header_t chkpt_header;
      memcpy(&chkpt_header, chkpt_buffer_h.data(), sizeof(header_t));
      uint32_t current_id = chkpt_header.chkpt_id;
      datalen = chkpt_header.datalen;
      chunk_size = chkpt_header.chunk_size;
      Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
      ref_id = chkpt_header.ref_id;
      cur_id = chkpt_header.chkpt_id;
      datalen = chkpt_header.datalen;
      chunk_size = chkpt_header.chunk_size;
      num_first_ocur = chkpt_header.num_first_ocur;
      num_prior_chkpts = chkpt_header.num_prior_chkpts;
      num_shift_dupl = chkpt_header.num_shift_dupl;

      STDOUT_PRINT("Ref ID:           %u\n",  chkpt_header.ref_id);
      STDOUT_PRINT("Chkpt ID:         %u\n",  chkpt_header.chkpt_id);
      STDOUT_PRINT("Datalen:          %lu\n", chkpt_header.datalen);
      STDOUT_PRINT("Chunk size:       %u\n",  chkpt_header.chunk_size);
      STDOUT_PRINT("Distinct size:    %u\n",  chkpt_header.num_first_ocur);
      STDOUT_PRINT("Num prior chkpts: %u\n",  chkpt_header.num_prior_chkpts);
      STDOUT_PRINT("Num shift dupl:   %u\n",  chkpt_header. num_shift_dupl);

      first_ocur_offset = sizeof(header_t);
      dupl_count_offset = first_ocur_offset + num_first_ocur*sizeof(uint32_t);
      dupl_map_offset = dupl_count_offset + num_prior_chkpts*2*sizeof(uint32_t);
      data_offset = dupl_map_offset + num_shift_dupl*2*sizeof(uint32_t);
      first_ocur_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(first_ocur_offset, dupl_count_offset));
      dupl_count_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(dupl_count_offset, dupl_map_offset));
      shift_dupl_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(dupl_map_offset, data_offset));
      data_subview  = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, filesize));
      STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
      STDOUT_PRINT("Checkpoint size: %lu\n", filesize);
      STDOUT_PRINT("First ocur offset: %lu\n", sizeof(header_t));
      STDOUT_PRINT("Dupl count offset: %lu\n", dupl_count_offset);
      STDOUT_PRINT("Dupl map offset: %lu\n", dupl_map_offset);
      STDOUT_PRINT("Data offset: %lu\n", data_offset);

      distinct_map.clear();
      repeat_map.clear();

      Kokkos::View<uint64_t[1]> counter_d("Write counter");
      auto counter_h = Kokkos::create_mirror_view(counter_d);
      Kokkos::deep_copy(counter_d, 0);
  
      Kokkos::resize(distinct_nodes, num_first_ocur);
      Kokkos::resize(chunk_len, num_first_ocur);
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+" Load maps");
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Calculate num chunks", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
        memcpy(&node, first_ocur_subview.data()+i*sizeof(uint32_t), sizeof(uint32_t));
        uint32_t len = num_leaf_descendents(node, num_nodes);
        distinct_nodes(i) = node;
        chunk_len(i) = len;
      });
      Kokkos::parallel_scan("Tree:"+std::to_string(idx)+":Calc offsets", num_first_ocur, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
        const uint32_t len = chunk_len(i);
        if(is_final) chunk_len(i) = partial_sum;
        partial_sum += len;
      });
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Restart Hashtree distinct", Kokkos::RangePolicy<>(0,num_first_ocur), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node = distinct_nodes(i);
        distinct_map.insert(NodeID(node, cur_id), chunk_len(i)*chunk_size);
        uint32_t start = leftmost_leaf(node, num_nodes);
        uint32_t left = 2*node+1;
        uint32_t right = 2*node+2;
        while(left < num_nodes) {
          if(right >= num_nodes)
            right = num_nodes;
          for(uint32_t u=left; u<=right; u++) {
            uint32_t leaf = leftmost_leaf(u, num_nodes);
            auto result = distinct_map.insert(NodeID(u, cur_id), chunk_len(i)*chunk_size + (leaf-start)*chunk_size);
            if(result.failed())
              printf("Failed to insert (%u,%u): %u\n", u, cur_id, chunk_len(i)*chunk_size+(leaf-start)*chunk_size);
          }
          left = 2*left+1;
          right = 2*right+2;
        }
      });
  
      Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t chkpt;
        memcpy(&chkpt, dupl_count_subview.data()+i*2*sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&repeat_region_sizes(chkpt), dupl_count_subview.data()+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
        DEBUG_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
      });
      Kokkos::parallel_scan("Tree:"+std::to_string(idx)+":Repeat offsets", cur_id+1, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
        partial_sum += repeat_region_sizes(i);
        if(is_final) repeat_region_sizes(i) = partial_sum;
      });

      DEBUG_PRINT("Num repeats: %u\n", num_shift_dupl);
  
      Kokkos::TeamPolicy<> repeat_policy(num_shift_dupl, Kokkos::AUTO);
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Restart Hash tree repeats middle chkpts", repeat_policy, 
                           KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
        uint32_t i = team_member.league_rank();
        uint32_t node, prev, tree=0;
        memcpy(&node, shift_dupl_subview.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
        memcpy(&prev, shift_dupl_subview.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
        for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
          if(i < repeat_region_sizes(j)) {
            tree = j;
          }
        }
        if(team_member.team_rank() == 0) {
          auto result = repeat_map.insert(node, NodeID(prev,tree));
          if(result.failed())
            STDOUT_PRINT("Failed to insert previous repeat %u: (%u,%u) into repeat map\n", node, prev, tree);
        }
        uint32_t curr_start = leftmost_leaf(node, num_nodes);
        uint32_t prev_start = leftmost_leaf(prev, num_nodes);
        uint32_t len = num_leaf_descendents(node, num_nodes);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, len), [&] (const uint32_t u) {
          repeat_map.insert(curr_start+u, NodeID(prev_start+u, tree));
        });
      });
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+" Fill chunks");

Kokkos::View<uint32_t[1]> curr_identical_counter("Num identical entries in current checkpoint");
Kokkos::View<uint32_t[1]>::HostMirror curr_identical_counter_h = Kokkos::create_mirror_view(curr_identical_counter);
Kokkos::deep_copy(curr_identical_counter, 0);
Kokkos::View<uint32_t[1]> prev_identical_counter("Num identical entries in previous checkpoint");
Kokkos::View<uint32_t[1]> base_identical_counter("Num identical entries in baseline checkpoint");
Kokkos::View<uint32_t[1]>::HostMirror prev_identical_counter_h = Kokkos::create_mirror_view(prev_identical_counter);;
Kokkos::View<uint32_t[1]>::HostMirror base_identical_counter_h = Kokkos::create_mirror_view(base_identical_counter);;
Kokkos::deep_copy(prev_identical_counter, 0);
Kokkos::deep_copy(base_identical_counter, 0);
Kokkos::deep_copy(total_region_size, 0);
Kokkos::View<uint32_t[1]> curr_chunks("Num identical entries in current checkpoint");
Kokkos::View<uint32_t[1]>::HostMirror curr_chunks_h = Kokkos::create_mirror_view(curr_chunks);
Kokkos::deep_copy(curr_chunks, 0);
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Fill data middle chkpts", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
        if(node_list(i).tree == current_id) {
Kokkos::atomic_add(&curr_chunks(0), 1);
          NodeID id = node_list(i);
          if(distinct_map.exists(id)) {
            uint32_t len = num_leaf_descendents(id.node, num_nodes);
Kokkos::atomic_add(&total_region_size(0), len);
            size_t offset = distinct_map.value_at(distinct_map.find(id));
            uint32_t writesize = chunk_size;
            if(i*chunk_size+writesize > datalen) 
              writesize = datalen-i*chunk_size;
            memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
            Kokkos::atomic_add(&counter_d(0), writesize);
Kokkos::atomic_add(&curr_identical_counter(0), writesize);
          } else if(repeat_map.exists(id.node)) {
            NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
            if(prev.tree == current_id) {
              if(!repeat_map.exists(id.node))
                printf("Failed to find repeat chunk %u\n", id.node);
              size_t offset = distinct_map.value_at(distinct_map.find(prev));
              uint32_t len = num_leaf_descendents(prev.node, num_nodes);
Kokkos::atomic_add(&total_region_size(0), len);
//              uint32_t start = leftmost_leaf(prev.node, num_nodes);
              uint32_t writesize = chunk_size;
              if(i*chunk_size+writesize > datalen) 
                writesize = datalen-i*chunk_size;
              memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
              Kokkos::atomic_add(&counter_d(0), writesize);
Kokkos::atomic_add(&curr_identical_counter(0), writesize);
            } else {
              node_list(i) = prev;
            }
          } else {
Kokkos::atomic_add(&prev_identical_counter(0), 1);
            node_list(i) = NodeID(node_list(i).node, current_id-1);
          }
        } else if(node_list(i).tree < current_id) {
Kokkos::atomic_add(&base_identical_counter(0), 1);
        }
      });
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::popRegion();
    }

    Kokkos::fence();
Kokkos::Profiling::popRegion();
Kokkos::Profiling::popRegion();
    std::chrono::high_resolution_clock::time_point c3 = std::chrono::high_resolution_clock::now();
    double copy_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
    double restart_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c3-t0).count());
    return std::make_pair(copy_time, restart_time);
}

#endif // TREE_APPROACH_HPP
