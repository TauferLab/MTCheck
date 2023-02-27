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

// Build full merkle tree and track all first occurrences of 
// new chunks and shifted duplicate chunks.
template <class Hasher, typename DataView>
void create_merkle_tree_deterministic(Hasher& hasher, 
                        MerkleTree& tree, 
                        DataView& data, 
                        uint32_t chunk_size, 
                        uint32_t tree_id, 
                        DistinctNodeIDMap& distinct_map, 
                        SharedNodeIDMap& shared_map) {
  // Calculate important constants
  uint32_t num_chunks = static_cast<uint32_t>(data.size()/static_cast<uint64_t>(chunk_size));
  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(chunk_size) < data.size())
    num_chunks += 1;
  const uint32_t num_nodes = 2*num_chunks-1;
  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  const uint32_t leaf_start = num_chunks-1;

  // Iterate through each level of the tree starting from the leaves
  for(int32_t level=num_levels-1; level>=0; level--) {
    // Calculate number of hashes in the level and the offsets within the flattened tree
    DEBUG_PRINT("Processing level %d\n", level);
    uint32_t nhashes = 1 << static_cast<uint32_t>(level);
    uint32_t start_offset = nhashes-1;
    if(start_offset + nhashes > num_nodes)
      nhashes = num_nodes - start_offset;
  
    // Compute hashes for each chunk in this level in parallel
    DEBUG_PRINT("Computing %u hashes\n", nhashes);
    auto range_policy = Kokkos::RangePolicy<>(start_offset, start_offset+nhashes);
    Kokkos::parallel_for("Build tree", range_policy, KOKKOS_LAMBDA(const uint32_t i) {
      // Number of bytes in the chunk
      uint32_t num_bytes = chunk_size;
      uint64_t offset = static_cast<uint64_t>(i-leaf_start)*static_cast<uint64_t>(chunk_size);
      if((i-leaf_start) == num_chunks-1)
        num_bytes = data.size()-offset;
      // Calc hash for leaf chunk or hash of child hashes
      if(i >= leaf_start) {
//        hasher.hash(data.data()+((i-leaf_start)*chunk_size), 
        hash(data.data()+offset, 
                    num_bytes, 
                    (uint8_t*)(tree(i).digest));
      } else {
//        hasher.hash((uint8_t*)&tree(2*i+1), 2*hasher.digest_size(), (uint8_t*)&tree(i));
        hash((uint8_t*)(&tree(2*i+1)), 2*sizeof(HashDigest), (uint8_t*)(tree(i).digest));
      }
      // Insert hash and node info into either the 
      // first occurrece table or the shifted duplicate table
      auto result = distinct_map.insert(tree(i), NodeID(i, tree_id));
      if(result.existing()) {
        auto& entry = distinct_map.value_at(result.index());
        Kokkos::atomic_min(&entry.node, i);
//        shared_map.insert(i,NodeID(entry.node, entry.tree));
      } else if(result.failed()) {
        printf("Failed to insert node %u into distinct map\n",i);
      }
    });
  }
  Kokkos::parallel_for("Save repeats", Kokkos::RangePolicy<>(0, num_nodes), KOKKOS_LAMBDA(const uint32_t node) {
    auto entry = distinct_map.value_at(distinct_map.find(tree(node)));
    if(entry.node != node) {
      shared_map.insert(node, entry);
    }
  });
  Kokkos::fence();
}

template<class Hasher, typename DataView>
void deduplicate_data_deterministic_baseline(DataView& data, 
                      const uint32_t chunk_size, 
                      const Hasher hasher, 
                      MerkleTree& curr_tree, 
                      const uint32_t chkpt_id, 
                      DistinctNodeIDMap& first_occur_d, 
                      CompactTable& shift_dupl_updates,
                      CompactTable& first_ocur_updates) {
  Kokkos::View<uint64_t[3]> chunk_counters("Chunk counters");
  Kokkos::View<uint64_t[3]> region_counters("Region counters");
  Kokkos::deep_copy(chunk_counters, 0);
  Kokkos::deep_copy(region_counters, 0);
  auto chunk_counters_h  = Kokkos::create_mirror_view(chunk_counters);
  auto region_counters_h = Kokkos::create_mirror_view(region_counters);
  Kokkos::Experimental::ScatterView<uint64_t[3]> chunk_counters_sv(chunk_counters);
  Kokkos::Experimental::ScatterView<uint64_t[3]> region_counters_sv(region_counters);

  uint32_t num_chunks = (curr_tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = curr_tree.tree_h.extent(0);
  STDOUT_PRINT("Num chunks: %u\n", num_chunks);
  STDOUT_PRINT("Num nodes: %u\n", num_nodes);

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
    auto region_counters_sa = region_counters_sv.access();
    uint32_t num_bytes = chunk_size;
    uint64_t offset = static_cast<uint64_t>(leaf-num_chunks+1)*static_cast<uint64_t>(chunk_size);
    if(leaf == num_nodes-1) // Calculate how much data to hash
      num_bytes = data.size()-offset;
    // Hash chunk
    HashDigest digest;
//    hasher.hash(data.data()+(leaf-(num_chunks-1))*chunk_size, num_bytes, digest.digest);
    hash(data.data()+offset, num_bytes, digest.digest);
    // Insert into table
    auto result = first_occur_d.insert(digest, NodeID(leaf, chkpt_id)); 
    if(digests_same(digest, curr_tree(leaf))) { // Fixed duplicate chunk
      labels(leaf) = FIXED_DUPL;
      chunk_counters_sa(labels(leaf)) += 1;
    } else if(result.success()) { // First occurrence chunk
      labels(leaf) = FIRST_OCUR;
      chunk_counters_sa(labels(leaf)) += 1;
    } else if(result.existing()) { // Shifted duplicate chunk
      auto& info = first_occur_d.value_at(result.index());
      if(info.tree == chkpt_id) {
        uint32_t min = Kokkos::atomic_fetch_min(&info.node, leaf);
        labels(leaf) = FIRST_OCUR;
      } else {
        labels(leaf) = SHIFT_DUPL;
        chunk_counters_sa(labels(leaf)) += 1;
      }
    }
    curr_tree(leaf) = digest;
  });
  Kokkos::parallel_for("Leaves", Kokkos::RangePolicy<>(num_chunks-1, num_nodes), KOKKOS_LAMBDA(const uint32_t leaf) {
    auto chunk_counters_sa = chunk_counters_sv.access();
    auto region_counters_sa = region_counters_sv.access();
    if(labels(leaf) == FIRST_OCUR) {
      auto info = first_occur_d.value_at(first_occur_d.find(curr_tree(leaf))); 
      if((info.tree == chkpt_id) && (leaf != info.node)) {
        labels(leaf) = SHIFT_DUPL;
        chunk_counters_sa(labels(leaf)) += 1;
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
  while(level_beg <= num_nodes) { // Intensional unsigned integer underflow
    Kokkos::parallel_for("Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
//        hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
//        first_occur_d.insert(curr_tree(node), NodeID(node, chkpt_id));
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
//          hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
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
    Kokkos::fence();
    Kokkos::parallel_for("Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
        first_occur_d.insert(curr_tree(node), NodeID(node, chkpt_id));
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }

  // Count regions
  Kokkos::parallel_for(tree_roots.size(), KOKKOS_LAMBDA(const uint32_t i) {
    auto chunk_counters_sa = chunk_counters_sv.access();
    auto region_counters_sa = region_counters_sv.access();
    uint32_t root = tree_roots.vector_d(i);
    if(labels(root) != DONE) {
//      region_counters_sa(labels(root)) += 1;
      NodeID node = first_occur_d.value_at(first_occur_d.find(curr_tree(root)));
      if(labels(root) == FIRST_OCUR) {
//        first_ocur_updates.insert(root, node);
        first_ocur_updates.insert(node.node, node);
      } else if(labels(root) == SHIFT_DUPL) {
        shift_dupl_updates.insert(root, node);
      }
    }
  });

//  Kokkos::deep_copy(tree_roots.vector_h, tree_roots.vector_d);
//  auto labels_h = Kokkos::create_mirror_view(labels);
//  Kokkos::deep_copy(labels_h, labels);
//  std::set<uint32_t> tree_root_set;
//  for(uint32_t i=0; i<tree_roots.size(); i++) {
//    uint32_t root = tree_roots.vector_h(i);
//    tree_root_set.insert(root);
//  }
//  
//  for(auto iter = tree_root_set.begin(); iter != tree_root_set.end(); iter++) {
//    if(labels_h(*iter) == FIRST_OCUR) {
//STDOUT_PRINT("%u FIRST_OCUR\n", *iter);
//    } else if(labels_h(*iter) == SHIFT_DUPL) {
//STDOUT_PRINT("%u SHIFT_DUPL\n", *iter);
//    }
//  }

  Kokkos::Experimental::contribute(region_counters, region_counters_sv);
  Kokkos::fence();
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

template<class Hasher, typename DataView>
void deduplicate_data_deterministic(DataView& data, 
                      const uint32_t chunk_size, 
                      const Hasher hasher, 
                      MerkleTree& curr_tree, 
                      const uint32_t chkpt_id, 
                      DistinctNodeIDMap& first_occur_d, 
                      CompactTable& shift_dupl_updates,
                      CompactTable& first_ocur_updates) {
  Kokkos::View<uint64_t[3]> chunk_counters("Chunk counters");
  Kokkos::View<uint64_t[3]> region_counters("Region counters");
  Kokkos::deep_copy(chunk_counters, 0);
  Kokkos::deep_copy(region_counters, 0);
  auto chunk_counters_h  = Kokkos::create_mirror_view(chunk_counters);
  auto region_counters_h = Kokkos::create_mirror_view(region_counters);
  Kokkos::Experimental::ScatterView<uint64_t[3]> chunk_counters_sv(chunk_counters);
  Kokkos::Experimental::ScatterView<uint64_t[3]> region_counters_sv(region_counters);

  uint32_t num_chunks = (curr_tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = curr_tree.tree_h.extent(0);
  STDOUT_PRINT("Num chunks: %u\n", num_chunks);
  STDOUT_PRINT("Num nodes: %u\n", num_nodes);

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
    auto region_counters_sa = region_counters_sv.access();
    uint32_t num_bytes = chunk_size;
    uint64_t offset = static_cast<uint64_t>(leaf-(num_chunks-1))*static_cast<uint64_t>(chunk_size);
    if(leaf == num_nodes-1) // Calculate how much data to hash
      num_bytes = data.size()-offset;
    // Hash chunk
    HashDigest digest;
//    hasher.hash(data.data()+(leaf-(num_chunks-1))*chunk_size, num_bytes, digest.digest);
    hash(data.data()+offset, num_bytes, digest.digest);
    // Insert into table
    auto result = first_occur_d.insert(digest, NodeID(leaf, chkpt_id)); 
    if(digests_same(digest, curr_tree(leaf))) { // Fixed duplicate chunk
      labels(leaf) = FIXED_DUPL;
      chunk_counters_sa(labels(leaf)) += 1;
    } else if(result.success()) { // First occurrence chunk
      labels(leaf) = FIRST_OCUR;
      chunk_counters_sa(labels(leaf)) += 1;
    } else if(result.existing()) { // Shifted duplicate chunk
      auto& info = first_occur_d.value_at(result.index());
      if(info.tree == chkpt_id) {
        uint32_t min = Kokkos::atomic_fetch_min(&info.node, leaf);
        labels(leaf) = FIRST_OCUR;
      } else {
        labels(leaf) = SHIFT_DUPL;
        chunk_counters_sa(labels(leaf)) += 1;
      }
    }
    curr_tree(leaf) = digest;
  });
  Kokkos::parallel_for("Leaves", Kokkos::RangePolicy<>(num_chunks-1, num_nodes), KOKKOS_LAMBDA(const uint32_t leaf) {
    auto chunk_counters_sa = chunk_counters_sv.access();
    auto region_counters_sa = region_counters_sv.access();
    if(labels(leaf) == FIRST_OCUR) {
      auto info = first_occur_d.value_at(first_occur_d.find(curr_tree(leaf))); 
      if((info.tree == chkpt_id) && (leaf != info.node)) {
        labels(leaf) = SHIFT_DUPL;
        chunk_counters_sa(labels(leaf)) += 1;
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
  while(level_beg <= num_nodes) { // Intensional unsigned integer underflow
    Kokkos::parallel_for("Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) == FIRST_OCUR && labels(child_r) == FIRST_OCUR) {
          labels(node) = FIRST_OCUR;
//          hasher.hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
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
//          hasher.hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
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
//    Kokkos::parallel_for("Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
//      if(node < num_chunks-1) {
//        uint32_t child_l = 2*node+1;
//        uint32_t child_r = 2*node+2;
//        hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
//        first_occur_d.insert(curr_tree(node), NodeID(node, chkpt_id));
//      }
//    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }

  // Count regions
  Kokkos::parallel_for(tree_roots.size(), KOKKOS_LAMBDA(const uint32_t i) {
    auto chunk_counters_sa = chunk_counters_sv.access();
    auto region_counters_sa = region_counters_sv.access();
    uint32_t root = tree_roots.vector_d(i);
    if(labels(root) != DONE) {
//      region_counters_sa(labels(root)) += 1;
      NodeID node = first_occur_d.value_at(first_occur_d.find(curr_tree(root)));
      if(labels(root) == FIRST_OCUR) {
        first_ocur_updates.insert(root, node);
      } else if(labels(root) == SHIFT_DUPL) {
        shift_dupl_updates.insert(root, node);
      }
    }
  });

//  Kokkos::deep_copy(tree_roots.vector_h, tree_roots.vector_d);
//  auto labels_h = Kokkos::create_mirror_view(labels);
//  Kokkos::deep_copy(labels_h, labels);
//  std::set<uint32_t> tree_root_set;
//  for(uint32_t i=0; i<tree_roots.size(); i++) {
//    uint32_t root = tree_roots.vector_h(i);
//    tree_root_set.insert(root);
//  }
//  
//  for(auto iter = tree_root_set.begin(); iter != tree_root_set.end(); iter++) {
//    if(labels_h(*iter) == FIRST_OCUR) {
//STDOUT_PRINT("%u FIRST_OCUR\n", *iter);
//    } else if(labels_h(*iter) == SHIFT_DUPL) {
//STDOUT_PRINT("%u SHIFT_DUPL\n", *iter);
//    }
//  }

  Kokkos::Experimental::contribute(region_counters, region_counters_sv);
  Kokkos::fence();
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

template<class Hasher, typename DataView>
void dedup_low_root(DataView& data,
                    const uint32_t chunk_size, 
                    const Hasher hasher, 
                    MerkleTree& curr_tree, 
                    const uint32_t chkpt_id, 
                    DistinctNodeIDMap& first_occur_d, 
                    CompactTable& shift_dupl_updates,
                    CompactTable& first_ocur_updates) {
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

  DigestListMap first_occurrences(num_nodes);
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
//    hasher.hash(data.data()+(leaf-(num_chunks-1))*chunk_size, num_bytes, digest.digest);
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

//      uint32_t id = digest_to_u32(digest);
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
//          hasher.hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
          hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);

          labels(node) = FIRST_DUPL;

//          uint32_t id = digest_to_u32(curr_tree(node));
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
//    Kokkos::fence();
  }
  DEBUG_PRINT("Processed leaves and trees\n");

  auto hash_id_counter_h = Kokkos::create_mirror_view(hash_id_counter);
  auto num_dupl_hash_h = Kokkos::create_mirror_view(num_dupl_hash);
//  Kokkos::deep_copy(hash_id_counter_h, hash_id_counter);
  Kokkos::deep_copy(num_dupl_hash_h, num_dupl_hash);
//  uint32_t num_hashes = hash_id_counter_h(0);
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
  uint32_t num_hashes = hash_id_counter_h(0);

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
//            uint32_t u = duplicates(num_duplicates(id));
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
//          hasher.hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
          hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
          first_occur_d.insert(curr_tree(node), NodeID(node, chkpt_id));
        }
        if(node == 0 && labels(0) == FIRST_OCUR)
          tree_roots.push(0);
      }
    });
//    Kokkos::fence();
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
//          hasher.hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
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
//    Kokkos::fence();
  }

  // Count regions
  Kokkos::parallel_for("Count regions", Kokkos::RangePolicy<>(0,tree_roots.size()), KOKKOS_LAMBDA(const uint32_t i) {
    auto chunk_counters_sa = chunk_counters_sv.access();
    auto region_counters_sa = region_counters_sv.access();
    uint32_t root = tree_roots.vector_d(i);
    if(labels(root) != DONE) {
//      region_counters_sa(labels(root)) += 1;
      NodeID node = first_occur_d.value_at(first_occur_d.find(curr_tree(root)));
      if(labels(root) == FIRST_OCUR) {
        first_ocur_updates.insert(root, node);
//printf("First Ocur: %u\n", root);
      } else if(labels(root) == SHIFT_DUPL) {
        shift_dupl_updates.insert(root, node);
//printf("Shift Dupl: %u\n", root);
      }
    }
  });

  Kokkos::Experimental::contribute(region_counters, region_counters_sv);
  Kokkos::fence();
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

