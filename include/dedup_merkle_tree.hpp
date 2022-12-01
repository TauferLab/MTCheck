#ifndef DEDUP_MERKLE_TREE_HPP
#define DEDUP_MERKLE_TREE_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_ScatterView.hpp>
#include <queue>
#include <iostream>
#include <climits>
#include "hash_functions.hpp"
#include "map_helpers.hpp"
#include "kokkos_merkle_tree.hpp"
#include "kokkos_hash_list.hpp"
#include "reference_impl.hpp"
#include "utils.hpp"

enum Label : uint8_t {
  FIRST_OCUR = 0,
  FIXED_DUPL = 1,
  SHIFT_DUPL = 2,
  DONE = 4
};

// Build full merkle tree and track all first occurrences of 
// new chunks and shifted duplicate chunks.
template <class Hasher>
void create_merkle_tree_deterministic(Hasher& hasher, 
                        MerkleTree& tree, 
                        Kokkos::View<uint8_t*>& data, 
                        uint32_t chunk_size, 
                        uint32_t tree_id, 
                        DistinctNodeIDMap& distinct_map, 
                        SharedNodeIDMap& shared_map) {
  // Calculate important constants
  uint32_t num_chunks = static_cast<uint32_t>(data.size()/chunk_size);
  if(num_chunks*chunk_size < data.size())
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
      if((i-leaf_start) == num_chunks-1)
        num_bytes = data.size()-((i-leaf_start)*chunk_size);
      // Calc hash for leaf chunk or hash of child hashes
      if(i >= leaf_start) {
        hasher.hash(data.data()+((i-leaf_start)*chunk_size), 
                    num_bytes, 
                    (uint8_t*)(tree(i).digest));
      } else {
        hasher.hash((uint8_t*)&tree(2*i+1), 2*hasher.digest_size(), (uint8_t*)&tree(i));
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

// Build full merkle tree and track all first occurrences of 
// new chunks and shifted duplicate chunks.
template <class Hasher>
void create_merkle_tree(Hasher& hasher, 
                        MerkleTree& tree, 
                        Kokkos::View<uint8_t*>& data, 
                        uint32_t chunk_size, 
                        uint32_t tree_id, 
                        DistinctNodeIDMap& distinct_map, 
                        SharedNodeIDMap& shared_map) {
  // Calculate important constants
  uint32_t num_chunks = static_cast<uint32_t>(data.size()/chunk_size);
  if(num_chunks*chunk_size < data.size())
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
      if((i-leaf_start) == num_chunks-1)
        num_bytes = data.size()-((i-leaf_start)*chunk_size);
      // Calc hash for leaf chunk or hash of child hashes
      if(i >= leaf_start) {
        hasher.hash(data.data()+((i-leaf_start)*chunk_size), 
                    num_bytes, 
                    (uint8_t*)(tree(i).digest));
      } else {
        hasher.hash((uint8_t*)&tree(2*i+1), 2*hasher.digest_size(), (uint8_t*)&tree(i));
      }
      // Insert hash and node info into either the 
      // first occurrece table or the shifted duplicate table
      auto result = distinct_map.insert(tree(i), NodeID(i, tree_id));
      if(result.existing()) {
        auto& entry = distinct_map.value_at(result.index());
        shared_map.insert(i,NodeID(entry.node, entry.tree));
      } else if(result.failed()) {
        printf("Failed to insert node %u into distinct map\n",i);
      }
    });
  }
  Kokkos::fence();
}

// Build full merkle tree and track all first occurrences of 
// new chunks and shifted duplicate chunks.
template <class Hasher>
void create_merkle_tree(Hasher& hasher, 
                        MerkleTree& tree, 
                        Kokkos::View<uint8_t*>& data, 
                        uint32_t chunk_size, 
                        uint32_t tree_id, 
                        DistinctNodeIDMap& distinct_map, 
                        NodeMap& node_map) {
  // Calculate important constants
  uint32_t num_chunks = static_cast<uint32_t>(data.size()/chunk_size);
  if(num_chunks*chunk_size < data.size())
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
      if((i-leaf_start) == num_chunks-1)
        num_bytes = data.size()-((i-leaf_start)*chunk_size);
      // Calc hash for leaf chunk or hash of child hashes
      if(i >= leaf_start) {
        hasher.hash(data.data()+((i-leaf_start)*chunk_size), 
                    num_bytes, 
                    (uint8_t*)(tree(i).digest));
      } else {
        hasher.hash((uint8_t*)&tree(2*i+1), 2*hasher.digest_size(), (uint8_t*)&tree(i));
      }
      // Insert hash and node info into either the 
      // first occurrece table or the shifted duplicate table
      auto result = distinct_map.insert(tree(i), NodeID(i, tree_id));
      if(result.existing()) {
        if(i >= leaf_start) {
          auto& entry = distinct_map.value_at(result.index());
          Node repeat_entry(entry.node, entry.tree, 1);
          auto res = node_map.insert(i, repeat_entry);
        }
      }
    });
  }
  Kokkos::fence();
}

template<class Hasher>
void deduplicate_data_deterministic(Kokkos::View<uint8_t*>& data, 
                      const uint32_t chunk_size, 
                      const Hasher hasher, 
                      MerkleTree& curr_tree, 
                      const uint32_t chkpt_id, 
                      DistinctNodeIDMap& first_occur_d, 
                      CompactTable& shift_dupl_updates,
                      CompactTable& first_ocur_updates) {
  const char FIRST_OCUR = 0;
  const char FIXED_DUPL = 1;
  const char SHIFT_DUPL = 2;
  const char DONE = 3;
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
  printf("Num chunks: %u\n", num_chunks);
  printf("Num nodes: %u\n", num_nodes);

  uint32_t level_beg = 0;
  uint32_t level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  Kokkos::View<char*> labels("Labels", num_nodes);
  Kokkos::deep_copy(labels, DONE);
  Vector tree_roots(num_chunks);

  // Process leaves first
  Kokkos::parallel_for("Leaves", Kokkos::RangePolicy<>(num_chunks-1, num_nodes), KOKKOS_LAMBDA(const uint32_t leaf) {
    auto chunk_counters_sa = chunk_counters_sv.access();
    auto region_counters_sa = region_counters_sv.access();
    uint32_t num_bytes = chunk_size;
    if(leaf == num_nodes-1) // Calculate how much data to hash
      num_bytes = data.size()-(leaf-(num_chunks-1))*chunk_size;
    // Hash chunk
    HashDigest digest;
    hasher.hash(data.data()+(leaf-(num_chunks-1))*chunk_size, num_bytes, digest.digest);
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
          hasher.hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
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
          hasher.hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
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
//printf("%u FIRST_OCUR\n", *iter);
//    } else if(labels_h(*iter) == SHIFT_DUPL) {
//printf("%u SHIFT_DUPL\n", *iter);
//    }
//  }

  Kokkos::Experimental::contribute(region_counters, region_counters_sv);
  Kokkos::fence();
  Kokkos::deep_copy(chunk_counters_h, chunk_counters);
  Kokkos::deep_copy(region_counters_h, region_counters);

  printf("Checkpoint %u\n", chkpt_id);
  printf("Number of first occurrence chunks:  %lu\n", chunk_counters_h(FIRST_OCUR));
  printf("Number of fixed duplicate chunks:   %lu\n", chunk_counters_h(FIXED_DUPL));
  printf("Number of shifted duplicate chunks: %lu\n", chunk_counters_h(SHIFT_DUPL));
  printf("Number of first occurrence regions:  %lu\n", region_counters_h(FIRST_OCUR));
  printf("Number of fixed duplicate regions:   %lu\n", region_counters_h(FIXED_DUPL));
  printf("Number of shifted duplicate regions: %lu\n", region_counters_h(SHIFT_DUPL));
  return;
}

template<class Hasher>
void deduplicate_data(Kokkos::View<uint8_t*>& data, 
                      const uint32_t chunk_size, 
                      const Hasher hasher, 
                      MerkleTree& curr_tree, 
                      const uint32_t chkpt_id, 
                      DistinctNodeIDMap& first_occur_d, 
                      CompactTable& shift_dupl_updates,
                      CompactTable& first_ocur_updates) {
  const char FIRST_OCUR = 0;
  const char FIXED_DUPL = 1;
  const char SHIFT_DUPL = 2;
  const char DONE = 3;
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
  printf("Num chunks: %u\n", num_chunks);
  printf("Num nodes: %u\n", num_nodes);

  uint32_t level_beg = 0;
  uint32_t level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  Kokkos::View<char*> labels("Labels", num_nodes);
  Kokkos::deep_copy(labels, DONE);
  Vector tree_roots(num_chunks);

  // Process leaves first
  Kokkos::parallel_for("Leaves", Kokkos::RangePolicy<>(num_chunks-1, num_nodes), KOKKOS_LAMBDA(const uint32_t leaf) {
    auto chunk_counters_sa = chunk_counters_sv.access();
    uint32_t num_bytes = chunk_size;
    if(leaf == num_nodes-1) // Calculate how much data to hash
      num_bytes = data.size()-(leaf-(num_chunks-1))*chunk_size;
    // Hash chunk
    HashDigest digest;
    hasher.hash(data.data()+(leaf-(num_chunks-1))*chunk_size, num_bytes, digest.digest);
    // Insert into table
    auto result = first_occur_d.insert(digest, NodeID(leaf, chkpt_id)); 
    if(digests_same(digest, curr_tree(leaf))) { // Fixed duplicate chunk
      labels(leaf) = FIXED_DUPL;
      chunk_counters_sa(labels(leaf)) += 1;
    } else if(result.success()) { // First occurrence chunk
      labels(leaf) = FIRST_OCUR;
      chunk_counters_sa(labels(leaf)) += 1;
    } else if(result.existing()) { // Shifted duplicate chunk
      labels(leaf) = SHIFT_DUPL;
      chunk_counters_sa(labels(leaf)) += 1;
    }
    curr_tree(leaf) = digest;
  });

  // Build up forest of Merkle Trees
//  level_beg = 0;
//  level_end = 0;
//  while(level_end < num_nodes) {
//    level_beg = 2*level_beg + 1;
//    level_end = 2*level_end + 2;
//  }
//  while(level_beg <= num_nodes) { // Intensional unsigned integer underflow
//    Kokkos::parallel_for("Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
//      if(node < num_chunks-1) {
//        uint32_t child_l = 2*node+1;
//        uint32_t child_r = 2*node+2;
//        if(labels(child_l) == FIRST_OCUR && labels(child_r) == FIRST_OCUR) {
//          labels(node) = FIRST_OCUR;
//          hasher.hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
//          first_occur_d.insert(curr_tree(node), NodeID(node, chkpt_id));
//        }
//        if(node == 0 && labels(0) == FIRST_OCUR)
//          tree_roots.push(0);
//      }
//    });
//    level_beg = (level_beg-1)/2;
//    level_end = (level_end-2)/2;
//  }
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  while(level_beg <= num_nodes) { // unsigned integer underflow
    Kokkos::parallel_for("Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_LAMBDA(const uint32_t node) {
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) != labels(child_r)) { // Children have different labels
          labels(node) = DONE;
          if((labels(child_l) != FIXED_DUPL) && (labels(child_l) != DONE)) {
            tree_roots.push(child_l);
          }
          if((labels(child_r) != FIXED_DUPL) && (labels(child_r) != DONE)) {
            tree_roots.push(child_r);
          }
        } else if(labels(child_l) == FIRST_OCUR && labels(child_r) == FIRST_OCUR) {
          labels(node) = FIRST_OCUR;
          hasher.hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
          first_occur_d.insert(curr_tree(node), NodeID(node, chkpt_id));
          if(node == 0 && labels(0) == FIRST_OCUR)
            tree_roots.push(0);
        } else if(labels(child_l) == FIXED_DUPL) { // Children are both fixed duplicates
          labels(node) = FIXED_DUPL;
        } else if(labels(child_l) == SHIFT_DUPL) { // Children are both shifted duplicates
          hasher.hash((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
          if(first_occur_d.exists(curr_tree(node))) { // This node is also a shifted duplicate
            labels(node) = SHIFT_DUPL;
          } else { // Node is not a shifted duplicate. Save child trees
            labels(node) = DONE; // Add children to tree root maps
            tree_roots.push(child_l);
            tree_roots.push(child_r);
          }
        }
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
      region_counters_sa(labels(root)) += 1;
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
//printf("%u FIRST_OCUR\n", *iter);
//    } else if(labels_h(*iter) == SHIFT_DUPL) {
//printf("%u SHIFT_DUPL\n", *iter);
//    }
//  }

  Kokkos::Experimental::contribute(region_counters, region_counters_sv);
  Kokkos::fence();
  Kokkos::deep_copy(chunk_counters_h, chunk_counters);
  Kokkos::deep_copy(region_counters_h, region_counters);

  printf("Checkpoint %u\n", chkpt_id);
  printf("Number of first occurrence chunks:  %lu\n", chunk_counters_h(FIRST_OCUR));
  printf("Number of fixed duplicate chunks:   %lu\n", chunk_counters_h(FIXED_DUPL));
  printf("Number of shifted duplicate chunks: %lu\n", chunk_counters_h(SHIFT_DUPL));
  printf("Number of first occurrence regions:  %lu\n", region_counters_h(FIRST_OCUR));
  printf("Number of fixed duplicate regions:   %lu\n", region_counters_h(FIXED_DUPL));
  printf("Number of shifted duplicate regions: %lu\n", region_counters_h(SHIFT_DUPL));
  return;
  Kokkos::Experimental::contribute(region_counters, region_counters_sv);
  Kokkos::fence();
  Kokkos::deep_copy(chunk_counters_h, chunk_counters);
  Kokkos::deep_copy(region_counters_h, region_counters);
//  Kokkos::parallel_for(1, KOKKOS_LAMBDA(const uint32_t _node) {
//    for(uint32_t node=0; node<num_nodes; node++) {
//      if(first_ocur_updates.exists(node)) {
//printf("%u FIRST_OCUR\n", node);
//      } else if(shift_dupl_updates.exists(node)) {
//printf("%u SHIFT_DUPL\n", node);
//      }
//    }
//  });

  printf("Checkpoint %u\n", chkpt_id);
  printf("Number of first occurrence chunks:  %lu\n", chunk_counters_h(FIRST_OCUR));
  printf("Number of fixed duplicate chunks:   %lu\n", chunk_counters_h(FIXED_DUPL));
  printf("Number of shifted duplicate chunks: %lu\n", chunk_counters_h(SHIFT_DUPL));
  printf("Number of first occurrence regions:  %lu\n", region_counters_h(FIRST_OCUR));
  printf("Number of fixed duplicate regions:   %lu\n", region_counters_h(FIXED_DUPL));
  printf("Number of shifted duplicate regions: %lu\n", region_counters_h(SHIFT_DUPL));
  return;
}

template<class Hasher>
void deduplicate_data(Kokkos::View<uint8_t*>& data, 
                      const uint32_t chunk_size, 
                      const Hasher hasher, 
                      MerkleTree& tree, 
                      const uint32_t tree_id, 
                      const SharedNodeIDMap& prior_identical_map, 
                      const SharedNodeIDMap& prior_shared_map, 
                      const DistinctNodeIDMap& prior_distinct_map, 
                      SharedNodeIDMap& identical_map, 
                      SharedNodeIDMap& shared_map, 
                      DistinctNodeIDMap& distinct_map, 
                      CompactTable& shared_updates,
                      CompactTable& distinct_updates) {
  STDOUT_PRINT("==========Start Deduplicate Data==========\n");
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  const uint32_t num_nodes = 2*num_chunks-1;
  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  const uint32_t leaf_start = num_chunks-1;
  const uint32_t num_prior_distinct = distinct_map.size();

  DEBUG_PRINT("Num prior shared: %u\n", prior_shared_map.size());
  DEBUG_PRINT("Num prior distinct: %u\n", prior_distinct_map.size());

  uint32_t prev_leftover = UINT32_MAX;
  uint32_t current_level = num_levels-1;
  uint32_t start_offset = (1 << (num_levels-1))-1;
  uint32_t end_offset = (1 << num_levels)-1;
  if(end_offset > num_nodes)
    end_offset = num_nodes;
  DEBUG_PRINT("Number of chunks: %u\n", num_chunks);
  DEBUG_PRINT("Number of nodes: %u\n", num_nodes);
  DEBUG_PRINT("Number of levels: %u\n", num_levels);
  DEBUG_PRINT("leaf start: %u\n", leaf_start);
  DEBUG_PRINT("Start, end offsets: (%u,%u)\n", start_offset, end_offset);
  Kokkos::View<uint32_t[1]> nodes_leftover("Leftover nodes to process");
  Kokkos::View<uint32_t[1]>::HostMirror nodes_leftover_h = Kokkos::create_mirror_view(nodes_leftover);
  Kokkos::deep_copy(nodes_leftover, 0);
  nodes_leftover_h(0) = 0;
#ifdef STATS
  Kokkos::View<uint32_t[1]> num_same("Number of chunks that remain the same");
  Kokkos::View<uint32_t[1]> num_new("Number of chunks that are new");
  Kokkos::View<uint32_t[1]> num_shift("Number of chunks that exist but in different spaces");
  Kokkos::View<uint32_t[1]> num_comp_d("Number of compressed distinct nodes");
  Kokkos::View<uint32_t[1]> num_comp_s("Number of compressed shared nodes");
  Kokkos::View<uint32_t[1]> num_dupl("Number of new duplicate nodes");
  Kokkos::View<uint32_t[1]> num_other("Number of other nodes");
  Kokkos::View<uint32_t[1]>::HostMirror num_same_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_new_h = Kokkos::create_mirror_view(num_new);
  Kokkos::View<uint32_t[1]>::HostMirror num_shift_h = Kokkos::create_mirror_view(num_shift);
  Kokkos::View<uint32_t[1]>::HostMirror num_comp_d_h = Kokkos::create_mirror_view(num_comp_d);
  Kokkos::View<uint32_t[1]>::HostMirror num_comp_s_h = Kokkos::create_mirror_view(num_comp_s);
  Kokkos::View<uint32_t[1]>::HostMirror num_dupl_h = Kokkos::create_mirror_view(num_dupl);
  Kokkos::View<uint32_t[1]>::HostMirror num_other_h = Kokkos::create_mirror_view(num_other);
  Kokkos::UnorderedMap<HashDigest, void, Kokkos::DefaultExecutionSpace, digest_hash, digest_equal_to> table(num_chunks);
  Kokkos::View<uint32_t[10]> num_prior_chunks_d("Number of chunks from prior checkpoints");
  Kokkos::View<uint32_t[10]>::HostMirror num_prior_chunks_h = Kokkos::create_mirror_view(num_prior_chunks_d);
  Kokkos::deep_copy(num_same, 0);
  Kokkos::deep_copy(num_new, 0);
  Kokkos::deep_copy(num_shift, 0);
  Kokkos::deep_copy(num_comp_d, 0);
  Kokkos::deep_copy(num_comp_s, 0);
  Kokkos::deep_copy(num_dupl, 0);
  Kokkos::deep_copy(num_other, 0);
  Kokkos::deep_copy(num_prior_chunks_d, 0);
#endif

  while(nodes_leftover_h(0) != prev_leftover) {
    prev_leftover = nodes_leftover_h(0);
    if(start_offset > num_chunks-1)
      prev_leftover = UINT32_MAX;
    Kokkos::parallel_for("Insert/compare hashes", Kokkos::RangePolicy<>(start_offset,end_offset), KOKKOS_LAMBDA(const uint32_t i) {
//    Kokkos::parallel_for("Insert/compare hashes", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const uint32_t _i) {
//for(uint32_t i=start_offset; i<end_offset; i++) {
      uint32_t node = i;
      if(node >= leaf_start) {
        uint32_t num_bytes = chunk_size;
        if(node == num_nodes-1)
          num_bytes = data.size()-(node-leaf_start)*chunk_size;
        hasher.hash(data.data()+((node-leaf_start)*chunk_size), num_bytes, tree(node).digest);

        if(tree_id != 0) {
          NodeID info = NodeID(node,tree_id);
          uint32_t index = prior_distinct_map.find(tree(node));
          if(!prior_distinct_map.valid_at(index)) { // Chunk not in prior map
            auto result = distinct_map.insert(tree(node), info);
            if(result.success()) { // Chunk is brand new
              tree.distinct_children_d(node) = 2;
#ifdef STATS
              Kokkos::atomic_add(&(num_new(0)), 1);
#endif
              nodes_leftover(0) += static_cast<uint32_t>(1);
            } else if(result.existing()) { // Chunk already exists locally
              NodeID& existing_info = distinct_map.value_at(result.index());
              tree.distinct_children_d(node) = 8;
              shared_map.insert(node, NodeID(existing_info.node, existing_info.tree));
              nodes_leftover(0) += static_cast<uint32_t>(1);
#ifdef STATS
              Kokkos::atomic_add(&num_dupl(0), static_cast<uint32_t>(1));
#endif
            } else if(result.failed()) {
              printf("Failed to insert new chunk into distinct or shared map (tree %u). Shouldn't happen.", tree_id);
            }
          } else { // Chunk already exists
            NodeID old_distinct = prior_distinct_map.value_at(index);
            if(prior_identical_map.exists(i)) {
              uint32_t old_identical_idx = prior_identical_map.find(i);
              NodeID old_identical = prior_identical_map.value_at(old_identical_idx);
              if(old_distinct.node == old_identical.node && old_distinct.tree == old_identical.tree) {
                #ifdef STATS
                Kokkos::atomic_add(&num_same(0), static_cast<uint32_t>(1));
                #endif
                tree.distinct_children_d(node) = 0;
                identical_map.insert(i, old_distinct);
              } else {
                uint32_t prior_shared_idx = prior_shared_map.find(node);
                if(prior_shared_map.valid_at(prior_shared_idx)) { // Node was repeat last checkpoint 
                  NodeID prior_shared = prior_shared_map.value_at(prior_shared_idx);
                  if(prior_shared.node == old_distinct.node && prior_shared.tree == old_distinct.tree) {
                    tree.distinct_children_d(node) = 0;
                    identical_map.insert(node, old_distinct);
                    #ifdef STATS
                    Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
                    #endif
                  } else {
                    shared_map.insert(node, NodeID(old_distinct.node, old_distinct.tree));
                    tree.distinct_children_d(node) = 8;
                    nodes_leftover(0) += static_cast<uint32_t>(1);
                    #ifdef STATS
                    auto res = table.insert(tree(node));
                    if(res.success())
                      Kokkos::atomic_add(&num_prior_chunks_d(old_distinct.tree), 1);
                    Kokkos::atomic_add(&(num_shift(0)), static_cast<uint32_t>(1));
                    #endif
                  }
                } else { // Node was distinct last checkpoint
                  if(node == old_distinct.node) { // No change since last checkpoint
                    tree.distinct_children_d(node) = 0;
                    identical_map.insert(node, old_distinct);
                    #ifdef STATS
                    Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
                    #endif
                  } else { // Node changed since last checkpoint
                    shared_map.insert(node, NodeID(old_distinct.node, old_distinct.tree));
                    tree.distinct_children_d(node) = 8;
                    nodes_leftover(0) += static_cast<uint32_t>(1);
                    #ifdef STATS
                    auto res = table.insert(tree(node));
                    if(res.success())
                      Kokkos::atomic_add(&num_prior_chunks_d(old_distinct.tree), 1);
                    Kokkos::atomic_add(&(num_shift(0)), static_cast<uint32_t>(1));
                    #endif
                  }
                }
              }
            } else {
              uint32_t prior_shared_idx = prior_shared_map.find(node);
              if(prior_shared_map.valid_at(prior_shared_idx)) { // Node was repeat last checkpoint 
                NodeID prior_shared = prior_shared_map.value_at(prior_shared_idx);
                if(prior_shared.node == old_distinct.node && prior_shared.tree == old_distinct.tree) {
                  tree.distinct_children_d(node) = 0;
                  identical_map.insert(node, old_distinct);
                  #ifdef STATS
                  Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
                  #endif
                } else {
                  shared_map.insert(node, NodeID(old_distinct.node, old_distinct.tree));
                  tree.distinct_children_d(node) = 8;
                  nodes_leftover(0) += static_cast<uint32_t>(1);
                  #ifdef STATS
                  auto res = table.insert(tree(node));
                  if(res.success())
                    Kokkos::atomic_add(&num_prior_chunks_d(old_distinct.tree), 1);
                  Kokkos::atomic_add(&(num_shift(0)), static_cast<uint32_t>(1));
                  #endif
                }
              } else { // Node was distinct last checkpoint
                if(node == old_distinct.node) { // No change since last checkpoint
                  tree.distinct_children_d(node) = 0;
                  identical_map.insert(node, old_distinct);
                  #ifdef STATS
                  Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
                  #endif
                } else { // Node changed since last checkpoint
                  shared_map.insert(node, NodeID(old_distinct.node, old_distinct.tree));
                  tree.distinct_children_d(node) = 8;
                  nodes_leftover(0) += static_cast<uint32_t>(1);
                  #ifdef STATS
                  auto res = table.insert(tree(node));
                  if(res.success())
                    Kokkos::atomic_add(&num_prior_chunks_d(old_distinct.tree), 1);
                  Kokkos::atomic_add(&(num_shift(0)), static_cast<uint32_t>(1));
                  #endif
                }
              }
            }
          }
        }
      } else {
        uint32_t child_l = 2*node + 1;
        uint32_t child_r = 2*node + 2;
        tree.distinct_children_d(node) = tree.distinct_children_d(child_l)/2 + tree.distinct_children_d(child_r)/2;
        if(tree.distinct_children_d(node) == 2) {
          hasher.hash((uint8_t*)&tree(2*(node)+1), 2*hasher.digest_size(), (uint8_t*)&tree(node));
          distinct_map.insert(tree(node), NodeID(node, tree_id));
          nodes_leftover(0) += static_cast<uint32_t>(1);
        } else if(tree.distinct_children_d(node) == 8) {
          hasher.hash((uint8_t*)&tree(2*(node)+1), 2*hasher.digest_size(), (uint8_t*)&tree(node));
          if(prior_distinct_map.exists(tree(node))) {
            nodes_leftover(0) += static_cast<uint32_t>(1);
          } else {
            uint32_t child_l = 2*(node)+1;
            uint32_t child_r = 2*(node)+2;
            if(prior_distinct_map.exists(tree(child_l))) {
              NodeID info = prior_distinct_map.value_at(prior_distinct_map.find(tree(child_l)));
              insert_entry(shared_updates, child_l, num_nodes, tree_id, info);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_s(0)), static_cast<uint32_t>(1));
#endif
            } else if(shared_map.exists(child_l)) {
              insert_entry(shared_updates, child_l, num_nodes, tree_id, shared_map.value_at(shared_map.find(child_l)));
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
#endif
            }
            if(prior_distinct_map.exists(tree(child_r))) {
              NodeID info = prior_distinct_map.value_at(prior_distinct_map.find(tree(child_r)));
              insert_entry(shared_updates, child_r, num_nodes, tree_id, info);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_s(0)), static_cast<uint32_t>(1));
#endif
            } else if(shared_map.exists(child_r)) {
              insert_entry(shared_updates, child_r, num_nodes, tree_id, shared_map.value_at(shared_map.find(child_r)));
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
#endif
            }
            tree.distinct_children_d(node) = 0;
          }
        } else if(tree.distinct_children_d(node) == 5) {
          uint32_t child_l = 2*(node)+1;
          uint32_t child_r = 2*(node)+2;
          if(child_l < num_nodes) {
            if((tree.distinct_children_d(child_l) == 2)) {
              insert_entry(distinct_updates, child_l, num_nodes, tree_id, NodeID(child_l, tree_id));
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
#endif
            } else if((tree.distinct_children_d(child_l) == 8)) {
              if(prior_distinct_map.exists(tree(child_l))) {
                NodeID info = prior_distinct_map.value_at(prior_distinct_map.find(tree(child_l)));
                insert_entry(shared_updates, child_l, num_nodes, tree_id, info);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_s(0)), static_cast<uint32_t>(1));
#endif
              } else if(shared_map.exists(child_l)) {
                auto repeat = shared_map.value_at(shared_map.find(child_l));
                insert_entry(shared_updates, child_l, num_nodes, tree_id, repeat);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
#endif
              }
            }
          }
          if(child_r < num_nodes) {
            if((tree.distinct_children_d(child_r) == 2)) {
              insert_entry(distinct_updates, child_r, num_nodes, tree_id, NodeID(child_r, tree_id));
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
#endif
            } else if((tree.distinct_children_d(child_r) == 8)) {
              if(prior_distinct_map.exists(tree(child_r))) {
                NodeID info = prior_distinct_map.value_at(prior_distinct_map.find(tree(child_r)));
                insert_entry(shared_updates, child_r, num_nodes, tree_id, info);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_s(0)), static_cast<uint32_t>(1));
#endif
              } else if(shared_map.exists(child_r)) {
                NodeID repeat = shared_map.value_at(shared_map.find(child_r));
                insert_entry(shared_updates, child_r, num_nodes, tree_id, repeat);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
#endif
              }
            }
          }
          tree.distinct_children_d(node) = 0;
        } else if(tree.distinct_children_d(node) == 4) {
          uint32_t child_l = 2*(node)+1;
          uint32_t child_r = 2*(node)+2;
          if((child_l < num_nodes) && (tree.distinct_children_d(child_l) == 8)) {
            if(prior_distinct_map.exists(tree(child_l))) {
              NodeID info = prior_distinct_map.value_at(prior_distinct_map.find(tree(child_l)));
              insert_entry(shared_updates, child_l, num_nodes, tree_id, info);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_s(0)), static_cast<uint32_t>(1));
#endif
            } else if(shared_map.exists(child_l)) {
                insert_entry(shared_updates, child_l, num_nodes, tree_id, shared_map.value_at(shared_map.find(child_l)));
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
#endif
            }
          } else if((child_r < num_nodes) && (tree.distinct_children_d(child_r) == 8)) {
            if(prior_distinct_map.exists(tree(child_r))) {
              NodeID info = prior_distinct_map.value_at(prior_distinct_map.find(tree(child_r)));
              insert_entry(shared_updates, child_r, num_nodes, tree_id, info);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_s(0)), static_cast<uint32_t>(1));
#endif
            } else if(shared_map.exists(child_r)) {
                insert_entry(shared_updates, child_r, num_nodes, tree_id, shared_map.value_at(shared_map.find(child_r)));
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
#endif
            }
          }
          tree.distinct_children_d(node) = 0;
        } else if(tree.distinct_children_d(node) == 1) {
          uint32_t child_l = 2*(node)+1;
          uint32_t child_r = 2*(node)+2;
          if((child_l < num_nodes) && (tree.distinct_children_d(child_l) == 2)) {
            insert_entry(distinct_updates, child_l, num_nodes, tree_id, NodeID(child_l, tree_id));
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
#endif
          } else if((child_r < num_nodes) && (tree.distinct_children_d(child_r) == 2)) {
            insert_entry(distinct_updates, child_r, num_nodes, tree_id, NodeID(child_r, tree_id));
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
#endif
          }
          tree.distinct_children_d(node) = 0;
        }
      }
//}
    });
#ifdef STATS
if(start_offset >= leaf_start-(num_chunks/2)) {
  Kokkos::fence();
  printf("------------------------------\n");
  uint32_t n_distinct = 0;
  Kokkos::RangePolicy<> reduce_policy(start_offset, end_offset);
  Kokkos::parallel_reduce("Count number of distinct", reduce_policy, KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
  //  if(tree.distinct_children_d(start_offset+i) == 2) {
    if(tree.distinct_children_d(i) == 2) {
      update += 1;
    }
  }, n_distinct);
  uint32_t n_same = 0;
  Kokkos::fence();
  Kokkos::parallel_reduce("Count number of same", reduce_policy, KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
    if(tree.distinct_children_d(i) == 0) {
      update += 1;
    }
  }, n_same);
  uint32_t n_shared = 0;
  Kokkos::parallel_reduce("Count number of shared", reduce_policy, KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
    if(tree.distinct_children_d(i) == 8) {
      update += 1;
    }
  }, n_shared);
  uint32_t n_distinct_shared = 0;
  Kokkos::parallel_reduce("Count number of distinct shared", reduce_policy, KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
    if(tree.distinct_children_d(i) == 5) {
      update += 1;
    }
  }, n_distinct_shared);
  uint32_t n_distinct_same = 0;
  Kokkos::parallel_reduce("Count number of distinct_same", reduce_policy, KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
    if(tree.distinct_children_d(i) == 1) {
      update += 1;
    }
  }, n_distinct_same);
  uint32_t n_shared_same = 0;
  Kokkos::parallel_reduce("Count number of shared_same", reduce_policy, KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
    if(tree.distinct_children_d(i) == 4) {
      update += 1;
    }
  }, n_shared_same);
  Kokkos::fence();
  STDOUT_PRINT("Count number of distinct chunks: %u\n", n_distinct);
  STDOUT_PRINT("Count number of same chunks: %u\n", n_same);
  STDOUT_PRINT("Count number of shared chunks: %u\n", n_shared);
  STDOUT_PRINT("Count number of distinct shared chunks: %u\n", n_distinct_shared);
  STDOUT_PRINT("Count number of distinct_same chunks: %u\n", n_distinct_same);
  STDOUT_PRINT("Count number of shared_same chunks: %u\n", n_shared_same);
  STDOUT_PRINT("------------------------------\n");
}
#endif
    Kokkos::fence();
    Kokkos::deep_copy(nodes_leftover_h, nodes_leftover);
    current_level -= 1;
    start_offset = (1 << current_level) - 1;
    end_offset = (1 << (current_level+1)) - 1;
  }
  Kokkos::fence();
#ifdef STATS
  Kokkos::deep_copy(num_same_h, num_same);
  Kokkos::deep_copy(num_new_h, num_new);
  Kokkos::deep_copy(num_shift_h, num_shift);
  Kokkos::deep_copy(num_comp_d_h, num_comp_d);
  Kokkos::deep_copy(num_comp_s_h, num_comp_s);
  Kokkos::deep_copy(num_dupl_h, num_dupl);
  Kokkos::deep_copy(num_other_h, num_other);
  Kokkos::fence();
  STDOUT_PRINT("Number of chunks: %u\n", num_chunks);
  STDOUT_PRINT("Number of new chunks: %u\n", num_new_h(0));
  STDOUT_PRINT("Number of same chunks: %u\n", num_same_h(0));
  STDOUT_PRINT("Number of shift chunks: %u\n", num_shift_h(0));
  STDOUT_PRINT("Number of distinct comp nodes: %u\n", num_comp_d_h(0));
  STDOUT_PRINT("Number of shared comp nodes: %u\n", num_comp_s_h(0));
  STDOUT_PRINT("Number of dupl nodes: %u\n", num_dupl_h(0));
  STDOUT_PRINT("Number of other nodes: %u\n", num_other_h(0));
  Kokkos::deep_copy(num_prior_chunks_h, num_prior_chunks_d);
  for(int i=0; i<10; i++) {
    STDOUT_PRINT("Number of chunks repeating from checkpoint %d: %u\n", i, num_prior_chunks_h(i));
  }
#endif
  STDOUT_PRINT("==========End Deduplicate Data==========\n");
}

template<class Hasher>
void deduplicate_data(Kokkos::View<uint8_t*>& data, 
                      const uint32_t chunk_size, 
                      const Hasher hasher, 
                      MerkleTree& tree, 
                      const uint32_t tree_id, 
                      const NodeMap& prior_node_map,
                      const DistinctNodeIDMap& prior_distinct_map, 
                      NodeMap& node_map,
                      DistinctNodeIDMap& distinct_map, 
                      NodeMap& updates) {
  STDOUT_PRINT("==========Start Deduplicate Data==========\n");
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  const uint32_t num_nodes = 2*num_chunks-1;
  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  const uint32_t leaf_start = num_chunks-1;
  const uint32_t num_prior_distinct = distinct_map.size();

//  DEBUG_PRINT("Num prior shared: %u\n", prior_shared_map.size());
//  DEBUG_PRINT("Num prior distinct: %u\n", prior_distinct_map.size());

  uint32_t prev_leftover = UINT32_MAX;
  uint32_t current_level = num_levels-1;
  uint32_t start_offset = (1 << (num_levels-1))-1;
  uint32_t end_offset = (1 << num_levels)-1;
  if(end_offset > num_nodes)
    end_offset = num_nodes;
  DEBUG_PRINT("Number of chunks: %u\n", num_chunks);
  DEBUG_PRINT("Number of nodes: %u\n", num_nodes);
  DEBUG_PRINT("Number of levels: %u\n", num_levels);
  DEBUG_PRINT("leaf start: %u\n", leaf_start);
  DEBUG_PRINT("Start, end offsets: (%u,%u)\n", start_offset, end_offset);
  Kokkos::View<uint32_t[1]> nodes_leftover("Leftover nodes to process");
  Kokkos::View<uint32_t[1]>::HostMirror nodes_leftover_h = Kokkos::create_mirror_view(nodes_leftover);
  Kokkos::deep_copy(nodes_leftover, 0);
  nodes_leftover_h(0) = 0;
#ifdef STATS
  Kokkos::View<uint32_t[1]> num_same("Number of chunks that remain the same");
  Kokkos::View<uint32_t[1]> num_new("Number of chunks that are new");
  Kokkos::View<uint32_t[1]> num_shift("Number of chunks that exist but in different spaces");
  Kokkos::View<uint32_t[1]> num_comp_d("Number of compressed distinct nodes");
  Kokkos::View<uint32_t[1]> num_comp_s("Number of compressed shared nodes");
  Kokkos::View<uint32_t[1]> num_dupl("Number of new duplicate nodes");
  Kokkos::View<uint32_t[1]> num_other("Number of other nodes");
  Kokkos::View<uint32_t[1]>::HostMirror num_same_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_new_h = Kokkos::create_mirror_view(num_new);
  Kokkos::View<uint32_t[1]>::HostMirror num_shift_h = Kokkos::create_mirror_view(num_shift);
  Kokkos::View<uint32_t[1]>::HostMirror num_comp_d_h = Kokkos::create_mirror_view(num_comp_d);
  Kokkos::View<uint32_t[1]>::HostMirror num_comp_s_h = Kokkos::create_mirror_view(num_comp_s);
  Kokkos::View<uint32_t[1]>::HostMirror num_dupl_h = Kokkos::create_mirror_view(num_dupl);
  Kokkos::View<uint32_t[1]>::HostMirror num_other_h = Kokkos::create_mirror_view(num_other);
  Kokkos::UnorderedMap<HashDigest, void, Kokkos::DefaultExecutionSpace, digest_hash, digest_equal_to> table(num_chunks);
  Kokkos::View<uint32_t[10]> num_prior_chunks_d("Number of chunks from prior checkpoints");
  Kokkos::View<uint32_t[10]>::HostMirror num_prior_chunks_h = Kokkos::create_mirror_view(num_prior_chunks_d);
  Kokkos::deep_copy(num_same, 0);
  Kokkos::deep_copy(num_new, 0);
  Kokkos::deep_copy(num_shift, 0);
  Kokkos::deep_copy(num_comp_d, 0);
  Kokkos::deep_copy(num_comp_s, 0);
  Kokkos::deep_copy(num_dupl, 0);
  Kokkos::deep_copy(num_other, 0);
  Kokkos::deep_copy(num_prior_chunks_d, 0);
#endif

  while(nodes_leftover_h(0) != prev_leftover) {
    prev_leftover = nodes_leftover_h(0);
    if(start_offset > num_chunks-1)
      prev_leftover = UINT32_MAX;
      Kokkos::parallel_for("Insert/compare hashes", Kokkos::RangePolicy<>(start_offset,end_offset), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node = i;
        if(node >= leaf_start) {
          uint32_t num_bytes = chunk_size;
          if(node == num_nodes-1)
            num_bytes = data.size()-(node-leaf_start)*chunk_size;
          hasher.hash(data.data()+((node-leaf_start)*chunk_size), num_bytes, tree(node).digest);
  
          if(tree_id != 0) {
            NodeID info = NodeID(node,tree_id);
            uint32_t index = prior_distinct_map.find(tree(node));
            if(!prior_distinct_map.valid_at(index)) { // Chunk not in prior map
              auto result = distinct_map.insert(tree(node), info);
              if(result.success()) { // Chunk is brand new
                tree.distinct_children_d(node) = 2;
                #ifdef STATS
                Kokkos::atomic_add(&(num_new(0)), 1);
                #endif
                nodes_leftover(0) += static_cast<uint32_t>(1);
              } else if(result.existing()) { // Chunk already exists locally
                NodeID& existing_info = distinct_map.value_at(result.index());
                tree.distinct_children_d(node) = 8;
                node_map.insert(node, Node(existing_info.node, existing_info.tree, Repeat));
                nodes_leftover(0) += static_cast<uint32_t>(1);
                #ifdef STATS
                Kokkos::atomic_add(&num_dupl(0), static_cast<uint32_t>(1));
                #endif
              } else if(result.failed()) {
                printf("Failed to insert new chunk into distinct or shared map (tree %u). Shouldn't happen.", tree_id);
              }
            } else { // Chunk already exists
              NodeID old_distinct = prior_distinct_map.value_at(index);
              if(prior_node_map.exists(i) && prior_node_map.value_at(prior_node_map.find(i)).nodetype == Identical) {
                uint32_t old_identical_idx = prior_node_map.find(i);
                Node old_identical = prior_node_map.value_at(old_identical_idx);
                if(old_distinct.node == old_identical.node && old_distinct.tree == old_identical.tree) {
                  #ifdef STATS
                  Kokkos::atomic_add(&num_same(0), static_cast<uint32_t>(1));
                  #endif
                  tree.distinct_children_d(node) = 0;
                  node_map.insert(i, Node(old_distinct.node, old_distinct.tree, Identical));
                } else {
                  uint32_t prior_shared_idx = prior_node_map.find(node);
                  Node prior_repeat_node = prior_node_map.value_at(prior_shared_idx);
                  if(prior_node_map.valid_at(prior_shared_idx) && prior_repeat_node.nodetype == Repeat) { // Node was repeat last checkpoint 
                    Node prior_shared = prior_node_map.value_at(prior_shared_idx);
                    if(prior_shared.node == old_distinct.node && prior_shared.tree == old_distinct.tree) {
                      tree.distinct_children_d(node) = 0;
                      node_map.insert(node, Node(old_distinct.node, old_distinct.tree, Identical));
                      #ifdef STATS
                      Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
                      #endif
                    } else {
                      node_map.insert(node, Node(old_distinct.node, old_distinct.tree, Repeat));
                      tree.distinct_children_d(node) = 8;
                      nodes_leftover(0) += static_cast<uint32_t>(1);
                      #ifdef STATS
                      auto res = table.insert(tree(node));
                      if(res.success())
                        Kokkos::atomic_add(&num_prior_chunks_d(old_distinct.tree), 1);
                      Kokkos::atomic_add(&(num_shift(0)), static_cast<uint32_t>(1));
                      #endif
                    }
                  } else { // Node was distinct last checkpoint
                    if(node == old_distinct.node) { // No change since last checkpoint
                      tree.distinct_children_d(node) = 0;
                      node_map.insert(node, Node(old_distinct.node, old_distinct.tree, Identical));
                      #ifdef STATS
                      Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
                      #endif
                    } else { // Node changed since last checkpoint
                      node_map.insert(node, Node(old_distinct.node, old_distinct.tree, Repeat));
                      tree.distinct_children_d(node) = 8;
                      nodes_leftover(0) += static_cast<uint32_t>(1);
                      #ifdef STATS
                      auto res = table.insert(tree(node));
                      if(res.success())
                        Kokkos::atomic_add(&num_prior_chunks_d(old_distinct.tree), 1);
                      Kokkos::atomic_add(&(num_shift(0)), static_cast<uint32_t>(1));
                      #endif
                    }
                  }
                }
              } else {
                uint32_t prior_shared_idx = prior_node_map.find(node);
                Node prior_repeat_node = prior_node_map.value_at(prior_shared_idx);
                if(prior_node_map.valid_at(prior_shared_idx) && prior_repeat_node.nodetype == Repeat) { // Node was repeat last checkpoint 
                  Node prior_shared = prior_node_map.value_at(prior_shared_idx);
                  if(prior_shared.node == old_distinct.node && prior_shared.tree == old_distinct.tree) {
                    tree.distinct_children_d(node) = 0;
                    node_map.insert(node, Node(old_distinct.node, old_distinct.tree, Identical));
                    #ifdef STATS
                    Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
                    #endif
                  } else {
                    node_map.insert(node, Node(old_distinct.node, old_distinct.tree, Repeat));
                    tree.distinct_children_d(node) = 8;
                    nodes_leftover(0) += static_cast<uint32_t>(1);
                    #ifdef STATS
                    auto res = table.insert(tree(node));
                    if(res.success())
                      Kokkos::atomic_add(&num_prior_chunks_d(old_distinct.tree), 1);
                    Kokkos::atomic_add(&(num_shift(0)), static_cast<uint32_t>(1));
                    #endif
                  }
                } else { // Node was distinct last checkpoint
                  if(node == old_distinct.node) { // No change since last checkpoint
                    tree.distinct_children_d(node) = 0;
                    node_map.insert(node, Node(old_distinct.node, old_distinct.tree, Identical));
                    #ifdef STATS
                    Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
                    #endif
                  } else { // Node changed since last checkpoint
                    node_map.insert(node, Node(old_distinct.node, old_distinct.tree, Repeat));
                    tree.distinct_children_d(node) = 8;
                    nodes_leftover(0) += static_cast<uint32_t>(1);
                    #ifdef STATS
                    auto res = table.insert(tree(node));
                    if(res.success())
                      Kokkos::atomic_add(&num_prior_chunks_d(old_distinct.tree), 1);
                    Kokkos::atomic_add(&(num_shift(0)), static_cast<uint32_t>(1));
                    #endif
                  }
                }
              }
            }
          }
        } else {
          uint32_t child_l = 2*node + 1;
          uint32_t child_r = 2*node + 2;
          tree.distinct_children_d(node) = tree.distinct_children_d(child_l)/2 + tree.distinct_children_d(child_r)/2;
          if(tree.distinct_children_d(node) == 2) { // Both children are distinct
            hasher.hash((uint8_t*)&tree(2*(node)+1), 2*hasher.digest_size(), (uint8_t*)&tree(node));
            distinct_map.insert(tree(node), NodeID(node, tree_id));
            nodes_leftover(0) += static_cast<uint32_t>(1);
          } else if(tree.distinct_children_d(node) == 8) { // Both children are repeats
            hasher.hash((uint8_t*)&tree(2*(node)+1), 2*hasher.digest_size(), (uint8_t*)&tree(node));
            if(prior_distinct_map.exists(tree(node))) {
              nodes_leftover(0) += static_cast<uint32_t>(1);
            } else {
              uint32_t child_l = 2*(node)+1;
              uint32_t child_r = 2*(node)+2;
              if(prior_distinct_map.exists(tree(child_l))) {
                NodeID info = prior_distinct_map.value_at(prior_distinct_map.find(tree(child_l)));
                updates.insert(child_l, Node(info.node, info.tree, Repeat));
                #ifdef STATS
                Kokkos::atomic_add(&(num_comp_s(0)), static_cast<uint32_t>(1));
                #endif
              } else if(node_map.exists(child_l) && node_map.value_at(node_map.find(child_l)).nodetype == Repeat) {
                Node repeat_node = node_map.value_at(node_map.find(child_l));
                updates.insert(child_l, Node(repeat_node.node, repeat_node.tree, Repeat));
                #ifdef STATS
                Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
                #endif
              }
              if(prior_distinct_map.exists(tree(child_r))) {
                NodeID info = prior_distinct_map.value_at(prior_distinct_map.find(tree(child_r)));
                updates.insert(child_r, Node(info.node, info.tree, Repeat));
                #ifdef STATS
                Kokkos::atomic_add(&(num_comp_s(0)), static_cast<uint32_t>(1));
                #endif
              } else if(node_map.exists(child_r) && node_map.value_at(node_map.find(child_r)).nodetype == Repeat) {
                Node repeat_node = node_map.value_at(node_map.find(child_r));
                updates.insert(child_r, Node(repeat_node.node, repeat_node.tree, Repeat));
                #ifdef STATS
                Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
                #endif
              }
              tree.distinct_children_d(node) = 0;
            }
          } else if(tree.distinct_children_d(node) == 5) { // One child is distincting and one is a repeat
            uint32_t child_l = 2*(node)+1;
            uint32_t child_r = 2*(node)+2;
            if(child_l < num_nodes) {
              if((tree.distinct_children_d(child_l) == 2)) {
                updates.insert(child_l, Node(child_l, tree_id, Distinct));
                #ifdef STATS
                Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
                #endif
              } else if((tree.distinct_children_d(child_l) == 8)) {
                if(prior_distinct_map.exists(tree(child_l))) {
                  NodeID info = prior_distinct_map.value_at(prior_distinct_map.find(tree(child_l)));
                  updates.insert(child_l, Node(info.node, info.tree, Repeat));
                  #ifdef STATS
                  Kokkos::atomic_add(&(num_comp_s(0)), static_cast<uint32_t>(1));
                  #endif
                } else if(node_map.exists(child_l) && node_map.value_at(node_map.find(child_l)).nodetype == Repeat) {
                  auto repeat = node_map.value_at(node_map.find(child_l));
                  updates.insert(child_l, Node(repeat.node, repeat.tree, Repeat));
                  #ifdef STATS
                  Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
                  #endif
                }
              }
            }
            if(child_r < num_nodes) {
              if((tree.distinct_children_d(child_r) == 2)) {
                updates.insert(child_r, Node(child_r, tree_id, Distinct));
                #ifdef STATS
                Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
                #endif
              } else if((tree.distinct_children_d(child_r) == 8)) {
                if(prior_distinct_map.exists(tree(child_r))) {
                  NodeID info = prior_distinct_map.value_at(prior_distinct_map.find(tree(child_r)));
                  updates.insert(child_r, Node(info.node, info.tree, Repeat));
                  #ifdef STATS
                  Kokkos::atomic_add(&(num_comp_s(0)), static_cast<uint32_t>(1));
                  #endif
                } else if(node_map.exists(child_r) && node_map.value_at(node_map.find(child_r)).nodetype == Repeat) {
                  auto repeat = node_map.value_at(node_map.find(child_r));
                  updates.insert(child_r, Node(repeat.node, repeat.tree, Repeat));
                  #ifdef STATS
                  Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
                  #endif
                }
              }
            }
            tree.distinct_children_d(node) = 0;
          } else if(tree.distinct_children_d(node) == 4) { // One child is a repeat and the other is identical
            uint32_t child_l = 2*(node)+1;
            uint32_t child_r = 2*(node)+2;
            if((child_l < num_nodes) && (tree.distinct_children_d(child_l) == 8)) {
              if(prior_distinct_map.exists(tree(child_l))) {
                NodeID info = prior_distinct_map.value_at(prior_distinct_map.find(tree(child_l)));
                updates.insert(child_l, Node(info.node, info.tree, Repeat));
                #ifdef STATS
                Kokkos::atomic_add(&(num_comp_s(0)), static_cast<uint32_t>(1));
                #endif
              } else if(node_map.exists(child_l) && node_map.value_at(node_map.find(child_l)).nodetype == Repeat) {
                auto repeat = node_map.value_at(node_map.find(child_l));
                updates.insert(child_l, Node(repeat.node, repeat.tree, Repeat));
                #ifdef STATS
                Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
                #endif
              }
            } else if((child_r < num_nodes) && (tree.distinct_children_d(child_r) == 8)) {
              if(prior_distinct_map.exists(tree(child_r))) {
                NodeID info = prior_distinct_map.value_at(prior_distinct_map.find(tree(child_r)));
                updates.insert(child_r, Node(info.node, info.tree, Repeat));
                #ifdef STATS
                Kokkos::atomic_add(&(num_comp_s(0)), static_cast<uint32_t>(1));
                #endif
              } else if(node_map.exists(child_r) && node_map.value_at(node_map.find(child_r)).nodetype == Repeat) {
                auto repeat = node_map.value_at(node_map.find(child_r));
                updates.insert(child_r, Node(repeat.node, repeat.tree, Repeat));
                #ifdef STATS
                Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
                #endif
              }
            }
            tree.distinct_children_d(node) = 0;
          } else if(tree.distinct_children_d(node) == 1) { // One child is distinct and the other is identical
            uint32_t child_l = 2*(node)+1;
            uint32_t child_r = 2*(node)+2;
            if((child_l < num_nodes) && (tree.distinct_children_d(child_l) == 2)) {
              updates.insert(child_l, Node(child_l, tree_id, Distinct));
              #ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
              #endif
            } else if((child_r < num_nodes) && (tree.distinct_children_d(child_r) == 2)) {
              updates.insert(child_r, Node(child_r, tree_id, Distinct));
              #ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), static_cast<uint32_t>(1));
              #endif
            }
            tree.distinct_children_d(node) = 0;
          }
        }
      });
#ifdef STATS
if(start_offset >= leaf_start-(num_chunks/2)) {
  Kokkos::fence();
  STDOUT_PRINT("------------------------------\n");
  uint32_t n_distinct = 0;
  Kokkos::parallel_reduce("Count number of distinct", Kokkos::RangePolicy<>(start_offset, end_offset), KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
  //  if(tree.distinct_children_d(start_offset+i) == 2) {
    if(tree.distinct_children_d(i) == 2) {
      update += 1;
    }
  }, n_distinct);
  uint32_t n_same = 0;
  Kokkos::fence();
  Kokkos::parallel_reduce("Count number of same", Kokkos::RangePolicy<>(start_offset, end_offset), KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
    if(tree.distinct_children_d(i) == 0) {
      update += 1;
    }
  }, n_same);
  uint32_t n_shared = 0;
  Kokkos::parallel_reduce("Count number of shared", Kokkos::RangePolicy<>(start_offset, end_offset), KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
    if(tree.distinct_children_d(i) == 8) {
      update += 1;
    }
  }, n_shared);
  uint32_t n_distinct_shared = 0;
  Kokkos::parallel_reduce("Count number of distinct shared", Kokkos::RangePolicy<>(start_offset, end_offset), KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
    if(tree.distinct_children_d(i) == 5) {
      update += 1;
    }
  }, n_distinct_shared);
  uint32_t n_distinct_same = 0;
  Kokkos::parallel_reduce("Count number of distinct_same", Kokkos::RangePolicy<>(start_offset, end_offset), KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
    if(tree.distinct_children_d(i) == 1) {
      update += 1;
    }
  }, n_distinct_same);
  uint32_t n_shared_same = 0;
  Kokkos::parallel_reduce("Count number of shared_same", Kokkos::RangePolicy<>(start_offset, end_offset), KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
    if(tree.distinct_children_d(i) == 4) {
      update += 1;
    }
  }, n_shared_same);
  Kokkos::fence();
  //  STDOUT_PRINT("Num distinct:  %u\n", distinct_map.size()-num_prior_distinct);
  //  STDOUT_PRINT("Num repeats:   %u\n", shared_map.size());
  //  STDOUT_PRINT("Num identical: %u\n", identical_map.size());
  STDOUT_PRINT("Count number of distinct chunks: %u\n", n_distinct);
  STDOUT_PRINT("Count number of same chunks: %u\n", n_same);
  STDOUT_PRINT("Count number of shared chunks: %u\n", n_shared);
  STDOUT_PRINT("Count number of distinct shared chunks: %u\n", n_distinct_shared);
  STDOUT_PRINT("Count number of distinct_same chunks: %u\n", n_distinct_same);
  STDOUT_PRINT("Count number of shared_same chunks: %u\n", n_shared_same);
  STDOUT_PRINT("------------------------------\n");
}
#endif
Kokkos::fence();
    Kokkos::deep_copy(nodes_leftover_h, nodes_leftover);
    current_level -= 1;
    start_offset = (1 << current_level) - 1;
    end_offset = (1 << (current_level+1)) - 1;
  }
  Kokkos::fence();
#ifdef STATS
  Kokkos::deep_copy(num_same_h, num_same);
  Kokkos::deep_copy(num_new_h, num_new);
  Kokkos::deep_copy(num_shift_h, num_shift);
  Kokkos::deep_copy(num_comp_d_h, num_comp_d);
  Kokkos::deep_copy(num_comp_s_h, num_comp_s);
  Kokkos::deep_copy(num_dupl_h, num_dupl);
  Kokkos::deep_copy(num_other_h, num_other);
  Kokkos::fence();
  STDOUT_PRINT("Number of chunks: %u\n", num_chunks);
  STDOUT_PRINT("Number of new chunks: %u\n", num_new_h(0));
  STDOUT_PRINT("Number of same chunks: %u\n", num_same_h(0));
  STDOUT_PRINT("Number of shift chunks: %u\n", num_shift_h(0));
  STDOUT_PRINT("Number of distinct comp nodes: %u\n", num_comp_d_h(0));
  STDOUT_PRINT("Number of shared comp nodes: %u\n", num_comp_s_h(0));
  STDOUT_PRINT("Number of dupl nodes: %u\n", num_dupl_h(0));
  STDOUT_PRINT("Number of other nodes: %u\n", num_other_h(0));
  Kokkos::deep_copy(num_prior_chunks_h, num_prior_chunks_d);
  for(int i=0; i<10; i++) {
    STDOUT_PRINT("Number of chunks repeating from checkpoint %d: %u\n", i, num_prior_chunks_h(i));
  }
#endif
  STDOUT_PRINT("==========End Deduplicate Data==========\n");
}

#endif // DEDUP_MERKLE_TREE_HPP

