#ifndef REFERENCE_IMPL_HPP
#define REFERENCE_IMPL_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_ScatterView.hpp>
#include <vector>
#include <set>
#include "hash_functions.hpp"
#include "map_helpers.hpp"
#include "kokkos_merkle_tree.hpp"
#include "kokkos_hash_list.hpp"
#include "utils.hpp"

template<typename DataView>
int dedup_low_offset_ref(DataView& data_d,
                      const uint32_t chunk_size,
                      MerkleTree& tree, 
                      const uint32_t chkpt_id,
                      DigestNodeIDDeviceMap& first_occur_d,
                      Vector<uint32_t>& shift_dupl_vec,
                      Vector<uint32_t>& first_ocur_vec) {
  Kokkos::View<uint8_t*>::HostMirror data_h = Kokkos::create_mirror_view(data_d);
  Kokkos::deep_copy(data_h, data_d);
  Kokkos::View<HashDigest*,Kokkos::DefaultHostExecutionSpace> prev_tree("Prev tree", tree.tree_h.extent(0));
  Kokkos::deep_copy(prev_tree, tree.tree_d);
  Kokkos::View<HashDigest*,Kokkos::DefaultHostExecutionSpace> curr_tree("Curr tree", tree.tree_h.extent(0));

  DigestNodeIDHostMap first_occur_h(first_occur_d.capacity());
  Kokkos::deep_copy(first_occur_h, first_occur_d);

//  IdxNodeIDHostMap shift_dupl_map_h(shift_dupl_map_d.capacity());
//  IdxNodeIDHostMap first_ocur_map_h(first_ocur_map_d.capacity());

//  const char FIRST_OCUR = 0;
//  const char FIXED_DUPL = 1;
//  const char SHIFT_DUPL = 2;
//  const char DONE = 4;
  uint64_t chunk_counters[4]  = {0,0,0,0};
  uint64_t region_counters[4] = {0,0,0,0};
  uint32_t num_chunks = (tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = tree.tree_h.extent(0);
  printf("Num chunks: %u\n", num_chunks);
  printf("Num nodes: %u\n", num_nodes);

  std::vector<uint8_t> labels(num_nodes, DONE);
  std::set<uint32_t> tree_roots;
  std::unordered_map<std::string, std::vector<uint32_t>> first_ocur_dupl;

  // Process leaves first
  for(uint32_t leaf=num_chunks-1; leaf<num_nodes; leaf++) {
    uint32_t num_bytes = chunk_size;
    if(leaf == num_nodes-1) // Calculate how much data to hash
      num_bytes = data_h.size()-(leaf-(num_chunks-1))*chunk_size;
    // Hash chunk
    MD5(data_h.data()+(leaf-(num_chunks-1))*chunk_size, num_bytes, curr_tree(leaf).digest);
    // Insert into table
    auto result = first_occur_h.insert(curr_tree(leaf), NodeID(leaf, chkpt_id)); 
    if(digests_same(prev_tree(leaf), curr_tree(leaf))) { // Fixed duplicate chunk
      labels[leaf] = FIXED_DUPL;
    } else if(result.success()) { // First occurrence chunk
      labels[leaf] = FIRST_OCUR;
    } else if(result.existing()) { // Shifted duplicate chunk
      auto& info = first_occur_h.value_at(result.index());
      if(info.tree == chkpt_id) { // Ensure node with lowest offset is the first occurrence
        if(info.node > leaf) {
          labels[leaf] = FIRST_OCUR;
          labels[info.node] = SHIFT_DUPL;
          info.node = leaf;
        } else {
          labels[leaf] = SHIFT_DUPL;
        }
      } else {
        labels[leaf] = SHIFT_DUPL;
      }
    }
    chunk_counters[labels[leaf]] += 1;
  }

  // Build up forest of Merkle Trees
  for(uint32_t node=num_chunks-2; node < num_chunks-1; node--) {
    uint32_t child_l = 2*node+1;
    uint32_t child_r = 2*node+2;
    if(labels[child_l] == FIRST_OCUR && labels[child_r] == FIRST_OCUR) {
      labels[node] = FIRST_OCUR;
      MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
      first_occur_h.insert(curr_tree(node), NodeID(node, chkpt_id));
    }
  }
  for(uint32_t node=num_chunks-2; node < num_chunks-1; node--) {
    uint32_t child_l = 2*node+1;
    uint32_t child_r = 2*node+2;
    if(labels[child_l] != labels[child_r]) { // Children have different labels
      labels[node] = DONE;
      if((labels[child_l] != FIXED_DUPL) && (labels[child_l] != DONE))
        tree_roots.insert(child_l);
      if((labels[child_r] != FIXED_DUPL) && (labels[child_r] != DONE))
        tree_roots.insert(child_r);
    } else if(labels[child_l] == FIXED_DUPL) { // Children are both fixed duplicates
      labels[node] = FIXED_DUPL;
    } else if(labels[child_l] == SHIFT_DUPL) { // Children are both shifted duplicates
      MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
      if(first_occur_h.exists(curr_tree(node))) { // This node is also a shifted duplicate
        labels[node] = SHIFT_DUPL;
      } else { // Node is not a shifted duplicate. Save child trees
        labels[node] = DONE; // Add children to tree root maps
        tree_roots.insert(child_l);
        tree_roots.insert(child_r);
      }
    }
  }
  if(labels[0] == FIRST_OCUR) 
    tree_roots.insert(0);

  // Count regions
  for(auto set_iter=tree_roots.begin(); set_iter!=tree_roots.end(); set_iter++) {
    uint32_t root = *set_iter;
    if(labels[root] != DONE) {
      region_counters[labels[root]] += 1;
//      NodeID node = first_occur_h.value_at(first_occur_h.find(curr_tree(root)));
      if(labels[root] == FIRST_OCUR) {
        first_ocur_vec.host_push(root);
//        first_ocur_map_h.insert(root, node);
//printf("%u FIRST_OCUR\n", root);
      } else if(labels[root] == SHIFT_DUPL) {
        shift_dupl_vec.host_push(root);
//        shift_dupl_map_h.insert(root, node);
//printf("%u SHIFT_DUPL\n", root);
      }
    }
  }

//  Kokkos::deep_copy(shift_dupl_map_d, shift_dupl_map_h);
//  Kokkos::deep_copy(first_ocur_map_d, first_ocur_map_h);
  Kokkos::deep_copy(tree.tree_d, curr_tree);
  Kokkos::deep_copy(first_occur_d, first_occur_h);

  printf("Checkpoint %u\n", chkpt_id);
  printf("Number of first occurrence chunks:  %lu\n", chunk_counters[FIRST_OCUR]);
  printf("Number of fixed duplicate chunks:   %lu\n", chunk_counters[FIXED_DUPL]);
  printf("Number of shifted duplicate chunks: %lu\n", chunk_counters[SHIFT_DUPL]);
  printf("Number of first occurrence regions:  %lu\n", region_counters[FIRST_OCUR]);
  printf("Number of fixed duplicate regions:   %lu\n", region_counters[FIXED_DUPL]);
  printf("Number of shifted duplicate regions: %lu\n", region_counters[SHIFT_DUPL]);
  return region_counters[FIRST_OCUR]+region_counters[FIXED_DUPL]+region_counters[SHIFT_DUPL];
}

template<typename DataView>
int dedup_low_offset_ref(DataView& data_d,
                      const uint32_t chunk_size,
                      MerkleTree& tree, 
                      const uint32_t chkpt_id,
                      DigestNodeIDDeviceMap& first_occur_d,
                      IdxNodeIDDeviceMap& shift_dupl_map_d,
                      IdxNodeIDDeviceMap& first_ocur_map_d) {
  Kokkos::View<uint8_t*>::HostMirror data_h = Kokkos::create_mirror_view(data_d);
  Kokkos::deep_copy(data_h, data_d);
  Kokkos::View<HashDigest*,Kokkos::DefaultHostExecutionSpace> prev_tree("Prev tree", tree.tree_h.extent(0));
  Kokkos::deep_copy(prev_tree, tree.tree_d);
  Kokkos::View<HashDigest*,Kokkos::DefaultHostExecutionSpace> curr_tree("Curr tree", tree.tree_h.extent(0));

  DigestNodeIDHostMap first_occur_h(first_occur_d.capacity());
  Kokkos::deep_copy(first_occur_h, first_occur_d);

  IdxNodeIDHostMap shift_dupl_map_h(shift_dupl_map_d.capacity());
  IdxNodeIDHostMap first_ocur_map_h(first_ocur_map_d.capacity());

//  const char FIRST_OCUR = 0;
//  const char FIXED_DUPL = 1;
//  const char SHIFT_DUPL = 2;
//  const char DONE = 4;
  uint64_t chunk_counters[4]  = {0,0,0,0};
  uint64_t region_counters[4] = {0,0,0,0};
  uint32_t num_chunks = (tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = tree.tree_h.extent(0);
  printf("Num chunks: %u\n", num_chunks);
  printf("Num nodes: %u\n", num_nodes);

  std::vector<uint8_t> labels(num_nodes, DONE);
  std::set<uint32_t> tree_roots;
  std::unordered_map<std::string, std::vector<uint32_t>> first_ocur_dupl;

  // Process leaves first
  for(uint32_t leaf=num_chunks-1; leaf<num_nodes; leaf++) {
    uint32_t num_bytes = chunk_size;
    if(leaf == num_nodes-1) // Calculate how much data to hash
      num_bytes = data_h.size()-(leaf-(num_chunks-1))*chunk_size;
    // Hash chunk
    MD5(data_h.data()+(leaf-(num_chunks-1))*chunk_size, num_bytes, curr_tree(leaf).digest);
    // Insert into table
    auto result = first_occur_h.insert(curr_tree(leaf), NodeID(leaf, chkpt_id)); 
    if(digests_same(prev_tree(leaf), curr_tree(leaf))) { // Fixed duplicate chunk
      labels[leaf] = FIXED_DUPL;
    } else if(result.success()) { // First occurrence chunk
      labels[leaf] = FIRST_OCUR;
    } else if(result.existing()) { // Shifted duplicate chunk
      auto& info = first_occur_h.value_at(result.index());
      if(info.tree == chkpt_id) { // Ensure node with lowest offset is the first occurrence
        if(info.node > leaf) {
          labels[leaf] = FIRST_OCUR;
          labels[info.node] = SHIFT_DUPL;
          info.node = leaf;
        } else {
          labels[leaf] = SHIFT_DUPL;
        }
      } else {
        labels[leaf] = SHIFT_DUPL;
      }
    }
    chunk_counters[labels[leaf]] += 1;
  }

  // Build up forest of Merkle Trees
  for(uint32_t node=num_chunks-2; node < num_chunks-1; node--) {
    uint32_t child_l = 2*node+1;
    uint32_t child_r = 2*node+2;
    if(labels[child_l] == FIRST_OCUR && labels[child_r] == FIRST_OCUR) {
      labels[node] = FIRST_OCUR;
      MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
      first_occur_h.insert(curr_tree(node), NodeID(node, chkpt_id));
    }
  }
  for(uint32_t node=num_chunks-2; node < num_chunks-1; node--) {
    uint32_t child_l = 2*node+1;
    uint32_t child_r = 2*node+2;
    if(labels[child_l] != labels[child_r]) { // Children have different labels
      labels[node] = DONE;
      if((labels[child_l] != FIXED_DUPL) && (labels[child_l] != DONE))
        tree_roots.insert(child_l);
      if((labels[child_r] != FIXED_DUPL) && (labels[child_r] != DONE))
        tree_roots.insert(child_r);
    } else if(labels[child_l] == FIXED_DUPL) { // Children are both fixed duplicates
      labels[node] = FIXED_DUPL;
    } else if(labels[child_l] == SHIFT_DUPL) { // Children are both shifted duplicates
      MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
      if(first_occur_h.exists(curr_tree(node))) { // This node is also a shifted duplicate
        labels[node] = SHIFT_DUPL;
      } else { // Node is not a shifted duplicate. Save child trees
        labels[node] = DONE; // Add children to tree root maps
        tree_roots.insert(child_l);
        tree_roots.insert(child_r);
      }
    }
  }
  if(labels[0] == FIRST_OCUR) 
    tree_roots.insert(0);

  // Count regions
  for(auto set_iter=tree_roots.begin(); set_iter!=tree_roots.end(); set_iter++) {
    uint32_t root = *set_iter;
    if(labels[root] != DONE) {
      region_counters[labels[root]] += 1;
      NodeID node = first_occur_h.value_at(first_occur_h.find(curr_tree(root)));
      if(labels[root] == FIRST_OCUR) {
        first_ocur_map_h.insert(root, node);
//printf("%u FIRST_OCUR\n", root);
      } else if(labels[root] == SHIFT_DUPL) {
        shift_dupl_map_h.insert(root, node);
//printf("%u SHIFT_DUPL\n", root);
      }
    }
  }

  Kokkos::deep_copy(shift_dupl_map_d, shift_dupl_map_h);
  Kokkos::deep_copy(first_ocur_map_d, first_ocur_map_h);
  Kokkos::deep_copy(tree.tree_d, curr_tree);
  Kokkos::deep_copy(first_occur_d, first_occur_h);

  printf("Checkpoint %u\n", chkpt_id);
  printf("Number of first occurrence chunks:  %lu\n", chunk_counters[FIRST_OCUR]);
  printf("Number of fixed duplicate chunks:   %lu\n", chunk_counters[FIXED_DUPL]);
  printf("Number of shifted duplicate chunks: %lu\n", chunk_counters[SHIFT_DUPL]);
  printf("Number of first occurrence regions:  %lu\n", region_counters[FIRST_OCUR]);
  printf("Number of fixed duplicate regions:   %lu\n", region_counters[FIXED_DUPL]);
  printf("Number of shifted duplicate regions: %lu\n", region_counters[SHIFT_DUPL]);
  return region_counters[FIRST_OCUR]+region_counters[FIXED_DUPL]+region_counters[SHIFT_DUPL];
}

template <typename DataView>
int dedup_low_root_ref(DataView& data_d,
                      const uint32_t chunk_size,
                      const MerkleTree& tree, 
                      const uint32_t chkpt_id,
                      DigestNodeIDDeviceMap& first_occur_d,
                      Vector<uint32_t>& shift_dupl_vec,
                      Vector<uint32_t>& first_ocur_vec) {
  Kokkos::View<uint8_t*>::HostMirror data_h = Kokkos::create_mirror_view(data_d);
  Kokkos::deep_copy(data_h, data_d);
  Kokkos::View<HashDigest*,Kokkos::DefaultHostExecutionSpace> prev_tree("Prev tree", tree.tree_h.extent(0));
  Kokkos::deep_copy(prev_tree, tree.tree_d);
  Kokkos::View<HashDigest*,Kokkos::DefaultHostExecutionSpace> curr_tree("Curr tree", tree.tree_h.extent(0));

  DigestNodeIDHostMap first_occur_h(first_occur_d.capacity());
  Kokkos::deep_copy(first_occur_h, first_occur_d);

//  IdxNodeIDHostMap shift_dupl_map_h(shift_dupl_map_d.capacity());
//  IdxNodeIDHostMap first_ocur_map_h(first_ocur_map_d.capacity());

//  const char FIRST_OCUR = 0;
//  const char FIXED_DUPL = 1;
//  const char SHIFT_DUPL = 2;
//  const char FIRST_DUPL = 3;
//  const char DONE = 4;
  uint64_t chunk_counters[4]  = {0,0,0,0};
  uint64_t region_counters[4] = {0,0,0,0};
  uint32_t num_chunks = (tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = tree.tree_h.extent(0);
  printf("Num chunks: %u\n", num_chunks);
  printf("Num nodes: %u\n", num_nodes);

  std::vector<uint8_t> labels(num_nodes, DONE);
  std::set<uint32_t> tree_roots;
  std::unordered_map<std::string, std::vector<uint32_t>> first_ocur_dupl;

  // Process leaves first
  for(uint32_t leaf=num_chunks-1; leaf<num_nodes; leaf++) {
    uint32_t num_bytes = chunk_size;
    if(leaf == num_nodes-1) // Calculate how much data to hash
      num_bytes = data_h.size()-(leaf-(num_chunks-1))*chunk_size;
    // Hash chunk
    MD5(data_h.data()+(leaf-(num_chunks-1))*chunk_size, num_bytes, curr_tree(leaf).digest);
    // Insert into table
    if(digests_same(prev_tree(leaf), curr_tree(leaf))) { // Fixed duplicate chunk
      labels[leaf] = FIXED_DUPL;
    } else if(first_occur_h.exists(curr_tree(leaf))) {
      labels[leaf] = SHIFT_DUPL;
    } else {
      labels[leaf] = FIRST_DUPL;
      if(first_ocur_dupl.find(digest_to_str(curr_tree(leaf))) == first_ocur_dupl.end()) {
        std::vector<uint32_t> entry;
        entry.push_back(leaf);
        first_ocur_dupl.insert(std::make_pair(digest_to_str(curr_tree(leaf)), entry));
      } else {
        auto& entry = first_ocur_dupl[digest_to_str(curr_tree(leaf))];
        entry.push_back(leaf);
      }
    }
    chunk_counters[labels[leaf]] += 1;
  }

  // Build up forest of first occurrence trees
  for(uint32_t node=num_chunks-2; node<num_nodes; node--) {
    uint32_t child_l = 2*node+1;
    uint32_t child_r = 2*node+2;
    if((labels[child_l] == FIRST_DUPL) && (labels[child_r] == FIRST_DUPL)) {
      labels[node] = FIRST_DUPL;
      MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
      if(first_ocur_dupl.find(digest_to_str(curr_tree(node))) == first_ocur_dupl.end()) {
        // Add new entry to table
        std::vector<uint32_t> entry;
        entry.push_back(node);
        first_ocur_dupl.insert(std::make_pair(digest_to_str(curr_tree(node)), entry));
      } else {
        // Update existing entry
        auto& entry = first_ocur_dupl[digest_to_str(curr_tree(node))];
        entry.push_back(node);
      }
    }
  }

  // Remove roots that have duplicate leaves
  for(auto it=first_ocur_dupl.begin(); it!=first_ocur_dupl.end(); it++) {
    auto& dup_list = it->second;
    uint32_t root = num_nodes;
    bool found_dup = true;
    while(found_dup) {
      found_dup = false;
      root = num_nodes;
      for(uint32_t idx=0; idx<dup_list.size(); idx++) {
        uint32_t u = dup_list[idx];
        uint32_t possible_root = u;
        if(possible_root > 0) {
          while(possible_root > 0 && first_ocur_dupl.find(digest_to_str(curr_tree((possible_root-1)/2))) != first_ocur_dupl.end()) {
            possible_root = (possible_root-1)/2;
          }
        }
        if(possible_root < root) {
          root = possible_root;
        } else if(possible_root == root) {
          first_ocur_dupl.erase(digest_to_str(curr_tree(root)));
          found_dup = true;
          break;
        }
      }
    }
  }

  // Select which leaves will be first occurrences
  for(uint32_t node=num_chunks-1; node<num_nodes; node++) {
    if(labels[node] == FIRST_DUPL) {
      uint32_t select = num_nodes;
      uint32_t root = num_nodes;
      auto dup_list = first_ocur_dupl[digest_to_str(curr_tree(node))];
      for(uint32_t idx=0; idx<dup_list.size(); idx++) {
        uint32_t u = dup_list[idx];
        uint32_t possible_root = u;
        while(possible_root > 0 && first_ocur_dupl.find(digest_to_str(curr_tree((possible_root-1)/2))) != first_ocur_dupl.end()) {
          possible_root = (possible_root-1)/2;
        }
        if(possible_root < root) {
          root = possible_root;
          select = u;
        }
      }
      for(uint32_t idx=0; idx<dup_list.size(); idx++) {
        labels[dup_list[idx]] = SHIFT_DUPL;
        chunk_counters[labels[dup_list[idx]]] += 1;
      }
      labels[select] = FIRST_OCUR;
      chunk_counters[FIRST_OCUR] += 1;
      first_occur_h.insert(curr_tree(select), NodeID(select, chkpt_id));
    }
  }

  // Build up forest of Merkle Trees
  for(uint32_t node=num_chunks-2; node < num_chunks-1; node--) {
    uint32_t child_l = 2*node+1;
    uint32_t child_r = 2*node+2;
    if(labels[child_l] == FIRST_OCUR && labels[child_r] == FIRST_OCUR) {
      labels[node] = FIRST_OCUR;
      MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
      auto res = first_occur_h.insert(curr_tree(node), NodeID(node, chkpt_id));
      if(res.existing()) {
        auto& entry = first_occur_h.value_at(res.index());
        entry.node = node;
      }
    }
  }

  for(uint32_t node=num_chunks-2; node < num_chunks-1; node--) {
    uint32_t child_l = 2*node+1;
    uint32_t child_r = 2*node+2;
    if(labels[child_l] != labels[child_r]) { // Children have different labels
      labels[node] = DONE;
      if((labels[child_l] != FIXED_DUPL) && (labels[child_l] != DONE)) {
        tree_roots.insert(child_l);
      }
      if((labels[child_r] != FIXED_DUPL) && (labels[child_r] != DONE)) {
        tree_roots.insert(child_r);
      }
    } else if(labels[child_l] == FIXED_DUPL) { // Children are both fixed duplicates
      labels[node] = FIXED_DUPL;
    } else if(labels[child_l] == SHIFT_DUPL) { // Children are both shifted duplicates
      MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
      if(first_occur_h.exists(curr_tree(node))) { // This node is also a shifted duplicate
        labels[node] = SHIFT_DUPL;
      } else { // Node is not a shifted duplicate. Save child trees
        labels[node] = DONE; // Add children to tree root maps
        tree_roots.insert(child_l);
        tree_roots.insert(child_r);
      }
    }
  }
  if(labels[0] == FIRST_OCUR) 
    tree_roots.insert(0);

  // Count regions
  for(auto set_iter=tree_roots.begin(); set_iter!=tree_roots.end(); set_iter++) {
    uint32_t root = *set_iter;
    if(labels[root] != DONE) {
      region_counters[labels[root]] += 1;
//      NodeID node = first_occur_h.value_at(first_occur_h.find(curr_tree(root)));
      if(labels[root] == FIRST_OCUR) {
        first_ocur_vec.host_push(root);
//        first_ocur_map_h.insert(root, node);
      } else if(labels[root] == SHIFT_DUPL) {
        shift_dupl_vec.host_push(root);
//        shift_dupl_map_h.insert(root, node);
      }
    }
  }

//  Kokkos::deep_copy(shift_dupl_map_d, shift_dupl_map_h);
//  Kokkos::deep_copy(first_ocur_map_d, first_ocur_map_h);
  Kokkos::deep_copy(tree.tree_d, curr_tree);
  Kokkos::deep_copy(first_occur_d, first_occur_h);

  printf("Checkpoint %u\n", chkpt_id);
  printf("Number of first occurrence chunks:  %lu\n", chunk_counters[FIRST_OCUR]);
  printf("Number of fixed duplicate chunks:   %lu\n", chunk_counters[FIXED_DUPL]);
  printf("Number of shifted duplicate chunks: %lu\n", chunk_counters[SHIFT_DUPL]);
  printf("Number of first occurrence regions:  %lu\n", region_counters[FIRST_OCUR]);
  printf("Number of fixed duplicate regions:   %lu\n", region_counters[FIXED_DUPL]);
  printf("Number of shifted duplicate regions: %lu\n", region_counters[SHIFT_DUPL]);
  return region_counters[FIRST_OCUR]+region_counters[FIXED_DUPL]+region_counters[SHIFT_DUPL];
}

template <typename DataView>
int dedup_low_root_ref(DataView& data_d,
                      const uint32_t chunk_size,
                      const MerkleTree& tree, 
                      const uint32_t chkpt_id,
                      DigestNodeIDDeviceMap& first_occur_d,
                      IdxNodeIDDeviceMap& shift_dupl_map_d,
                      IdxNodeIDDeviceMap& first_ocur_map_d) {
  Kokkos::View<uint8_t*>::HostMirror data_h = Kokkos::create_mirror_view(data_d);
  Kokkos::deep_copy(data_h, data_d);
  Kokkos::View<HashDigest*,Kokkos::DefaultHostExecutionSpace> prev_tree("Prev tree", tree.tree_h.extent(0));
  Kokkos::deep_copy(prev_tree, tree.tree_d);
  Kokkos::View<HashDigest*,Kokkos::DefaultHostExecutionSpace> curr_tree("Curr tree", tree.tree_h.extent(0));

  DigestNodeIDHostMap first_occur_h(first_occur_d.capacity());
  Kokkos::deep_copy(first_occur_h, first_occur_d);

  IdxNodeIDHostMap shift_dupl_map_h(shift_dupl_map_d.capacity());
  IdxNodeIDHostMap first_ocur_map_h(first_ocur_map_d.capacity());

//  const char FIRST_OCUR = 0;
//  const char FIXED_DUPL = 1;
//  const char SHIFT_DUPL = 2;
//  const char FIRST_DUPL = 3;
//  const char DONE = 4;
  uint64_t chunk_counters[4]  = {0,0,0,0};
  uint64_t region_counters[4] = {0,0,0,0};
  uint32_t num_chunks = (tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = tree.tree_h.extent(0);
  printf("Num chunks: %u\n", num_chunks);
  printf("Num nodes: %u\n", num_nodes);

  std::vector<uint8_t> labels(num_nodes, DONE);
  std::set<uint32_t> tree_roots;
  std::unordered_map<std::string, std::vector<uint32_t>> first_ocur_dupl;

  // Process leaves first
  for(uint32_t leaf=num_chunks-1; leaf<num_nodes; leaf++) {
    uint32_t num_bytes = chunk_size;
    if(leaf == num_nodes-1) // Calculate how much data to hash
      num_bytes = data_h.size()-(leaf-(num_chunks-1))*chunk_size;
    // Hash chunk
    MD5(data_h.data()+(leaf-(num_chunks-1))*chunk_size, num_bytes, curr_tree(leaf).digest);
    // Insert into table
    if(digests_same(prev_tree(leaf), curr_tree(leaf))) { // Fixed duplicate chunk
      labels[leaf] = FIXED_DUPL;
    } else if(first_occur_h.exists(curr_tree(leaf))) {
      labels[leaf] = SHIFT_DUPL;
    } else {
      labels[leaf] = FIRST_DUPL;
      if(first_ocur_dupl.find(digest_to_str(curr_tree(leaf))) == first_ocur_dupl.end()) {
        std::vector<uint32_t> entry;
        entry.push_back(leaf);
        first_ocur_dupl.insert(std::make_pair(digest_to_str(curr_tree(leaf)), entry));
      } else {
        auto& entry = first_ocur_dupl[digest_to_str(curr_tree(leaf))];
        entry.push_back(leaf);
      }
    }
    chunk_counters[labels[leaf]] += 1;
  }

  // Build up forest of first occurrence trees
  for(uint32_t node=num_chunks-2; node<num_nodes; node--) {
    uint32_t child_l = 2*node+1;
    uint32_t child_r = 2*node+2;
    if((labels[child_l] == FIRST_DUPL) && (labels[child_r] == FIRST_DUPL)) {
      labels[node] = FIRST_DUPL;
      MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
      if(first_ocur_dupl.find(digest_to_str(curr_tree(node))) == first_ocur_dupl.end()) {
        // Add new entry to table
        std::vector<uint32_t> entry;
        entry.push_back(node);
        first_ocur_dupl.insert(std::make_pair(digest_to_str(curr_tree(node)), entry));
      } else {
        // Update existing entry
        auto& entry = first_ocur_dupl[digest_to_str(curr_tree(node))];
        entry.push_back(node);
      }
    }
  }

  // Remove roots that have duplicate leaves
  for(auto it=first_ocur_dupl.begin(); it!=first_ocur_dupl.end(); it++) {
    auto& dup_list = it->second;
    uint32_t root = num_nodes;
    bool found_dup = true;
    while(found_dup) {
      found_dup = false;
      root = num_nodes;
      for(uint32_t idx=0; idx<dup_list.size(); idx++) {
        uint32_t u = dup_list[idx];
        uint32_t possible_root = u;
        if(possible_root > 0) {
          while(possible_root > 0 && first_ocur_dupl.find(digest_to_str(curr_tree((possible_root-1)/2))) != first_ocur_dupl.end()) {
            possible_root = (possible_root-1)/2;
          }
        }
        if(possible_root < root) {
          root = possible_root;
        } else if(possible_root == root) {
          first_ocur_dupl.erase(digest_to_str(curr_tree(root)));
          found_dup = true;
          break;
        }
      }
    }
  }

  // Select which leaves will be first occurrences
  for(uint32_t node=num_chunks-1; node<num_nodes; node++) {
    if(labels[node] == FIRST_DUPL) {
      uint32_t select = num_nodes;
      uint32_t root = num_nodes;
      auto dup_list = first_ocur_dupl[digest_to_str(curr_tree(node))];
      for(uint32_t idx=0; idx<dup_list.size(); idx++) {
        uint32_t u = dup_list[idx];
        uint32_t possible_root = u;
        while(possible_root > 0 && first_ocur_dupl.find(digest_to_str(curr_tree((possible_root-1)/2))) != first_ocur_dupl.end()) {
          possible_root = (possible_root-1)/2;
        }
        if(possible_root < root) {
          root = possible_root;
          select = u;
        }
      }
      for(uint32_t idx=0; idx<dup_list.size(); idx++) {
        labels[dup_list[idx]] = SHIFT_DUPL;
        chunk_counters[labels[dup_list[idx]]] += 1;
      }
      labels[select] = FIRST_OCUR;
      chunk_counters[FIRST_OCUR] += 1;
      first_occur_h.insert(curr_tree(select), NodeID(select, chkpt_id));
    }
  }

  // Build up forest of Merkle Trees
  for(uint32_t node=num_chunks-2; node < num_chunks-1; node--) {
    uint32_t child_l = 2*node+1;
    uint32_t child_r = 2*node+2;
    if(labels[child_l] == FIRST_OCUR && labels[child_r] == FIRST_OCUR) {
      labels[node] = FIRST_OCUR;
      MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
      auto res = first_occur_h.insert(curr_tree(node), NodeID(node, chkpt_id));
      if(res.existing()) {
        auto& entry = first_occur_h.value_at(res.index());
        entry.node = node;
      }
    }
  }

  for(uint32_t node=num_chunks-2; node < num_chunks-1; node--) {
    uint32_t child_l = 2*node+1;
    uint32_t child_r = 2*node+2;
    if(labels[child_l] != labels[child_r]) { // Children have different labels
      labels[node] = DONE;
      if((labels[child_l] != FIXED_DUPL) && (labels[child_l] != DONE)) {
        tree_roots.insert(child_l);
      }
      if((labels[child_r] != FIXED_DUPL) && (labels[child_r] != DONE)) {
        tree_roots.insert(child_r);
      }
    } else if(labels[child_l] == FIXED_DUPL) { // Children are both fixed duplicates
      labels[node] = FIXED_DUPL;
    } else if(labels[child_l] == SHIFT_DUPL) { // Children are both shifted duplicates
      MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
      if(first_occur_h.exists(curr_tree(node))) { // This node is also a shifted duplicate
        labels[node] = SHIFT_DUPL;
      } else { // Node is not a shifted duplicate. Save child trees
        labels[node] = DONE; // Add children to tree root maps
        tree_roots.insert(child_l);
        tree_roots.insert(child_r);
      }
    }
  }
  if(labels[0] == FIRST_OCUR) 
    tree_roots.insert(0);

  // Count regions
  for(auto set_iter=tree_roots.begin(); set_iter!=tree_roots.end(); set_iter++) {
    uint32_t root = *set_iter;
    if(labels[root] != DONE) {
      region_counters[labels[root]] += 1;
      NodeID node = first_occur_h.value_at(first_occur_h.find(curr_tree(root)));
      if(labels[root] == FIRST_OCUR) {
        first_ocur_map_h.insert(root, node);
      } else if(labels[root] == SHIFT_DUPL) {
        shift_dupl_map_h.insert(root, node);
      }
    }
  }

  Kokkos::deep_copy(shift_dupl_map_d, shift_dupl_map_h);
  Kokkos::deep_copy(first_ocur_map_d, first_ocur_map_h);
  Kokkos::deep_copy(tree.tree_d, curr_tree);
  Kokkos::deep_copy(first_occur_d, first_occur_h);

  printf("Checkpoint %u\n", chkpt_id);
  printf("Number of first occurrence chunks:  %lu\n", chunk_counters[FIRST_OCUR]);
  printf("Number of fixed duplicate chunks:   %lu\n", chunk_counters[FIXED_DUPL]);
  printf("Number of shifted duplicate chunks: %lu\n", chunk_counters[SHIFT_DUPL]);
  printf("Number of first occurrence regions:  %lu\n", region_counters[FIRST_OCUR]);
  printf("Number of fixed duplicate regions:   %lu\n", region_counters[FIXED_DUPL]);
  printf("Number of shifted duplicate regions: %lu\n", region_counters[SHIFT_DUPL]);
  return region_counters[FIRST_OCUR]+region_counters[FIXED_DUPL]+region_counters[SHIFT_DUPL];
}


#endif // REFERENCE_IMPL
