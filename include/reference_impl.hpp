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

//struct ReferenceImpl {
//  const char FIRST_OCUR = 0;
//  const char FIXED_DUPL = 1;
//  const char SHIFT_DUPL = 2;
//  const char DONE = 3;

int num_subtree_roots(Kokkos::View<uint8_t*>& data_d,
                      const uint32_t chunk_size,
                      const MerkleTree& tree, 
                      const uint32_t chkpt_id,
                      DistinctNodeIDMap& first_occur_d,
                      CompactTable& shift_dupl_map_d,
                      CompactTable& first_ocur_map_d) {
  Kokkos::View<uint8_t*>::HostMirror data_h = Kokkos::create_mirror_view(data_d);
  Kokkos::deep_copy(data_h, data_d);
  Kokkos::View<HashDigest*,Kokkos::DefaultHostExecutionSpace> prev_tree("Prev tree", tree.tree_h.extent(0));
  Kokkos::deep_copy(prev_tree, tree.tree_h);
  Kokkos::View<HashDigest*,Kokkos::DefaultHostExecutionSpace> curr_tree("Curr tree", tree.tree_h.extent(0));

  DistinctHostNodeIDMap first_occur_h(first_occur_d.capacity());
  Kokkos::deep_copy(first_occur_h, first_occur_d);

  CompactHostTable shift_dupl_map_h(shift_dupl_map_d.capacity());
  CompactHostTable first_ocur_map_h(first_ocur_map_d.capacity());

  const char FIRST_OCUR = 0;
  const char FIXED_DUPL = 1;
  const char SHIFT_DUPL = 2;
  const char DONE = 4;
  uint64_t chunk_counters[4]  = {0,0,0,0};
  uint64_t region_counters[4] = {0,0,0,0};
  uint32_t num_chunks = (tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = tree.tree_h.extent(0);
  printf("Num chunks: %u\n", num_chunks);
  printf("Num nodes: %u\n", num_nodes);

  std::vector<char> labels(num_nodes, DONE);
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

int num_subtree_roots_experiment(Kokkos::View<uint8_t*>& data_d,
                      const uint32_t chunk_size,
                      const MerkleTree& tree, 
                      const uint32_t chkpt_id,
                      DistinctNodeIDMap& first_occur_d,
                      CompactTable& shift_dupl_map_d,
                      CompactTable& first_ocur_map_d) {
  Kokkos::View<uint8_t*>::HostMirror data_h = Kokkos::create_mirror_view(data_d);
  Kokkos::deep_copy(data_h, data_d);
  Kokkos::View<HashDigest*,Kokkos::DefaultHostExecutionSpace> prev_tree("Prev tree", tree.tree_h.extent(0));
  Kokkos::deep_copy(prev_tree, tree.tree_h);
  Kokkos::View<HashDigest*,Kokkos::DefaultHostExecutionSpace> curr_tree("Curr tree", tree.tree_h.extent(0));

  DistinctHostNodeIDMap first_occur_h(first_occur_d.capacity());
  Kokkos::deep_copy(first_occur_h, first_occur_d);

  CompactHostTable shift_dupl_map_h(shift_dupl_map_d.capacity());
  CompactHostTable first_ocur_map_h(first_ocur_map_d.capacity());
//    Kokkos::deep_copy(shift_dupl_map_h, shift_dupl_map_d);
//    Kokkos::deep_copy(first_ocur_map_h, first_ocur_map_d);

  const char FIRST_OCUR = 0;
  const char FIXED_DUPL = 1;
  const char SHIFT_DUPL = 2;
  const char FIRST_DUPL = 3;
  const char DONE = 4;
  uint64_t chunk_counters[4]  = {0,0,0,0};
  uint64_t region_counters[4] = {0,0,0,0};
  uint32_t num_chunks = (tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = tree.tree_h.extent(0);
  printf("Num chunks: %u\n", num_chunks);
  printf("Num nodes: %u\n", num_nodes);

  std::vector<char> labels(num_nodes, DONE);
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
//      labels[leaf] = FIRST_OCUR;
      labels[leaf] = FIRST_DUPL;
std::vector<uint32_t> dupl_list;
dupl_list.push_back(leaf);
first_ocur_dupl.insert({digest_to_str(curr_tree(leaf)),dupl_list});
chunk_counters[FIRST_OCUR] += 1;
    } else if(result.existing() && (first_occur_h.value_at(result.index()).tree == chkpt_id)) { // Shifted duplicate chunk
//        labels[leaf] = FIRST_OCUR;
      labels[leaf] = FIRST_DUPL;
first_ocur_dupl[digest_to_str(curr_tree(leaf))].push_back(leaf);
    } else if(result.existing()) { // Shifted duplicate chunk
      labels[leaf] = SHIFT_DUPL;
    }
    chunk_counters[labels[leaf]] += 1;
  }

  // Build up forest of Merkle Trees
  for(uint32_t node=num_chunks-2; node < num_chunks-1; node--) {
    uint32_t child_l = 2*node+1;
    uint32_t child_r = 2*node+2;
    if(labels[child_l] != labels[child_r]) { // Children have different labels
      labels[node] = DONE;
      if((labels[child_l] != FIXED_DUPL) && (labels[child_l] != DONE))
        tree_roots.insert(child_l);
      if((labels[child_r] != FIXED_DUPL) && (labels[child_r] != DONE))
        tree_roots.insert(child_r);
    } else if(labels[child_l] == FIRST_DUPL) { // Children are both first occurrences
uint32_t leftmost = leftmost_leaf(node, num_nodes);
uint32_t rightmost = rightmost_leaf(node, num_nodes);
//printf("Node %u: [%u,%u]\n", node, leftmost, rightmost);
std::set<std::string> leaf_set;
bool node_valid = true;
// Check if leaves are unique within the subtree
for(uint32_t leaf=leftmost; leaf<=rightmost; leaf++) {
  auto result = leaf_set.insert(digest_to_str(curr_tree(leaf)));
  if(!result.second) {
    node_valid = false;
    break;
  }
}
//printf("Node %u is valid? %d\n", node, node_valid);
if(node_valid) {
//      labels[node] = FIRST_DUPL;
//    } else if(labels[child_l] == FIRST_OCUR) { // Children are both first occurrences
      labels[node] = FIRST_DUPL;
      MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
      if(first_ocur_dupl.find(digest_to_str(curr_tree(node))) == first_ocur_dupl.end()) {
        std::vector<uint32_t> dupl_list;
        dupl_list.push_back(node);
        first_ocur_dupl.insert({digest_to_str(curr_tree(node)),dupl_list});
      } else {
        first_ocur_dupl[digest_to_str(curr_tree(node))].push_back(node);
      }
      first_occur_h.insert(curr_tree(node), NodeID(node, chkpt_id));
} else {
  labels[node] = DONE;
  tree_roots.insert(child_l);
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
  if(labels[0] == FIRST_DUPL) 
    tree_roots.insert(0);

//for(int i=0; i<tree_roots.size(); i++) {
//  printf("Pre tree roots: %u\n", tree_roots[i]);
//}

//for(auto iter = first_ocur_dupl.begin(); iter != first_ocur_dupl.end(); ++iter) {
//  auto digest = iter->first;
//  auto list = iter->second;
//  printf("%s: (%u) ", digest.c_str(), list.size());
//  for(int i=0; i<list.size(); i++) {
//    printf("%u ", list[i]);
//  }
//  printf("\n");
//}
std::set<std::string> chunks_seen;
for(auto set_iter=tree_roots.begin(); set_iter!=tree_roots.end(); set_iter++) {
  uint32_t node = *set_iter;
  if(labels[node] == FIRST_DUPL) {
    uint32_t leaf_beg = leftmost_leaf(node, num_nodes);
    uint32_t leaf_end = rightmost_leaf(node, num_nodes);
    uint32_t num_match = 0;
    uint32_t leaf_start = first_occur_h.value_at(first_occur_h.find(curr_tree(leaf_beg))).node;
    bool contig = true;
//printf("Node: %u\n", node);
    // Count how many leaves have already been seen
    for(uint32_t u=leaf_beg; u<=leaf_end; u++) {
      // Check if leaf has been seen already
      if(chunks_seen.find(digest_to_str(curr_tree(u))) != chunks_seen.end()) {
        num_match += 1;
      }
      // Check if order of leaves matches 
//printf("\tComparing digests %u and %u\n", leaf_start+(u-leaf_beg), u);
      if(!digests_same(curr_tree(leaf_start+(u-leaf_beg)), curr_tree(u))) {
        contig = false;
      }
    }
//printf("Node %u: Num match: %u, contig: %d\n", node, num_match, contig);
    if((num_match > 0) && !contig) {
      labels[node] = DONE;
      labels[2*node+1] = FIRST_DUPL;
      labels[2*node+2] = FIRST_DUPL;
      tree_roots.insert(2*node+1);
      tree_roots.insert(2*node+2);
      continue;
    }

    auto iter = first_ocur_dupl.find(digest_to_str(curr_tree(node)));
    if(iter != first_ocur_dupl.end()) {
      auto list = iter->second;
      std::sort(list.begin(), list.end());
      uint32_t highest_parent = num_nodes;
      uint32_t root = num_nodes;
      for(uint32_t j=0; j<list.size(); j++) {
        uint32_t curr_node = list[j];
        while(curr_node != 0) {
          if(tree_roots.find(curr_node) != tree_roots.end()) {
            if(highest_parent > curr_node) {
              if(labels[curr_node] != DONE) {
                highest_parent = curr_node;
                root = list[j];
              }
            }
          }
          curr_node = (curr_node-1)/2;
        }
        if(tree_roots.find(0) != tree_roots.end()) {
          if(highest_parent > curr_node) {
            if(labels[curr_node] != DONE) {
              highest_parent = 0;
              root = list[j];
            }
          }
        }
      }
//printf("Node: %u, root: %u, highest_parent: %u, label: %u\n", node, root, highest_parent, static_cast<uint32_t>(labels[root]));
      if(root == highest_parent && labels[root] == FIRST_DUPL) {
//printf("Node: %u, num_match: %u, leaves: %u\n", node, num_match, leaf_end-leaf_beg+1);
        if((num_match == leaf_end-leaf_beg+1) && contig) { 
          labels[highest_parent] = SHIFT_DUPL;
        } else {
          labels[highest_parent] = FIRST_OCUR;
          auto& node_meta = first_occur_h.value_at(first_occur_h.find(curr_tree(highest_parent)));
          node_meta.node = highest_parent;
        }
        uint32_t beg = leftmost_leaf(highest_parent, num_nodes);
        uint32_t end = rightmost_leaf(highest_parent, num_nodes);
        for(uint32_t u=beg; u<=end; u++) {
          auto res = chunks_seen.insert(digest_to_str(curr_tree(u)));
          if(!res.second)
            printf("Failed to insert %u\n", u);
        }
      }
      
      for(uint32_t j=0; j<list.size(); j++) {
        if(list[j] != root) {
          labels[list[j]] = SHIFT_DUPL;
        }
      }
    }
  }
}
//printf("Number of chunks covered: %u\n", chunks_seen.size());
//for(uint32_t node=0; node<num_nodes; node++) {
//  if(labels[node] == FIRST_DUPL)
//    printf("Error: %u is a FIRST_DUPL\n", node);
//}

  // Count regions
  for(auto set_iter=tree_roots.begin(); set_iter!=tree_roots.end(); set_iter++) {
    uint32_t root = *set_iter;
//printf("Root %u\n", root);
    if(labels[root] != DONE) {
      region_counters[labels[root]] += 1;
      NodeID node = first_occur_h.value_at(first_occur_h.find(curr_tree(root)));
      if(labels[root] == FIRST_OCUR) {
//printf("First occurrence: %u: (%u,%u)\n", root, node.node, node.tree);
        first_ocur_map_h.insert(root, node);
      } else if(labels[root] == SHIFT_DUPL || labels[root] == FIRST_DUPL) {
//printf("Shifted duplicate: %u: (%u,%u)\n", root, node.node, node.tree);
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
