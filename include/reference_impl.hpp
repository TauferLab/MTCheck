#ifndef REFERENCE_IMPL_HPP
#define REFERENCE_IMPL_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_ScatterView.hpp>
//#include <queue>
//#include <iostream>
//#include <climits>
#include <vector>
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
    Kokkos::deep_copy(shift_dupl_map_h, shift_dupl_map_d);
    Kokkos::deep_copy(first_ocur_map_h, first_ocur_map_d);
  
    const char FIRST_OCUR = 0;
    const char FIXED_DUPL = 1;
    const char SHIFT_DUPL = 2;
    const char DONE = 3;
    uint64_t chunk_counters[3] = {0,0,0};
    uint64_t region_counters[3] = {0,0,0};
    uint32_t num_chunks = (tree.tree_h.extent(0)+1)/2;
    uint32_t num_nodes = tree.tree_h.extent(0);
    printf("Num chunks: %u\n", num_chunks);
    printf("Num nodes: %u\n", num_nodes);
  
    std::vector<char> labels(num_nodes, DONE);
    std::vector<uint32_t> tree_roots;
  
    // Process leaves first
    for(uint32_t leaf=num_chunks-1; leaf<num_nodes; leaf++) {
      uint32_t num_bytes = chunk_size;
      if(leaf == num_nodes-1) // Calculate how much data to hash
        num_bytes = data_h.size()-(leaf-(num_chunks-1))*chunk_size;
      MD5(data_h.data()+(leaf-(num_chunks-1))*chunk_size, num_bytes, curr_tree(leaf).digest); // Hash chunk
      auto result = first_occur_h.insert(curr_tree(leaf), NodeID(leaf, chkpt_id)); // Insert into table
      if(digests_same(prev_tree(leaf), curr_tree(leaf))) { // Fixed duplicate chunk
        labels[leaf] = FIXED_DUPL;
      } else if(result.success()) { // First occurrence chunk
        labels[leaf] = FIRST_OCUR;
      } else if(result.existing()) { // Shifted duplicate chunk
        labels[leaf] = SHIFT_DUPL;
      }
      chunk_counters[labels[leaf]] += 1;
    }
  
    // Build up forest of Merkle Trees
    for(uint32_t node=num_chunks-2; node>=0 && node < num_chunks-1; node--) {
      uint32_t child_l = 2*node+1;
      uint32_t child_r = 2*node+2;
      if(labels[child_l] != labels[child_r]) { // Children have different labels
        labels[node] = DONE;
        tree_roots.push_back(child_l);
        tree_roots.push_back(child_r);
      } else if(labels[child_l] == FIRST_OCUR) { // Children are both first occurrences
        labels[node] = FIRST_OCUR;
        MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
        first_occur_h.insert(curr_tree(node), NodeID(node, chkpt_id));
        if(node == 0) {
          tree_roots.push_back(node);
        }
      } else if(labels[child_l] == FIXED_DUPL) { // Children are both fixed duplicates
        labels[node] = FIXED_DUPL;
      } else if(labels[child_l] == SHIFT_DUPL) { // Children are both shifted duplicates
        MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
        if(first_occur_h.exists(curr_tree(node))) { // This node is also a shifted duplicate
          labels[node] = SHIFT_DUPL;
        } else { // Node is not a shifted duplicate. Save child trees
          labels[node] = DONE; // Add children to tree root maps
          tree_roots.push_back(child_l);
          tree_roots.push_back(child_r);
        }
      }
    }

    // Count regions
    for(uint32_t i=0; i<tree_roots.size(); i++) {
      uint32_t root = tree_roots[i];
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
//};


#endif // REFERENCE_IMPL
