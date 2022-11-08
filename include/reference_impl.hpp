#ifndef REFERENCE_IMPL_HPP
#define REFERENCE_IMPL_HPP
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
#include "utils.hpp"

//enum Case {
//  BothFixedDupl,
//  BothFirstOcur,
//  BothShiftDupl,
//  DiffShiftDupl,
//  DiffChildren,
//};

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

  const int BothFixedDupl=0;
  const int BothFirstOcur=1;
  const int BothShiftDupl=2;
  const int BothDone=3;
//  const int DiffShiftDupl=4;
  const int DiffChildren=5;

  const char FIRST_OCUR = 1;
  const char FIXED_DUPL = 2;
  const char SHIFT_DUPL = 3;
  const char DONE = 4;
  uint64_t num_first_ocur_chunks = 0;
  uint64_t num_fixed_dupl_chunks = 0;
  uint64_t num_shift_dupl_chunks = 0;
  uint64_t num_first_ocur_regs = 0;
  uint64_t num_fixed_dupl_regs = 0;
  uint64_t num_shift_dupl_regs = 0;
  uint32_t num_chunks = (tree.tree_h.extent(0)+1)/2;
  uint32_t num_nodes = tree.tree_h.extent(0);
  printf("Num chunks: %u\n", num_chunks);
  printf("Num nodes: %u\n", num_nodes);

  uint32_t level_beg = 0;
  uint32_t level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  std::vector<char> labels(num_nodes, DONE);
  while(level_beg <= num_nodes) { // Intensional unsigned integer underflow
    for(uint32_t node=level_beg; node<=level_end; node++) {
      if(node < num_nodes) {
        if(node >= num_chunks-1) { // Leave nodes
          uint32_t num_bytes = chunk_size;
          if(node == num_nodes-1) // Calculate how much data to hash
            num_bytes = data_h.size()-(node-(num_chunks-1))*chunk_size;
          MD5(data_h.data()+(node-(num_chunks-1))*chunk_size, num_bytes, curr_tree(node).digest); // Hash chunk
          auto result = first_occur_h.insert(curr_tree(node), NodeID(node, chkpt_id)); // Insert into table
          if(digests_same(prev_tree(node), curr_tree(node))) { // Fixed duplicate chunk
            labels[node] = FIXED_DUPL;
            num_fixed_dupl_chunks += 1;
          } else if(result.success()) { // First occurrence chunk
            labels[node] = FIRST_OCUR;
            num_first_ocur_chunks += 1;
          } else if(result.existing()) { // Shifted duplicate chunk
            labels[node] = SHIFT_DUPL;
            num_shift_dupl_chunks += 1;
          }
        } else { // Inner nodes
          uint32_t child_l = 2*node+1;
          uint32_t child_r = 2*node+2;
          int state = 5;
          if(labels[child_l] != labels[child_r]) {
            state = DiffChildren;
          } else if(labels[child_l] == FIXED_DUPL && labels[child_r] == FIXED_DUPL) {
            state = BothFixedDupl;
          } else if(labels[child_l] == SHIFT_DUPL && labels[child_r] == SHIFT_DUPL) {
            state = BothShiftDupl;
          } else if(labels[child_l] == FIRST_OCUR && labels[child_r] == FIRST_OCUR) {
            state = BothFirstOcur;
          } else if(labels[child_l] == DONE && labels[child_r] == DONE) {
            state = BothDone;
          } else {
            printf("Invalid state!\n");
          }

          switch (state) {
            case BothFirstOcur:
              labels[node] = FIRST_OCUR;
              MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
              first_occur_h.insert(curr_tree(node), NodeID(node, chkpt_id));
              if(node == 0) {
                first_ocur_map_h.insert(node, NodeID(node, chkpt_id));
                num_first_ocur_regs += 1;
              }
              break;
            case BothShiftDupl:
              {
                MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
                if(first_occur_h.exists(curr_tree(node))) { // This node is also a shifted duplicate
                  labels[node] = SHIFT_DUPL;
                } else { // Node is not a shifted duplicate. Save child trees
                  labels[node] = DONE; // Add children to tree root maps
                  NodeID child_l_node = first_occur_h.value_at(first_occur_h.find(curr_tree(child_l)));
                  NodeID child_r_node = first_occur_h.value_at(first_occur_h.find(curr_tree(child_r)));

                  shift_dupl_map_h.insert(child_l, child_l_node);
                  num_shift_dupl_regs += 1;

                  shift_dupl_map_h.insert(child_r, child_r_node);
                  num_shift_dupl_regs += 1;
                }
              }
              break;
            case BothFixedDupl:
              labels[node] = FIXED_DUPL;
              break;
            case DiffChildren:
              {
                labels[node] = DONE; // Add children to tree root maps
                NodeID child_l_node = first_occur_h.value_at(first_occur_h.find(curr_tree(child_l)));
                NodeID child_r_node = first_occur_h.value_at(first_occur_h.find(curr_tree(child_r)));
                if(labels[child_l] == FIRST_OCUR) {        // First occurrences
                  first_ocur_map_h.insert(child_l, child_l_node);
                  num_first_ocur_regs += 1;
                } else if(labels[child_l] == SHIFT_DUPL) { // Shifted duplicates
                  shift_dupl_map_h.insert(child_l, child_l_node);
                  num_shift_dupl_regs += 1;
                } else if(labels[child_l] == FIXED_DUPL) { // Ignore fixed duplicates
                  num_fixed_dupl_regs += 1;
                }
                if(labels[child_r] == FIRST_OCUR) {        // First occurrences
                  first_ocur_map_h.insert(child_r, child_r_node);
                  num_first_ocur_regs += 1;
                } else if(labels[child_r] == SHIFT_DUPL) { // Shifted duplicates
                  shift_dupl_map_h.insert(child_r, child_r_node);
                  num_shift_dupl_regs += 1;
                } else if(labels[child_r] == FIXED_DUPL) { // Ignore fixed duplicates
                  num_fixed_dupl_regs += 1;
                }
              }
              break;
            case BothDone:
              labels[node] = DONE;
              break;
            default:
              printf("Reached default case. Should not happen!\n");
          }
//          if(state == BothFirstOcur) { // First occurrence region
//            labels[node] = FIRST_OCUR;
//            MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
//            first_occur_h.insert(curr_tree(node), NodeID(node, chkpt_id));
//          } else if(state == BothShiftDupl) { // Shifted duplicate region
//            MD5((uint8_t*)&curr_tree(child_l), 2*sizeof(HashDigest), curr_tree(node).digest);
//            if(first_occur_h.exists(curr_tree(node))) { // This node is also a shifted duplicate
//              labels[node] = SHIFT_DUPL;
//            } else { // Node is not a shifted duplicate. Save child trees
//              labels[node] = DONE; // Add children to tree root maps
//              NodeID child_l_node = first_occur_h.value_at(first_occur_h.find(curr_tree(child_l)));
//              NodeID child_r_node = first_occur_h.value_at(first_occur_h.find(curr_tree(child_r)));
//
//              shift_dupl_map_h.insert(child_l, child_l_node);
//              num_shift_dupl_regs += 1;
//
//              shift_dupl_map_h.insert(child_r, child_r_node);
//              num_shift_dupl_regs += 1;
//            }
//          } else if(state == BothFixedDupl) { // Fixed duplicate region
//            labels[node] = FIXED_DUPL;
//          } else { // Children have different labels so we save the tree roots for each child
//            labels[node] = DONE; // Add children to tree root maps
//            NodeID child_l_node = first_occur_h.value_at(first_occur_h.find(curr_tree(child_l)));
//            NodeID child_r_node = first_occur_h.value_at(first_occur_h.find(curr_tree(child_r)));
//            if(labels[child_l] == FIRST_OCUR) {        // First occurrences
//              first_ocur_map_h.insert(child_l, child_l_node);
//              num_first_ocur_regs += 1;
//            } else if(labels[child_l] == SHIFT_DUPL) { // Shifted duplicates
//              shift_dupl_map_h.insert(child_l, child_l_node);
//              num_shift_dupl_regs += 1;
//            } else if(labels[child_l] == FIXED_DUPL) { // Ignore fixed duplicates
//              num_fixed_dupl_regs += 1;
//            }
//            if(labels[child_r] == FIRST_OCUR) {        // First occurrences
//              first_ocur_map_h.insert(child_r, child_r_node);
//              num_first_ocur_regs += 1;
//            } else if(labels[child_r] == SHIFT_DUPL) { // Shifted duplicates
//              shift_dupl_map_h.insert(child_r, child_r_node);
//              num_shift_dupl_regs += 1;
//            } else if(labels[child_r] == FIXED_DUPL) { // Ignore fixed duplicates
//              num_fixed_dupl_regs += 1;
//            }
//          }
        }
      }
    }
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }

  Kokkos::deep_copy(shift_dupl_map_d, shift_dupl_map_h);
  Kokkos::deep_copy(first_ocur_map_d, first_ocur_map_h);
  Kokkos::deep_copy(tree.tree_d, curr_tree);
  Kokkos::deep_copy(first_occur_d, first_occur_h);

  printf("Checkpoint %u\n", chkpt_id);
  printf("Size of first occurrence map: %lu\n", first_ocur_map_d.size());
  printf("Size of shift duplicate map: %lu\n", shift_dupl_map_d.size());
  printf("Number of first occurrence chunks:  %lu\n", num_first_ocur_chunks);
  printf("Number of fixed duplicate chunks:   %lu\n", num_fixed_dupl_chunks);
  printf("Number of shifted duplicate chunks: %lu\n", num_shift_dupl_chunks);
  printf("Number of first occurrence regions:  %lu\n", num_first_ocur_regs);
  printf("Number of fixed duplicate regions:   %lu\n", num_fixed_dupl_regs);
  printf("Number of shifted duplicate regions: %lu\n", num_shift_dupl_regs);
  return num_first_ocur_regs+num_fixed_dupl_regs+num_shift_dupl_regs;
}

#endif // REFERENCE_IMPL
