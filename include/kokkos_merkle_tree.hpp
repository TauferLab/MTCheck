#ifndef KOKKOS_MERKLE_TREE_HPP
#define KOKKOS_MERKLE_TREE_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <climits>
#include "hash_functions.hpp"
#include "map_helpers.hpp"
#include "kokkos_queue.hpp"
#include <iostream>
#include "utils.hpp"


//template<uint32_t N>
class MerkleTree {
public:
  Kokkos::View<HashDigest*> tree_d;
  Kokkos::View<HashDigest*>::HostMirror tree_h;
  Kokkos::View<uint8_t*> distinct_children_d;
  Kokkos::View<uint8_t*>::HostMirror distinct_children_h;

  MerkleTree(const uint32_t num_leaves) {
    tree_d = Kokkos::View<HashDigest*>("Merkle tree", (2*num_leaves-1));
    tree_h = Kokkos::create_mirror_view(tree_d);
    distinct_children_d = Kokkos::View<uint8_t*>("Num distinct children", (2*num_leaves-1));
    distinct_children_h = Kokkos::create_mirror_view(distinct_children_d);
    Kokkos::deep_copy(distinct_children_d, 0);
  }
  
  KOKKOS_INLINE_FUNCTION HashDigest& operator()(int32_t i) const {
    return tree_d(i);
  }
 
  void digest_to_hex_(const uint8_t digest[16], char* output) {
    int i,j;
    char* c = output;
    for(i=0; i<16/4; i++) {
      for(j=0; j<4; j++) {
        sprintf(c, "%02X", digest[i*4 + j]);
        c += 2;
      }
      sprintf(c, " ");
      c += 1;
    }
    *(c-1) = '\0';
  }

  void print_leaves() {
    Kokkos::deep_copy(tree_h, tree_d);
    uint32_t num_leaves = (tree_h.extent(0)+1)/2;
    printf("============================================================\n");
    char buffer[64];
    unsigned int counter = 2;
    for(unsigned int i=num_leaves-1; i<tree_h.extent(0); i++) {
      digest_to_hex_((uint8_t*)(tree_h(i).digest), buffer);
      printf("Node: %u: %s \n", i, buffer);
      if(i == counter) {
        printf("\n");
        counter += 2*counter;
      }
    }
    printf("============================================================\n");
  }

  void print() {
    Kokkos::deep_copy(tree_h, tree_d);
printf("Num digests: %lu\n", tree_h.extent(0));
    uint32_t num_leaves = (tree_h.extent(0)+1)/2;
    printf("============================================================\n");
    char buffer[64];
    unsigned int counter = 2;
    for(unsigned int i=16777215; i<16777315; i++) {
      digest_to_hex_((uint8_t*)(tree_h(i).digest), buffer);
      printf("Node: %u: %s \n", i, buffer);
      if(i == counter) {
        printf("\n");
        counter += 2*counter;
      }
    }
    printf("============================================================\n");
  }
};

//template <class Hasher>
//void create_merkle_tree(Hasher& hasher, MerkleTree& tree, Kokkos::View<uint8_t*>& data, const uint32_t chunk_size, const int32_t n_levels=INT_MAX) {
//  uint32_t num_chunks = data.size()/chunk_size;
//  if(num_chunks*chunk_size < data.size())
//    num_chunks += 1;
//  const uint32_t num_nodes = 2*num_chunks-1;
//  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
//  int32_t stop_level = 0;
//  if(n_levels < num_levels)
//    stop_level = num_levels-n_levels;
//  const uint32_t leaf_start = num_chunks-1;
//  for(int32_t level=num_levels-1; level>=stop_level; level--) {
//    uint32_t nhashes = 1 << level;
//    uint32_t start_offset = nhashes-1;
//    if(start_offset + nhashes > num_nodes)
//      nhashes = num_nodes - start_offset;
//    auto range_policy = Kokkos::RangePolicy<>(start_offset, start_offset+nhashes);
//    Kokkos::parallel_for("Build tree", range_policy, KOKKOS_LAMBDA(const int i) {
//      uint32_t num_bytes = chunk_size;
//      if((i-leaf_start) == num_chunks-1)
//        num_bytes = data.size()-((i-leaf_start)*chunk_size);
//      if(i >= leaf_start) {
//        hasher.hash(data.data()+((i-leaf_start)*chunk_size), 
//                    num_bytes, 
//                    (uint8_t*)(tree(i).digest));
//      } else {
//        hasher.hash((uint8_t*)&tree(2*i+1), 2*hasher.digest_size(), (uint8_t*)&tree(i));
//      }
//    });
//  }
//  Kokkos::fence();
//}
//
//template <class Hasher>
//MerkleTree create_merkle_tree(Hasher& hasher, Kokkos::View<uint8_t*>& data, const uint32_t chunk_size) {
//  uint32_t num_chunks = data.size()/chunk_size;
//  if(num_chunks*chunk_size < data.size())
//    num_chunks += 1;
//  MerkleTree tree = MerkleTree(num_chunks);
//  create_merkle_tree(hasher, tree, data, chunk_size, INT_MAX);
//  return tree;
//}

template <class Hasher>
void create_merkle_tree(Hasher& hasher, MerkleTree& tree, Kokkos::View<uint8_t*>& data, uint32_t chunk_size, uint32_t tree_id, DistinctNodeIDMap& distinct_map, SharedNodeIDMap& shared_map) {
  uint32_t num_chunks = static_cast<uint32_t>(data.size()/chunk_size);
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  const uint32_t num_nodes = 2*num_chunks-1;
  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  const uint32_t leaf_start = num_chunks-1;
  for(int32_t level=num_levels-1; level>=0; level--) {
    DEBUG_PRINT("Processing level %d\n", level);
    uint32_t nhashes = 1 << static_cast<uint32_t>(level);
    uint32_t start_offset = nhashes-1;
    if(start_offset + nhashes > num_nodes)
      nhashes = num_nodes - start_offset;
    DEBUG_PRINT("Computing %u hashes\n", nhashes);
    auto range_policy = Kokkos::RangePolicy<>(start_offset, start_offset+nhashes);
    Kokkos::parallel_for("Build tree", range_policy, KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t num_bytes = chunk_size;
      if((i-leaf_start) == num_chunks-1)
        num_bytes = data.size()-((i-leaf_start)*chunk_size);
      if(i >= leaf_start) {
        hasher.hash(data.data()+((i-leaf_start)*chunk_size), 
                    num_bytes, 
                    (uint8_t*)(tree(i).digest));
      } else {
        hasher.hash((uint8_t*)&tree(2*i+1), 2*hasher.digest_size(), (uint8_t*)&tree(i));
      }
      auto result = distinct_map.insert(tree(i), NodeID(i, tree_id));
      if(result.existing()) {
        auto& entry = distinct_map.value_at(result.index());
//          shared_map.insert(i,entry.node);
          shared_map.insert(i,NodeID(entry.node, entry.tree));
      } else if(result.failed()) {
          printf("Failed to insert node %u into distinct map\n",i);
      }
    });
  }
  Kokkos::fence();
}

KOKKOS_INLINE_FUNCTION uint32_t num_leaf_descendents(uint32_t node, uint32_t num_nodes) {
  uint32_t leftmost = (2*node)+1;
  uint32_t rightmost = (2*node)+2;
  while(leftmost < num_nodes) {
    leftmost = (2*leftmost)+1;
    rightmost = (2*rightmost)+2;
  }
  leftmost = (leftmost-1)/2;
  rightmost = (rightmost-2)/2;
  if(rightmost > num_nodes)
    rightmost = num_nodes-1;
  return static_cast<uint32_t>(rightmost-leftmost+1);
}

KOKKOS_INLINE_FUNCTION uint32_t leftmost_leaf(uint32_t node, uint32_t num_nodes) {
  uint32_t leftmost = (2*node)+1;
  uint32_t rightmost = (2*node)+2;
  while(leftmost < num_nodes) {
    leftmost = (2*leftmost)+1;
    rightmost = (2*rightmost)+2;
  }
  leftmost = (leftmost-1)/2;
  rightmost = (rightmost-2)/2;
  if(rightmost > num_nodes)
    rightmost = num_nodes-1;
  return static_cast<uint32_t>(leftmost);
}

KOKKOS_INLINE_FUNCTION uint32_t rightmost_leaf(uint32_t node, uint32_t num_nodes) {
  uint32_t leftmost = (2*node)+1;
  uint32_t rightmost = (2*node)+2;
  while(leftmost < num_nodes) {
    leftmost = (2*leftmost)+1;
    rightmost = (2*rightmost)+2;
  }
  leftmost = (leftmost-1)/2;
  rightmost = (rightmost-2)/2;
  if(rightmost > num_nodes)
    rightmost = num_nodes-1;
  return static_cast<uint32_t>(rightmost);
}

//template<uint32_t N>
//KOKKOS_INLINE_FUNCTION
//void insert_entry(const CompactTable<N>& updates, const uint32_t node, const uint32_t num_nodes, const uint32_t tree_id, const uint32_t prior_node) {
//if(node > num_nodes)
//printf("Something very wrong happened.\n");
////  uint32_t num_chunks = (num_nodes+1)/2;
////  uint32_t size = num_leaf_descendents(node, num_nodes);
////  uint32_t leaf = leftmost_leaf(node, num_nodes);
//  CompactNodeInfo info(node, prior_node);
//  auto result = updates.insert(info);
//  auto& update = updates.value_at(result.index());
//  update.push(tree_id);
//}

KOKKOS_INLINE_FUNCTION
void insert_entry(const CompactTable& updates, const uint32_t node, const uint32_t num_nodes, const uint32_t tree_id, const NodeID node_id) {
  if(node > num_nodes)
    printf("Something very wrong happened.\n");
  auto result = updates.insert(node, NodeID(node_id.node, node_id.tree));
  if(result.failed()) {
    DEBUG_PRINT("Failed to insert entry %u: (%u,%u)\n", node, node_id.node, node_id.tree);
  }
}

//template<class Hasher, uint32_t N>
template<class Hasher>
void deduplicate_data(Kokkos::View<uint8_t*>& data, 
                      const uint32_t chunk_size, 
                      const Hasher hasher, 
                      MerkleTree& tree, 
                      const uint32_t tree_id, 
                      const SharedNodeIDMap& prior_shared_map, 
                      const DistinctNodeIDMap& prior_distinct_map, 
                      SharedNodeIDMap& shared_map, 
                      DistinctNodeIDMap& distinct_map, 
                      CompactTable& shared_updates,
                      CompactTable& distinct_updates) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  const uint32_t num_nodes = 2*num_chunks-1;
  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  const uint32_t leaf_start = num_chunks-1;

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
  Kokkos::UnorderedMap<HashDigest, void, Kokkos::CudaUVMSpace, digest_hash, digest_equal_to> table(num_chunks);
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

        if(tree_id == 0) {
        } else {
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
//              auto res = shared_map.insert(node, existing_info.node);
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
            uint32_t prior_shared_idx = prior_shared_map.find(node);
            if(prior_shared_map.valid_at(prior_shared_idx)) { // Node was repeat last checkpoint 
              NodeID prior_shared = prior_shared_map.value_at(prior_shared_idx);
              if(prior_shared.node == old_distinct.node && prior_shared.tree == old_distinct.tree) {
                tree.distinct_children_d(node) = 0;
#ifdef STATS
                Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
#endif
              } else {
                shared_map.insert(node, NodeID(old_distinct.node, old_distinct.tree));
#ifdef STATS
auto res = table.insert(tree(node));
if(res.success())
  Kokkos::atomic_add(&num_prior_chunks_d(old_distinct.tree), 1);
#endif
                tree.distinct_children_d(node) = 8;
                nodes_leftover(0) += static_cast<uint32_t>(1);
#ifdef STATS
                Kokkos::atomic_add(&(num_shift(0)), static_cast<uint32_t>(1));
#endif
              }
            } else { // Node was distinct last checkpoint
              if(node == old_distinct.node) { // No change since last checkpoint
                tree.distinct_children_d(node) = 0;
#ifdef STATS
                Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
#endif
              } else { // Node changed since last checkpoint
                shared_map.insert(node, NodeID(old_distinct.node, old_distinct.tree));
#ifdef STATS
auto res = table.insert(tree(node));
if(res.success())
  Kokkos::atomic_add(&num_prior_chunks_d(old_distinct.tree), 1);
#endif
                tree.distinct_children_d(node) = 8;
                nodes_leftover(0) += static_cast<uint32_t>(1);
#ifdef STATS
                Kokkos::atomic_add(&(num_shift(0)), static_cast<uint32_t>(1));
#endif
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
//              insert_entry(distinct_updates, child_l, num_nodes, tree_id, shared_map.value_at(shared_map.find(child_l)));
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
//              insert_entry(distinct_updates, child_r, num_nodes, tree_id, shared_map.value_at(shared_map.find(child_r)));
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
//                insert_entry(distinct_updates, child_l, num_nodes, tree_id, repeat);
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
//                insert_entry(distinct_updates, child_r, num_nodes, tree_id, repeat);
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
//                insert_entry(distinct_updates, child_l, num_nodes, tree_id, shared_map.value_at(shared_map.find(child_l)));
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
//                insert_entry(distinct_updates, child_r, num_nodes, tree_id, shared_map.value_at(shared_map.find(child_r)));
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
printf("Count number of distinct chunks: %u\n", n_distinct);
printf("Count number of same chunks: %u\n", n_same);
printf("Count number of shared chunks: %u\n", n_shared);
printf("Count number of distinct shared chunks: %u\n", n_distinct_shared);
printf("Count number of distinct_same chunks: %u\n", n_distinct_same);
printf("Count number of shared_same chunks: %u\n", n_shared_same);
printf("------------------------------\n");
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
  printf("Number of chunks: %u\n", num_chunks);
  printf("Number of new chunks: %u\n", num_new_h(0));
  printf("Number of same chunks: %u\n", num_same_h(0));
  printf("Number of shift chunks: %u\n", num_shift_h(0));
  printf("Number of distinct comp nodes: %u\n", num_comp_d_h(0));
  printf("Number of shared comp nodes: %u\n", num_comp_s_h(0));
  printf("Number of dupl nodes: %u\n", num_dupl_h(0));
  printf("Number of other nodes: %u\n", num_other_h(0));
Kokkos::deep_copy(num_prior_chunks_h, num_prior_chunks_d);
for(int i=0; i<10; i++) {
  printf("Number of chunks repeating from checkpoint %d: %u\n", i, num_prior_chunks_h(i));
}
#endif
}

int 
restart_incr_chkpt_hashtree(std::vector<std::string>& chkpt_files,
                            const int file_idx, 
                            Kokkos::View<uint8_t*>& data) {
  // Read main incremental checkpoint header
  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  file.open(chkpt_files[file_idx], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
  size_t filesize = file.tellg();
  file.seekg(0);

  DEBUG_PRINT("File size: %zd\n", filesize);
  header_t header;
  file.read((char*)&header, sizeof(header_t));
  STDOUT_PRINT("Ref ID: %u\n",               header.ref_id);
  STDOUT_PRINT("Chkpt ID: %u\n",             header.chkpt_id);
  STDOUT_PRINT("Data len: %lu\n",            header.datalen);
  STDOUT_PRINT("Chunk size: %u\n",           header.chunk_size);
  STDOUT_PRINT("Window size: %u\n",          header.window_size);
  STDOUT_PRINT("Distinct size: %u\n",        header.distinct_size);
  STDOUT_PRINT("Current Repeat size: %u\n",  header.curr_repeat_size);
  STDOUT_PRINT("Previous Repeat size: %u\n", header.prev_repeat_size);

  Kokkos::View<uint8_t*> buffer_d("Buffer", filesize);
  Kokkos::deep_copy(buffer_d, 0);
  auto buffer_h = Kokkos::create_mirror_view(buffer_d);
  Kokkos::deep_copy(buffer_h, 0);
  file.close();

  uint32_t num_chunks = header.datalen / header.chunk_size;
  if(num_chunks*header.chunk_size < header.datalen) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;
  Kokkos::resize(data, header.datalen);
printf("Setup data buffers\n");

  if(header.window_size == 0) {
    // Main checkpoint
    file.open(chkpt_files[file_idx], std::ifstream::in | std::ifstream::binary);
    file.read((char*)(buffer_h.data()), filesize);
    file.close();
    Kokkos::deep_copy(buffer_d, buffer_h);
    Kokkos::View<NodeID*> node_list("List of NodeIDs", num_chunks);
    Kokkos::deep_copy(node_list, NodeID());
    uint32_t ref_id = header.ref_id;
    uint32_t cur_id = header.chkpt_id;
    size_t prev_repeat_offset = filesize - header.prev_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
    size_t curr_repeat_offset = prev_repeat_offset - header.curr_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
    auto curr_repeat   = Kokkos::subview(buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    auto prev_repeat   = Kokkos::subview(buffer_d, std::make_pair(prev_repeat_offset, filesize));
    auto distinct = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));

    uint32_t chunk_size = header.chunk_size;
    size_t datalen = header.datalen;
    uint32_t num_distinct = header.distinct_size;
    Kokkos::UnorderedMap<NodeID, size_t, Kokkos::CudaUVMSpace> distinct_map(num_nodes);
printf("Setup reference offsets and subviews\n");
    Kokkos::parallel_for("Restart Hashtree distinct", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const uint32_t index) {
      size_t read_offset = 0;
      for(uint32_t i=0; i<num_distinct; i++) {
        uint32_t node;
        memcpy(&node, distinct.data() + read_offset,  sizeof(uint32_t));
DEBUG_PRINT("%u: %c, %lu\n", node, *((const char*)(distinct.data()+read_offset+sizeof(uint32_t))), read_offset+sizeof(uint32_t));
        auto result_d = distinct_map.insert(NodeID(node,cur_id),  read_offset + sizeof(uint32_t));
if(result_d.failed())
STDOUT_PRINT("Failed to insert distinct entry (%u,%u): %lu\n", node, cur_id, read_offset+sizeof(uint32_t));
//        node_list(node) = NodeID(node,cur_id);
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
    auto result = distinct_map.insert(NodeID(u, cur_id), read_offset+sizeof(uint32_t)+(leaf-start)*chunk_size);
//printf("Inserting distinct node (%u,%u): %lu\n", u, cur_id, read_offset+sizeof(uint32_t)+(leaf-start)*chunk_size);
    if(result.failed())
      printf("Failed to insert (%u,%u): %lu\n", u, cur_id, read_offset+sizeof(uint32_t)+(leaf-start)*chunk_size);
  }
  left = 2*left+1;
  right = 2*right+2;
}
        for(uint32_t j=0; j<len; j++) {
//auto result = distinct_map.insert(NodeID(start+j, cur_id), read_offset+sizeof(uint32_t)+j*chunk_size);
//if(result.failed())
//STDOUT_PRINT("Failed ot insert (%u,%u): %lu\n", start, cur_id, read_offset+sizeof(uint32_t)+j*chunk_size);
          node_list(start-num_chunks+1+j) = NodeID(start-num_chunks+1+j, UINT32_MAX);
//if(len > 1)
//STDOUT_PRINT("Entry %u updated: (%u,%u) (distinct node %u) (%u chunks long)\n", start-num_chunks+1+j, start-num_chunks+1+j, UINT32_MAX, node, len);
        }
        uint32_t datasize = len*chunk_size;
        if(end == num_nodes-1)
          datasize = datalen - (start-num_chunks+1)*chunk_size;
DEBUG_PRINT("Copying %c to position %u, len: %u\n", (char)(*(distinct.data()+read_offset+sizeof(uint32_t))), chunk_size*(start-num_chunks+1), datasize);
        memcpy(data.data()+chunk_size*(start-num_chunks+1), distinct.data()+read_offset+sizeof(uint32_t), datasize);
        read_offset += sizeof(uint32_t) + chunk_size*len;
      }
    });
    Kokkos::parallel_for("Restart Hashtree current repeat", Kokkos::RangePolicy<>(0, header.curr_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      memcpy(&node, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
//      node_list(node) = NodeID(prev, cur_id);
//if(!distinct_map.exists(NodeID(prev, cur_id)))
//  STDOUT_PRINT("Repeat not found in distinct map: %u,%u\n", prev, cur_id);
      size_t offset = distinct_map.value_at(distinct_map.find(NodeID(prev, cur_id)));
      uint32_t node_start = leftmost_leaf(node, num_nodes);
      uint32_t prev_start = leftmost_leaf(prev, num_nodes);
      uint32_t len = num_leaf_descendents(prev, num_nodes);
//if(distinct_map.exists(NodeID(prev, cur_id)))
//STDOUT_PRINT("Searching for region (%u,%u) (aka (%u,%u)), len %u\n", prev, cur_id, prev-num_chunks+1, cur_id, len);
      for(uint32_t j=0; j<len; j++) {
        node_list(node_start+j-num_chunks+1) = NodeID(prev_start-num_chunks+1+j, UINT32_MAX);
DEBUG_PRINT("Entry %u updated: (%u,%u) (current repeat node %u prev %u)\n", node_start-num_chunks+1+j, prev_start+j, UINT32_MAX, node, prev);
      }
      uint32_t copysize = chunk_size;
      if(node == num_nodes-1)
        copysize = data.size() - chunk_size*(num_chunks-1);
      memcpy(data.data()+chunk_size*(node_start-num_chunks+1), distinct.data()+offset, copysize);
DEBUG_PRINT("Duplicated (%u) %c%c to %u (offset %lu)\n", node_start-num_chunks+1, *((char*)(distinct.data()+offset)), *((char*)(distinct.data()+offset+1)),  (node_start-num_chunks+1)*chunk_size, offset);
    });
    Kokkos::parallel_for("Restart Hashtree previous repeat", Kokkos::RangePolicy<>(0, header.prev_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      memcpy(&node, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)) +sizeof(uint32_t), sizeof(uint32_t));
DEBUG_PRINT("Node: %u, Prev: %u\n", node, prev);
//      node_list(node) = NodeID(prev,ref_id);
      uint32_t node_start = leftmost_leaf(node, num_nodes);
      uint32_t prev_start = leftmost_leaf(prev, num_nodes);
      uint32_t len = num_leaf_descendents(prev, num_nodes);
      for(uint32_t j=0; j<len; j++) {
        node_list(node_start-num_chunks+1+j) = NodeID(prev_start+j, ref_id);
DEBUG_PRINT("Entry %u updated: (%u,%u) (previous repeat node %u, prev %u)\n", node_start-num_chunks+1, prev_start+j, ref_id, node, prev);
      }
    });
Kokkos::fence();
DEBUG_PRINT("Restarted previous repeats\n");
    Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      NodeID entry = node_list(i);
      if(entry.node == UINT_MAX) {
        node_list(i) = NodeID(i+num_chunks-1, ref_id);
      }
    });
    Kokkos::fence();
//auto node_list_h = Kokkos::create_mirror_view(node_list);
//Kokkos::deep_copy(node_list_h, node_list);
//for(uint32_t i=0; i<node_list.size(); i++) {
//  printf("Node: %u: (%u,%u)\n", i, node_list_h(i).node, node_list_h(i).tree);
//}
//auto data_h = Kokkos::create_mirror_view(data);
//printf("Done setting up the restart\n");

    if(header.ref_id != header.chkpt_id) {
      // Reference
      file.open(chkpt_files[header.ref_id], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
      size_t chkpt_size = file.tellg();
      file.seekg(0);
      Kokkos::View<uint8_t*> chkpt_buffer_d("Checkpoint buffer", chkpt_size);
      auto chkpt_buffer_h = Kokkos::create_mirror_view(chkpt_buffer_d);
      file.read((char*)(chkpt_buffer_h.data()), chkpt_size);
      file.close();
      header_t chkpt_header;
      memcpy(&chkpt_header, chkpt_buffer_h.data(), sizeof(header_t));
      uint32_t current_id = chkpt_header.chkpt_id;
      datalen = chkpt_header.datalen;
      chunk_size = chkpt_header.chunk_size;
      uint32_t num_distinct = chkpt_header.distinct_size;
      Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
      cur_id = chkpt_header.chkpt_id;
      ref_id = chkpt_header.ref_id;
      prev_repeat_offset = chkpt_size - chkpt_header.prev_repeat_size*(2*sizeof(uint32_t));
      curr_repeat_offset = prev_repeat_offset - chkpt_header.curr_repeat_size*(2*sizeof(uint32_t));
      curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
      prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, chkpt_size));
      distinct = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
      
      distinct_map.clear();
      distinct_map.rehash(chkpt_header.distinct_size);
      Kokkos::fence();
      Kokkos::UnorderedMap<uint32_t, NodeID, Kokkos::CudaUVMSpace> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
      Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const uint32_t j) {
//        uint32_t node;
//        memcpy(&node, distinct.data()+i*(sizeof(uint32_t)+chunk_size), sizeof(uint32_t));
//        distinct_map.insert(NodeID(node, current_id), i*(sizeof(uint32_t)+chunk_size)+sizeof(uint32_t));
        size_t read_offset = 0;
        for(uint32_t i=0; i<num_distinct; i++) {
          uint32_t node;
          memcpy(&node, distinct.data() + read_offset,  sizeof(uint32_t));
          distinct_map.insert(NodeID(node,cur_id),  read_offset + sizeof(uint32_t));
//STDOUT_PRINT("Inserting distinct node (%u,%u): %lu\n", node, cur_id, read_offset+sizeof(uint32_t));
          uint32_t len = num_leaf_descendents(node, num_nodes);
          read_offset += sizeof(uint32_t) + chunk_size*len;
        }
      });
      uint32_t num_repeat = chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size;
      Kokkos::parallel_for("Fill repeat map", Kokkos::RangePolicy<>(0, num_repeat), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
        uint32_t prev;
        memcpy(&node, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
        memcpy(&prev, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
        repeat_map.insert(node, NodeID(prev, ref_id));
//STDOUT_PRINT("Inserting repeat node %u: (%u,%u)\n", node, prev, ref_id);
      });
      Kokkos::parallel_for("Fill data", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
        if(node_list(i).tree == current_id) {
          NodeID id = node_list(i);
          if(distinct_map.exists(id)) {
            size_t offset = distinct_map.value_at(distinct_map.find(id));
            uint32_t start = leftmost_leaf(id.node, num_nodes);
            uint32_t len = num_leaf_descendents(id.node, num_nodes);
            uint32_t end = start+len-1;
            uint32_t writesize = chunk_size*len;
            if(end == num_nodes-1)
              writesize = datalen-(start-num_chunks+1)*chunk_size;
            memcpy(data.data()+chunk_size*(i), distinct.data()+offset, writesize);
//STDOUT_PRINT("Updating distinct node %u with (%u,%u). Start: %u, Len: %u, End: %u, writesize: %u, %c\n", i, id.node, id.tree, start, len, end, writesize, *((char*)(data.data()+chunk_size*i)));
          } else if(repeat_map.exists(id.node)) {
            NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
//printf("Found repeat node: %u: (%u,%u)\n", id.node, prev.node, prev.tree);
            if(prev.tree == current_id) {
              size_t offset = distinct_map.value_at(distinct_map.find(prev));
              uint32_t start = leftmost_leaf(prev.node, num_nodes);
              uint32_t len = num_leaf_descendents(prev.node, num_nodes);
              uint32_t writesize = chunk_size*len;
              if(datalen <= (start-num_chunks+1)*chunk_size+len*chunk_size)
                writesize = datalen-(start-num_chunks+1)*chunk_size;
              memcpy(data.data()+chunk_size*(i), distinct.data()+offset, writesize); 
//STDOUT_PRINT("Updating repeat node %u with (%u,%u). Start: %u, Len: %u, writesize: %u, %c\n", i, id.node, id.tree, start, len, writesize, *((char*)(data.data()+chunk_size*i)));
            } else {
              node_list(i) = prev;
            }
          } else {
            node_list(i) = NodeID(node_list(i).node, current_id-1);
          }
        }
      });
    }
    Kokkos::fence();
//Kokkos::deep_copy(data_h, data);
//for(uint32_t idx=0; idx<data_h.size()/2; idx++) {
//  printf("\"%c%c\"", *((char*)(data_h.data()+chunk_size*idx)), *((char*)(data_h.data()+chunk_size*idx+1)));
//}
//printf("\n");
  } else {
    // Main checkpoint
printf("Global checkpoint\n");
    file.open(chkpt_files[file_idx], std::ifstream::in | std::ifstream::binary);
    file.read((char*)(buffer_h.data()), filesize);
    file.close();
    Kokkos::deep_copy(buffer_d, buffer_h);
    Kokkos::View<NodeID*> node_list("List of NodeIDs", num_chunks);
    Kokkos::deep_copy(node_list, NodeID());
    uint32_t ref_id = header.ref_id;
    uint32_t cur_id = header.chkpt_id;
    size_t prev_repeat_offset = filesize - header.prev_repeat_size*(sizeof(NodeID)+sizeof(uint32_t));
    size_t curr_repeat_offset = prev_repeat_offset - header.curr_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
    auto curr_repeat   = Kokkos::subview(buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    auto prev_repeat   = Kokkos::subview(buffer_d, std::make_pair(prev_repeat_offset, filesize));
    auto distinct = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
printf("Read file and setup subviews\n");

Kokkos::View<uint64_t[1]> counter_d("Write counter");
auto counter_h = Kokkos::create_mirror_view(counter_d);
Kokkos::deep_copy(counter_d, 0);

    uint32_t chunk_size = header.chunk_size;
    size_t datalen = header.datalen;
    Kokkos::UnorderedMap<NodeID, size_t> distinct_map(num_nodes);
    uint32_t num_distinct = header.distinct_size;
    Kokkos::parallel_for("Restart distinct", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const uint32_t index) {
      size_t read_offset = 0;
      for(uint32_t i=0; i<num_distinct; i++) {
        uint32_t node;
        memcpy(&node, distinct.data() + read_offset,  sizeof(uint32_t));
        distinct_map.insert(NodeID(node,cur_id),  read_offset + sizeof(uint32_t));
        uint32_t start = leftmost_leaf(node, num_nodes);
        uint32_t len = num_leaf_descendents(node, num_nodes);
        uint32_t end = start+len-1;
//STDOUT_PRINT("Inserting node %u with length %u\n", node, len);
        uint32_t left = 2*node+1;
        uint32_t right = 2*node+2;
        while(left < num_nodes) {
          if(right > num_nodes)
            right = num_nodes;
          for(uint32_t u=left; u<=right; u++) {
            uint32_t leaf = leftmost_leaf(u, num_nodes);
            auto result = distinct_map.insert(NodeID(u, cur_id), read_offset+sizeof(uint32_t)+(leaf-start)*chunk_size);
            if(result.failed())
              printf("Distinct: Failed to insert (%u,%u): %lu\n", u, cur_id, read_offset+sizeof(uint32_t)+(leaf-start)*chunk_size);
          }
          left = 2*left+1;
          right = 2*right+2;
        }
        for(uint32_t j=0; j<len; j++) {
//          node_list(start-num_chunks+1+j) = NodeID(start-num_chunks+1+j, cur_id);
          node_list(start-num_chunks+1+j) = NodeID(start+j, cur_id);
        }
        uint32_t datasize = len*chunk_size;
Kokkos::atomic_add(&counter_d(0), datasize);
        if(end == num_nodes-1)
          datasize = datalen - (start-num_chunks+1)*chunk_size;
//        STDOUT_PRINT("Copying %c to position %u, len: %u\n", (char)(*(distinct.data()+read_offset+sizeof(uint32_t))), chunk_size*(start-num_chunks+1), datasize);
        memcpy(data.data()+chunk_size*(start-num_chunks+1), distinct.data()+read_offset+sizeof(uint32_t), datasize);
//STDOUT_PRINT("Replacing %u chunks for distinct node %u. %lu to %u (%u bytes)\n", len, node, read_offset+sizeof(uint32_t), chunk_size*(start-num_chunks+1), datasize);
        read_offset += sizeof(uint32_t) + chunk_size*len;
      }
    });
//Kokkos::parallel_for("Find non leaf entries", Kokkos::RangePolicy<>(0,node_list.size()), KOKKOS_LAMBDA(const uint32_t i) {
//  if(node_list(i).node < num_chunks-1) {
//    STDOUT_PRINT("Found non leaf entry %u: (%u,%u)\n", i, node_list(i).node, node_list(i).tree);
//  }
//});
    Kokkos::fence();
Kokkos::deep_copy(counter_h, counter_d);
printf("Bytes written for distinct: %lu\n", counter_h(0));
Kokkos::deep_copy(counter_d, 0);
    Kokkos::parallel_for("Restart Hashlist current repeat", Kokkos::RangePolicy<>(0, header.curr_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      memcpy(&node, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
//STDOUT_PRINT("Current repeat: %u:%u\n", node, prev);
if(!distinct_map.exists(NodeID(prev, cur_id)))
STDOUT_PRINT("Couldn't find repeat node %u: (%u,%u)\n", node, prev, cur_id);
      uint32_t idx = distinct_map.find(NodeID(prev, cur_id));
      if(!distinct_map.valid_at(idx))
        STDOUT_PRINT("Main: Failed to find (%u,%u) in distinct map\n", prev, cur_id);
      size_t offset = distinct_map.value_at(distinct_map.find(NodeID(prev, cur_id)));
      uint32_t node_start = leftmost_leaf(node, num_nodes);
      uint32_t prev_start = leftmost_leaf(prev, num_nodes);
      uint32_t len = num_leaf_descendents(prev, num_nodes);
      for(uint32_t j=0; j<len; j++) {
//        node_list(node_start+j-num_chunks+1) = NodeID(prev_start-num_chunks+1+j, cur_id);
        node_list(node_start+j-num_chunks+1) = NodeID(prev_start+j, cur_id);
//        STDOUT_PRINT("Entry %u updated: (%u,%u) (current repeat node %u prev %u)\n", node_start-num_chunks+1+j, prev_start+j, cur_id, node, prev);
      }
      uint32_t copysize = chunk_size*len;
Kokkos::atomic_add(&counter_d(0), copysize);
      if(node_start+len-1 == num_nodes-1)
        copysize = data.size() - chunk_size*(node_start-num_chunks+1);
      memcpy(data.data()+chunk_size*(node_start-num_chunks+1), distinct.data()+offset, copysize);
//STDOUT_PRINT("Replacing %u chunks for current node %u with %u. %lu to %u (%u bytes)\n", len, node, prev, offset, chunk_size*(node_start-num_chunks+1), copysize);
//      STDOUT_PRINT("Duplicated (%u) %c%c to %u (offset %lu)\n", node_start-num_chunks+1, *((char*)(distinct.data()+offset)), *((char*)(distinct.data()+offset+1)),  (node_start-num_chunks+1)*chunk_size, offset);
    });
    Kokkos::fence();
Kokkos::deep_copy(counter_h, counter_d);
printf("Bytes written for current repeats: %lu\n", counter_h(0));
Kokkos::deep_copy(counter_d, 0);
    Kokkos::parallel_for("Restart Hashlist previous repeat", Kokkos::RangePolicy<>(0, header.prev_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      NodeID prev;
      memcpy(&node, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(NodeID)), sizeof(uint32_t));
      memcpy(&prev, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(NodeID)) +sizeof(uint32_t), sizeof(NodeID));
//      STDOUT_PRINT("Node: %u, Prev: (%u,%u)\n", node, prev.node, prev.tree);
      uint32_t node_start = leftmost_leaf(node, num_nodes);
      uint32_t prev_start = leftmost_leaf(prev.node, num_nodes);
      uint32_t len = num_leaf_descendents(prev.node, num_nodes);
      for(uint32_t j=0; j<len; j++) {
        node_list(node_start-num_chunks+1+j) = NodeID(prev_start+j, prev.tree);
if(prev.tree > 9)
  STDOUT_PRINT("Restart Prior repeat (main): Problematic entry for checkpoint %u. Chunk %u: (%u,%u)\n", cur_id, node_start-num_chunks+1+j, prev_start+j, prev.tree);
//        STDOUT_PRINT("Entry %u updated: (%u,%u) (previous repeat node %u, prev %u)\n", node_start-num_chunks+1, prev_start+j, ref_id, node, prev.node);
      }
    });
    Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      NodeID entry = node_list(i);
      if(entry.node == UINT_MAX) {
        node_list(i) = NodeID(i+num_chunks-1, cur_id-1);
      }
    });
    Kokkos::fence();
//auto node_list_h = Kokkos::create_mirror_view(node_list);
//Kokkos::deep_copy(node_list_h, node_list);
//for(uint32_t i=0; i<node_list_h.size(); i++) {
//  printf("Node %u: (%u,%u)\n", i, node_list_h(i).node, node_list_h(i).tree);
//}
//auto data_h = Kokkos::create_mirror_view(data);
//Kokkos::deep_copy(data_h, data);
//for(uint32_t i=0; i<data_h.size(); i++) {
//  if(node_list_h(i).tree == cur_id) {
//    printf("%c", *((char*)(data_h.data()+i)));
//  } else {
//    printf("\"*\"");
//  }
//}
//printf("\n");
STDOUT_PRINT("Post main pre history\n");
//Kokkos::parallel_for("Find non missing entries", Kokkos::RangePolicy<>(0,node_list.size()), KOKKOS_LAMBDA(const uint32_t i) {
//  if(node_list(i).tree < UINT32_MAX-1 && node_list(i).tree > 9) {
//    STDOUT_PRINT("Found skipped entry %u: (%u,%u)\n", i, node_list(i).node, node_list(i).tree);
//  }
//});

    for(int idx=static_cast<int>(file_idx)-1; idx>static_cast<int>(ref_id); idx--) {
      printf("Processing checkpoint %u\n", idx);
      file.open(chkpt_files[idx], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
      size_t chkpt_size = file.tellg();
      STDOUT_PRINT("Checkpoint size: %zd\n", chkpt_size);
      file.seekg(0);
      Kokkos::View<uint8_t*> chkpt_buffer_d("Checkpoint buffer", chkpt_size);
      auto chkpt_buffer_h = Kokkos::create_mirror_view(chkpt_buffer_d);
      file.read((char*)(chkpt_buffer_h.data()), chkpt_size);
      file.close();
      header_t chkpt_header;
      memcpy(&chkpt_header, chkpt_buffer_h.data(), sizeof(header_t));
      uint32_t current_id = chkpt_header.chkpt_id;
      datalen = chkpt_header.datalen;
      chunk_size = chkpt_header.chunk_size;
      Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
      ref_id = chkpt_header.ref_id;
      cur_id = chkpt_header.chkpt_id;
      STDOUT_PRINT("Cur ID: %u\n", cur_id);
      STDOUT_PRINT("Ref ID: %u\n", ref_id);
      STDOUT_PRINT("Num distinct: %u\n", chkpt_header.distinct_size);
      STDOUT_PRINT("Num current: %u\n", chkpt_header.curr_repeat_size);
      STDOUT_PRINT("Num previous: %u\n", chkpt_header.prev_repeat_size);
      prev_repeat_offset = chkpt_size - chkpt_header.prev_repeat_size*(sizeof(uint32_t)+sizeof(NodeID));
      curr_repeat_offset = prev_repeat_offset - chkpt_header.curr_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
      curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
      prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, chkpt_size));
      distinct = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
      num_distinct = chkpt_header.distinct_size;
      
      distinct_map.clear();
      distinct_map.rehash(num_nodes);
//      Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
      Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(2*num_nodes-1);
      Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const uint32_t index) {
        size_t read_offset = 0;
        for(uint32_t i=0; i<num_distinct; i++) {
          uint32_t node;
          memcpy(&node, distinct.data() + read_offset,  sizeof(uint32_t));
          uint32_t start = leftmost_leaf(node, num_nodes);
          uint32_t len = num_leaf_descendents(node, num_nodes);
          uint32_t end = rightmost_leaf(node, num_nodes);
          uint32_t left = 2*node+1;
          uint32_t right = 2*node+2;
          distinct_map.insert(NodeID(node,cur_id),  read_offset + sizeof(uint32_t));
//STDOUT_PRINT("Found distinct node %u with length %u\n", node, len);
          while(left < num_nodes) {
            if(right >= num_nodes)
              right = num_nodes;
            for(uint32_t u=left; u<=right; u++) {
              uint32_t leaf = leftmost_leaf(u, num_nodes);
              auto result = distinct_map.insert(NodeID(u, cur_id), read_offset+sizeof(uint32_t)+(leaf-start)*chunk_size);
              if(result.failed())
                printf("Failed to insert (%u,%u): %lu\n", u, cur_id, read_offset+sizeof(uint32_t)+(leaf-start)*chunk_size);
            }
            left = 2*left+1;
            right = 2*right+2;
          }
          read_offset += sizeof(uint32_t) + chunk_size*len;
        }
      });
      Kokkos::parallel_for("Fill current repeat map", Kokkos::RangePolicy<>(0, chkpt_header.curr_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
        uint32_t prev;
        memcpy(&node, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
        memcpy(&prev, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
        auto result = repeat_map.insert(node, NodeID(prev, cur_id));
        uint32_t curr_start = leftmost_leaf(node, num_nodes);
        uint32_t prev_start = leftmost_leaf(prev, num_nodes);
        uint32_t len = num_leaf_descendents(node, num_nodes);
        for(uint32_t u=0; u<len; u++) {
          repeat_map.insert(curr_start+u, NodeID(prev_start+u, cur_id));
        }
//        uint32_t left = 2*node+1;
//        uint32_t right = 2*node+2;
//        while(left < num_nodes) {
//          if(right >= num_nodes)
//            right = num_nodes;
//          for(uint32_t u=left; u<=right; u++) {
//          }
//        }
        if(result.failed())
          STDOUT_PRINT("Failed to insert current repeat %u: (%u,%u) into repeat map\n", node, prev, cur_id);
//STDOUT_PRINT("Found current repeat %u: (%u,%u)\n", node, prev, cur_id);
      });
      Kokkos::parallel_for("Fill previous repeat map", Kokkos::RangePolicy<>(0, chkpt_header.prev_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
        NodeID prev;
        memcpy(&node, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(NodeID)), sizeof(uint32_t));
        memcpy(&prev, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(NodeID))+sizeof(uint32_t), sizeof(NodeID));
        auto result = repeat_map.insert(node, prev);
        if(result.failed())
          STDOUT_PRINT("Failed to insert previous repeat %u: (%u,%u) into repeat map\n", node, prev.node, prev.tree);
        uint32_t curr_start = leftmost_leaf(node, num_nodes);
        uint32_t prev_start = leftmost_leaf(prev.node, num_nodes);
        uint32_t len = num_leaf_descendents(node, num_nodes);
        for(uint32_t u=0; u<len; u++) {
          repeat_map.insert(curr_start+u, NodeID(prev_start+u, prev.tree));
        }
//STDOUT_PRINT("Found prior repeat %u: (%u,%u)\n", node, prev.node, prev.tree);
      });
      Kokkos::parallel_for("Fill data", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
        if(node_list(i).tree == current_id) {
          NodeID id = node_list(i);
          if(distinct_map.exists(id)) {
            uint32_t len = num_leaf_descendents(id.node, num_nodes);
            uint32_t start = leftmost_leaf(id.node, num_nodes);
            uint32_t end = rightmost_leaf(id.node, num_nodes);
            size_t offset = distinct_map.value_at(distinct_map.find(id));
            uint32_t writesize = chunk_size;
            if(i*chunk_size+writesize > datalen) 
              writesize = datalen-i*chunk_size;
            memcpy(data.data()+chunk_size*i, distinct.data()+offset, writesize);
Kokkos::atomic_add(&counter_d(0), writesize);
//node_list(i) = NodeID(id.node, UINT32_MAX-1);
//            uint32_t writesize = chunk_size*len;
//Kokkos::atomic_add(&counter_d(0), writesize);
//            if(end == num_nodes-1)
//              writesize = datalen-(start-num_chunks+1)*chunk_size;
//if(!distinct_map.exists(id))
//printf("Failed to find distinct chunk %u: (%u,%u)\n", i, id.node, id.tree);
//            memcpy(data.data()+chunk_size*i, distinct.data()+offset, writesize);
          } else if(repeat_map.exists(id.node)) {
            NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
//printf("Repaeat value: %u: (%u,%u)\n", id.node, prev.node, prev.tree);
            if(prev.tree == current_id) {
if(!repeat_map.exists(id.node))
printf("Failed to find repeat chunk %u\n", id.node);
              size_t offset = distinct_map.value_at(distinct_map.find(prev));
              uint32_t len = num_leaf_descendents(prev.node, num_nodes);
              uint32_t start = leftmost_leaf(prev.node, num_nodes);
              uint32_t writesize = chunk_size;
              if(i*chunk_size+writesize > datalen) 
                writesize = datalen-i*chunk_size;
              memcpy(data.data()+chunk_size*i, distinct.data()+offset, writesize);
Kokkos::atomic_add(&counter_d(0), writesize);
//node_list(i) = NodeID(id.node, UINT32_MAX-1);
//              uint32_t writesize = chunk_size*len;
//Kokkos::atomic_add(&counter_d(0), writesize);
//              if((start+len-1-num_chunks+1)*chunk_size >= datalen)
//                writesize = datalen-(start-num_chunks+1)*chunk_size;
//              memcpy(data.data()+chunk_size*i, distinct.data()+offset, writesize); 
            } else {
              node_list(i) = prev;
if(node_list(i).tree > 9 && node_list(i).tree < UINT32_MAX)
  STDOUT_PRINT("Restart Prior repeat (main): Problematic entry for checkpoint %u. Chunk %u: (%u,%u) (update prior) \n", cur_id, i, node_list(i).node, node_list(i).tree);
            }
          } else {
            node_list(i) = NodeID(node_list(i).node, current_id-1);
if(node_list(i).tree > 9 && node_list(i).tree < UINT32_MAX)
  STDOUT_PRINT("Restart Prior repeat (main): Problematic entry for checkpoint %u. Chunk %u: (%u,%u) (update identical)\n", cur_id, i, node_list(i).node, node_list(i).tree);
          }
        } else if(node_list(i).tree > 9 && node_list(i).tree < UINT32_MAX-1) {
  STDOUT_PRINT("Restart Prior repeat (main): Problematic entry for checkpoint %u. Chunk %u: (%u,%u)\n", cur_id, i, node_list(i).node, node_list(i).tree);
        }
      });
    Kokkos::fence();
Kokkos::deep_copy(counter_h, counter_d);
printf("Bytes written for checkpoint %u: %lu\n", cur_id, counter_h(0));
Kokkos::deep_copy(counter_d, 0);
//Kokkos::deep_copy(node_list_h, node_list);
//for(uint32_t i=0; i<node_list_h.size(); i++) {
//  printf("Node %u: (%u,%u)\n", i, node_list_h(i).node, node_list_h(i).tree);
//}
    }
STDOUT_PRINT("Post history pre baseline\n");
//Kokkos::parallel_for("Find non missing entries", Kokkos::RangePolicy<>(0,node_list.size()), KOKKOS_LAMBDA(const uint32_t i) {
//  if(node_list(i).tree != UINT32_MAX-1 && node_list(i).tree > 9) {
//    STDOUT_PRINT("Found skipped entry %u: (%u,%u)\n", i, node_list(i).node, node_list(i).tree);
//  }
//});

    // Reference
    file.open(chkpt_files[header.ref_id], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
    size_t chkpt_size = file.tellg();
    file.seekg(0);
    Kokkos::View<uint8_t*> chkpt_buffer_d("Checkpoint buffer", chkpt_size);
    auto chkpt_buffer_h = Kokkos::create_mirror_view(chkpt_buffer_d);
    file.read((char*)(chkpt_buffer_h.data()), chkpt_size);
    file.close();
    header_t chkpt_header;
    memcpy(&chkpt_header, chkpt_buffer_h.data(), sizeof(header_t));
    uint32_t current_id = chkpt_header.chkpt_id;
    datalen = chkpt_header.datalen;
    chunk_size = chkpt_header.chunk_size;
    Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
    ref_id = chkpt_header.ref_id;
    cur_id = chkpt_header.chkpt_id;
    prev_repeat_offset = chkpt_size - chkpt_header.prev_repeat_size*(2*sizeof(uint32_t));
    curr_repeat_offset = prev_repeat_offset - chkpt_header.curr_repeat_size*(2*sizeof(uint32_t));
    curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, chkpt_size));
    distinct = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
    
    distinct_map.clear();
    distinct_map.rehash(chkpt_header.distinct_size);
    Kokkos::fence();
    Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
    Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, distinct.data()+i*(sizeof(uint32_t)+chunk_size), sizeof(uint32_t));
      auto result = distinct_map.insert(NodeID(node, current_id), i*(sizeof(uint32_t)+chunk_size)+sizeof(uint32_t));
      if(result.failed())
        STDOUT_PRINT("Reference: Failed to insert distinct node (%u,%u): %lu\n", node, cur_id, i*(sizeof(uint32_t)+chunk_size)+sizeof(uint32_t));
//STDOUT_PRINT("Reference: Inserted distinct node (%u,%u): %lu\n", node, current_id, i*(sizeof(uint32_t)+chunk_size)+sizeof(uint32_t));
    });
    uint32_t num_repeat = chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size;
    Kokkos::parallel_for("Fill repeat map", Kokkos::RangePolicy<>(0, num_repeat), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      memcpy(&node, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
      auto result = repeat_map.insert(node, NodeID(prev, ref_id));
      if(result.failed())
        STDOUT_PRINT("Reference: Failed to insert repeat node %u: (%u,%u)\n", node, prev, ref_id);
//STDOUT_PRINT("Reference: Inserted repeat node %u: (%u,%u): %lu\n", node, prev, ref_id, i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t));
    });
    Kokkos::parallel_for("Fill data", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      if(node_list(i).tree == current_id) {
        NodeID id = node_list(i);
        if(distinct_map.exists(id)) {
          uint32_t len = num_leaf_descendents(id.node, num_nodes);
          uint32_t start = leftmost_leaf(id.node, num_nodes);
          size_t offset = distinct_map.value_at(distinct_map.find(id));
          uint32_t writesize = chunk_size*len;
Kokkos::atomic_add(&counter_d(0), writesize);
          if((start+len-1-num_chunks+1)*chunk_size >= datalen)
            writesize = datalen-(start-num_chunks+1)*chunk_size;
          memcpy(data.data()+chunk_size*i, distinct.data()+offset, writesize);
//node_list(i) = NodeID(id.node, UINT32_MAX-1);
//STDOUT_PRINT("Copied distinct %u (%c) to chunk %u. %lu to %u (%u bytes)\n", id.node, *((char*)(distinct.data()+offset)), i, offset, chunk_size*i, writesize);
        } else if(repeat_map.exists(id.node)) {
          NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
          if(prev.tree == current_id) {
            size_t offset = distinct_map.value_at(distinct_map.find(prev));
            
            uint32_t len = num_leaf_descendents(prev.node, num_nodes);
            uint32_t start = leftmost_leaf(prev.node, num_nodes);
            uint32_t writesize = chunk_size*len;
Kokkos::atomic_add(&counter_d(0), writesize);
            if((start+len-1-num_chunks+1)*chunk_size >= datalen)
              writesize = datalen-(start-num_chunks+1)*chunk_size;

            memcpy(data.data()+chunk_size*i, distinct.data()+offset, writesize); 
//node_list(i) = NodeID(id.node, UINT32_MAX-1);
//STDOUT_PRINT("Copied repeat %u,%u (%c) to chunk %u. %lu to %u (%u bytes)\n", id.node, prev.node, *((char*)(distinct.data()+offset)), i, offset, i*chunk_size, writesize);
          } else {
            node_list(i) = prev;
          }
        } else {
          node_list(i) = NodeID(node_list(i).node, current_id-1);
        }
      }
    });
    Kokkos::fence();
Kokkos::deep_copy(counter_h, counter_d);
printf("Bytes written for checkpoint %u: %lu\n", cur_id, counter_h(0));
Kokkos::deep_copy(counter_d, 0);
//Kokkos::parallel_for("Find non missing entries", Kokkos::RangePolicy<>(0,node_list.size()), KOKKOS_LAMBDA(const uint32_t i) {
//  if(node_list(i).tree != UINT32_MAX-1) {
//    STDOUT_PRINT("Found skipped entry %u: (%u,%u)\n", i, node_list(i).node, node_list(i).tree);
//  }
//});
    Kokkos::fence();
//Kokkos::deep_copy(data_h, data);
//for(int i=0; i<data_h.size(); i++) {
// STDOUT_PRINT("%c", *((char*)(data_h.data()+i)));
//}
//STDOUT_PRINT("\n");
  }
  Kokkos::fence();
  STDOUT_PRINT("Restarted checkpoint\n");
  return 0;
}

std::pair<uint64_t,uint64_t> 
write_incr_chkpt_hashtree_local_mode( const std::string& filename, 
                                const Kokkos::View<uint8_t*>& data, 
                                Kokkos::View<uint8_t*>& buffer_d,
                                uint32_t chunk_size, 
                                const DistinctNodeIDMap& distinct, 
                                const SharedNodeIDMap& shared,
                                uint32_t prior_chkpt_id,
                                uint32_t chkpt_id,
                                header_t& header) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size()) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;

  // Write whether we are storing the hashes, length full checkpoint, chunk size, number of repeat chunks, number of distinct chunks
  DEBUG_PRINT("Wrote header\n");
  uint32_t distinct_size = 0;
  Kokkos::parallel_reduce("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i, uint32_t& sum) {
    if(distinct.valid_at(i)) {
      auto info = distinct.value_at(i);
      if(info.node >= num_chunks-1) {
        sum += sizeof(uint32_t)+chunk_size;
      }
    }
  }, distinct_size);
  uint32_t repeat_size = 0;
  Kokkos::parallel_reduce("Count shared updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i, uint32_t& sum) {
    if(shared.valid_at(i)) {
      auto node = shared.key_at(i);
      if(node >= num_chunks-1) {
        sum += sizeof(uint32_t)*2;
      }
    }
  }, repeat_size);
  Kokkos::fence();
  buffer_d = Kokkos::View<uint8_t*>("Buffer", repeat_size + distinct_size);
  Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
  Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
  Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
  Kokkos::View<uint64_t[1]> num_distinct_d("Number of distinct entries");
  Kokkos::View<uint64_t[1]>::HostMirror num_distinct_h = Kokkos::create_mirror_view(num_distinct_d);
  Kokkos::deep_copy(num_distinct_d, 0);
  Kokkos::View<uint64_t[1]> num_curr_repeat_d("Number of curr repeat entries");
  Kokkos::View<uint64_t[1]>::HostMirror num_curr_repeat_h = Kokkos::create_mirror_view(num_curr_repeat_d);
  Kokkos::deep_copy(num_curr_repeat_d, 0);
  Kokkos::View<uint64_t[1]> num_prev_repeat_d("Number of prev repeat entries");
  Kokkos::View<uint64_t[1]>::HostMirror num_prev_repeat_h = Kokkos::create_mirror_view(num_prev_repeat_d);
  Kokkos::deep_copy(num_prev_repeat_d, 0);
  Kokkos::deep_copy(num_bytes_d, 0);
  Kokkos::deep_copy(num_bytes_data_d, 0);
  Kokkos::deep_copy(num_bytes_metadata_d, 0);
  DEBUG_PRINT("Setup counters and buffers\n");
  DEBUG_PRINT("Distinct capacity: %u, size: %u\n", distinct.capacity(), distinct.size());
  DEBUG_PRINT("Repeat capacity: %u, size: %u\n", shared.capacity(), shared.size());

  Kokkos::parallel_for("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      auto info = distinct.value_at(i);
      if(info.node >= num_chunks-1) {
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
        Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + chunk_size);
        memcpy(buffer_d.data()+pos, &info.node, sizeof(uint32_t));
        uint32_t writesize = chunk_size;
        if(info.node == num_nodes-1) {
          writesize = data.size()-(info.node-num_chunks+1)*chunk_size;
        }
        memcpy(buffer_d.data()+pos+sizeof(uint32_t), data.data()+chunk_size*(info.node-num_chunks+1), writesize);
        Kokkos::atomic_add(&num_distinct_d(0), 1);
//printf("Writing distinct chunk: %u at %lu\n", info.node, pos);
      }
    }
  });
Kokkos::View<uint32_t[1]> counter_d("Counter");
Kokkos::deep_copy(counter_d, 0);
auto counter_h = Kokkos::create_mirror_view(counter_d);
  Kokkos::parallel_for("Count curr repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      uint32_t k = shared.key_at(i);
      if(k >= num_chunks-1) {
        NodeID v = shared.value_at(i);
        if(v.tree == chkpt_id) {
          Kokkos::atomic_add(&num_curr_repeat_d(0), 1);
          Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
          uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
          memcpy(buffer_d.data()+pos, &k, sizeof(uint32_t));
          memcpy(buffer_d.data()+pos+sizeof(uint32_t), &v.node, sizeof(uint32_t));
Kokkos::atomic_add(&counter_d(0), 1);
//printf("Writing current repeat chunk: %u:%u at %lu\n", k, v.node, pos);
        }
      }
    }
  });
Kokkos::deep_copy(counter_h, counter_d);
DEBUG_PRINT("Number of current repeats: %u\n", counter_h(0));
  Kokkos::parallel_for("Count prior repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      uint32_t k = shared.key_at(i);
      if(k >= num_chunks-1) {
        NodeID v = shared.value_at(i);
        if(v.tree != chkpt_id) {
          Kokkos::atomic_add(&num_prev_repeat_d(0), 1);
          Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
          uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
          memcpy(buffer_d.data()+pos, &k, sizeof(uint32_t));
          memcpy(buffer_d.data()+pos+sizeof(uint32_t), &v.node, sizeof(uint32_t));
        }
      }
    }
  });
  Kokkos::fence();
  Kokkos::deep_copy(num_distinct_h, num_distinct_d);
  Kokkos::deep_copy(num_curr_repeat_h, num_curr_repeat_d);
  Kokkos::deep_copy(num_prev_repeat_h, num_prev_repeat_d);
  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
  Kokkos::fence();
  header.ref_id = prior_chkpt_id;
  header.chkpt_id = chkpt_id;
  header.datalen = data.size();
  header.chunk_size = chunk_size;
  header.window_size = 0;
  header.distinct_size = num_distinct_h(0);
  header.curr_repeat_size = num_curr_repeat_h(0);
  header.prev_repeat_size = num_prev_repeat_h(0);
  STDOUT_PRINT("Buffer size: %lu\n", buffer_d.size());
  STDOUT_PRINT("Ref ID: %u\n"          , header.ref_id);
  STDOUT_PRINT("Chkpt ID: %u\n"        , header.chkpt_id);
  STDOUT_PRINT("Data len: %lu\n"       , header.datalen);
  STDOUT_PRINT("Chunk size: %u\n"      , header.chunk_size);
  STDOUT_PRINT("Window size: %u\n"     , header.window_size);
  STDOUT_PRINT("Distinct size: %u\n"   , header.distinct_size);
  STDOUT_PRINT("Curr repeat size: %u\n", header.curr_repeat_size);
  STDOUT_PRINT("prev repeat size: %u\n", header.prev_repeat_size);
  DEBUG_PRINT("Copied data to host\n");
  STDOUT_PRINT("Number of bytes written for compact incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_h(0));
  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));
  return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
}

std::pair<uint64_t,uint64_t> 
write_incr_chkpt_hashtree_global_mode( const std::string& filename, 
                                const Kokkos::View<uint8_t*>& data, 
                                Kokkos::View<uint8_t*>& buffer_d,
                                uint32_t chunk_size, 
                                const DistinctNodeIDMap& distinct, 
                                const SharedNodeIDMap& shared,
                                uint32_t prior_chkpt_id,
                                uint32_t chkpt_id,
                                header_t& header) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size()) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;

  // Write whether we are storing the hashes, length full checkpoint, chunk size, number of repeat chunks, number of distinct chunks
  DEBUG_PRINT("Wrote header\n");
  uint32_t distinct_size = 0;
  Kokkos::parallel_reduce("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i, uint32_t& sum) {
    if(distinct.valid_at(i)) {
      auto info = distinct.value_at(i);
      if(info.node >= num_chunks-1) {
        sum += sizeof(uint32_t)+chunk_size;
      }
    }
  }, distinct_size);
  uint32_t repeat_size = 0;
  Kokkos::parallel_reduce("Count shared updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i, uint32_t& sum) {
    if(shared.valid_at(i)) {
      auto node = shared.key_at(i);
      if(node >= num_chunks-1) {
        sum += sizeof(uint32_t)*2;
      }
    }
  }, repeat_size);
  Kokkos::fence();
  buffer_d = Kokkos::View<uint8_t*>("Buffer", repeat_size + distinct_size);
  Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
  Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
  Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
  Kokkos::View<uint64_t[1]> num_distinct_d("Number of distinct entries");
  Kokkos::View<uint64_t[1]>::HostMirror num_distinct_h = Kokkos::create_mirror_view(num_distinct_d);
  Kokkos::deep_copy(num_distinct_d, 0);
  Kokkos::View<uint64_t[1]> num_curr_repeat_d("Number of curr repeat entries");
  Kokkos::View<uint64_t[1]>::HostMirror num_curr_repeat_h = Kokkos::create_mirror_view(num_curr_repeat_d);
  Kokkos::deep_copy(num_curr_repeat_d, 0);
  Kokkos::View<uint64_t[1]> num_prev_repeat_d("Number of prev repeat entries");
  Kokkos::View<uint64_t[1]>::HostMirror num_prev_repeat_h = Kokkos::create_mirror_view(num_prev_repeat_d);
  Kokkos::deep_copy(num_prev_repeat_d, 0);
  Kokkos::deep_copy(num_bytes_d, 0);
  Kokkos::deep_copy(num_bytes_data_d, 0);
  Kokkos::deep_copy(num_bytes_metadata_d, 0);
  DEBUG_PRINT("Setup counters and buffers\n");
  DEBUG_PRINT("Distinct capacity: %u, size: %u\n", distinct.capacity(), distinct.size());
  DEBUG_PRINT("Repeat capacity: %u, size: %u\n", shared.capacity(), shared.size());

  Kokkos::parallel_for("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      auto info = distinct.value_at(i);
      if(info.node >= num_chunks-1) {
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
        Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + chunk_size);
        memcpy(buffer_d.data()+pos, &info.node, sizeof(uint32_t));
        uint32_t writesize = chunk_size;
        if(info.node == num_nodes-1) {
          writesize = data.size()-(info.node-num_chunks+1)*chunk_size;
        }
        memcpy(buffer_d.data()+pos+sizeof(uint32_t), data.data()+chunk_size*(info.node-num_chunks+1), writesize);
        Kokkos::atomic_add(&num_distinct_d(0), 1);
//printf("Writing distinct chunk: %u at %lu\n", info.node, pos);
      }
    }
  });
Kokkos::View<uint32_t[1]> counter_d("Counter");
Kokkos::deep_copy(counter_d, 0);
auto counter_h = Kokkos::create_mirror_view(counter_d);
  Kokkos::parallel_for("Count curr repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      uint32_t k = shared.key_at(i);
      if(k >= num_chunks-1) {
        NodeID v = shared.value_at(i);
        if(v.tree == chkpt_id) {
          Kokkos::atomic_add(&num_curr_repeat_d(0), 1);
          Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
          uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
          memcpy(buffer_d.data()+pos, &k, sizeof(uint32_t));
          memcpy(buffer_d.data()+pos+sizeof(uint32_t), &v.node, sizeof(uint32_t));
Kokkos::atomic_add(&counter_d(0), 1);
//printf("Writing current repeat chunk: %u:%u at %lu\n", k, v.node, pos);
        }
      }
    }
  });
Kokkos::deep_copy(counter_h, counter_d);
DEBUG_PRINT("Number of current repeats: %u\n", counter_h(0));
  Kokkos::parallel_for("Count prior repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      uint32_t k = shared.key_at(i);
      if(k >= num_chunks-1) {
        NodeID v = shared.value_at(i);
        if(v.tree != chkpt_id) {
          Kokkos::atomic_add(&num_prev_repeat_d(0), 1);
          Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
          uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
          memcpy(buffer_d.data()+pos, &k, sizeof(uint32_t));
          memcpy(buffer_d.data()+pos+sizeof(uint32_t), &v.node, sizeof(uint32_t));
        }
      }
    }
  });
  Kokkos::fence();
  Kokkos::deep_copy(num_distinct_h, num_distinct_d);
  Kokkos::deep_copy(num_curr_repeat_h, num_curr_repeat_d);
  Kokkos::deep_copy(num_prev_repeat_h, num_prev_repeat_d);
  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
  Kokkos::fence();
  header.ref_id = prior_chkpt_id;
  header.chkpt_id = chkpt_id;
  header.datalen = data.size();
  header.chunk_size = chunk_size;
  header.window_size = 0;
  header.distinct_size = num_distinct_h(0);
  header.curr_repeat_size = num_curr_repeat_h(0);
  header.prev_repeat_size = num_prev_repeat_h(0);
  STDOUT_PRINT("Buffer size: %lu\n", buffer_d.size());
  STDOUT_PRINT("Ref ID: %u\n"          , header.ref_id);
  STDOUT_PRINT("Chkpt ID: %u\n"        , header.chkpt_id);
  STDOUT_PRINT("Data len: %lu\n"       , header.datalen);
  STDOUT_PRINT("Chunk size: %u\n"      , header.chunk_size);
  STDOUT_PRINT("Window size: %u\n"     , header.window_size);
  STDOUT_PRINT("Distinct size: %u\n"   , header.distinct_size);
  STDOUT_PRINT("Curr repeat size: %u\n", header.curr_repeat_size);
  STDOUT_PRINT("prev repeat size: %u\n", header.prev_repeat_size);
  DEBUG_PRINT("Copied data to host\n");
  STDOUT_PRINT("Number of bytes written for compact incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_h(0));
  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));
  return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
}

std::pair<uint64_t,uint64_t> 
write_incr_chkpt_hashtree_local_mode( const std::string& filename, 
                                const Kokkos::View<uint8_t*>& data, 
                                Kokkos::View<uint8_t*>& buffer_d, 
                                uint32_t chunk_size, 
                                const CompactTable& distinct, 
                                const CompactTable& shared,
                                uint32_t prior_chkpt_id,
                                uint32_t chkpt_id,
                                header_t& header) {
  DEBUG_PRINT("File: %s\n", filename.c_str());
  
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size()) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;

  Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
  Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
  Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
  Kokkos::View<uint64_t[1]> num_curr_repeat_d("Number of curr repeat entries");
  Kokkos::View<uint64_t[1]>::HostMirror num_curr_repeat_h = Kokkos::create_mirror_view(num_curr_repeat_d);
  Kokkos::deep_copy(num_curr_repeat_d, 0);
  Kokkos::deep_copy(num_bytes_d, 0);
  Kokkos::deep_copy(num_bytes_data_d, 0);
  Kokkos::deep_copy(num_bytes_metadata_d, 0);
  DEBUG_PRINT("Setup Views\n");

  DEBUG_PRINT("Wrote shared metadata\n");
  Kokkos::View<uint32_t[1]> max_reg("Max region size");
  Kokkos::View<uint32_t[1]>::HostMirror max_reg_h = Kokkos::create_mirror_view(max_reg);
  max_reg_h(0) = 0;
  Kokkos::deep_copy(max_reg, max_reg_h);
  Kokkos::parallel_for("Count distinct bytes", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      uint32_t node = distinct.key_at(i);
      NodeID prev = distinct.value_at(i);
      if(node == prev.node && chkpt_id == prev.tree) {
        uint32_t size = num_leaf_descendents(node, num_nodes);
//        Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t) + size*chunk_size);
        Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t) + size*chunk_size);
        Kokkos::atomic_max(&max_reg(0), size);
      } else {
//        Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
        Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t));
      }
    }
  });

  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
  num_bytes_h(0) += shared.size()*(sizeof(uint32_t)+sizeof(uint32_t));
  Kokkos::deep_copy(max_reg_h, max_reg);

  STDOUT_PRINT("Length of buffer: %lu\n", num_bytes_h(0));
  buffer_d = Kokkos::View<uint8_t*>("Buffer", num_bytes_h(0));

  Kokkos::deep_copy(num_bytes_d, 0);

  DEBUG_PRINT("Largest region: %u\n", max_reg_h(0));
  if(max_reg_h(0) < 2048) {
STDOUT_PRINT("Using memcpy\n");
    Kokkos::parallel_for("Write distinct bytes", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
      if(distinct.valid_at(i)) {
        uint32_t node = distinct.key_at(i);
        NodeID prev = distinct.value_at(i);
        if(node == prev.node && chkpt_id == prev.tree) {
          uint32_t size = num_leaf_descendents(node, num_nodes);
          uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
          Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
          Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(size*chunk_size));
          size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + size*chunk_size);
          memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
          uint32_t writesize = chunk_size*size;
          if(start*chunk_size+writesize > data.size())
            writesize = data.size()-start*chunk_size;
          memcpy(buffer_d.data()+pos+sizeof(uint32_t), data.data()+start*chunk_size, writesize);
//STDOUT_PRINT("Wrote distinct %u in distinct loop\n", node);
DEBUG_PRINT("Writing %u bytes for node %u: %u\t\"%c\" at pos %lu\n", writesize, node, prev.node, *((const char*)(data.data()+start*chunk_size)), pos);
        } else {
//STDOUT_PRINT("Wrote distinct (%u,%u) in distinct loop\n", node, prev.node);
          Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(uint32_t));
          size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
          memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
          memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
        }
      }
    });
  } else {
STDOUT_PRINT("Using explicit copy\n");
    Kokkos::parallel_for("Write distinct bytes", Kokkos::TeamPolicy<>(distinct.capacity(), Kokkos::AUTO) , 
                         KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
  //    uint32_t i=team_member.league_rank()*team_member.team_size()+team_member.team_rank();
      uint32_t i=team_member.league_rank();
      if(distinct.valid_at(i)) {
        uint32_t node = distinct.key_at(i);
        NodeID prev = distinct.value_at(i);
        if(node == prev.node && chkpt_id == prev.tree) {
          uint32_t size = num_leaf_descendents(node, num_nodes);
          uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
          size_t pos;
          if(team_member.team_rank() == 0) {
            Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
            Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(size*chunk_size));
            pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + size*chunk_size);
            memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
//            memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
          }
          team_member.team_barrier();
          team_member.team_broadcast(pos, 0);
          uint32_t writesize = chunk_size*size;
          if(start*chunk_size+writesize > data.size())
            writesize = data.size()-start*chunk_size;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, writesize), [&] (const uint64_t& j) {
            buffer_d(pos+sizeof(uint32_t)+j) = data(start*chunk_size+j);
          });
        } else {
          if(team_member.team_rank() == 0) {
            Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(uint32_t));
            size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
            memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
            memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
          }
        }
      }
    });
  }
Kokkos::fence();
printf("Done writing distinct bytes\n");
  STDOUT_PRINT("Start writing shared metadata (%u total entries)\n", shared.size());
  Kokkos::parallel_for("Write curr repeat bytes", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      uint32_t node = shared.key_at(i);
      NodeID prev = shared.value_at(i);
      if(prev.tree == chkpt_id) {
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(uint32_t));
        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
DEBUG_PRINT("Trying to write 8 bytes starting at %lu. Max size: %lu\n", pos, buffer_d.size());
        memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
        memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
//STDOUT_PRINT("Write current repeat: %u: (%u,%u)\n", node, prev.node, prev.tree);
Kokkos::atomic_add(&num_curr_repeat_d(0), 1);
      }
    }
  });
Kokkos::fence();
DEBUG_PRINT("Done writing current repeat bytes\n");
  Kokkos::parallel_for("Write prior repeat bytes", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      uint32_t node = shared.key_at(i);
      NodeID prev = shared.value_at(i);
      if(prev.tree != chkpt_id) {
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(uint32_t));
        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
        memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
        memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
      }
    }
  });
Kokkos::fence();
DEBUG_PRINT("Done writing previous repeat bytes\n");
  DEBUG_PRINT("Wrote shared metadata\n");
  Kokkos::fence();
  DEBUG_PRINT("Finished collecting data\n");
  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
  Kokkos::deep_copy(num_curr_repeat_h, num_curr_repeat_d);
  Kokkos::fence();
  header.ref_id = prior_chkpt_id;
  header.chkpt_id = chkpt_id;
  header.datalen = data.size();
  header.chunk_size = chunk_size;
  header.window_size = 0;
  header.distinct_size = distinct.size();
  header.curr_repeat_size = num_curr_repeat_h(0);
  header.prev_repeat_size = shared.size() - num_curr_repeat_h(0);
  STDOUT_PRINT("Ref ID: %u\n"          , header.ref_id);
  STDOUT_PRINT("Chkpt ID: %u\n"        , header.chkpt_id);
  STDOUT_PRINT("Data len: %lu\n"       , header.datalen);
  STDOUT_PRINT("Chunk size: %u\n"      , header.chunk_size);
  STDOUT_PRINT("Window size: %u\n"     , header.window_size);
  STDOUT_PRINT("Distinct size: %u\n"   , header.distinct_size);
  STDOUT_PRINT("Curr repeat size: %u\n", header.curr_repeat_size);
  STDOUT_PRINT("prev repeat size: %u\n", header.prev_repeat_size);
  STDOUT_PRINT("Number of bytes written for compact incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_h(0));
  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));
  DEBUG_PRINT("Closed file\n");
//auto buffer_h = Kokkos::create_mirror_view(buffer_d);
//Kokkos::deep_copy(buffer_h, buffer_d);
//printf("Buffer entry: %u: %c\n", *((uint32_t*)(buffer_h.data())), *((char*)(buffer_h.data()+sizeof(uint32_t))));
  return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
}

//template<uint32_t N>
std::pair<uint64_t,uint64_t> 
write_incr_chkpt_hashtree_global_mode( const std::string& filename, 
                                const Kokkos::View<uint8_t*>& data, 
                                Kokkos::View<uint8_t*>& buffer_d, 
                                uint32_t chunk_size, 
                                const CompactTable& distinct, 
                                const CompactTable& shared,
                                uint32_t prior_chkpt_id,
                                uint32_t chkpt_id,
                                header_t& header) {
  DEBUG_PRINT("File: %s\n", filename.c_str());
  
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size()) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;

  Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
  Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
  Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
  Kokkos::View<uint64_t[1]> num_curr_repeat_d("Number of curr repeat entries");
  Kokkos::View<uint64_t[1]>::HostMirror num_curr_repeat_h = Kokkos::create_mirror_view(num_curr_repeat_d);
  Kokkos::deep_copy(num_curr_repeat_d, 0);
  Kokkos::deep_copy(num_bytes_d, 0);
  Kokkos::deep_copy(num_bytes_data_d, 0);
  Kokkos::deep_copy(num_bytes_metadata_d, 0);
  STDOUT_PRINT("Setup Views\n");

  Kokkos::View<uint32_t[1]> max_reg("Max region size");
  Kokkos::View<uint32_t[1]>::HostMirror max_reg_h = Kokkos::create_mirror_view(max_reg);
  max_reg_h(0) = 0;
  Kokkos::deep_copy(max_reg, max_reg_h);
  Kokkos::parallel_for("Count distinct bytes", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      uint32_t node = distinct.key_at(i);
      NodeID prev = distinct.value_at(i);
      if(node == prev.node && chkpt_id == prev.tree) {
        uint32_t size = num_leaf_descendents(node, num_nodes);
        uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
        Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t) + size*chunk_size);
        Kokkos::atomic_max(&max_reg(0), size);
      } else {
        printf("Distinct node with differint node/tree. Shouldn't happen.\n");
      }
    }
  });
  DEBUG_PRINT("Start writing shared metadata\n");
  Kokkos::parallel_for("Write curr repeat bytes", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      uint32_t node = shared.key_at(i);
      NodeID prev = shared.value_at(i);
      if(prev.tree == chkpt_id) {
        Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
        Kokkos::atomic_add(&num_curr_repeat_d(0), 1);
      } else {
        Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(NodeID));
      }
    }
  });

  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
  Kokkos::deep_copy(max_reg_h, max_reg);

  DEBUG_PRINT("Length of buffer: %lu\n", num_bytes_h(0));
  buffer_d = Kokkos::View<uint8_t*>("Buffer", num_bytes_h(0));

  Kokkos::deep_copy(num_bytes_d, 0);

  DEBUG_PRINT("Largest region: %u\n", max_reg_h(0));
  if(max_reg_h(0) < 2048) {
    Kokkos::parallel_for("Write distinct bytes", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
      if(distinct.valid_at(i)) {
        uint32_t node = distinct.key_at(i);
        NodeID prev = distinct.value_at(i);
        if(node == prev.node && chkpt_id == prev.tree) {
          uint32_t size = num_leaf_descendents(node, num_nodes);
          uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
          Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
          Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(size*chunk_size));
          size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + size*chunk_size);
          memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
//          memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev, sizeof(NodeID));
          uint32_t writesize = chunk_size*size;
          if(start*chunk_size+writesize > data.size())
            writesize = data.size()-start*chunk_size;
          memcpy(buffer_d.data()+pos+sizeof(uint32_t), data.data()+start*chunk_size, writesize);
//printf("Storing distinct node %u: %lu, starting at %u, length %u\n", node, pos+sizeof(uint32_t), start, size);
        } else {
          printf("Distinct node with differint node/tree. Shouldn't happen.\n");
//          Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(NodeID));
//          size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(NodeID));
//          memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
//          memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev, sizeof(NodeID));
        }
      }
    });
  } else {
    Kokkos::parallel_for("Write distinct bytes", Kokkos::TeamPolicy<>(distinct.capacity(), Kokkos::AUTO) , 
                         KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
  //    uint32_t i=team_member.league_rank()*team_member.team_size()+team_member.team_rank();
      uint32_t i=team_member.league_rank();
      if(distinct.valid_at(i)) {
        uint32_t node = distinct.key_at(i);
        NodeID prev = distinct.value_at(i);
        if(node == prev.node && chkpt_id == prev.tree) {
          uint32_t size = num_leaf_descendents(node, num_nodes);
          uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
          size_t pos;
          if(team_member.team_rank() == 0) {
            Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
            Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(size*chunk_size));
            pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + size*chunk_size);
            memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
//            memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev, sizeof(NodeID));
          }
          team_member.team_barrier();
          team_member.team_broadcast(pos, 0);
          uint32_t writesize = chunk_size*size;
          if(start*chunk_size+writesize > data.size())
            writesize = data.size()-start*chunk_size;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, writesize), [&] (const uint64_t& j) {
            buffer_d(pos+sizeof(uint32_t)+j) = data(start*chunk_size+j);
          });
        } else {
          if(team_member.team_rank() == 0) {
            printf("Distinct node with differint node/tree. Shouldn't happen.\n");
//            Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(NodeID));
//            size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(NodeID));
//            memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
//            memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev, sizeof(NodeID));
          }
        }
      }
    });
  }
  DEBUG_PRINT("Start writing shared metadata\n");
  Kokkos::parallel_for("Write curr repeat bytes", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      uint32_t node = shared.key_at(i);
      NodeID prev = shared.value_at(i);
      if(prev.tree == chkpt_id) {
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(uint32_t));
        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
        memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
        memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
//printf("Writing current repeat: %u: %u\n", node, prev.node);
//        memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev, sizeof(NodeID));
      }
    }
  });
  Kokkos::parallel_for("Write prior repeat bytes", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      uint32_t node = shared.key_at(i);
      NodeID prev = shared.value_at(i);
      if(prev.tree != chkpt_id) {
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(NodeID));
        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(NodeID));
        memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
        memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev, sizeof(NodeID));
//printf("Writing previous repeat: %u: (%u,%u)\n", node, prev.node, prev.tree);
      }
    }
  });
  DEBUG_PRINT("Wrote shared metadata\n");
  Kokkos::fence();
  DEBUG_PRINT("Finished collecting data\n");
  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
  Kokkos::deep_copy(num_curr_repeat_h, num_curr_repeat_d);
  Kokkos::fence();
  header.ref_id = prior_chkpt_id;
  header.chkpt_id = chkpt_id;
  header.datalen = data.size();
  header.chunk_size = chunk_size;
  header.window_size = UINT32_MAX;
  header.distinct_size = distinct.size();
  header.curr_repeat_size = num_curr_repeat_h(0);
  header.prev_repeat_size = shared.size() - num_curr_repeat_h(0);
  STDOUT_PRINT("Ref ID: %u\n"          , header.ref_id);
  STDOUT_PRINT("Chkpt ID: %u\n"        , header.chkpt_id);
  STDOUT_PRINT("Data len: %lu\n"       , header.datalen);
  STDOUT_PRINT("Chunk size: %u\n"      , header.chunk_size);
  STDOUT_PRINT("Window size: %u\n"     , header.window_size);
  STDOUT_PRINT("Distinct size: %u\n"   , header.distinct_size);
  STDOUT_PRINT("Curr repeat size: %u\n", header.curr_repeat_size);
  STDOUT_PRINT("prev repeat size: %u\n", header.prev_repeat_size);
  STDOUT_PRINT("Number of bytes written for compact incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_h(0));
  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));
  DEBUG_PRINT("Closed file\n");
  return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
}

#endif // KOKKOS_MERKLE_TREE_HPP
