#ifndef DEDUP_MERKLE_TREE_HPP
#define DEDUP_MERKLE_TREE_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_ScatterView.hpp>
#include <iostream>
#include <climits>
#include "hash_functions.hpp"
#include "map_helpers.hpp"
//#include "kokkos_queue.hpp"
#include "kokkos_merkle_tree.hpp"
#include "utils.hpp"

template <class Hasher>
void create_merkle_tree(Hasher& hasher, 
                        MerkleTree& tree, 
                        Kokkos::View<uint8_t*>& data, 
                        uint32_t chunk_size, 
                        uint32_t tree_id, 
                        DistinctNodeIDMap& distinct_map, 
                        SharedNodeIDMap& shared_map) {
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
        shared_map.insert(i,NodeID(entry.node, entry.tree));
      } else if(result.failed()) {
        printf("Failed to insert node %u into distinct map\n",i);
      }
    });
  }
  Kokkos::fence();
}

template <class Hasher>
void create_merkle_tree(Hasher& hasher, 
                        MerkleTree& tree, 
                        Kokkos::View<uint8_t*>& data, 
                        uint32_t chunk_size, 
                        uint32_t tree_id, 
                        DistinctNodeIDMap& distinct_map, 
                        NodeMap& node_map) {
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
              } else if(node_map.exists(child_l) && node_map.value_at(node_map.find(child_l)).nodetype == Repeat) {
                Node repeat_node = node_map.value_at(node_map.find(child_l));
                insert_entry(shared_updates, child_l, num_nodes, tree_id, NodeID(repeat_node.node, repeat_node.tree));
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
              } else if(node_map.exists(child_r) && node_map.value_at(node_map.find(child_r)).nodetype == Repeat) {
                Node repeat_node = node_map.value_at(node_map.find(child_r));
                insert_entry(shared_updates, child_r, num_nodes, tree_id, NodeID(repeat_node.node, repeat_node.tree));
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
                } else if(node_map.exists(child_l) && node_map.value_at(node_map.find(child_l)).nodetype == Repeat) {
                  auto repeat = node_map.value_at(node_map.find(child_l));
                  insert_entry(shared_updates, child_l, num_nodes, tree_id, NodeID(repeat.node, repeat.tree));
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
                } else if(node_map.exists(child_r) && node_map.value_at(node_map.find(child_r)).nodetype == Repeat) {
                  auto repeat = node_map.value_at(node_map.find(child_r));
                  insert_entry(shared_updates, child_r, num_nodes, tree_id, NodeID(repeat.node, repeat.tree));
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
              } else if(node_map.exists(child_l) && node_map.value_at(node_map.find(child_l)).nodetype == Repeat) {
                auto repeat = node_map.value_at(node_map.find(child_l));
                insert_entry(shared_updates, child_l, num_nodes, tree_id, NodeID(repeat.node, repeat.tree));
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
              } else if(node_map.exists(child_r) && node_map.value_at(node_map.find(child_r)).nodetype == Repeat) {
                auto repeat = node_map.value_at(node_map.find(child_r));
                insert_entry(shared_updates, child_r, num_nodes, tree_id, NodeID(repeat.node, repeat.tree));
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

