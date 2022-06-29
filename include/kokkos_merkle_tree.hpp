#ifndef KOKKOS_MERKLE_TREE_HPP
#define KOKKOS_MERKLE_TREE_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <climits>
#include "hash_functions.hpp"
#include "map_helpers.hpp"
#include "kokkos_queue.hpp"

//#define STATS

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
 
  void digest_to_hex_(const uint8_t digest[20], char* output) {
    int i,j;
    char* c = output;
    for(i=0; i<20/4; i++) {
      for(j=0; j<4; j++) {
        sprintf(c, "%02X", digest[i*4 + j]);
        c += 2;
      }
      sprintf(c, " ");
      c += 1;
    }
    *(c-1) = '\0';
  }

  void print() {
    Kokkos::deep_copy(tree_h, tree_d);
    uint32_t num_leaves = (tree_h.extent(0)+1)/2;
    printf("============================================================\n");
    char buffer[80];
    unsigned int counter = 2;
    for(unsigned int i=0; i<2*num_leaves-1; i++) {
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

template <class Hasher>
void create_merkle_tree(Hasher& hasher, MerkleTree& tree, Kokkos::View<uint8_t*>& data, const uint32_t chunk_size, const int32_t n_levels=INT_MAX) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  const uint32_t num_nodes = 2*num_chunks-1;
  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  int32_t stop_level = 0;
  if(n_levels < num_levels)
    stop_level = num_levels-n_levels;
  const uint32_t leaf_start = num_chunks-1;
  for(int32_t level=num_levels-1; level>=stop_level; level--) {
    uint32_t nhashes = 1 << level;
    uint32_t start_offset = nhashes-1;
    if(start_offset + nhashes > num_nodes)
      nhashes = num_nodes - start_offset;
    auto range_policy = Kokkos::RangePolicy<>(start_offset, start_offset+nhashes);
    Kokkos::parallel_for("Build tree", range_policy, KOKKOS_LAMBDA(const int i) {
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
    });
  }
  Kokkos::fence();
}

template <class Hasher>
MerkleTree create_merkle_tree(Hasher& hasher, Kokkos::View<uint8_t*>& data, const uint32_t chunk_size) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  MerkleTree tree = MerkleTree(num_chunks);
  create_merkle_tree(hasher, tree, data, chunk_size, INT_MAX);
  return tree;
}

template <class Hasher>
void create_merkle_tree(Hasher& hasher, MerkleTree& tree, Kokkos::View<uint8_t*>& data, const uint32_t chunk_size, uint32_t tree_id, DistinctMap& distinct_map, SharedMap& shared_map, int32_t n_levels=UINT_MAX) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  const uint32_t num_nodes = 2*num_chunks-1;
  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  int32_t stop_level = 0;
  if(n_levels < num_levels)
    stop_level = num_levels-n_levels;
  const uint32_t leaf_start = num_chunks-1;
  for(int32_t level=num_levels-1; level>=stop_level; level--) {
    uint32_t nhashes = 1 << level;
    uint32_t start_offset = nhashes-1;
    if(start_offset + nhashes > num_nodes)
      nhashes = num_nodes - start_offset;
    auto range_policy = Kokkos::RangePolicy<>(start_offset, start_offset+nhashes);
    Kokkos::parallel_for("Build tree", range_policy, KOKKOS_LAMBDA(const int i) {
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
      auto result = distinct_map.insert(tree(i), NodeInfo(i, i, tree_id));
      if(result.existing()) {
        auto& entry = distinct_map.value_at(result.index());
        uint32_t prev = Kokkos::atomic_fetch_min(&entry.node, i);
        if(prev > i) {
          entry.src = i;
          NodeInfo new_info(prev, prev, entry.tree);
          shared_map.insert(prev, result.index());
        } else {
          shared_map.insert(i, result.index());
        }
      } else if(result.failed()) {
          printf("Failed to insert node %u into distinct map\n");
      }
    });
  }
  Kokkos::fence();
}

//template <class Hasher>
//void create_merkle_tree_subtrees(Hahser& hasher, MerkleTree& tree, Kokkos::View<uint8_t*>& data, const uint32_t chunk_size) {
//  uint32_t num_chunks = data.size()/chunk_size;
//  if(num_chunks*chunk_size < data.size())
//    num_chunks += 1;
//  const uint32_t num_nodes = 2*num_chunks-1;
//  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
//  constexpr uint32_t num_threads = 128;
//  const uint32_t leaf_start = num_chunks-1;
//  uint32_t num_leagues = num_chunks/num_threads;
//  if(num_threads*num_leagues < num_chunks)
//    num_leagues += 1;
//  Kokkos::TeamPolicy<> team_policy(num_leagues, Kokkos::AUTO());
//  using team_member_type = Kokkos::TeamPolicy<>::member_type;
//  Kokkos::parallel_for("Build tree by subtrees", team_policy, KOKKOS_LAMBDA(team_member_type team_member) {
//    uint32_t league_offset = team_member.league_rank()*num_threads;
//    uint32_t active_threads = 128;
//    uint32_t n_level = num_levels-1;
//    while(active_threads > 0) {
//      Kokkos::parallel_for("Compute level of subtree", Kokkos::RangePolicy<>(0, active_threads), KOKKOS_LAMBDA(const uint32_t j) {
//        uint32_t i = league_offset + j;
//        uint32_t num_bytes = chunk_size;
//        if((i-leaf_start) == num_chunks-1)
//          num_bytes = data.size()-((i-leaf_start)*chunk_size);
//        if(i >= leaf_start) {
//          hasher.hash(data.data()+((i-leaf_start)*chunk_size), 
//                      num_bytes, 
//                      (uint8_t*)(tree(i).digest));
//        } else {
//          hasher.hash((uint8_t*)&tree(2*i+1), 2*hasher.digest_size(), (uint8_t*)&tree(i));
//        }
//      });
//    }
//  });
//}

void compare_trees_fused(const MerkleTree& tree, Queue& queue, const uint32_t tree_id, DistinctMap& distinct_map) {
  uint32_t num_comp = 0;
  uint32_t q_size = queue.size();
  while(q_size > 0) {
    num_comp += q_size;
    Kokkos::parallel_for("Compare trees", Kokkos::RangePolicy<>(0, q_size), KOKKOS_LAMBDA(const uint32_t entry) {
      uint32_t node = queue.pop();
      HashDigest digest = tree(node);
      NodeInfo info(node, node, tree_id);
      auto result = distinct_map.insert(digest, info); // Try to insert
      if(result.success()) { // Node is distinct
        uint32_t child_l = 2*node+1;
        if(child_l < queue.capacity())
          queue.push(child_l);
        uint32_t child_r = 2*node+2;
        if(child_r < queue.capacity())
          queue.push(child_r);
#ifdef STATS
      } else {
        printf("Failed to insert (%u,%u,%u). Already exists.\n", info.node, info.src, info.tree);
#endif
      }
    });
    q_size = queue.size();
  }

  printf("Number of comparisons (Merkle Tree): %u\n", num_comp);
  Kokkos::fence();
}

//template<typename Scheduler>
//struct CompareTreeTask {
//  using sched_type  = Scheduler;
//  using future_type = Kokkos::BasicFuture<uint32_t, Scheduler>;
//  using value_type  = uint32_t;
//
//  uint32_t node;
//  uint32_t tree_id;
//  MerkleTree tree;
//  DistinctMap distinct_map;
//  future_type child_l_fut;
//  future_type child_r_fut;
//  bool l_active;
//  bool r_active;
//
//  KOKKOS_INLINE_FUNCTION
//  CompareTreeTask(const uint32_t n, const MerkleTree& merkle_tree, const uint32_t treeID, DistinctMap& distinct) : 
//                  node(n), tree_id(treeID), tree(merkle_tree), distinct_map(distinct), l_active(true), r_active(true) {}
//
//  KOKKOS_INLINE_FUNCTION
//  void operator()(typename sched_type::member_type& member, uint32_t& result) {
//    auto& sched = member.scheduler();
//
//    bool child_l_ready = ( (l_active && !child_l_fut.is_null()) || (l_active == false) );
//    bool child_r_ready = ( (r_active && !child_r_fut.is_null()) || (r_active == false) );
//    if((child_l_ready && child_r_ready)) { // On task respawn
//      result = 1 + child_l_fut.get() + child_r_fut.get();
//    } else { // Perform task and spawn for children if needed
//      uint32_t active_children = 0;
//      HashDigest digest = tree(node);
//      NodeInfo info(node, node, tree_id);
//      auto insert_result = distinct_map.insert(digest, info); // Try to insert
//      if(insert_result.success()) { // Node is distinct
//        uint32_t child_l = 2*node+1;
//        if(child_l < tree.tree_d.extent(0)) {
//          child_l_fut = Kokkos::task_spawn(Kokkos::TaskSingle(sched, Kokkos::TaskPriority::High), CompareTreeTask(child_l, tree, tree_id, distinct_map));
//          active_children += 1;
//        } else {
//          l_active = false;
//        }
//        uint32_t child_r = 2*node+2;
//        if(child_r < tree.tree_d.extent(0)) {
//          child_r_fut = Kokkos::task_spawn(Kokkos::TaskSingle(sched, Kokkos::TaskPriority::High), CompareTreeTask(child_r, tree, tree_id, distinct_map));
//          active_children += 1;
//        } else {
//          l_active = false;
//        }
//      }
//      if(active_children == 2) {
//        Kokkos::BasicFuture<void, Scheduler> dep[] = {child_l_fut, child_r_fut};
//        Kokkos::BasicFuture<void, Scheduler> all_children = sched.when_all(dep, 2);
//        Kokkos::respawn(this, all_children, Kokkos::TaskPriority::High);
//      } else if(active_children == 1) {
//        if(l_active) { 
//          Kokkos::respawn(this, child_l_fut, Kokkos::TaskPriority::High);
//        } else {
//          Kokkos::respawn(this, child_r_fut, Kokkos::TaskPriority::High);
//        }
//      } else {
//        result = 1;
//      }
//    }
//  }
//};
//
//void compare_trees_tasks(const MerkleTree& tree, Queue& queue, const uint32_t tree_id, DistinctMap& distinct_map) {
//  using scheduler_type = Kokkos::TaskScheduler<Kokkos::DefaultExecutionSpace>;
//  using memory_space = typename scheduler_type::memory_space;
//  using memory_pool = typename scheduler_type::memory_pool;
//  auto mpool = memory_pool(memory_space{}, estimate_required_memory(tree.tree_d.extent(0)));
//  auto root_sched = scheduler_type(mpool);
//  Kokkos::BasicFuture<uint32_t, scheduler_type> f = Kokkos::host_spawn(Kokkos::TaskSingle(root_sched), 
//                                                                  CompareTreeTask<scheduler_type>(0, tree, tree_id, distinct_map));
//  Kokkos::wait(root_sched);
//  printf("Number of comparisons (Merkle Tree Task): %u\n", f.get());
//}

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

template<class Hasher, uint32_t N>
void deduplicate_data(Kokkos::View<uint8_t*>& data, 
                      const uint32_t chunk_size, 
                      const Hasher hasher, 
                      MerkleTree& tree, 
                      const uint32_t tree_id, 
                      const SharedMap& prior_shared_map, 
                      const DistinctMap& prior_distinct_map, 
                      SharedMap& shared_map, 
                      DistinctMap& distinct_map, 
                      CompactTable<N>& updates) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  const uint32_t num_nodes = 2*num_chunks-1;
  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  const uint32_t leaf_start = num_chunks-1;

  uint32_t prev_leftover = UINT32_MAX;
  uint32_t num_nodes_left = num_chunks;
  uint32_t num_threads = num_chunks;
  uint32_t start_offset = leaf_start;
  Kokkos::View<uint32_t[1]> nodes_leftover("Leftover nodes to process");
  Kokkos::View<uint32_t[1]>::HostMirror nodes_leftover_h = Kokkos::create_mirror_view(nodes_leftover);
  Kokkos::deep_copy(nodes_leftover, 0);
  nodes_leftover_h(0) = 0;
#ifdef STATS
  Kokkos::View<uint32_t[1]> num_same("Number of chunks that remain the same");
  Kokkos::View<uint32_t[1]> num_new("Number of chunks that are new");
  Kokkos::View<uint32_t[1]> num_shift("Number of chunks that exist but in different spaces");
  Kokkos::View<uint32_t[1]> num_comp("Number of compressed nodes");
  Kokkos::View<uint32_t[1]> num_dupl("Number of new duplicate nodes");
  Kokkos::View<uint32_t[1]>::HostMirror num_same_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_new_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_shift_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_comp_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_dupl_h = Kokkos::create_mirror_view(num_same);
  Kokkos::deep_copy(num_same, 0);
  Kokkos::deep_copy(num_new, 0);
  Kokkos::deep_copy(num_shift, 0);
  Kokkos::deep_copy(num_comp, 0);
  Kokkos::deep_copy(num_dupl, 0);
#endif

  while(nodes_leftover_h(0) != prev_leftover) {
    prev_leftover = nodes_leftover_h(0);
    Kokkos::parallel_for("Insert/compare hashes", Kokkos::RangePolicy<>(0,num_threads), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node = i+start_offset;
      if(start_offset == leaf_start) {
        uint32_t num_bytes = chunk_size;
        if(i == num_chunks-1)
          num_bytes = data.size()-i*chunk_size;
        hasher.hash(data.data()+(i*chunk_size), num_bytes, tree(node).digest);
        if(tree_id == 0) {
          NodeInfo info(node, node, tree_id);
          auto result = prior_distinct_map.insert(tree(node), info);
          if(result.existing()) {
            NodeInfo& old = prior_distinct_map.value_at(result.index());
            uint32_t prev = Kokkos::atomic_fetch_min(&old.node, node);
            if(prev > node) {
              old.src = node;
              NodeInfo new_info(prev, prev, old.tree);
              shared_map.insert(prev, result.index());
            } else {
              shared_map.insert(node, result.index());
            }
#ifdef STATS
            Kokkos::atomic_add(&(num_shift(0)), 1);
          } else if(result.success()) {
            Kokkos::atomic_add(&(num_new(0)), 1);
          } else if(result.failed()) {
            printf("Failed to insert node into distinct or shared map (tree 0). Shouldn't happen.");
#endif
          }
        } else {
          NodeInfo info = NodeInfo(node, node, tree_id);
          uint32_t index = prior_distinct_map.find(tree(node));
          if(!prior_distinct_map.valid_at(index)) { // New chunk 
            auto result = distinct_map.insert(tree(node), info);
            if(result.success()) {
Kokkos::atomic_exchange(&tree.distinct_children_d(node) , 2);
#ifdef STATS
              Kokkos::atomic_add(&(num_new(0)), 1);
#endif
              Kokkos::atomic_add(&(tree.distinct_children_d((node-1)/2)), 1);
              Kokkos::atomic_add(&nodes_leftover(0), 1);
            } else if(result.existing()) {
              NodeInfo& existing_info = distinct_map.value_at(result.index());
              uint32_t existing_node = Kokkos::atomic_fetch_min(&existing_info.node, node);
              if(existing_node > node) {
Kokkos::atomic_exchange(&tree.distinct_children_d(node) , 2);
Kokkos::atomic_exchange(&tree.distinct_children_d(existing_node) , 0);
                Kokkos::atomic_sub(&(tree.distinct_children_d((existing_node-1)/2)), 1);
                Kokkos::atomic_add(&(tree.distinct_children_d((node-1)/2)), 1);
                existing_info.src = node;
                shared_map.insert(existing_node, result.index());
              } else {
Kokkos::atomic_exchange(&tree.distinct_children_d(node) , 0);
                shared_map.insert(node, result.index());
              }
#ifdef STATS
              Kokkos::atomic_add(&num_dupl(0), 1);
#endif
            } else if(result.failed()) {
              printf("Failed to insert new chunk into distinct or shared map (tree %u). Shouldn't happen.", tree_id);
            }
          } else { // Chunk already exists
            NodeInfo old_distinct = prior_distinct_map.value_at(index);
            if(node != old_distinct.node) { // Chunk exists but at a different offset
              uint32_t prior_shared_idx = prior_shared_map.find(node);
              if(prior_shared_map.valid_at(prior_shared_idx)) { // Node is in prior shared map
                uint32_t prior_node = prior_distinct_map.value_at(prior_shared_map.value_at(prior_shared_idx)).node;
                if(prior_node != node) { // Chunk has changed since prior checkpoint
                  shared_map.insert(node, index);
#ifdef STATS
                  Kokkos::atomic_add(&(num_shift(0)), 1);
                } else {
                  Kokkos::atomic_add(&(num_same(0)), 1);
#endif
                }
              } else { // Node not in prior shared map
                shared_map.insert(node, index);
#ifdef STATS
                Kokkos::atomic_add(&(num_shift(0)), 1);
#endif
              }
#ifdef STATS
            } else { // Chunk exists and hasn't changed node
              Kokkos::atomic_add(&(num_same(0)), 1);
#endif
            }
          }
        }
      } else if(tree.distinct_children_d(node) == 2) {
        hasher.hash((uint8_t*)&tree(2*(node)+1), 2*hasher.digest_size(), (uint8_t*)&tree(node));
        Kokkos::atomic_add(&(tree.distinct_children_d((node-1)/2)), 1);
        Kokkos::atomic_add(&nodes_leftover(0), 1);
      } else if(tree.distinct_children_d(node) == 1) {
        uint32_t child_l = 2*(node)+1;
        uint32_t child_r = 2*(node)+2;
        if((tree.distinct_children_d(child_l) == 2)) {
          uint32_t size = num_leaf_descendents(child_l, num_nodes);
          CompactNodeInfo info(child_l, size);
          uint32_t existing_idx = updates.find(info);
          if(updates.valid_at(existing_idx)) {
            auto& old = updates.value_at(existing_idx);
            old.push(tree_id);
          } else {
            auto insert_res = updates.insert(info);
            Array<N>& vec = updates.value_at(insert_res.index());
            vec.push(tree_id);
#ifdef STATS
            Kokkos::atomic_add(&(num_comp(0)), 1);
#endif
          }
        } else if((tree.distinct_children_d(child_r) == 2)) {
          uint32_t size = num_leaf_descendents(child_r, num_nodes);
          CompactNodeInfo info(child_r, size);
          uint32_t existing_idx = updates.find(info);
          if(updates.valid_at(existing_idx)) {
            auto& old = updates.value_at(existing_idx);
            old.push(tree_id);
          } else {
            auto insert_res = updates.insert(info);
            Array<N>& vec = updates.value_at(insert_res.index());
            vec.push(tree_id);
#ifdef STATS
            Kokkos::atomic_add(&(num_comp(0)), 1);
#endif
          }
        }
      }
    });
    Kokkos::deep_copy(nodes_leftover_h, nodes_leftover);
//    printf("Found %u nodes to process\n", nodes_leftover_h(0));
    num_threads /= 2;
    start_offset -= num_threads;
  }
  Kokkos::fence();
#ifdef STATS
  Kokkos::deep_copy(num_same_h, num_same);
  Kokkos::deep_copy(num_new_h, num_new);
  Kokkos::deep_copy(num_shift_h, num_shift);
  Kokkos::deep_copy(num_comp_h, num_comp);
  Kokkos::deep_copy(num_dupl_h, num_dupl);
  printf("Number of chunks: %u\n", num_chunks);
  printf("Number of new chunks: %u\n", num_new_h(0));
  printf("Number of same chunks: %u\n", num_same_h(0));
  printf("Number of shift chunks: %u\n", num_shift_h(0));
  printf("Number of comp nodes: %u\n", num_comp_h(0));
  printf("Number of dupl nodes: %u\n", num_dupl_h(0));
#endif
}

//template<uint32_t N>
//KOKKOS_INLINE_FUNCTION
//void insert_entry(const CompactTable<N>& updates, const uint32_t node, const uint32_t num_nodes, const uint32_t tree_id, const Kokkos::View<uint32_t[1]>& num_comp) {
//if(node > num_nodes)
//printf("Something very wrong happened.\n");
//  uint32_t size = num_leaf_descendents(node, num_nodes);
//  uint32_t leaf = leftmost_leaf(node, num_nodes);
////for(uint32_t n = leaf; n<leaf+size; n++) {
////  printf("Insert entry %u\n", n);
////}
//  CompactNodeInfo info(leaf, size);
//  auto result = updates.insert(info);
//  auto& update = updates.value_at(result.index());
//  update.push(tree_id);
//  if(result.success()) {
//    Kokkos::atomic_add(&(num_comp(0)), 1);
//  } else if(result.existing()) {
//    printf("Tried to insert existing node %u: (%u,%u)\n", node, leaf, size);
//  } else if(result.failed()) {
//    printf("Failed to update compact represntation.\n");
//  }
//}

template<uint32_t N>
KOKKOS_INLINE_FUNCTION
void insert_entry(const CompactTable<N>& updates, const uint32_t node, const uint32_t num_nodes, const uint32_t tree_id) {
if(node > num_nodes)
printf("Something very wrong happened.\n");
  uint32_t size = num_leaf_descendents(node, num_nodes);
  uint32_t leaf = leftmost_leaf(node, num_nodes);
  CompactNodeInfo info(leaf, size);
  auto result = updates.insert(info);
  auto& update = updates.value_at(result.index());
  update.push(tree_id);
}

template<class Hasher, uint32_t N>
void deduplicate_data(Kokkos::View<uint8_t*>& data, 
                      const uint32_t chunk_size, 
                      const Hasher hasher, 
                      MerkleTree& tree, 
                      const uint32_t tree_id, 
                      const SharedMap& prior_shared_map, 
                      const DistinctMap& prior_distinct_map, 
                      SharedMap& shared_map, 
                      DistinctMap& distinct_map, 
                      CompactTable<N>& shared_updates,
                      CompactTable<N>& distinct_updates) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  const uint32_t num_nodes = 2*num_chunks-1;
  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  const uint32_t leaf_start = num_chunks-1;

  uint32_t prev_leftover = UINT32_MAX;
  uint32_t num_nodes_left = num_chunks;
  uint32_t num_threads = num_chunks;
  uint32_t current_level = num_levels-1;
  uint32_t start_offset = (1 << num_levels-1)-1;
  uint32_t end_offset = (1 << num_levels)-1;
  if(end_offset > num_nodes)
    end_offset = num_nodes;
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
  Kokkos::View<uint32_t[1]>::HostMirror num_new_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_shift_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_comp_d_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_comp_s_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_dupl_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_other_h = Kokkos::create_mirror_view(num_same);
  Kokkos::deep_copy(num_same, 0);
  Kokkos::deep_copy(num_new, 0);
  Kokkos::deep_copy(num_shift, 0);
  Kokkos::deep_copy(num_comp_d, 0);
  Kokkos::deep_copy(num_comp_s, 0);
  Kokkos::deep_copy(num_dupl, 0);
  Kokkos::deep_copy(num_other, 0);
#endif

  while(nodes_leftover_h(0) != prev_leftover) {
    prev_leftover = nodes_leftover_h(0);
    Kokkos::parallel_for("Insert/compare hashes", Kokkos::RangePolicy<>(start_offset,end_offset), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node = i;
      if(node >= leaf_start) {
        uint32_t num_bytes = chunk_size;
        if(node-leaf_start == num_chunks-1)
          num_bytes = data.size()-(node-leaf_start)*chunk_size;
        hasher.hash(data.data()+((node-leaf_start)*chunk_size), num_bytes, tree(node).digest);
        if(tree_id == 0) {
          NodeInfo info(node, node, tree_id);
          auto result = prior_distinct_map.insert(tree(node), info);
          if(result.existing()) {
            NodeInfo& old = prior_distinct_map.value_at(result.index());
            uint32_t prev = Kokkos::atomic_fetch_min(&old.node, node);
            if(prev > node) {
              old.src = node;
              NodeInfo new_info(prev, prev, old.tree);
              shared_map.insert(prev, result.index());
            } else {
              shared_map.insert(node, result.index());
            }
#ifdef STATS
            Kokkos::atomic_add(&(num_shift(0)), 1);
          } else if(result.success()) {
            Kokkos::atomic_add(&(num_new(0)), 1);
          } else if(result.failed()) {
            printf("Failed to insert node into distinct or shared map (tree 0). Shouldn't happen.");
#endif
          }
        } else {
          NodeInfo info = NodeInfo(node, node, tree_id);
          uint32_t index = prior_distinct_map.find(tree(node));
          if(!prior_distinct_map.valid_at(index)) { // Chunk not in prior map
            auto result = distinct_map.insert(tree(node), info);
            if(result.success()) { // Chunk is brand new
              tree.distinct_children_d(node) = 2;
#ifdef STATS
              Kokkos::atomic_add(&(num_new(0)), 1);
#endif
              Kokkos::atomic_add(&nodes_leftover(0), 1);
            } else if(result.existing()) { // Chunk already exists locally
              NodeInfo& existing_info = distinct_map.value_at(result.index());
              tree.distinct_children_d(node) = 8;
              Kokkos::atomic_add(&nodes_leftover(0), 1);
              shared_map.insert(node, result.index());
#ifdef STATS
              Kokkos::atomic_add(&num_dupl(0), 1);
#endif
            } else if(result.failed()) {
              printf("Failed to insert new chunk into distinct or shared map (tree %u). Shouldn't happen.", tree_id);
            }
          } else { // Chunk already exists
            NodeInfo old_distinct = prior_distinct_map.value_at(index);
            if(node != old_distinct.node) { // Chunk exists but at a different offset
              uint32_t prior_shared_idx = prior_shared_map.find(node);
              if(prior_shared_map.valid_at(prior_shared_idx)) { // Node is in prior shared map
                uint32_t prior_node = prior_distinct_map.value_at(prior_shared_map.value_at(prior_shared_idx)).node;
                if(prior_node != node) { // Chunk has changed since prior checkpoint
                  shared_map.insert(node, index);
                  tree.distinct_children_d(node) = 8;
                  Kokkos::atomic_add(&nodes_leftover(0), 1);
#ifdef STATS
                  Kokkos::atomic_add(&(num_shift(0)), 1);
                } else {
                  Kokkos::atomic_add(&(num_same(0)), 1);
                  tree.distinct_children_d(node) = 0;
#endif
                }
              } else { // Node not in prior shared map
                shared_map.insert(node, index);
                tree.distinct_children_d(node) = 8;
                Kokkos::atomic_add(&nodes_leftover(0), 1);
#ifdef STATS
                Kokkos::atomic_add(&(num_shift(0)), 1);
#endif
              }
#ifdef STATS
            } else { // Chunk exists and hasn't changed node
              Kokkos::atomic_add(&(num_same(0)), 1);
              tree.distinct_children_d(node) = 0;
#endif
            }
          }
        }
      } else {
        uint32_t child_l = 2*node + 1;
        uint32_t child_r = 2*node + 2;
        tree.distinct_children_d(node) = tree.distinct_children_d(child_l)/2 + tree.distinct_children_d(child_r)/2;
        if(tree.distinct_children_d(node) == 2) {
          hasher.hash((uint8_t*)&tree(2*(node)+1), 2*hasher.digest_size(), (uint8_t*)&tree(node));
          Kokkos::atomic_add(&nodes_leftover(0), 1);
        } else if(tree.distinct_children_d(node) == 8) {
          hasher.hash((uint8_t*)&tree(2*(node)+1), 2*hasher.digest_size(), (uint8_t*)&tree(node));
          if(prior_distinct_map.exists(tree(node))) {
            Kokkos::atomic_add(&nodes_leftover(0), 1);
          } else {
            uint32_t child_l = 2*(node)+1;
            uint32_t child_r = 2*(node)+2;
            if(prior_distinct_map.exists(tree(child_l))) {
              insert_entry(shared_updates, child_l, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_s(0)), 1);
#endif
            } else if(shared_map.exists(child_l)) {
              insert_entry(distinct_updates, child_l, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), 1);
#endif
            }
            if(prior_distinct_map.exists(tree(child_r))) {
              insert_entry(shared_updates, child_r, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_s(0)), 1);
#endif
            } else if(shared_map.exists(child_r)) {
              insert_entry(distinct_updates, child_r, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), 1);
#endif
            }
            tree.distinct_children_d(node) = 0;
          }
        } else if(tree.distinct_children_d(node) == 5) {
          uint32_t child_l = 2*(node)+1;
          uint32_t child_r = 2*(node)+2;
          if(child_l < num_nodes) {
            if((tree.distinct_children_d(child_l) == 2)) {
              insert_entry(distinct_updates, child_l, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), 1);
#endif
            } else if((tree.distinct_children_d(child_l) == 8)) {
              if(prior_distinct_map.exists(tree(child_l))) {
                insert_entry(shared_updates, child_l, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_s(0)), 1);
#endif
              } else if(shared_map.exists(child_l)) {
                insert_entry(distinct_updates, child_l, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), 1);
#endif
              }
            }
          }
          if(child_r < num_nodes) {
            if((tree.distinct_children_d(child_r) == 2)) {
              insert_entry(distinct_updates, child_r, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), 1);
#endif
            } else if((tree.distinct_children_d(child_r) == 8)) {
              if(prior_distinct_map.exists(tree(child_r))) {
                insert_entry(shared_updates, child_r, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_s(0)), 1);
#endif
              } else if(shared_map.exists(child_r)) {
                insert_entry(distinct_updates, child_r, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), 1);
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
              insert_entry(shared_updates, child_l, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_s(0)), 1);
#endif
            } else if(shared_map.exists(child_l)) {
              insert_entry(distinct_updates, child_l, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), 1);
#endif
            }
          } else if((child_r < num_nodes) && (tree.distinct_children_d(child_r) == 8)) {
            if(prior_distinct_map.exists(tree(child_r))) {
              insert_entry(shared_updates, child_r, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_s(0)), 1);
#endif
            } else if(shared_map.exists(child_r)) {
              insert_entry(distinct_updates, child_r, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), 1);
#endif
            }
          }
          tree.distinct_children_d(node) = 0;
        } else if(tree.distinct_children_d(node) == 1) {
          uint32_t child_l = 2*(node)+1;
          uint32_t child_r = 2*(node)+2;
          if((child_l < num_nodes) && (tree.distinct_children_d(child_l) == 2)) {
            insert_entry(distinct_updates, child_l, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), 1);
#endif
          } else if((child_r < num_nodes) && (tree.distinct_children_d(child_r) == 2)) {
            insert_entry(distinct_updates, child_r, num_nodes, tree_id);
#ifdef STATS
              Kokkos::atomic_add(&(num_comp_d(0)), 1);
#endif
          }
          tree.distinct_children_d(node) = 0;
        }
      }
    });
#ifdef STATS
if(start_offset >= leaf_start-(num_chunks/2)) {
printf("------------------------------\n");
uint32_t n_distinct = 0;
Kokkos::parallel_reduce("Count number of distinct", Kokkos::RangePolicy<>(start_offset, end_offset), KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
  if(tree.distinct_children_d(start_offset+i) == 2) {
    update += 1;
  }
}, n_distinct);
printf("Count number of distinct chunks: %u\n", n_distinct);
uint32_t n_same = 0;
Kokkos::parallel_reduce("Count number of same", Kokkos::RangePolicy<>(start_offset, end_offset), KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
  if(tree.distinct_children_d(start_offset+i) == 0) {
    update += 1;
  }
}, n_same);
printf("Count number of same chunks: %u\n", n_same);
uint32_t n_shared = 0;
Kokkos::parallel_reduce("Count number of shared", Kokkos::RangePolicy<>(start_offset, end_offset), KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
  if(tree.distinct_children_d(start_offset+i) == 8) {
    update += 1;
  }
}, n_shared);
printf("Count number of shared chunks: %u\n", n_shared);
uint32_t n_distinct_shared = 0;
Kokkos::parallel_reduce("Count number of distinct shared", Kokkos::RangePolicy<>(start_offset, end_offset), KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
  if(tree.distinct_children_d(start_offset+i) == 5) {
    update += 1;
  }
}, n_distinct_shared);
printf("Count number of distinct shared chunks: %u\n", n_distinct_shared);
uint32_t n_distinct_same = 0;
Kokkos::parallel_reduce("Count number of distinct_same", Kokkos::RangePolicy<>(start_offset, end_offset), KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
  if(tree.distinct_children_d(start_offset+i) == 1) {
    update += 1;
  }
}, n_distinct_same);
printf("Count number of distinct_same chunks: %u\n", n_distinct_same);
uint32_t n_shared_same = 0;
Kokkos::parallel_reduce("Count number of shared_same", Kokkos::RangePolicy<>(start_offset, end_offset), KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
  if(tree.distinct_children_d(start_offset+i) == 4) {
    update += 1;
  }
}, n_shared_same);
printf("Count number of shared_same chunks: %u\n", n_shared_same);
printf("------------------------------\n");
}
#endif
    Kokkos::deep_copy(nodes_leftover_h, nodes_leftover);
    current_level -= 1;
    start_offset = (1 << current_level) - 1;
    end_offset = (1 << current_level+1) - 1;
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
  printf("Number of chunks: %u\n", num_chunks);
  printf("Number of new chunks: %u\n", num_new_h(0));
  printf("Number of same chunks: %u\n", num_same_h(0));
  printf("Number of shift chunks: %u\n", num_shift_h(0));
  printf("Number of distinct comp nodes: %u\n", num_comp_d_h(0));
  printf("Number of shared comp nodes: %u\n", num_comp_s_h(0));
  printf("Number of dupl nodes: %u\n", num_dupl_h(0));
  printf("Number of other nodes: %u\n", num_other_h(0));
#endif
}

template<class Hasher, uint32_t N>
void deduplicate_data_team( Kokkos::View<uint8_t*>& data, 
                            const uint32_t chunk_size, 
                            const Hasher hasher, 
                            const uint32_t chunks_per_league,
                            MerkleTree& tree, 
                            const uint32_t tree_id, 
                            const SharedMap& prior_shared_map, 
                            const DistinctMap& prior_distinct_map, 
                            SharedMap& shared_map, 
                            DistinctMap& distinct_map, 
                            CompactTable<N>& updates) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  const uint32_t num_nodes = 2*num_chunks-1;
  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  const uint32_t leaf_start = num_chunks-1;

  uint32_t prev_leftover = UINT32_MAX;
  uint32_t num_nodes_left = num_chunks;
  uint32_t num_threads = num_chunks;
  Kokkos::View<uint32_t[1]> nodes_leftover("Leftover nodes to process");
  Kokkos::View<uint32_t[1]>::HostMirror nodes_leftover_h = Kokkos::create_mirror_view(nodes_leftover);
  Kokkos::deep_copy(nodes_leftover, 0);
  nodes_leftover_h(0) = 0;
//DistinctMap local_distinct = DistinctMap(prior_distinct_map.capacity());
//Kokkos::deep_copy(local_distinct, prior_distinct_map);
#ifdef STATS
  Kokkos::View<uint32_t[1]> num_same("Number of chunks that remain the same");
  Kokkos::View<uint32_t[1]> num_new("Number of chunks that are new");
  Kokkos::View<uint32_t[1]> num_shift("Number of chunks that exist but in different spaces");
  Kokkos::View<uint32_t[1]> num_comp("Number of compressed nodes");
  Kokkos::View<uint32_t[1]> num_dupl("Number of new duplicate nodes");
  Kokkos::View<uint32_t[1]>::HostMirror num_same_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_new_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_shift_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_comp_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_dupl_h = Kokkos::create_mirror_view(num_same);
  Kokkos::deep_copy(num_same, 0);
  Kokkos::deep_copy(num_new, 0);
  Kokkos::deep_copy(num_shift, 0);
  Kokkos::deep_copy(num_comp, 0);
  Kokkos::deep_copy(num_dupl, 0);
#endif

  uint32_t level_width = (1 << (num_levels-1));
  uint32_t per_league = chunks_per_league;
  uint32_t num_leagues = level_width/per_league;
  if(num_leagues*per_league < level_width)
    num_leagues += 1;
//  uint32_t per_league = 1024;
//  uint32_t num_leagues = num_chunks / per_league;
//  if(num_leagues * per_league < num_chunks) 
//    num_leagues += 1;
  typedef Kokkos::DefaultExecutionSpace::scratch_memory_space ScratchSpace;
  typedef Kokkos::View<uint32_t[1], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_uint32_t;
printf("Width: %u, Num leagues: %u, per_league: %u\n", level_width, num_leagues, per_league);

  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>(num_leagues, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(2*(4+sizeof(shared_uint32_t))));
  Kokkos::parallel_for("Compare chkpts", team_policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type team_member) {
    shared_uint32_t prev_parents(team_member.team_scratch(0));
    shared_uint32_t curr_parents(team_member.team_scratch(0));
    if(team_member.team_rank() == 0) {
      prev_parents(0) = UINT32_MAX;
      curr_parents(0) = 0;
    }
    team_member.team_barrier();

    uint32_t num_iter = num_chunks;
    uint32_t start_offset = leaf_start;
    uint32_t num_threads = per_league;
    while(curr_parents(0) != prev_parents(0)) {
      if(team_member.team_rank() == 0)  {
        prev_parents(0) = curr_parents(0);
      }
//      team_member.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_threads), [=] (const uint32_t _i) {
        uint32_t i = team_member.league_rank()*num_threads + _i;
        uint32_t node = i+start_offset;
        if(node < num_nodes) {
          if(start_offset == leaf_start) {
            uint32_t num_bytes = chunk_size;
            if(i == num_chunks-1)
              num_bytes = data.size()-i*chunk_size;
            hasher.hash(data.data()+(i*chunk_size), num_bytes, tree(node).digest);
            if(tree_id == 0) {
              NodeInfo info(node, node, tree_id);
              auto result = prior_distinct_map.insert(tree(node), info);
              if(result.existing()) {
                NodeInfo& old = prior_distinct_map.value_at(result.index());
                uint32_t prev = Kokkos::atomic_fetch_min(&old.node, node);
                if(prev > node) {
                  old.src = node;
                  NodeInfo new_info(prev, prev, old.tree);
                  shared_map.insert(prev, result.index());
                } else {
                  shared_map.insert(node, result.index());
                }
#ifdef STATS
                Kokkos::atomic_add(&(num_shift(0)), 1);
              } else if(result.success()) {
                Kokkos::atomic_add(&(num_new(0)), 1);
              } else if(result.failed()) {
                printf("Failed to insert node into distinct or shared map (tree 0). Shouldn't happen.");
#endif
              }
            } else {
              NodeInfo info = NodeInfo(node, node, tree_id);
              uint32_t index = prior_distinct_map.find(tree(node));
              if(!prior_distinct_map.valid_at(index)) { // New chunk 
                auto result = distinct_map.insert(tree(node), info);
                if(result.success()) {
#ifdef STATS
                  Kokkos::atomic_add(&(num_new(0)), 1);
#endif
                  Kokkos::atomic_add(&(tree.distinct_children_d((node-1)/2)), 1);
                  Kokkos::atomic_add(&curr_parents(0), 1);
                } else if(result.existing()) {
                  NodeInfo& existing_info = distinct_map.value_at(result.index());
                  uint32_t existing_node = Kokkos::atomic_fetch_min(&existing_info.node, node);
                  if(existing_node > node) {
                    Kokkos::atomic_sub(&(tree.distinct_children_d((existing_node-1)/2)), 1);
                    Kokkos::atomic_add(&(tree.distinct_children_d((node-1)/2)), 1);
                    existing_info.src = node;
                    shared_map.insert(existing_node, result.index());
                  } else {
                    shared_map.insert(node, result.index());
                  }
#ifdef STATS
                  Kokkos::atomic_add(&num_dupl(0), 1);
#endif
                } else if(result.failed()) {
                  printf("Failed to insert new chunk into distinct or shared map (tree %u). Shouldn't happen.", tree_id);
                }
              } else { // Chunk already exists
                NodeInfo old_distinct = prior_distinct_map.value_at(index);
                if(node != old_distinct.node) { // Chunk exists but at a different offset
                  uint32_t prior_shared_idx = prior_shared_map.find(node);
                  if(prior_shared_map.valid_at(prior_shared_idx)) { // Node is in prior shared map
                    uint32_t prior_node = prior_distinct_map.value_at(prior_shared_map.value_at(prior_shared_idx)).node;
                    if(prior_node != node) { // Chunk has changed since prior checkpoint
                      shared_map.insert(node, index);
#ifdef STATS
                      Kokkos::atomic_add(&(num_shift(0)), 1);
                    } else {
                      Kokkos::atomic_add(&(num_same(0)), 1);
#endif
                    }
                  } else { // Node not in prior shared map
                    shared_map.insert(node, index);
#ifdef STATS
                    Kokkos::atomic_add(&(num_shift(0)), 1);
#endif
                  }
#ifdef STATS
                } else { // Chunk exists and hasn't changed node
                  Kokkos::atomic_add(&(num_same(0)), 1);
#endif
                }
              }
            }
          } else if(tree.distinct_children_d(node) == 2) {
            hasher.hash((uint8_t*)&tree(2*(node)+1), 2*hasher.digest_size(), (uint8_t*)&tree(node));
            Kokkos::atomic_add(&(tree.distinct_children_d((node-1)/2)), 1);
            Kokkos::atomic_add(&curr_parents(0), 1);
          } else if(tree.distinct_children_d(node) == 1) {
            uint32_t child_l = 2*(node)+1;
            uint32_t child_r = 2*(node)+2;
            if((tree.distinct_children_d(child_l) == 2) || (child_l >= leaf_start)) {
              uint32_t size = num_leaf_descendents(child_l, num_nodes);
              CompactNodeInfo info(child_l, size);
              uint32_t existing_idx = updates.find(info);
              if(updates.valid_at(existing_idx)) {
                auto& old = updates.value_at(existing_idx);
                old.push(tree_id);
              } else {
                auto insert_res = updates.insert(info);
                Array<N>& vec = updates.value_at(insert_res.index());
                vec.push(tree_id);
#ifdef STATS
                Kokkos::atomic_add(&(num_comp(0)), 1);
#endif
              }
            } else if((tree.distinct_children_d(child_r) == 2) || (child_r >= leaf_start)) {
              uint32_t size = num_leaf_descendents(child_r, num_nodes);
              CompactNodeInfo info(child_r, size);
              uint32_t existing_idx = updates.find(info);
              if(updates.valid_at(existing_idx)) {
                auto& old = updates.value_at(existing_idx);
                old.push(tree_id);
              } else {
                auto insert_res = updates.insert(info);
                Array<N>& vec = updates.value_at(insert_res.index());
                vec.push(tree_id);
#ifdef STATS
                Kokkos::atomic_add(&(num_comp(0)), 1);
#endif
              }
            }
          }
        }
      });
//      team_member.team_barrier();
      num_threads /= 2;
      num_iter /= 2;
      start_offset -= num_iter;
      team_member.team_barrier();
    }
  }); 
#ifdef STATS
  Kokkos::deep_copy(num_same_h, num_same);
  Kokkos::deep_copy(num_new_h, num_new);
  Kokkos::deep_copy(num_shift_h, num_shift);
  Kokkos::deep_copy(num_comp_h, num_comp);
  Kokkos::deep_copy(num_dupl_h, num_dupl);
  printf("Number of chunks: %u\n", num_chunks);
  printf("Number of new chunks: %u\n", num_new_h(0));
  printf("Number of same chunks: %u\n", num_same_h(0));
  printf("Number of shift chunks: %u\n", num_shift_h(0));
  printf("Number of comp nodes: %u\n", num_comp_h(0));
  printf("Number of dupl nodes: %u\n", num_dupl_h(0));
#endif
  Kokkos::fence();
}

template<uint32_t N>
bool restart( const Kokkos::View<uint8_t*> data,
              const Kokkos::View<uint8_t*> current, 
              const Kokkos::View<uint8_t*> restart, 
              const uint32_t chunk_size, 
              const uint32_t tree_id, 
              const CompactTable<N>& distinct, 
              const CompactTable<N>& shared) {
  uint64_t data_len = data.size();
  uint32_t num_chunks = data.size() / chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  uint32_t num_nodes = 2*num_chunks-1;
  uint32_t leaf_start = num_chunks-1;
  Kokkos::Bitset<Kokkos::DefaultExecutionSpace> done(num_chunks);
  done.reset();

  Kokkos::parallel_for("Restart distinct", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      Array<N> hist = distinct.value_at(i);
      for(uint32_t j=0; j<hist.size(); j++) {
        if(hist(j) == tree_id) {
          CompactNodeInfo info = distinct.key_at(i);
          uint32_t start = info.node;
          uint32_t size = info.size;
          for(uint32_t k=start-leaf_start; k<start-leaf_start+size; k++) {
            done.set(k);
          }
          uint32_t end = ((start-leaf_start)+info.size)*chunk_size;
          if(end > data.size())
            end = data.size();
          for(uint32_t k=(start-leaf_start)*chunk_size; k<end; k++) {
            restart(k) = current(k);
          }
          break;
        }
      }
    }
  });
  Kokkos::fence();

  uint32_t num_distinct = done.count();
  printf("Restarted %u chunks\n", num_distinct);

  Kokkos::parallel_for("Restart shared", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      Array<N> hist = shared.value_at(i);
      CompactNodeInfo info = shared.key_at(i);
      for(uint32_t j=0; j<hist.size(); j++) {
        if(hist(j) == tree_id) {
          uint32_t start = info.node;
          for(uint32_t k=start-leaf_start; k<start-leaf_start+info.size; k++) {
//if(done.test(k))
//printf("Overlap %u Shouldn't happen.\n", k);
            done.set(k);
          }
          uint32_t end = ((start-leaf_start)+info.size)*chunk_size;
          if(end > data.size())
            end = data.size();
          for(uint32_t k=(start-leaf_start)*chunk_size; k<end; k++) {
            restart(k) = current(k);
          }
          break;
        }
      }
    }
  });
  Kokkos::fence();

  uint32_t num_shared = done.count() - num_distinct;
  printf("Restarted %u shared chunks\n", num_shared);

  Kokkos::parallel_for("Restart identical", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
    if(!done.test(i)) {
      uint32_t size = chunk_size;
      if(i == num_chunks-1) {
        size = data.size()-i*chunk_size;
      } 
      for(uint32_t j=i*chunk_size; j<i*chunk_size + size; j++) {
//        restart(j) = data(j);
        restart(j) = current(j);
      }
      done.set(i);
    }
  });
  Kokkos::fence();
  uint32_t num_same = done.count() - num_distinct - num_shared;
  printf("Restarted %u identical chunks\n", num_same);
  
  uint32_t num_diff = 0;
  Kokkos::parallel_reduce("Check is same", Kokkos::RangePolicy<>(0, data.size()), KOKKOS_LAMBDA(const uint32_t i, uint32_t& update) {
    if(current(i) != restart(i))
      update += 1;
  }, num_diff);

  if(num_diff == 0) {
    return true;
  } else {
printf("Number of differences: %u Bytes out of %u\n", num_diff, data.size());
    return false;
  }
}

void count_distinct_nodes(const MerkleTree& tree, Queue& queue, const uint32_t tree_id, const DistinctMap& distinct) {
  Kokkos::View<uint32_t[1]> n_distinct("Num distinct\n");
  Kokkos::View<uint32_t[1]>::HostMirror n_distinct_h = Kokkos::create_mirror_view(n_distinct);
  Kokkos::deep_copy(n_distinct, 0);
  uint32_t q_size = queue.size();
  while(q_size > 0) {
    Kokkos::parallel_for(q_size, KOKKOS_LAMBDA(const uint32_t entry) {
      uint32_t node = queue.pop();
      HashDigest digest = tree(node);
      if(distinct.exists(digest)) {
        uint32_t existing_id = distinct.find(digest);
        NodeInfo info = distinct.value_at(existing_id);
        Kokkos::atomic_add(&n_distinct(0), 1);
        if(info.node == node && info.tree == tree_id) {
          uint32_t child_l = 2*node+1;
          if(child_l < queue.capacity())
            queue.push(child_l);
          uint32_t child_r = 2*node+2;
          if(child_r < queue.capacity())
            queue.push(child_r);
	}
      } else {
        printf("Node %u digest not in map. This shouldn't happen.\n", node);
      }
    });
    q_size = queue.size();
  }
  Kokkos::deep_copy(n_distinct_h, n_distinct);
  Kokkos::fence();
  printf("Number of distinct nodes: %u out of %u\n", n_distinct_h(0), tree.tree_d.extent(0));
}

void print_nodes(const MerkleTree& tree, const uint32_t tree_id, const DistinctMap& distinct) {
  Kokkos::View<uint32_t[1]> n_distinct("Num distinct\n");
  Kokkos::View<uint32_t[1]>::HostMirror n_distinct_h = Kokkos::create_mirror_view(n_distinct);
  Kokkos::deep_copy(n_distinct, 0);
  Queue queue(tree.tree_d.extent(0));
  queue.host_push(0);
  uint32_t q_size = queue.size();
  while(q_size > 0) {
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(const uint32_t entry) {
      for(uint32_t i=0; i<q_size; i++) {
        uint32_t node = queue.pop();
        HashDigest digest = tree(node);
        if(distinct.exists(digest)) {
          uint32_t existing_id = distinct.find(digest);
          NodeInfo info = distinct.value_at(existing_id);
          printf("Distinct Node %u: (%u,%u,%u)\n", node, info.node, info.src, info.tree);
          Kokkos::atomic_add(&n_distinct(0), 1);
          if(info.node == node && info.tree == tree_id) {
            uint32_t child_l = 2*node+1;
            if(child_l < queue.capacity())
              queue.push(child_l);
            uint32_t child_r = 2*node+2;
            if(child_r < queue.capacity())
              queue.push(child_r);
          }
        } else {
          printf("Node %u digest not in map. This shouldn't happen.\n", node);
        }
      }
    });
    q_size = queue.size();
  }
  Kokkos::deep_copy(n_distinct_h, n_distinct);
  Kokkos::fence();
  printf("Number of distinct nodes: %u out of %u\n", n_distinct_h(0), tree.tree_d.extent(0));
}

#endif // KOKKOS_MERKLE_TREE_HPP
