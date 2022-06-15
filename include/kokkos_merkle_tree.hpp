#ifndef KOKKOS_MERKLE_TREE_HPP
#define KOKKOS_MERKLE_TREE_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <climits>
#include "hash_functions.hpp"
#include "map_helpers.hpp"
#include "kokkos_queue.hpp"

#define STATS

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

//template<typename Scheduler, class Hasher>
//struct CreateTreeTask {
//  using sched_type  = Scheduler;
//  using future_type = Kokkos::BasicFuture<uint32_t, Scheduler>;
//  using value_type  = uint32_t;
//
//  uint32_t node;
//  MerkleTree tree;
//  Kokkos::View<uint8_t*> data;
//  uint32_t chunk_size;
//  Hasher hasher;
//  future_type child_l_fut;
//  future_type child_r_fut;
//
//  KOKKOS_INLINE_FUNCTION
//  CreateTreeTask(const uint32_t n, Hasher& _hasher, const MerkleTree& merkle_tree, const Kokkos::View<uint8_t*>& _data, const uint32_t size) : 
//                  node(n), hasher(_hasher), tree(merkle_tree), data(_data), chunk_size(size) {}
//
//  KOKKOS_INLINE_FUNCTION
//  void operator()(typename sched_type::member_type& member, uint32_t& result) {
//    auto& sched = member.scheduler();
//
//    uint32_t num_chunks = data.size()/chunk_size;
//    if(num_chunks*chunk_size < data.size())
//      num_chunks += 1;
//    const uint32_t leaf_start = num_chunks-1;
//
//    if((node >= leaf_start) || !child_l_fut.is_null() && !child_r_fut.is_null()) {
//      uint32_t num_bytes = chunk_size;
//      if((node-leaf_start) == num_chunks-1)
//        num_bytes = data.size()-((node-leaf_start)*chunk_size);
//      if(node >= leaf_start) {
//        hasher.hash(data.data()+((node-leaf_start)*chunk_size), 
//                    num_bytes, 
//                    (uint8_t*)(tree(node).digest));
//      } else {
//        hasher.hash((uint8_t*)&tree(2*node+1), 2*hasher.digest_size(), (uint8_t*)&tree(node));
//      }
//      result = 1;
//    } else {
//      int active_children = 0;
//      uint32_t child_l = 2*node+1;
//      if(child_l < tree.tree_d.extent(0)) {
//        child_l_fut = Kokkos::task_spawn(Kokkos::TaskSingle(sched, Kokkos::TaskPriority::High), CreateTreeTask(child_l, hasher, tree, data, chunk_size));
//        active_children += 1;
//      }
//      uint32_t child_r = 2*node+2;
//      if(child_r < tree.tree_d.extent(0)) {
//        child_r_fut = Kokkos::task_spawn(Kokkos::TaskSingle(sched, Kokkos::TaskPriority::High), CreateTreeTask(child_r, hasher, tree, data, chunk_size));
//        active_children += 1;
//      }
//      if(active_children == 2) {
//        Kokkos::BasicFuture<void, Scheduler> dep[] = {child_l_fut, child_r_fut};
//        Kokkos::BasicFuture<void, Scheduler> all_children = sched.when_all(dep, 2);
//        Kokkos::respawn(this, all_children, Kokkos::TaskPriority::High);
//      }
//    }
//  }
//};
//
//size_t estimate_required_memory(int n_nodes) {
//  return n_nodes*2000;
//}
//
//
//template <class Hasher>
//void create_merkle_tree_task(Hasher& hasher, MerkleTree& tree, Kokkos::View<uint8_t*>& data, const uint32_t chunk_size) {
//  using scheduler_type = Kokkos::TaskScheduler<Kokkos::DefaultExecutionSpace>;
//  using memory_space = typename scheduler_type::memory_space;
//  using memory_pool = typename scheduler_type::memory_pool;
//  uint32_t num_chunks = data.size()/chunk_size;
//  if(num_chunks*chunk_size < data.size())
//    num_chunks += 1;
//  const uint32_t num_nodes = 2*num_chunks-1;
//  auto mpool = memory_pool(memory_space{}, estimate_required_memory(2*num_nodes-1));
//  auto root_sched = scheduler_type(mpool);
//  Kokkos::BasicFuture<uint32_t, scheduler_type> f = Kokkos::host_spawn(Kokkos::TaskSingle(root_sched), 
//                                                                  CreateTreeTask<scheduler_type, Hasher>(0, hasher, tree, data, chunk_size));
//  Kokkos::wait(root_sched);
//}

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
  uint64_t leftmost = (2*node)+1;
  uint64_t rightmost = (2*node)+2;
  while(leftmost < num_nodes) {
    leftmost = (2*leftmost)+1;
    rightmost = (2*rightmost)+2;
  }
  leftmost = (leftmost-1)/2;
  rightmost = (rightmost-2)/2;
  if(rightmost > num_nodes)
    rightmost = num_nodes;
  return static_cast<uint32_t>(rightmost-leftmost+1);
}

KOKKOS_INLINE_FUNCTION uint32_t leftmost_leaf(uint32_t node, uint32_t num_nodes) {
  uint64_t leftmost = (2*node)+1;
  uint64_t rightmost = (2*node)+2;
  while(leftmost < num_nodes) {
    leftmost = (2*leftmost)+1;
    rightmost = (2*rightmost)+2;
  }
//  leftmost = (leftmost-1)/2;
//  rightmost = (rightmost-2)/2;
  if(rightmost > num_nodes)
    rightmost = num_nodes;
  return static_cast<uint32_t>(leftmost);
}

KOKKOS_INLINE_FUNCTION uint32_t rightmost_leaf(uint32_t node, uint32_t num_nodes) {
  uint64_t leftmost = (2*node)+1;
  uint64_t rightmost = (2*node)+2;
  while(leftmost < num_nodes) {
    leftmost = (2*leftmost)+1;
    rightmost = (2*rightmost)+2;
  }
//  leftmost = (leftmost-1)/2;
//  rightmost = (rightmost-2)/2;
  if(rightmost > num_nodes)
    rightmost = num_nodes;
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
DistinctMap local_distinct = DistinctMap(prior_distinct_map.capacity());
Kokkos::deep_copy(local_distinct, prior_distinct_map);
#ifdef STATS
  Kokkos::View<uint32_t[1]> num_same("Number of chunks that remain the same");
  Kokkos::View<uint32_t[1]> num_new("Number of chunks that are new");
  Kokkos::View<uint32_t[1]> num_shift("Number of chunks that exist but in different spaces");
  Kokkos::View<uint32_t[1]> num_comp("Number of compressed nodes");
  Kokkos::View<uint32_t[1]>::HostMirror num_same_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_new_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_shift_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_comp_h = Kokkos::create_mirror_view(num_same);
  Kokkos::deep_copy(num_same, 0);
  Kokkos::deep_copy(num_new, 0);
  Kokkos::deep_copy(num_shift, 0);
  Kokkos::deep_copy(num_comp, 0);
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
            NodeInfo old = prior_distinct_map.value_at(result.index());
            shared_map.insert(node, old);
#ifdef STATS
            Kokkos::atomic_add(&(num_shift(0)), 1);
          } else if(result.success()) {
            Kokkos::atomic_add(&(num_new(0)), 1);
#endif
          }
        } else {
          NodeInfo info = NodeInfo(node, node, tree_id);
          uint32_t index = prior_distinct_map.find(tree(node));
          if(!prior_distinct_map.valid_at(index)) { // New chunk 
            auto result = local_distinct.insert(tree(node), info);
            if(result.success()) {
#ifdef STATS
              Kokkos::atomic_add(&(num_new(0)), 1);
#endif
              Kokkos::atomic_add(&(tree.distinct_children_d((node-1)/2)), 1);
              Kokkos::atomic_add(&nodes_leftover(0), 1);
            }
          } else { // Chunk already exists
            NodeInfo old_distinct = prior_distinct_map.value_at(index);
if(old_distinct.node == node) {
  Kokkos::atomic_add(&num_same(0), 1);
} else {
  uint32_t idx = prior_shared_map.find(node);
  if(prior_shared_map.valid_at(idx) && prior_shared_map.value_at(node).node == node) {
    Kokkos::atomic_add(&num_same(0), 1);
  } else if(prior_shared_map.valid_at(idx)) {
    Kokkos::atomic_add(&num_shift(0), 1);
  } else if(!prior_shared_map.valid_at(idx)) {
    Kokkos::atomic_add(&num_shift(0), 1);
  }
}
            if(node != old_distinct.node) { // Chunk exists but at a different offset
              uint32_t prior_shared_idx = prior_shared_map.find(node);
              if(prior_shared_map.valid_at(prior_shared_idx)) { // Node is in prior shared map
                NodeInfo prior_info = prior_shared_map.value_at(prior_shared_idx);
                if(prior_info.node != node) { // Chunk has changed since prior checkpoint
                  shared_map.insert(node, prior_info);
//#ifdef STATS
//                  Kokkos::atomic_add(&(num_shift(0)), 1);
//                } else {
//                  Kokkos::atomic_add(&(num_same(0)), 1);
//#endif
                }
              } else { // Node not in prior shared map
                shared_map.insert(node, old_distinct);
//#ifdef STATS
//                Kokkos::atomic_add(&(num_shift(0)), 1);
//#endif
              }
//#ifdef STATS
//            } else { // Chunk exists and hasn't changed node
//              Kokkos::atomic_add(&(num_same(0)), 1);
//#endif
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
        if((tree.distinct_children_d(child_l) == 2) || (child_l >= leaf_start)) {
          uint32_t size = num_leaf_descendents(child_l, num_nodes);
//uint32_t left = leftmost_leaf(child_l, num_nodes);
//uint32_t right = rightmost_leaf(child_l, num_nodes);
//printf("Node: %u, num nodes: %u, leftmost: %u, rightmost: %u, size: %u\n", node, num_nodes, left, right, size);
//for(uint32_t leaf=left; leaf<=right; leaf++) {
//  prior_distinct_map.erase(tree(leaf));
//}
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
//uint32_t left = leftmost_leaf(child_r, num_nodes);
//uint32_t right = rightmost_leaf(child_r, num_nodes);
//printf("Node: %u, num nodes: %u, leftmost: %u, rightmost: %u, size: %u\n", node, num_nodes, left, right, size);
//for(uint32_t leaf=left; leaf<=right; leaf++) {
//  prior_distinct_map.erase(tree(leaf));
//}
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
//    prev_leftover = nodes_leftover_h(0);
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
  printf("Number of chunks: %u\n", num_chunks);
  printf("Number of new chunks: %u\n", num_new_h(0));
  printf("Number of same chunks: %u\n", num_same_h(0));
  printf("Number of shift chunks: %u\n", num_shift_h(0));
  printf("Number of comp nodes: %u\n", num_comp_h(0));
#endif
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
