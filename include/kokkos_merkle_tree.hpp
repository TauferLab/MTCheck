#ifndef KOKKOS_MERKLE_TREE_HPP
#define KOKKOS_MERKLE_TREE_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <climits>
#include "hash_functions.hpp"
#include "map_helpers.hpp"
#include "kokkos_queue.hpp"

//template<uint32_t N>
class MerkleTree {
public:
  Kokkos::View<HashDigest*> tree_d;
  Kokkos::View<HashDigest*>::HostMirror tree_h;

  MerkleTree(const uint32_t num_leaves) {
    tree_d = Kokkos::View<HashDigest*>("Merkle tree", (2*num_leaves-1));
    tree_h = Kokkos::create_mirror_view(tree_d);
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
MerkleTree create_merkle_tree(Hasher& hasher, Kokkos::View<uint8_t*>& data, const uint32_t chunk_size) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  const uint32_t num_nodes = 2*num_chunks-1;
  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  const uint32_t leaf_start = num_chunks-1;
  MerkleTree tree = MerkleTree(num_chunks);
  for(int32_t level=num_levels-1; level>=0; level--) {
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
  return tree;
}

template <class Hasher>
void create_merkle_tree(Hasher& hasher, MerkleTree& tree, Kokkos::View<uint8_t*>& data, const uint32_t chunk_size) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  const uint32_t num_nodes = 2*num_chunks-1;
  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  const uint32_t leaf_start = num_chunks-1;
  for(int32_t level=num_levels-1; level>=0; level--) {
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

void find_distinct_subtrees(const MerkleTree& tree, const uint32_t tree_id, DistinctMap& distinct_map, SharedMap& shared_map) {
  auto policy = Kokkos::RangePolicy<>(0, tree.tree_d.extent(0));
  Kokkos::parallel_for("Insert nodes", policy, KOKKOS_LAMBDA(const uint32_t i) {
    NodeInfo node_info(i, i, tree_id);
    HashDigest digest = tree.tree_d(i);
    auto result = distinct_map.insert(digest, node_info);
    if(result.existing()) {
      NodeInfo& old_info = distinct_map.value_at(result.index());
      if(i < old_info.node) {
        uint32_t prior_node = old_info.node;
        old_info.node = node_info.node;
        old_info.src = node_info.src;
        old_info.tree = node_info.tree;
        auto shared_insert = shared_map.insert(prior_node, i);
        if(shared_insert.failed())
          printf("Failed to insert in the distinct and shared map\n");
      } else {
        auto shared_insert = shared_map.insert(i, old_info.node);
        if(shared_insert.failed())
          printf("Failed to insert in the distinct and shared map\n");
      }
    } 
  });
  Kokkos::fence();
}

template <class Hasher>
MerkleTree create_merkle_tree_find_distinct_subtrees(Hasher& hasher, 
                                                     Kokkos::View<uint8_t*>& data, 
                                                     const uint32_t chunk_size, 
                                                     const uint32_t tree_id, 
                                                     DistinctMap& distinct_map, 
                                                     SharedMap& shared_map) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  const uint32_t num_nodes = 2*num_chunks-1;
  const uint32_t num_levels = static_cast<uint32_t>(ceil(log2(num_nodes+1)));
  const uint32_t leaf_start = num_chunks-1;
  MerkleTree tree = MerkleTree(num_chunks);
  for(int32_t level=num_levels-1; level>=0; level--) {
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
      NodeInfo node_info(i, i, tree_id);
      auto result = distinct_map.insert(tree(i), node_info);
      if(result.existing()) {
        NodeInfo& old_info = distinct_map.value_at(result.index());
        if(i < old_info.node) {
          uint32_t prior_node = old_info.node;
          old_info.node = node_info.node;
          old_info.src = node_info.src;
          old_info.tree = node_info.tree;
          auto shared_insert = shared_map.insert(prior_node, i);
          if(shared_insert.failed())
            printf("Failed to insert in the distinct and shared map\n");
        } else {
          auto shared_insert = shared_map.insert(i, old_info.node);
          if(shared_insert.failed())
            printf("Failed to insert in the distinct and shared map\n");
        }
      } 
    });
  }
  Kokkos::fence();
  return tree;
}

void compare_trees(const MerkleTree& tree, const uint32_t tree_id, DistinctMap& distinct_map, DistinctMap& prior_map, Queue& queue) {
  queue.host_push(0);
  uint32_t num_comp = 0;
  uint32_t q_size = queue.size();
  while(q_size > 0) {
    num_comp += q_size;
    Kokkos::parallel_for("Compare trees", Kokkos::RangePolicy<>(0, q_size), KOKKOS_LAMBDA(const uint32_t entry) {
      uint32_t node = queue.pop();
      HashDigest digest = tree(node);
      if(distinct_map.exists(digest)) {
        uint32_t distinct_index = distinct_map.find(digest);
        NodeInfo& info = distinct_map.value_at(distinct_index);
        if(info.node == node) {
          if(prior_map.exists(digest)) {
            uint32_t prior_index = prior_map.find(digest);
            NodeInfo old = prior_map.value_at(prior_index);
            info.src = old.src;
            info.tree = old.tree;
          } else {
            uint32_t child_l = 2*node+1;
            uint32_t child_r = 2*node+2;
            if(child_l < queue.capacity()) {
              queue.push(child_l);
            }
            if(child_r < queue.capacity()) {
              queue.push(child_r);
            }
            
          }
        }
      }
    });
    q_size = queue.size();
  }
  Kokkos::fence();
  printf("Number of comparisons (Merkle Tree): %u\n", num_comp);
}

void compare_trees(const MerkleTree& tree, const uint32_t tree_id, DistinctMap& distinct_map, DistinctMap& prior_map) {
  Kokkos::View<uint32_t*> queue = Kokkos::View<uint32_t*>("queue", tree.tree_d.extent(0));
  Kokkos::deep_copy(queue, 0);
  Kokkos::View<uint32_t[1]> q_start("Start index");
  Kokkos::View<uint32_t[1]> q_end("End index");
  Kokkos::View<uint32_t[1]> q_len("Length");
  Kokkos::View<uint32_t[1]>::HostMirror q_len_h = Kokkos::create_mirror_view(q_len);
  Kokkos::deep_copy(q_start, 0);
  Kokkos::deep_copy(q_end, 1);
  Kokkos::deep_copy(q_len, 1);
  q_len_h(0) = 1;
  uint32_t num_comp = 0;
  while(q_len_h(0) > 0) {
    Kokkos::deep_copy(q_len_h, q_len);
    num_comp += q_len_h(0);
    Kokkos::parallel_for("Compare trees", Kokkos::RangePolicy<>(0, q_len_h(0)), KOKKOS_LAMBDA(const uint32_t entry) {
      uint32_t start = Kokkos::atomic_fetch_add(&q_start(0), 1);
      start = start % queue.extent(0);
      Kokkos::atomic_decrement(&q_len(0));
      uint32_t node = queue(start);
      HashDigest digest = tree(node);
      if(distinct_map.exists(digest)) {
        uint32_t distinct_index = distinct_map.find(digest);
        NodeInfo& info = distinct_map.value_at(distinct_index);
        if(info.node == node) {
          if(prior_map.exists(digest)) {
            uint32_t prior_index = prior_map.find(digest);
            NodeInfo old = prior_map.value_at(prior_index);
            info.src = old.src;
            info.tree = old.tree;
          } else {
            uint32_t child_l = 2*node+1;
            uint32_t child_r = 2*node+2;
            if(child_l < queue.extent(0)) {
              uint32_t end = Kokkos::atomic_fetch_add(&q_end(0), 1);
              end = end % queue.extent(0);
              Kokkos::atomic_increment(&q_len(0));
              queue[end] = child_l;
            }
            if(child_r < queue.extent(0)) {
              uint32_t end = Kokkos::atomic_fetch_add(&q_end(0), 1);
              end = end % queue.extent(0);
              Kokkos::atomic_increment(&q_len(0));
              queue[end] = child_r;
            }
            
          }
        }
      }
    });
  }
  Kokkos::fence();
  printf("Number of comparisons (Merkle Tree): %u\n", num_comp);
}

void compare_trees_fused(const MerkleTree& tree, const uint32_t tree_id, DistinctMap& distinct_map, SharedMap& shared_map, DistinctMap& prior_map) {
  Kokkos::View<uint32_t*> queue = Kokkos::View<uint32_t*>("queue", tree.tree_d.extent(0));
  Kokkos::deep_copy(queue, 0);
  Kokkos::View<uint32_t[1]> q_start("Start index");
  Kokkos::View<uint32_t[1]> q_end("End index");
  Kokkos::View<uint32_t[1]> q_len("Length");
  Kokkos::View<uint32_t[1]>::HostMirror q_len_h = Kokkos::create_mirror_view(q_len);
  Kokkos::deep_copy(q_start, 0);
  Kokkos::deep_copy(q_end, 1);
  Kokkos::deep_copy(q_len, 1);
  q_len_h(0) = 1;
  uint32_t num_comp = 0;
  while(q_len_h(0) > 0) {
    Kokkos::deep_copy(q_len_h, q_len);
    num_comp += q_len_h(0);
    Kokkos::parallel_for(q_len_h(0), KOKKOS_LAMBDA(const uint32_t entry) {
      uint32_t start = Kokkos::atomic_fetch_add(&q_start(0), 1);
      start = start % queue.extent(0);
      Kokkos::atomic_decrement(&q_len(0));
      uint32_t node = queue(start);
//printf("Processing node %u\n", node);
      HashDigest digest = tree(node);
      NodeInfo info(node, node, tree_id);
      if(!distinct_map.exists(digest)) {
        if(prior_map.exists(digest)) {
//printf("Node (%u): Hash does not exist in the distinct map but exists in the prior map\n", node);
          uint32_t prior_idx = prior_map.find(digest);
          NodeInfo prior_info = prior_map.value_at(prior_idx);
//printf("Node (%u): Inserting prior entry (%u,%u,%u)\n", node, prior_info.node, prior_info.src, prior_info.tree);
          info.src = prior_info.src;
          info.tree = prior_info.tree;
          auto insert_res = distinct_map.insert(digest, info);
          if(insert_res.failed()) {
            //printf("Failed to insert prior existing entry into distinct map\n");
          }
        } else {
//printf("Node (%u): Hash does not exist in the distinct map or the prior map\n", node);
//printf("Node (%u): Inserting new entry (%u,%u,%u)\n", node, info.node, info.src, info.tree);
          auto insert_res = distinct_map.insert(digest, info);
          if(insert_res.failed()) {
            //printf("Failed to insert new entry into distinct map\n");
          }
          uint32_t child_l = 2*node+1;
          uint32_t child_r = 2*node+2;
          if(child_l < queue.extent(0)) {
            uint32_t end = Kokkos::atomic_fetch_add(&q_end(0), 1);
            end = end % queue.extent(0);
            Kokkos::atomic_increment(&q_len(0));
            queue[end] = child_l;
          }
          if(child_r < queue.extent(0)) {
            uint32_t end = Kokkos::atomic_fetch_add(&q_end(0), 1);
            end = end % queue.extent(0);
            Kokkos::atomic_increment(&q_len(0));
            queue[end] = child_r;
          }
        }
      } else if(distinct_map.exists(digest)) {
        uint32_t existing_idx = distinct_map.find(digest);
        NodeInfo& existing = distinct_map.value_at(existing_idx);
//printf("Node (%u): Already exists (%u,%u,%u)\n", node, existing.node, existing.src, existing.tree);
        if(node < existing.node) {
            if(existing.tree == tree_id) {
              existing.node = node;
              existing.src = node;
            } else {
              existing.node = node;
            }
//            uint32_t child_l = 2*node+1;
//            uint32_t child_r = 2*node+2;
//            if(child_l < queue.extent(0)) {
//              uint32_t end = Kokkos::atomic_fetch_add(&q_end(0), 1);
//              end = end % queue.extent(0);
//              Kokkos::atomic_increment(&q_len(0));
//              queue[end] = child_l;
//            }
//            if(child_r < queue.extent(0)) {
//              uint32_t end = Kokkos::atomic_fetch_add(&q_end(0), 1);
//              end = end % queue.extent(0);
//              Kokkos::atomic_increment(&q_len(0));
//              queue[end] = child_r;
//            }
//          }
        }
      } else {
        printf("Failed to insert in either the distinct and shared maps\n");
      }
    });
  }
  printf("Number of comparisons (Merkle Tree): %u\n", num_comp);
  Kokkos::fence();
}

void print_distinct_nodes(const MerkleTree& tree, const uint32_t tree_id, const DistinctMap& distinct) {
//  printf("True distinct nodes \n");
  Kokkos::View<uint32_t*> queue = Kokkos::View<uint32_t*>("queue", tree.tree_d.extent(0));
  Kokkos::deep_copy(queue, 0);
  Kokkos::View<uint32_t[1]> q_start("Start index");
  Kokkos::View<uint32_t[1]> q_end("End index");
  Kokkos::View<uint32_t[1]> q_len("Length");
  Kokkos::View<uint32_t[1]>::HostMirror q_len_h = Kokkos::create_mirror_view(q_len);
  Kokkos::View<uint32_t[1]> n_distinct("Num distinct\n");
  Kokkos::View<uint32_t[1]>::HostMirror n_distinct_h = Kokkos::create_mirror_view(n_distinct);
  Kokkos::deep_copy(q_start, 0);
  Kokkos::deep_copy(q_end, 1);
  Kokkos::deep_copy(q_len, 1);
  Kokkos::deep_copy(q_len_h, q_len);
  Kokkos::deep_copy(n_distinct, 0);
  while(q_len_h(0) > 0) {
    Kokkos::deep_copy(q_len_h, q_len);
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(const uint32_t entry) {
      for(uint32_t i=0; i<q_len(0); i++) {
        uint32_t start = Kokkos::atomic_fetch_add(&q_start(0), 1);
        start = start % queue.extent(0);
        Kokkos::atomic_decrement(&q_len(0));
        uint32_t node = queue(start);
        HashDigest digest = tree(node);
        if(distinct.exists(digest)) {
          uint32_t distinct_index = distinct.find(digest);
          NodeInfo info = distinct.value_at(distinct_index);
          if(node == info.node)  {
//            printf("Node %u: (%u,%u,%u)\n", node, info.node, info.src, info.tree);
            if(info.tree == tree_id) {
            n_distinct(0) += 1;
              uint32_t child_l = 2*node+1;
              uint32_t child_r = 2*node+2;
              if(child_l < queue.extent(0)) {
                uint32_t end = Kokkos::atomic_fetch_add(&q_end(0), 1);
                end = end % queue.extent(0);
                Kokkos::atomic_increment(&q_len(0));
                queue[end] = child_l;
              }
              if(child_r < queue.extent(0)) {
                uint32_t end = Kokkos::atomic_fetch_add(&q_end(0), 1);
                end = end % queue.extent(0);
                Kokkos::atomic_increment(&q_len(0));
                queue[end] = child_r;
              }
            }
          }
        }
      }
    });
  }
  Kokkos::deep_copy(n_distinct_h, n_distinct);
  Kokkos::fence();
  printf("Number of distinct nodes: %u\n", n_distinct_h(0));
}

#endif // KOKKOS_MERKLE_TREE_HPP
