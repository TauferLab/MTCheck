#ifndef KOKKOS_HASH_LIST_HPP
#define KOKKOS_HASH_LIST_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <climits>
#include "hash_functions.hpp"
#include "map_helpers.hpp"

class HashList {
public:
  Kokkos::View<HashDigest*> list_d;
  Kokkos::View<HashDigest*>::HostMirror list_h;

  HashList(const uint32_t num_leaves) {
    list_d = Kokkos::View<HashDigest*>("Hash list", num_leaves);
    list_h = Kokkos::create_mirror_view(list_d);
  }

  KOKKOS_INLINE_FUNCTION HashDigest& operator()(int32_t i) const {
    return list_d(i);
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
    Kokkos::deep_copy(list_h, list_d);
    uint32_t num_leaves = (list_h.extent(0));
    printf("============================================================\n");
    char buffer[80];
    for(unsigned int i=0; i<num_leaves; i++) {
      digest_to_hex_((uint8_t*)(list_h(i).digest), buffer);
      printf("Node: %u: %s \n", i, buffer);
    }
    printf("============================================================\n");
  }
};

template <class Hasher>
void create_hash_list(Hasher& hasher, HashList& list, Kokkos::View<uint8_t*>& data, const uint32_t chunk_size) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  Kokkos::parallel_for("Create Hash list", Kokkos::RangePolicy<>(0,num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t num_bytes = chunk_size;
        if(i == num_chunks-1)
          num_bytes = data.size()-i*chunk_size;
        hasher.hash(data.data()+(i*chunk_size), 
                    num_bytes, 
                    list(i).digest);
  });
  Kokkos::fence();
}

template <class Hasher>
HashList create_hash_list(Hasher& hasher, Kokkos::View<uint8_t*>& data, const uint32_t chunk_size) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  HashList list = HashList(num_chunks);
  create_hash_list(hasher, list, data, chunk_size);
  Kokkos::fence();
  return list;
}

void find_distinct_chunks(const HashList& list, const uint32_t list_id, DistinctMap& distinct_map, SharedMap& shared_map, DistinctMap& prior_map) {
  Kokkos::parallel_for("Find distinct chunks", Kokkos::RangePolicy<>(0,list.list_d.extent(0)), KOKKOS_LAMBDA(const uint32_t i) {
    HashDigest digest = list.list_d(i);
    NodeInfo info(i, i, list_id);
    auto result = distinct_map.insert(digest, info);
    if(result.failed()) 
      printf("Warning: Failed to insert (%u,%u,%u) into map for hashlist.\n", info.node, info.src, info.tree);

//    if(prior_map.exists(digest)) {
//      auto old_res = prior_map.find(digest);
//      NodeInfo old_node = prior_map.value_at(old_res);
//      NodeInfo new_node(i, old_node.src, old_node.tree);
//      auto distinct_res = distinct_map.insert(digest, new_node);
//      if(distinct_res.failed())
//        printf("Failed to insert entry into distinct map\n");
//    } else {
//      auto result = distinct_map.insert(digest, info);
//      if(result.existing()) {
//        NodeInfo old_info = distinct_map.value_at(result.index());
//        auto shared_res = shared_map.insert(i, old_info.node);
//        if(shared_res.failed())
//          printf("Failed to insert chunk into either distinct or shared maps\n");
//      }
//    }
  });
  printf("Number of comparisons (Hash List)  : %d\n", list.list_d.extent(0));
  Kokkos::fence();
}

template<class Hasher>
void compare_lists(Hasher& hasher, const HashList& list, const uint32_t list_id, const Kokkos::View<uint8_t*>& data, const uint32_t chunk_size, SharedMap& shared_map, DistinctMap& distinct_map, const SharedMap prior_shared_map, const DistinctMap& prior_distinct_map) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
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
  Kokkos::parallel_for("Find distinct chunks", Kokkos::RangePolicy<>(0,list.list_d.extent(0)), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t num_bytes = chunk_size;
    if(i == num_chunks-1)
      num_bytes = data.size()-i*chunk_size;
    hasher.hash(data.data()+(i*chunk_size), 
                num_bytes, 
                list(i).digest);
    HashDigest digest = list.list_d(i);
    NodeInfo info(i, i, list_id);
    uint32_t old_idx = prior_distinct_map.find(digest); // Search for digest in prior distinct map
    if(!prior_distinct_map.valid_at(old_idx)) { // Chunk not in prior chkpt
      auto result = distinct_map.insert(digest, info);
      if(result.existing()) {
        NodeInfo old = distinct_map.value_at(result.index());
        shared_map.insert(i, old);
#ifdef STATS
Kokkos::atomic_add(&num_dupl(0), 1);
#endif
      } else if(result.failed())  {
        printf("Warning: Failed to insert (%u,%u,%u) into map for hashlist.\n", info.node, info.src, info.tree);
#ifdef STATS
      } else {
Kokkos::atomic_add(&num_new(0), 1);
#endif
      }
    } else { // Chunk is in prior chkpt
      NodeInfo old_distinct = prior_distinct_map.value_at(old_idx);
      if(i != old_distinct.node) {
        uint32_t old_shared_idx = prior_shared_map.find(i);
        if(prior_shared_map.valid_at(old_shared_idx)) {
          NodeInfo old_shared = prior_shared_map.value_at(old_shared_idx);
          if(i != old_shared.node) {
            shared_map.insert(i, old_shared);
#ifdef STATS
Kokkos::atomic_add(&num_shift(0), 1);
          } else {
Kokkos::atomic_add(&num_same(0), 1);
#endif
          }
        } else {
          shared_map.insert(i, old_distinct);
#ifdef STATS
Kokkos::atomic_add(&num_shift(0), 1);
#endif
        }
#ifdef STATS
      } else {
Kokkos::atomic_add(&num_same(0), 1);
#endif
      }
    }
  });
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

void print_distinct_nodes(const HashList& list, const uint32_t id, const DistinctMap& distinct) {
  Kokkos::parallel_for(1, KOKKOS_LAMBDA(const uint32_t entry) {
    uint32_t num_distinct=0;
    for(uint32_t i=0; i<list.list_d.extent(0); i++) {
      HashDigest digest = list(i);
      if(distinct.exists(digest)) {
        uint32_t item = distinct.find(digest);
        NodeInfo info = distinct.value_at(item);
        if(info.tree == id) {
          num_distinct += 1;
//          printf("Node (%u): (%u,%u,%u)\n", i, info.node, info.src, info.tree);
        }
      }
    }
    printf("Num distinct chunks: %u\n", num_distinct);
  });
  Kokkos::fence();
}

void count_distinct_nodes(const HashList& list, const uint32_t tree_id, const DistinctMap& distinct) {
  Kokkos::View<uint32_t[1]> counter("Counter");
  Kokkos::View<uint32_t[1]>::HostMirror counter_h = Kokkos::create_mirror_view(counter);
  Kokkos::deep_copy(counter, 0);
  Kokkos::parallel_for("Count updated chunks", Kokkos::RangePolicy<>(0, list.list_d.extent(0)), KOKKOS_LAMBDA(const uint32_t chunk) {
    HashDigest digest = list(chunk);
    uint32_t idx = distinct.find(digest);
    if(distinct.exists(digest)) {
      NodeInfo info = distinct.value_at(idx);
      if((info.tree == tree_id) || ((info.tree != tree_id) && (info.node != chunk)) ) {
//      if(info.tree == tree_id) {
        Kokkos::atomic_add(&counter(0), 1);
      }
    }
  });
//  Kokkos::parallel_for("Count current nodes", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
//    if(distinct.valid_at(i)) {
//      NodeInfo info = distinct.value_at(i);
//      if(info.tree == tree_id) {
//        Kokkos::atomic_increment(&counter(0));
//      }
//    }
//  });
  Kokkos::deep_copy(counter_h, counter);
  printf("Number of distinct chunks: %u out of %u\n", counter_h(0), list.list_d.extent(0));
}


void count_distinct_nodes(const HashList& list, const uint32_t tree_id, const DistinctMap& distinct, const DistinctMap& prior) {
  Kokkos::View<uint32_t[1]> counter("Counter");
  Kokkos::View<uint32_t[1]>::HostMirror counter_h = Kokkos::create_mirror_view(counter);
  Kokkos::deep_copy(counter, 0);
  Kokkos::parallel_for("Count updated chunks", Kokkos::RangePolicy<>(0, list.list_d.extent(0)), KOKKOS_LAMBDA(const uint32_t chunk) {
    HashDigest digest = list(chunk);
    uint32_t idx = distinct.find(digest);
    uint32_t prior_idx = prior.find(digest);
    if(distinct.valid_at(idx)) {
      Kokkos::atomic_add(&counter(0), 1);
    } else if(prior.valid_at(prior_idx)) {
      NodeInfo info = prior.value_at(idx);
      if(info.node != chunk) {
        Kokkos::atomic_add(&counter(0), 1);
      }
    } else {
      printf("Could not find digest for chunk %u in prior or current map!\n", chunk);
    }
  });
  Kokkos::deep_copy(counter_h, counter);
  printf("Number of distinct chunks: %u out of %u\n", counter_h(0), list.list_d.extent(0));
}

//void estimate_metadata(const HashList& list, const uint32_t tree_id, const DistinctMap& distinct) {
//  Kokkos::View<uint32_t[1]> counter("Counter");
//  Kokkos::View<uint32_t[1]>::HostMirror counter_h = Kokkos::create_mirror_view(counter);
//  Kokkos::deep_copy(counter, 0);
//  Kokkos::parallel_for("Count updated chunks", Kokkos::RangePolicy<>(0, list.list_d.extent(0)), KOKKOS_LAMBDA(const uint32_t chunk) {
//    HashDigest digest = list(chunk);
//    uint32_t idx = distinct.find(digest);
//    if(distinct.exists(digest)) {
//      NodeInfo info = distinct.value_at(idx);
//      if((chunk == info.node && info.tree == tree_id) || (info.tree != tree_id)) {
//        Kokkos::atomic_add(&counter(0), 1);
//      }
//    }
//  });
////  Kokkos::parallel_for("Count current nodes", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
////    if(distinct.valid_at(i)) {
////      NodeInfo info = distinct.value_at(i);
////      if(info.tree == tree_id) {
////        Kokkos::atomic_increment(&counter(0));
////      }
////    }
////  });
//  Kokkos::deep_copy(counter_h, counter);
//  printf("Number of distinct chunks: %u out of %u\n", counter_h(0), list.list_d.extent(0));
//}

#endif

