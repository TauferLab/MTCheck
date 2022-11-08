#ifndef DEDUPLICATOR_HPP
#define DEDUPLICATOR_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <utility>
#include "stdio.h"
#include "dedup_merkle_tree.hpp"
#include "write_merkle_tree_chkpt.hpp"
#include "kokkos_hash_list.hpp"
#include "utils.hpp"

enum DedupMode {
  Full,
  Naive,
  List,
  Tree
};

template<typename HashFunc>
class Deduplicator {
  public:
    HashFunc hash_func;
    MerkleTree tree;
    HashList leaves;
    DigestNodeIDMap first_ocur_d;
    CompactTable first_ocur_updates_d;
    CompactTable shift_dupl_updates_d;
    uint32_t chunk_size;
    uint32_t num_chunks;
    uint32_t num_nodes;
    uint32_t current_id;
    uint64_t data_len;

    Deduplicator() {
      tree = MerkleTree(1);
      first_ocur_d = DigestNodeIDMap(1);
      first_ocur_updates_d = CompactTable(1);
      shift_dupl_updates_d = CompactTable(1);
      chunk_size = 4096;
      current_id = 0;
    }

    Deduplicator(uint32_t bytes_per_chunk) {
      tree = MerkleTree(1);
      first_ocur_d = DigestNodeIDMap(1);
      first_ocur_updates_d = CompactTable(1);
      shift_dupl_updates_d = CompactTable(1);
      chunk_size = bytes_per_chunk;
      current_id = 0;
    }

    void dedup(Kokkos::View<uint8_t*>& data, bool make_baseline) {
      data_len = data.size();
      num_chunks = data_len/chunk_size;
      if(num_chunks*chunk_size < data_len)
        num_chunks += 1;
      num_nodes = 2*num_chunks-1;

      if(current_id == 0) {
        tree = MerkleTree(num_chunks);
        first_ocur_d = DigestNodeIDMap(num_nodes);
        first_ocur_updates_d = CompactTable(num_chunks);
        shift_dupl_updates_d = CompactTable(num_chunks);
      }
      if(tree.tree_d.size() < num_nodes) {
        Kokkos::resize(tree.tree_d, num_nodes);
        Kokkos::resize(tree.tree_h, num_nodes);
      }
      if(first_ocur_d.capacity() < first_ocur_d.size()+num_nodes)
        first_ocur_d.rehash(num_nodes);
      if(num_chunks != first_ocur_updates_d.capacity()) {
        first_ocur_updates_d.rehash(num_chunks);
        shift_dupl_updates_d.rehash(num_chunks);
      }

      if((current_id == 0) || make_baseline) {
        first_ocur_updates_d.clear();
        shift_dupl_updates_d.clear();
        create_merkle_tree(hash_func, tree, data, chunk_size, current_id, first_ocur_d, shift_dupl_updates_d);
      } else {
        first_ocur_updates_d.clear();
        shift_dupl_updates_d.clear();
        deduplicate_data(data, chunk_size, hash_func, tree, current_id, first_ocur_d, shift_dupl_updates_d, first_ocur_updates_d);
//        num_subtree_roots(data, chunk_size, tree, current_id, first_ocur_d, shift_dupl_updates_d, first_ocur_updates_d);
      }
      printf("First occurrence map capacity: %lu, size: %lu\n", first_ocur_d.capacity(), first_ocur_d.size());
      printf("First occurrence update capacity: %lu, size: %lu\n", first_ocur_updates_d.capacity(), first_ocur_updates_d.size());
      printf("Shift duplicate update capacity: %lu, size: %lu\n", shift_dupl_updates_d.capacity(), shift_dupl_updates_d.size());

      Kokkos::deep_copy(tree.tree_h, tree.tree_d);
      current_id += 1;
    }
};

#endif // DEDUPLICATOR_HPP
