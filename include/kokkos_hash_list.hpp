#ifndef KOKKOS_HASH_LIST_HPP
#define KOKKOS_HASH_LIST_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_ScatterView.hpp>
#include <climits>
#include <chrono>
#include <fstream>
#include <vector>
#include <queue>
#include <utility>
#include "hash_functions.hpp"
#include "map_helpers.hpp"
#include "utils.hpp"

class HashList {
public:
  Kokkos::View<HashDigest*> list_d;
  Kokkos::View<HashDigest*>::HostMirror list_h;

  HashList() {}

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

template<class Hasher>
void compare_lists_naive( Hasher& hasher, 
                    const HashList& prior_list,
                    const HashList& list, 
                    Kokkos::Bitset<Kokkos::DefaultExecutionSpace>& changes,
                    const uint32_t list_id, 
                    const Kokkos::View<uint8_t*>& data, 
                    const uint32_t chunk_size 
                    ) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  changes.reset();
  #ifdef STATS
  Kokkos::View<uint64_t[1]> num_same_d("Num same");
  Kokkos::View<uint64_t[1]>::HostMirror num_same_h = Kokkos::create_mirror_view(num_same_d);
  Kokkos::deep_copy(num_same_d, 0);
  #endif
  Kokkos::parallel_for("Find distinct chunks", Kokkos::RangePolicy<>(0,list.list_d.extent(0)), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t num_bytes = chunk_size;
    if(i == num_chunks-1)
      num_bytes = data.size()-i*chunk_size;
    hasher.hash(data.data()+(i*chunk_size), 
                num_bytes, 
                list(i).digest);
    HashDigest digest = list.list_d(i);
    if(list_id > 0) {
      bool same = true;
      for(uint32_t j=0; j<16; j++) {
        if(digest.digest[j] != prior_list.list_d(i).digest[j]) {
          same = false;
          break;
        }
      }
      if(!same) {
        changes.set(i);
      #ifdef STATS
      } else {
        Kokkos::atomic_add(&num_same_d(0), 1);
      #endif
      }
    } else {
      changes.set(i);
    }
  });
  Kokkos::fence();
  #ifdef STATS
  Kokkos::deep_copy(num_same_h, num_same_d);
  printf("Number of identical chunks: %lu\n", num_same_h(0));
  printf("Number of changes: %u\n", changes.count());
  #endif
}

template<class Hasher>
void compare_lists_local( Hasher& hasher, 
                    const HashList& list, 
                    const uint32_t list_id, 
                    const Kokkos::View<uint8_t*>& data, 
                    const uint32_t chunk_size, 
                    SharedNodeIDMap& shared_map, 
                    DistinctNodeIDMap& distinct_map, 
                    const SharedNodeIDMap prior_shared_map, 
                    const DistinctNodeIDMap& prior_distinct_map) {
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
  Kokkos::View<uint32_t[1]>::HostMirror num_new_h = Kokkos::create_mirror_view(num_new);
  Kokkos::View<uint32_t[1]>::HostMirror num_shift_h = Kokkos::create_mirror_view(num_shift);
  Kokkos::View<uint32_t[1]>::HostMirror num_comp_h = Kokkos::create_mirror_view(num_comp);
  Kokkos::View<uint32_t[1]>::HostMirror num_dupl_h = Kokkos::create_mirror_view(num_dupl);
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
    uint32_t old_idx = prior_distinct_map.find(digest); // Search for digest in prior distinct map
    if(!prior_distinct_map.valid_at(old_idx)) { // Chunk not in prior chkpt
      auto result = distinct_map.insert(digest, NodeID(i, list_id));
      if(result.existing()) {
        NodeID& old = distinct_map.value_at(result.index());
          auto shared_result = shared_map.insert(i, old);
          if(shared_result.existing()) {
            printf("Update SharedMap: Node %u already in shared map.\n", i);
          } else if(shared_result.failed()) {
            printf("Update SharedMap: Failed to insert %u into the shared map.\n", i);
          }
#ifdef STATS
Kokkos::atomic_add(&num_dupl(0), static_cast<uint32_t>(1));
#endif
      } else if(result.failed())  {
        printf("Warning: Failed to insert (%u,%u) into map for hashlist.\n", i, list_id);
#ifdef STATS
      } else if(result.success()) {
Kokkos::atomic_add(&num_new(0), static_cast<uint32_t>(1));
#endif
      }
    } else { // Chunk is in prior chkpt
      NodeID old_distinct = prior_distinct_map.value_at(old_idx);
      if(i != old_distinct.node) {
        uint32_t old_shared_idx = prior_shared_map.find(i);
        if(prior_shared_map.valid_at(old_shared_idx)) {
          NodeID old_node = prior_shared_map.value_at(old_shared_idx);
          if(old_distinct.node != old_node.node) {
            shared_map.insert(i, old_distinct);
#ifdef STATS
Kokkos::atomic_add(&num_shift(0), static_cast<uint32_t>(1));
          } else {
Kokkos::atomic_add(&num_same(0), static_cast<uint32_t>(1));
#endif
          }
        } else {
          shared_map.insert(i, old_distinct);
#ifdef STATS
Kokkos::atomic_add(&num_shift(0), static_cast<uint32_t>(1));
#endif
        }
#ifdef STATS
      } else {
Kokkos::atomic_add(&num_same(0), static_cast<uint32_t>(1));
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
  STDOUT_PRINT("Number of chunks: %u\n", num_chunks);
  STDOUT_PRINT("Number of new chunks: %u\n", num_new_h(0));
  STDOUT_PRINT("Number of same chunks: %u\n", num_same_h(0));
  STDOUT_PRINT("Number of shift chunks: %u\n", num_shift_h(0));
  STDOUT_PRINT("Number of comp nodes: %u\n", num_comp_h(0));
  STDOUT_PRINT("Number of dupl nodes: %u\n", num_dupl_h(0));
#endif
}

int num_subtree_roots(const HashList& list, 
                      const DistinctNodeIDMap& first_occur_d, 
                      const SharedNodeIDMap&   fixed_dupl_d, 
                      const SharedNodeIDMap&   shifted_dupl_d) {
  Kokkos::deep_copy(list.list_h, list.list_d);
  DistinctHostNodeIDMap first_occur_h(first_occur_d.capacity());
  SharedHostNodeIDMap   fixed_dupl_h(fixed_dupl_d.capacity());
  SharedHostNodeIDMap   shifted_dupl_h(shifted_dupl_d.capacity());
  Kokkos::deep_copy(first_occur_h, first_occur_d);
  Kokkos::deep_copy(fixed_dupl_h, fixed_dupl_d);
  Kokkos::deep_copy(shifted_dupl_h, shifted_dupl_d);
  // First Occurrece: 1
  // Fixed Duplicate: 2
  // Shift Duplicate: 3
  const char OTHER = 0;
  const char FIRST_OCUR = 1;
  const char FIXED_DUPL = 2;
  const char SHIFT_DUPL = 3;
  uint64_t num_chunks = list.list_h.extent(0);
  std::vector<char> labels(2*num_chunks-1, 0);
  Kokkos::fence();
  uint64_t num_contig = 0;
  uint64_t num_first_ocur_regs = 0;
  uint64_t num_fixed_dupl_regs = 0;
  uint64_t num_shift_dupl_regs = 0;
  for(uint64_t i=2*num_chunks-2; i<2*num_chunks-1; i--) {
    if(i >= num_chunks-1) {
      if(shifted_dupl_h.exists(i-(num_chunks-1))) {
        labels[i] = SHIFT_DUPL;
      } else if(fixed_dupl_h.exists(i-(num_chunks-1))) {
        labels[i] = FIXED_DUPL;
      } else if(first_occur_h.exists(list.list_h(i-(num_chunks-1)))) {
        labels[i] = FIRST_OCUR;
      } else {
        printf("Incorrect label!\n");
      }
    } else {
      uint32_t child_l = 2*i+1;
      uint32_t child_r = 2*i+2;
      if(labels[child_l] == labels[child_r]) {
        labels[i] = labels[child_l];
      } else {
        labels[i] = OTHER;
      }
    }
  }

  std::queue<uint64_t> queue;
  queue.push(0);
  while(!queue.empty()) {
    uint64_t u = queue.front();
    queue.pop();
    uint64_t left = 2*u+1;
    uint64_t right = 2*u+2;
    if(labels[u] != OTHER) {
      num_contig += 1;
      if(labels[u] == SHIFT_DUPL) {
        num_shift_dupl_regs += 1;
      } else if(labels[u] == FIXED_DUPL) {
        num_fixed_dupl_regs += 1;
      } else if(labels[u] == FIRST_OCUR) {
        num_first_ocur_regs += 1;
      }
    } else {
      queue.push(left);
      queue.push(right);
    }
  }
  printf("Number of first occurrence regions:  %lu\n", num_first_ocur_regs);
  printf("Number of fixed duplicate regions:   %lu\n", num_fixed_dupl_regs);
  printf("Number of shifted duplicate regions: %lu\n", num_shift_dupl_regs);
  printf("Number of contiguous regions: %lu\n", num_contig);
  return num_contig;
}

template<class Hasher>
void compare_lists_global( Hasher& hasher, 
                    const HashList& list, 
                    const uint32_t list_id, 
                    const Kokkos::View<uint8_t*>& data, 
                    const uint32_t chunk_size, 
                    SharedNodeIDMap& identical_map,
                    SharedNodeIDMap& shared_map, 
                    DistinctNodeIDMap& distinct_map, 
                    const SharedNodeIDMap& prior_identical_map,
                    const SharedNodeIDMap& prior_shared_map, 
                    const DistinctNodeIDMap& prior_distinct_map) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size())
    num_chunks += 1;
  uint32_t num_prior_distinct = distinct_map.size();
#ifdef STATS
  Kokkos::View<uint32_t[1]> num_same("Number of chunks that remain the same");
  Kokkos::View<uint32_t[1]> num_new("Number of chunks that are new");
  Kokkos::View<uint32_t[1]> num_shift("Number of chunks that exist but in different spaces");
  Kokkos::View<uint32_t[1]> num_comp("Number of compressed nodes");
  Kokkos::View<uint32_t[1]> num_dupl("Number of new duplicate nodes");
  Kokkos::View<uint32_t[1]>::HostMirror num_same_h = Kokkos::create_mirror_view(num_same);
  Kokkos::View<uint32_t[1]>::HostMirror num_new_h = Kokkos::create_mirror_view(num_new);
  Kokkos::View<uint32_t[1]>::HostMirror num_shift_h = Kokkos::create_mirror_view(num_shift);
  Kokkos::View<uint32_t[1]>::HostMirror num_comp_h = Kokkos::create_mirror_view(num_comp);
  Kokkos::View<uint32_t[1]>::HostMirror num_dupl_h = Kokkos::create_mirror_view(num_dupl);
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
    NodeID info(i, list_id);

    uint32_t node = i;
    uint32_t index = prior_distinct_map.find(digest);
    if(!prior_distinct_map.valid_at(index)) { // Chunk not in prior map
      auto result = distinct_map.insert(digest, info);
      if(result.success()) { // Chunk is brand new
        #ifdef STATS
        Kokkos::atomic_add(&(num_new(0)), 1);
        #endif
      } else if(result.existing()) { // Chunk already exists locally
        NodeID& existing_info = distinct_map.value_at(result.index());
        shared_map.insert(node, NodeID(existing_info.node, existing_info.tree));
        #ifdef STATS
        Kokkos::atomic_add(&num_dupl(0), static_cast<uint32_t>(1));
        #endif
      } else if(result.failed()) {
        printf("Failed to insert new chunk into distinct or shared map (tree %u). Shouldn't happen.", list_id);
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
          identical_map.insert(i, old_distinct);
        } else {
          uint32_t prior_shared_idx = prior_shared_map.find(node);
          if(prior_shared_map.valid_at(prior_shared_idx)) { // Node was repeat last checkpoint 
            NodeID prior_shared = prior_shared_map.value_at(prior_shared_idx);
            if(prior_shared.node == old_distinct.node && prior_shared.tree == old_distinct.tree) {
              identical_map.insert(node, old_distinct);
              #ifdef STATS
              Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
              #endif
            } else {
              shared_map.insert(node, NodeID(old_distinct.node, old_distinct.tree));
              #ifdef STATS
              Kokkos::atomic_add(&(num_shift(0)), static_cast<uint32_t>(1));
              #endif
            }
          } else { // Node was distinct last checkpoint
            if(node == old_distinct.node) { // No change since last checkpoint
              identical_map.insert(node, old_distinct);
              #ifdef STATS
              Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
              #endif
            } else { // Node changed since last checkpoint
              shared_map.insert(node, NodeID(old_distinct.node, old_distinct.tree));
              #ifdef STATS
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
            identical_map.insert(node, old_distinct);
            #ifdef STATS
            Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
            #endif
          } else {
            shared_map.insert(node, NodeID(old_distinct.node, old_distinct.tree));
            #ifdef STATS
            Kokkos::atomic_add(&(num_shift(0)), static_cast<uint32_t>(1));
            #endif
          }
        } else { // Node was distinct last checkpoint
          if(node == old_distinct.node) { // No change since last checkpoint
            identical_map.insert(node, old_distinct);
            #ifdef STATS
            Kokkos::atomic_add(&(num_same(0)), static_cast<uint32_t>(1));
            #endif
          } else { // Node changed since last checkpoint
            shared_map.insert(node, NodeID(old_distinct.node, old_distinct.tree));
            #ifdef STATS
            Kokkos::atomic_add(&(num_shift(0)), static_cast<uint32_t>(1));
            #endif
          }
        }
      }
    }
  });
  Kokkos::fence();
  STDOUT_PRINT("Num distinct:  %u\n", distinct_map.size()-num_prior_distinct);
  STDOUT_PRINT("Num repeats:   %u\n", shared_map.size());
  STDOUT_PRINT("Num identical: %u\n", identical_map.size());
#ifdef STATS
  Kokkos::deep_copy(num_same_h, num_same);
  Kokkos::deep_copy(num_new_h, num_new);
  Kokkos::deep_copy(num_shift_h, num_shift);
  Kokkos::deep_copy(num_comp_h, num_comp);
  Kokkos::deep_copy(num_dupl_h, num_dupl);
  STDOUT_PRINT("Number of chunks: %u\n", num_chunks);
  STDOUT_PRINT("Number of new chunks: %u\n", num_new_h(0));
  STDOUT_PRINT("Number of same chunks: %u\n", num_same_h(0));
  STDOUT_PRINT("Number of shift chunks: %u\n", num_shift_h(0));
  STDOUT_PRINT("Number of comp nodes: %u\n", num_comp_h(0));
  STDOUT_PRINT("Number of dupl nodes: %u\n", num_dupl_h(0));
#endif
//  int num_regions = num_subtree_roots(list, distinct_map, identical_map, shared_map);
}

std::pair<double,double> restart_chkpt_naive(std::vector<std::vector<uint8_t> >& incr_chkpts, 
                             const int chkpt_idx, 
                             Kokkos::View<uint8_t*>& data,
                             size_t size,
                             uint32_t num_chunks,
                             header_t header,
                             Kokkos::View<uint8_t*>& buffer_d,
                             Kokkos::View<uint8_t*>::HostMirror& buffer_h
                             ) {
  // Main checkpoint
  Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace> checkpoint_h("Checkpoint", incr_chkpts[chkpt_idx].size());
  memcpy(checkpoint_h.data(), incr_chkpts[chkpt_idx].data(), incr_chkpts[chkpt_idx].size());
  std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();
  Kokkos::deep_copy(buffer_d, checkpoint_h);
  Kokkos::fence();
  std::chrono::high_resolution_clock::time_point c2 = std::chrono::high_resolution_clock::now();

  Kokkos::View<NodeID*> node_list("List of NodeIDs", num_chunks);
  Kokkos::deep_copy(node_list, NodeID());
  uint32_t ref_id = header.ref_id;
  uint32_t cur_id = header.chkpt_id;

  size_t curr_repeat_offset = sizeof(header_t) + header.distinct_size*sizeof(uint32_t);
  size_t prev_repeat_offset = curr_repeat_offset + header.curr_repeat_size*2*sizeof(uint32_t);
  size_t data_offset = prev_repeat_offset + header.prev_repeat_size*2*sizeof(uint32_t);
  auto curr_repeat   = Kokkos::subview(buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
  auto prev_repeat   = Kokkos::subview(buffer_d, std::make_pair(prev_repeat_offset, data_offset));
  auto distinct      = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
  auto data_subview  = Kokkos::subview(buffer_d, std::make_pair(data_offset, size));

  STDOUT_PRINT("Checkpoint %u\n", cur_id);
  STDOUT_PRINT("Checkpoint size: %lu\n", size);
  STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
  STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
  STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
  STDOUT_PRINT("Data offset: %lu\n", data_offset);

  uint32_t chunk_size = header.chunk_size;
  size_t datalen = header.datalen;
  Kokkos::UnorderedMap<NodeID, size_t> distinct_map(header.distinct_size);
  Kokkos::parallel_for("Restart Hashlist distinct", Kokkos::RangePolicy<>(0, header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t node;
    memcpy(&node, distinct.data() + i*sizeof(uint32_t),  sizeof(uint32_t));
    distinct_map.insert(NodeID(node,cur_id),  i*chunk_size);
    node_list(node) = NodeID(node,cur_id);
    uint32_t datasize = chunk_size;
    if(node == num_chunks-1)
      datasize = datalen - node*chunk_size;
    memcpy(data.data()+chunk_size*node, data_subview.data()+i*chunk_size, datasize);
  });
  Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
    NodeID entry = node_list(i);
    if(entry.node == UINT_MAX) {
      node_list(i) = NodeID(i, cur_id-1);
    }
  });
  Kokkos::fence();

  for(int idx=static_cast<int>(chkpt_idx)-1; idx>static_cast<int>(ref_id); idx--) {
    STDOUT_PRINT("Processing checkpoint %u\n", idx);
    size_t chkpt_size = incr_chkpts[idx].size();
    Kokkos::View<uint8_t*> chkpt_buffer_d("Checkpoint buffer", chkpt_size);
    auto chkpt_buffer_h = Kokkos::create_mirror_view(chkpt_buffer_d);
    memcpy(chkpt_buffer_h.data(), incr_chkpts[idx].data(), chkpt_size);
    header_t chkpt_header;
    memcpy(&chkpt_header, chkpt_buffer_h.data(), sizeof(header_t));
    uint32_t current_id = chkpt_header.chkpt_id;
    datalen = chkpt_header.datalen;
    chunk_size = chkpt_header.chunk_size;
    Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
    ref_id = chkpt_header.ref_id;
    cur_id = chkpt_header.chkpt_id;

    STDOUT_PRINT("Ref ID:           %u\n",  chkpt_header.ref_id);
    STDOUT_PRINT("Chkpt ID:         %u\n",  chkpt_header.chkpt_id);
    STDOUT_PRINT("Datalen:          %lu\n", chkpt_header.datalen);
    STDOUT_PRINT("Chunk size:       %u\n",  chkpt_header.chunk_size);
    STDOUT_PRINT("Window size:      %u\n",  chkpt_header.window_size);
    STDOUT_PRINT("Distinct size:    %u\n",  chkpt_header.distinct_size);
    STDOUT_PRINT("Curr repeat size: %u\n",  chkpt_header.curr_repeat_size);
    STDOUT_PRINT("Prev repeat size: %u\n",  chkpt_header.prev_repeat_size);
    STDOUT_PRINT("Num prior chkpts: %u\n",  chkpt_header.num_prior_chkpts);
    
    curr_repeat_offset = sizeof(header_t) + chkpt_header.distinct_size*sizeof(uint32_t);
    prev_repeat_offset = chkpt_header.num_prior_chkpts*2*sizeof(uint32_t) + curr_repeat_offset + chkpt_header.curr_repeat_size*2*sizeof(uint32_t);
    data_offset = prev_repeat_offset + chkpt_header.prev_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
    curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, data_offset));
    distinct      = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
    data_subview  = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, chkpt_size));
    STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", chkpt_size);
    STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
    STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
    STDOUT_PRINT("Data offset: %lu\n", data_offset);

    distinct_map.clear();
    distinct_map.rehash(chkpt_header.distinct_size);
    Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
    Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, distinct.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      distinct_map.insert(NodeID(node,cur_id), i*chunk_size);
    });

    Kokkos::parallel_for("Fill data middle chkpts", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      if(node_list(i).tree == current_id) {
        NodeID id = node_list(i);
        if(distinct_map.exists(id)) {
          size_t offset = distinct_map.value_at(distinct_map.find(id));
          uint32_t writesize = chunk_size;
          if(i*chunk_size+writesize > datalen) 
            writesize = datalen-i*chunk_size;
          memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
        } else {
          node_list(i) = NodeID(node_list(i).node, current_id-1);
        }
      }
    });
  }

  if(header.ref_id != header.chkpt_id) {
    // Reference
    size_t chkpt_size = incr_chkpts[header.ref_id].size();
    Kokkos::View<uint8_t*> chkpt_buffer_d("Checkpoint buffer", chkpt_size);
    auto chkpt_buffer_h = Kokkos::create_mirror_view(chkpt_buffer_d);
    memcpy(chkpt_buffer_h.data(), incr_chkpts[header.ref_id].data(), chkpt_size);
    
    header_t chkpt_header;
    memcpy(&chkpt_header, chkpt_buffer_h.data(), sizeof(header_t));
    uint32_t current_id = chkpt_header.chkpt_id;
    datalen = chkpt_header.datalen;
    chunk_size = chkpt_header.chunk_size;
    Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
    ref_id = chkpt_header.ref_id;

    curr_repeat_offset = sizeof(header_t) + chkpt_header.distinct_size*sizeof(uint32_t);
    prev_repeat_offset = curr_repeat_offset + chkpt_header.curr_repeat_size*2*sizeof(uint32_t);
    data_offset = prev_repeat_offset + chkpt_header.prev_repeat_size*2*sizeof(uint32_t);
    curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, data_offset));
    distinct = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
    data_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, size));
    STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", chkpt_size);
    STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
    STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
    STDOUT_PRINT("Data offset: %lu\n", data_offset);
    
    distinct_map.clear();
    distinct_map.rehash(chkpt_header.distinct_size);
    Kokkos::fence();
    Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
    Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, distinct.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      distinct_map.insert(NodeID(node, current_id), i*chunk_size);
    });
    Kokkos::parallel_for("Fill data", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      if(node_list(i).tree == current_id) {
        NodeID id = node_list(i);
        if(distinct_map.exists(id)) {
          size_t offset = distinct_map.value_at(distinct_map.find(id));
          uint32_t writesize = chunk_size;
          if(id.node == num_chunks-1)
            writesize = datalen-id.node*chunk_size;
          memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
        } else {
          node_list(i) = NodeID(node_list(i).node, current_id-1);
        }
      }
    });
  }
  Kokkos::fence();
  std::chrono::high_resolution_clock::time_point c3 = std::chrono::high_resolution_clock::now();
  double copy_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
  double restart_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c3-c2).count());
  return std::make_pair(copy_time, restart_time);
}

std::pair<double,double> restart_chkpt_naive(std::vector<std::string>& chkpt_files, 
                             const int file_idx, 
                             std::ifstream& file,
                             Kokkos::View<uint8_t*>& data,
                             size_t filesize,
                             uint32_t num_chunks,
                             header_t header,
                             Kokkos::View<uint8_t*>& buffer_d,
                             Kokkos::View<uint8_t*>::HostMirror& buffer_h
                             ) {
  // Main checkpoint
  file.open(chkpt_files[file_idx], std::ifstream::in | std::ifstream::binary);
  file.read((char*)(buffer_h.data()), filesize);
  file.close();

  Kokkos::fence();
  std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();
  Kokkos::deep_copy(buffer_d, buffer_h);
  Kokkos::fence();
  std::chrono::high_resolution_clock::time_point c2 = std::chrono::high_resolution_clock::now();

  Kokkos::View<NodeID*> node_list("List of NodeIDs", num_chunks);
  Kokkos::deep_copy(node_list, NodeID());
  uint32_t ref_id = header.ref_id;
  uint32_t cur_id = header.chkpt_id;

  size_t curr_repeat_offset = sizeof(header_t) + header.distinct_size*sizeof(uint32_t);
  size_t prev_repeat_offset = curr_repeat_offset + header.curr_repeat_size*2*sizeof(uint32_t);
  size_t data_offset = prev_repeat_offset + header.prev_repeat_size*2*sizeof(uint32_t);
  auto curr_repeat   = Kokkos::subview(buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
  auto prev_repeat   = Kokkos::subview(buffer_d, std::make_pair(prev_repeat_offset, data_offset));
  auto distinct = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
  auto data_subview = Kokkos::subview(buffer_d, std::make_pair(data_offset, filesize));

  STDOUT_PRINT("Checkpoint %u\n", cur_id);
  STDOUT_PRINT("Checkpoint size: %lu\n", filesize);
  STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
  STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
  STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
  STDOUT_PRINT("Data offset: %lu\n", data_offset);

  uint32_t chunk_size = header.chunk_size;
  size_t datalen = header.datalen;
  Kokkos::UnorderedMap<NodeID, size_t> distinct_map(header.distinct_size);
  Kokkos::parallel_for("Restart Hashlist distinct", Kokkos::RangePolicy<>(0, header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t node;
    memcpy(&node, distinct.data() + i*sizeof(uint32_t),  sizeof(uint32_t));
    distinct_map.insert(NodeID(node,cur_id),  i*chunk_size);
    node_list(node) = NodeID(node,cur_id);
    uint32_t datasize = chunk_size;
    if(node == num_chunks-1)
      datasize = datalen - node*chunk_size;
    memcpy(data.data()+chunk_size*node, data_subview.data()+i*chunk_size, datasize);
  });
  Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
    NodeID entry = node_list(i);
    if(entry.node == UINT_MAX) {
      node_list(i) = NodeID(i, cur_id-1);
    }
  });
  Kokkos::fence();

  for(int idx=static_cast<int>(file_idx)-1; idx>static_cast<int>(ref_id); idx--) {
    STDOUT_PRINT("Processing checkpoint %u\n", idx);
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

    STDOUT_PRINT("Ref ID:           %u\n",  chkpt_header.ref_id);
    STDOUT_PRINT("Chkpt ID:         %u\n",  chkpt_header.chkpt_id);
    STDOUT_PRINT("Datalen:          %lu\n", chkpt_header.datalen);
    STDOUT_PRINT("Chunk size:       %u\n",  chkpt_header.chunk_size);
    STDOUT_PRINT("Window size:      %u\n",  chkpt_header.window_size);
    STDOUT_PRINT("Distinct size:    %u\n",  chkpt_header.distinct_size);
    STDOUT_PRINT("Curr repeat size: %u\n",  chkpt_header.curr_repeat_size);
    STDOUT_PRINT("Prev repeat size: %u\n",  chkpt_header.prev_repeat_size);
    STDOUT_PRINT("Num prior chkpts: %u\n",  chkpt_header.num_prior_chkpts);
    
    curr_repeat_offset = sizeof(header_t) + chkpt_header.distinct_size*sizeof(uint32_t);
    prev_repeat_offset = chkpt_header.num_prior_chkpts*2*sizeof(uint32_t) + curr_repeat_offset + chkpt_header.curr_repeat_size*2*sizeof(uint32_t);
    data_offset = prev_repeat_offset + chkpt_header.prev_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
    curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, data_offset));
    distinct      = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
    data_subview  = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, chkpt_size));
    STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", chkpt_size);
    STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
    STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
    STDOUT_PRINT("Data offset: %lu\n", data_offset);

    distinct_map.clear();
    distinct_map.rehash(chkpt_header.distinct_size);
    Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
    Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, distinct.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      distinct_map.insert(NodeID(node,cur_id), i*chunk_size);
    });

    Kokkos::parallel_for("Fill data middle chkpts", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      if(node_list(i).tree == current_id) {
        NodeID id = node_list(i);
        if(distinct_map.exists(id)) {
          size_t offset = distinct_map.value_at(distinct_map.find(id));
          uint32_t writesize = chunk_size;
          if(i*chunk_size+writesize > datalen) 
            writesize = datalen-i*chunk_size;
          memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
        } else {
          node_list(i) = NodeID(node_list(i).node, current_id-1);
        }
      }
    });
  }

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
    Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
    ref_id = chkpt_header.ref_id;

    curr_repeat_offset = sizeof(header_t) + chkpt_header.distinct_size*sizeof(uint32_t);
    prev_repeat_offset = curr_repeat_offset + chkpt_header.curr_repeat_size*2*sizeof(uint32_t);
    data_offset = prev_repeat_offset + chkpt_header.prev_repeat_size*2*sizeof(uint32_t);
    curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, data_offset));
    distinct = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
    data_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, filesize));
    STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", chkpt_size);
    STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
    STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
    STDOUT_PRINT("Data offset: %lu\n", data_offset);
    
    distinct_map.clear();
    distinct_map.rehash(chkpt_header.distinct_size);
    Kokkos::fence();
    Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
    Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, distinct.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      distinct_map.insert(NodeID(node, current_id), i*chunk_size);
    });
    Kokkos::parallel_for("Fill data", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      if(node_list(i).tree == current_id) {
        NodeID id = node_list(i);
        if(distinct_map.exists(id)) {
          size_t offset = distinct_map.value_at(distinct_map.find(id));
          uint32_t writesize = chunk_size;
          if(id.node == num_chunks-1)
            writesize = datalen-id.node*chunk_size;
          memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
        } else {
          node_list(i) = NodeID(node_list(i).node, current_id-1);
        }
      }
    });
  }
  Kokkos::fence();
  std::chrono::high_resolution_clock::time_point c3 = std::chrono::high_resolution_clock::now();
  double copy_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
  double restart_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c3-c2).count());
  return std::make_pair(copy_time, restart_time);
}

std::pair<double,double> restart_chkpt_local(std::vector<std::vector<uint8_t> >& incr_chkpts, 
                             const int chkpt_idx, 
                             Kokkos::View<uint8_t*>& data,
                             size_t size,
                             uint32_t num_chunks,
                             header_t header,
                             Kokkos::View<uint8_t*>& buffer_d,
                             Kokkos::View<uint8_t*>::HostMirror& buffer_h
                             ) {
    // Main checkpoint
    Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace> checkpoint_h("Checkpoint", incr_chkpts[chkpt_idx].size());
    memcpy(checkpoint_h.data(), incr_chkpts[chkpt_idx].data(), incr_chkpts[chkpt_idx].size());

    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();
    Kokkos::deep_copy(buffer_d, checkpoint_h);
    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c2 = std::chrono::high_resolution_clock::now();

    Kokkos::View<NodeID*> node_list("List of NodeIDs", num_chunks);
    Kokkos::deep_copy(node_list, NodeID());
    uint32_t ref_id = header.ref_id;
    uint32_t cur_id = header.chkpt_id;

    size_t curr_repeat_offset = sizeof(header_t) + header.distinct_size*sizeof(uint32_t);
    size_t prev_repeat_offset = curr_repeat_offset + header.curr_repeat_size*2*sizeof(uint32_t);
    size_t data_offset = prev_repeat_offset + header.prev_repeat_size*2*sizeof(uint32_t);
    auto curr_repeat   = Kokkos::subview(buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    auto prev_repeat   = Kokkos::subview(buffer_d, std::make_pair(prev_repeat_offset, data_offset));
    auto distinct = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
    auto data_subview = Kokkos::subview(buffer_d, std::make_pair(data_offset, size));
    STDOUT_PRINT("Checkpoint %u\n", cur_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", size);
    STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
    STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
    STDOUT_PRINT("Data offset: %lu\n", data_offset);

    uint32_t chunk_size = header.chunk_size;
    size_t datalen = header.datalen;
    Kokkos::UnorderedMap<NodeID, size_t> distinct_map(header.distinct_size);
    Kokkos::parallel_for("Restart Hashlist distinct", Kokkos::RangePolicy<>(0, header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, distinct.data() + i*sizeof(uint32_t),  sizeof(uint32_t));
      distinct_map.insert(NodeID(node,cur_id),  i*chunk_size);
      node_list(node) = NodeID(node,cur_id);
      uint32_t datasize = chunk_size;
      if(node == num_chunks-1)
        datasize = datalen - node*chunk_size;
      memcpy(data.data()+chunk_size*node, data_subview.data()+i*chunk_size, datasize);
    });
    Kokkos::parallel_for("Restart Hashlist current repeat", Kokkos::RangePolicy<>(0, header.curr_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      memcpy(&node, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
      node_list(node) = NodeID(prev, cur_id);
      size_t offset = distinct_map.value_at(distinct_map.find(NodeID(prev, cur_id)));
      uint32_t copysize = chunk_size;
      if(node == num_chunks-1)
        copysize = data.size() - chunk_size*(num_chunks-1);
      memcpy(data.data()+chunk_size*node, data_subview.data()+offset, copysize);
    });
    Kokkos::parallel_for("Restart Hashlist previous repeat", Kokkos::RangePolicy<>(0, header.prev_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      memcpy(&node, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)) +sizeof(uint32_t), sizeof(uint32_t));
      node_list(node) = NodeID(prev,ref_id);
    });
    Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      NodeID entry = node_list(i);
      if(entry.node == UINT_MAX) {
        node_list(i) = NodeID(i, ref_id);
      }
    });
    Kokkos::fence();

    if(header.ref_id != header.chkpt_id) {
      // Reference
      size_t chkpt_size = incr_chkpts[header.ref_id].size();
      Kokkos::View<uint8_t*> chkpt_buffer_d("Checkpoint buffer", chkpt_size);
      auto chkpt_buffer_h = Kokkos::create_mirror_view(chkpt_buffer_d);
      memcpy(chkpt_buffer_h.data(), incr_chkpts[header.ref_id].data(), chkpt_size);

      header_t chkpt_header;
      memcpy(&chkpt_header, chkpt_buffer_h.data(), sizeof(header_t));
      uint32_t current_id = chkpt_header.chkpt_id;
      datalen = chkpt_header.datalen;
      chunk_size = chkpt_header.chunk_size;
      Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
      ref_id = chkpt_header.ref_id;

      curr_repeat_offset = sizeof(header_t) + chkpt_header.distinct_size*sizeof(uint32_t);
      prev_repeat_offset = curr_repeat_offset + chkpt_header.curr_repeat_size*2*sizeof(uint32_t);
      data_offset = prev_repeat_offset + chkpt_header.prev_repeat_size*2*sizeof(uint32_t);
      curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
      prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, data_offset));
      distinct = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
      data_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, size));
      STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
      STDOUT_PRINT("Checkpoint size: %lu\n", chkpt_size);
      STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
      STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
      STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
      STDOUT_PRINT("Data offset: %lu\n", data_offset);
      
      distinct_map.clear();
      distinct_map.rehash(chkpt_header.distinct_size);
      Kokkos::fence();
      Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
      Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
        memcpy(&node, distinct.data()+i*sizeof(uint32_t), sizeof(uint32_t));
        distinct_map.insert(NodeID(node, current_id), i*chunk_size);
      });
      uint32_t num_repeat = chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size;
      Kokkos::parallel_for("Fill repeat map", Kokkos::RangePolicy<>(0, num_repeat), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
        uint32_t prev;
        memcpy(&node, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
        memcpy(&prev, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
        repeat_map.insert(node, NodeID(prev, ref_id));
      });
      Kokkos::parallel_for("Fill data", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
        if(node_list(i).tree == current_id) {
          NodeID id = node_list(i);
          if(distinct_map.exists(id)) {
            size_t offset = distinct_map.value_at(distinct_map.find(id));
            uint32_t writesize = chunk_size;
            if(id.node == num_chunks-1)
              writesize = datalen-id.node*chunk_size;
            memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
          } else if(repeat_map.exists(id.node)) {
            NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
            if(prev.tree == current_id) {
              size_t offset = distinct_map.value_at(distinct_map.find(prev));
              uint32_t writesize = chunk_size;
              if(i == num_chunks-1)
                writesize = datalen-i*chunk_size;
              memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize); 
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
    std::chrono::high_resolution_clock::time_point c3 = std::chrono::high_resolution_clock::now();
    double copy_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
    double restart_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c3-c2).count());
    return std::make_pair(copy_time, restart_time);
}

std::pair<double,double> restart_chkpt_local(std::vector<std::string>& chkpt_files, 
                             const int file_idx, 
                             std::ifstream& file,
                             Kokkos::View<uint8_t*>& data,
                             size_t filesize,
                             uint32_t num_chunks,
                             header_t header,
                             Kokkos::View<uint8_t*>& buffer_d,
                             Kokkos::View<uint8_t*>::HostMirror& buffer_h
                             ) {
    // Main checkpoint
    file.open(chkpt_files[file_idx], std::ifstream::in | std::ifstream::binary);
    file.read((char*)(buffer_h.data()), filesize);
    file.close();

    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();
    Kokkos::deep_copy(buffer_d, buffer_h);
    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c2 = std::chrono::high_resolution_clock::now();

    Kokkos::View<NodeID*> node_list("List of NodeIDs", num_chunks);
    Kokkos::deep_copy(node_list, NodeID());
    uint32_t ref_id = header.ref_id;
    uint32_t cur_id = header.chkpt_id;

    size_t curr_repeat_offset = sizeof(header_t) + header.distinct_size*sizeof(uint32_t);
    size_t prev_repeat_offset = curr_repeat_offset + header.curr_repeat_size*2*sizeof(uint32_t);
    size_t data_offset = prev_repeat_offset + header.prev_repeat_size*2*sizeof(uint32_t);
    auto curr_repeat   = Kokkos::subview(buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    auto prev_repeat   = Kokkos::subview(buffer_d, std::make_pair(prev_repeat_offset, data_offset));
    auto distinct = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
    auto data_subview = Kokkos::subview(buffer_d, std::make_pair(data_offset, filesize));
    STDOUT_PRINT("Checkpoint %u\n", cur_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", filesize);
    STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
    STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
    STDOUT_PRINT("Data offset: %lu\n", data_offset);

    uint32_t chunk_size = header.chunk_size;
    size_t datalen = header.datalen;
    Kokkos::UnorderedMap<NodeID, size_t> distinct_map(header.distinct_size);
    Kokkos::parallel_for("Restart Hashlist distinct", Kokkos::RangePolicy<>(0, header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, distinct.data() + i*sizeof(uint32_t),  sizeof(uint32_t));
      distinct_map.insert(NodeID(node,cur_id),  i*chunk_size);
      node_list(node) = NodeID(node,cur_id);
      uint32_t datasize = chunk_size;
      if(node == num_chunks-1)
        datasize = datalen - node*chunk_size;
      memcpy(data.data()+chunk_size*node, data_subview.data()+i*chunk_size, datasize);
    });
    Kokkos::parallel_for("Restart Hashlist current repeat", Kokkos::RangePolicy<>(0, header.curr_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      memcpy(&node, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
      node_list(node) = NodeID(prev, cur_id);
      size_t offset = distinct_map.value_at(distinct_map.find(NodeID(prev, cur_id)));
      uint32_t copysize = chunk_size;
      if(node == num_chunks-1)
        copysize = data.size() - chunk_size*(num_chunks-1);
      memcpy(data.data()+chunk_size*node, data_subview.data()+offset, copysize);
    });
    Kokkos::parallel_for("Restart Hashlist previous repeat", Kokkos::RangePolicy<>(0, header.prev_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      memcpy(&node, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)) +sizeof(uint32_t), sizeof(uint32_t));
      node_list(node) = NodeID(prev,ref_id);
    });
    Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      NodeID entry = node_list(i);
      if(entry.node == UINT_MAX) {
        node_list(i) = NodeID(i, ref_id);
      }
    });
    Kokkos::fence();

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
      Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
      ref_id = chkpt_header.ref_id;

      curr_repeat_offset = sizeof(header_t) + chkpt_header.distinct_size*sizeof(uint32_t);
      prev_repeat_offset = curr_repeat_offset + chkpt_header.curr_repeat_size*2*sizeof(uint32_t);
      data_offset = prev_repeat_offset + chkpt_header.prev_repeat_size*2*sizeof(uint32_t);
      curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
      prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, data_offset));
      distinct = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
      data_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, filesize));
      STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
      STDOUT_PRINT("Checkpoint size: %lu\n", chkpt_size);
      STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
      STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
      STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
      STDOUT_PRINT("Data offset: %lu\n", data_offset);
      
      distinct_map.clear();
      distinct_map.rehash(chkpt_header.distinct_size);
      Kokkos::fence();
      Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
      Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
        memcpy(&node, distinct.data()+i*sizeof(uint32_t), sizeof(uint32_t));
        distinct_map.insert(NodeID(node, current_id), i*chunk_size);
      });
      uint32_t num_repeat = chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size;
      Kokkos::parallel_for("Fill repeat map", Kokkos::RangePolicy<>(0, num_repeat), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
        uint32_t prev;
        memcpy(&node, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
        memcpy(&prev, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
        repeat_map.insert(node, NodeID(prev, ref_id));
      });
      Kokkos::parallel_for("Fill data", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
        if(node_list(i).tree == current_id) {
          NodeID id = node_list(i);
          if(distinct_map.exists(id)) {
            size_t offset = distinct_map.value_at(distinct_map.find(id));
            uint32_t writesize = chunk_size;
            if(id.node == num_chunks-1)
              writesize = datalen-id.node*chunk_size;
            memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
          } else if(repeat_map.exists(id.node)) {
            NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
            if(prev.tree == current_id) {
              size_t offset = distinct_map.value_at(distinct_map.find(prev));
              uint32_t writesize = chunk_size;
              if(i == num_chunks-1)
                writesize = datalen-i*chunk_size;
              memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize); 
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
    std::chrono::high_resolution_clock::time_point c3 = std::chrono::high_resolution_clock::now();
    double copy_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
    double restart_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c3-c2).count());
    return std::make_pair(copy_time, restart_time);
}

std::pair<double,double> restart_chkpt_global(std::vector<std::vector<uint8_t> >& incr_chkpts, 
                             const int chkpt_idx, 
                             Kokkos::View<uint8_t*>& data,
                             size_t size,
                             uint32_t num_chunks,
                             header_t header,
                             Kokkos::View<uint8_t*>& buffer_d,
                             Kokkos::View<uint8_t*>::HostMirror& buffer_h
                             ) {
    // Main checkpoint
    Kokkos::View<uint8_t*, Kokkos::DefaultHostExecutionSpace> checkpoint_h("Checkpoint", incr_chkpts[chkpt_idx].size());
    memcpy(checkpoint_h.data(), incr_chkpts[chkpt_idx].data(), incr_chkpts[chkpt_idx].size());
    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();
    Kokkos::deep_copy(buffer_d, checkpoint_h);
    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c2 = std::chrono::high_resolution_clock::now();
    Kokkos::View<NodeID*> node_list("List of NodeIDs", num_chunks);
    Kokkos::deep_copy(node_list, NodeID());
    uint32_t ref_id = header.ref_id;
    uint32_t cur_id = header.chkpt_id;

    STDOUT_PRINT("Ref ID:           %u\n",  header.ref_id);
    STDOUT_PRINT("Chkpt ID:         %u\n",  header.chkpt_id);
    STDOUT_PRINT("Datalen:          %lu\n", header.datalen);
    STDOUT_PRINT("Chunk size:       %u\n",  header.chunk_size);
    STDOUT_PRINT("Window size:      %u\n",  header.window_size);
    STDOUT_PRINT("Distinct size:    %u\n",  header.distinct_size);
    STDOUT_PRINT("Curr repeat size: %u\n",  header.curr_repeat_size);
    STDOUT_PRINT("Prev repeat size: %u\n",  header.prev_repeat_size);
    STDOUT_PRINT("Num prior chkpts: %u\n",  header.num_prior_chkpts);

    size_t curr_repeat_offset = sizeof(header_t) + header.distinct_size*sizeof(uint32_t);
    size_t prev_repeat_offset = header.num_prior_chkpts*2*sizeof(uint32_t) + curr_repeat_offset + header.curr_repeat_size*2*sizeof(uint32_t);
    size_t data_offset = prev_repeat_offset + header.prev_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
    auto curr_repeat   = Kokkos::subview(buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    auto prev_repeat   = Kokkos::subview(buffer_d, std::make_pair(prev_repeat_offset, data_offset));
    auto distinct      = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
    auto data_subview  = Kokkos::subview(buffer_d, std::make_pair(data_offset, size));
    STDOUT_PRINT("Checkpoint %u\n", header.chkpt_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", size);
    STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
    STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
    STDOUT_PRINT("Data offset: %lu\n", data_offset);

    uint32_t chunk_size = header.chunk_size;
    size_t datalen = header.datalen;
    Kokkos::UnorderedMap<NodeID, size_t> distinct_map(header.distinct_size);
    Kokkos::parallel_for("Restart Hashlist distinct", Kokkos::RangePolicy<>(0, header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, distinct.data() + i*(sizeof(uint32_t)),  sizeof(uint32_t));
      distinct_map.insert(NodeID(node,cur_id),  i*chunk_size);
      node_list(node) = NodeID(node, cur_id);
      uint32_t datasize = chunk_size;
      if(node == num_chunks-1)
        datasize = datalen - node*chunk_size;
      memcpy(data.data()+chunk_size*node, data_subview.data()+i*chunk_size, datasize);
    });

    uint32_t num_prior_chkpts = header.num_prior_chkpts;
    Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
    auto repeat_region_sizes_h = Kokkos::create_mirror_view(repeat_region_sizes);
    Kokkos::deep_copy(repeat_region_sizes, 0);
    // Read map of repeats for each checkpoint
    Kokkos::parallel_for("Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t chkpt;
      memcpy(&chkpt, buffer_d.data()+curr_repeat_offset+i*2*sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&repeat_region_sizes(chkpt), buffer_d.data()+curr_repeat_offset+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
      STDOUT_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
    });
    Kokkos::deep_copy(repeat_region_sizes_h, repeat_region_sizes);
    // Perform exclusive scan to determine where regions start/stop
    Kokkos::parallel_scan("Repeat offsets", cur_id+1, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      partial_sum += repeat_region_sizes(i);
      if(is_final) repeat_region_sizes(i) = partial_sum;
    });
    STDOUT_PRINT("Num repeats: %u\n", header.curr_repeat_size+header.prev_repeat_size);
    // Load repeat entries and fill in metadata for chunks
    Kokkos::parallel_for("Restart Hash tree repeats main checkpoint", Kokkos::RangePolicy<>(0, header.curr_repeat_size+header.prev_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      uint32_t tree = 0;
      memcpy(&node, curr_repeat.data()+(num_prior_chkpts)*2*sizeof(uint32_t)+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, curr_repeat.data()+(num_prior_chkpts)*2*sizeof(uint32_t)+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
      // Determine ID 
      for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
        if(i < repeat_region_sizes(j)) {
          tree = j;
        }
      }
      uint32_t idx = distinct_map.find(NodeID(prev, tree));
      size_t offset = distinct_map.value_at(distinct_map.find(NodeID(prev, tree)));
      node_list(node) = NodeID(prev, tree);
      if(tree == cur_id) {
        uint32_t copysize = chunk_size;
        if(node == num_chunks-1)
          copysize = data.size() - chunk_size*node;
        memcpy(data.data()+chunk_size*node, data_subview.data()+offset, copysize);
      }
    });

    Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      NodeID entry = node_list(i);
      if(entry.node == UINT_MAX) {
        node_list(i) = NodeID(i, cur_id-1);
      }
    });
    Kokkos::fence();

    for(int idx=static_cast<int>(chkpt_idx)-1; idx>static_cast<int>(ref_id); idx--) {
      STDOUT_PRINT("Processing checkpoint %u\n", idx);
      size_t chkpt_size = incr_chkpts[idx].size();
      Kokkos::View<uint8_t*> chkpt_buffer_d("Checkpoint buffer", chkpt_size);
      auto chkpt_buffer_h = Kokkos::create_mirror_view(chkpt_buffer_d);
      memcpy(chkpt_buffer_h.data(), incr_chkpts[idx].data(), chkpt_size);
      header_t chkpt_header;
      memcpy(&chkpt_header, chkpt_buffer_h.data(), sizeof(header_t));
      uint32_t current_id = chkpt_header.chkpt_id;
      datalen = chkpt_header.datalen;
      chunk_size = chkpt_header.chunk_size;
      Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
      ref_id = chkpt_header.ref_id;
      cur_id = chkpt_header.chkpt_id;

      STDOUT_PRINT("Ref ID:           %u\n",  chkpt_header.ref_id);
      STDOUT_PRINT("Chkpt ID:         %u\n",  chkpt_header.chkpt_id);
      STDOUT_PRINT("Datalen:          %lu\n", chkpt_header.datalen);
      STDOUT_PRINT("Chunk size:       %u\n",  chkpt_header.chunk_size);
      STDOUT_PRINT("Window size:      %u\n",  chkpt_header.window_size);
      STDOUT_PRINT("Distinct size:    %u\n",  chkpt_header.distinct_size);
      STDOUT_PRINT("Curr repeat size: %u\n",  chkpt_header.curr_repeat_size);
      STDOUT_PRINT("Prev repeat size: %u\n",  chkpt_header.prev_repeat_size);
      STDOUT_PRINT("Num prior chkpts: %u\n",  chkpt_header.num_prior_chkpts);
      
      curr_repeat_offset = sizeof(header_t) + chkpt_header.distinct_size*sizeof(uint32_t);
      prev_repeat_offset = chkpt_header.num_prior_chkpts*2*sizeof(uint32_t) + curr_repeat_offset + chkpt_header.curr_repeat_size*2*sizeof(uint32_t);
      data_offset = prev_repeat_offset + chkpt_header.prev_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
      curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
      prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, data_offset));
      distinct      = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
      data_subview  = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, chkpt_size));
      STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
      STDOUT_PRINT("Checkpoint size: %lu\n", chkpt_size);
      STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
      STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
      STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
      STDOUT_PRINT("Data offset: %lu\n", data_offset);

      distinct_map.clear();
      distinct_map.rehash(chkpt_header.distinct_size);
      Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
      Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
        memcpy(&node, distinct.data()+i*sizeof(uint32_t), sizeof(uint32_t));
        distinct_map.insert(NodeID(node,cur_id), i*chunk_size);
      });
      num_prior_chkpts = chkpt_header.num_prior_chkpts;
      Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
      Kokkos::parallel_for("Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t chkpt;
        memcpy(&chkpt, chkpt_buffer_d.data()+curr_repeat_offset+i*2*sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&repeat_region_sizes(chkpt), chkpt_buffer_d.data()+curr_repeat_offset+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
        STDOUT_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
      });
      Kokkos::parallel_scan("Repeat offsets", cur_id+1, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
        partial_sum += repeat_region_sizes(i);
        if(is_final) repeat_region_sizes(i) = partial_sum;
      });

      STDOUT_PRINT("Num repeats: %u\n", chkpt_header.curr_repeat_size+chkpt_header.prev_repeat_size);
  
      uint32_t num_repeat_entries = chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size;

      Kokkos::parallel_for("Restart Hash tree repeats middle chkpts", Kokkos::RangePolicy<>(0,num_repeat_entries), KOKKOS_LAMBDA(const uint32_t i) { 
        uint32_t node, prev, tree=0;
        memcpy(&node, curr_repeat.data()+(num_prior_chkpts)*2*sizeof(uint32_t)+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
        memcpy(&prev, curr_repeat.data()+(num_prior_chkpts)*2*sizeof(uint32_t)+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
        for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
          if(i < repeat_region_sizes(j)) {
            tree = j;
          }
        }
        auto result = repeat_map.insert(node, NodeID(prev,tree));
        if(result.failed())
          STDOUT_PRINT("Failed to insert previous repeat %u: (%u,%u) into repeat map\n", node, prev, tree);
      });

      Kokkos::parallel_for("Fill data middle chkpts", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
        if(node_list(i).tree == current_id) {
          NodeID id = node_list(i);
          if(distinct_map.exists(id)) {
            size_t offset = distinct_map.value_at(distinct_map.find(id));
            uint32_t writesize = chunk_size;
            if(i*chunk_size+writesize > datalen) 
              writesize = datalen-i*chunk_size;
            memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
          } else if(repeat_map.exists(id.node)) {
            NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
            DEBUG_PRINT("Repaeat value: %u: (%u,%u)\n", id.node, prev.node, prev.tree);
            if(prev.tree == current_id) {
              if(!repeat_map.exists(id.node))
                printf("Failed to find repeat chunk %u\n", id.node);
              size_t offset = distinct_map.value_at(distinct_map.find(prev));
              uint32_t writesize = chunk_size;
              if(i*chunk_size+writesize > datalen) 
                writesize = datalen-i*chunk_size;
              memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
            } else {
              node_list(i) = prev;
            }
          } else {
            node_list(i) = NodeID(node_list(i).node, current_id-1);
          }
        }
      });
    }

    // Reference
    size_t chkpt_size = incr_chkpts[header.ref_id].size();
    Kokkos::View<uint8_t*> chkpt_buffer_d("Checkpoint buffer", chkpt_size);
    auto chkpt_buffer_h = Kokkos::create_mirror_view(chkpt_buffer_d);
    memcpy(chkpt_buffer_h.data(), incr_chkpts[header.ref_id].data(), chkpt_size);
    header_t chkpt_header;
    memcpy(&chkpt_header, chkpt_buffer_h.data(), sizeof(header_t));
    uint32_t current_id = chkpt_header.chkpt_id;
    datalen = chkpt_header.datalen;
    chunk_size = chkpt_header.chunk_size;
    Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
    ref_id = chkpt_header.ref_id;
    curr_repeat_offset = sizeof(header_t) + chkpt_header.distinct_size*sizeof(uint32_t);
    prev_repeat_offset = curr_repeat_offset + chkpt_header.curr_repeat_size*2*sizeof(uint32_t);
    data_offset = prev_repeat_offset + chkpt_header.prev_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
    curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, data_offset));
    distinct = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
    data_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, chkpt_size));
    
    distinct_map.clear();
    distinct_map.rehash(chkpt_header.distinct_size);
    Kokkos::fence();
    Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
    Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, distinct.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      distinct_map.insert(NodeID(node, current_id), i*chunk_size);
    });
    uint32_t num_repeat = chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size;
    Kokkos::parallel_for("Fill repeat map", Kokkos::RangePolicy<>(0, num_repeat), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      memcpy(&node, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
      repeat_map.insert(node, NodeID(prev, ref_id));
    });
    Kokkos::parallel_for("Fill list reference data", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      if(node_list(i).tree == current_id) {
        NodeID id = node_list(i);
        if(distinct_map.exists(id)) {
          size_t offset = distinct_map.value_at(distinct_map.find(id));
          uint32_t writesize = chunk_size;
          if(id.node == num_chunks-1)
            writesize = datalen-id.node*chunk_size;
          memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
        } else if(repeat_map.exists(id.node)) {
          NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
          if(prev.tree == current_id) {
            size_t offset = distinct_map.value_at(distinct_map.find(prev));
            uint32_t writesize = chunk_size;
            if(i == num_chunks-1)
              writesize = datalen-i*chunk_size;
            memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize); 
          } else {
            node_list(i) = prev;
          }
        } else {
          node_list(i) = NodeID(node_list(i).node, current_id-1);
        }
      }
    });
    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c3 = std::chrono::high_resolution_clock::now();
    double copy_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
    double restart_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c3-c2).count());
    return std::make_pair(copy_time, restart_time);
}

std::pair<double,double> restart_chkpt_global(std::vector<std::string>& chkpt_files, 
                             const int file_idx, 
                             std::ifstream& file,
                             Kokkos::View<uint8_t*>& data,
                             size_t filesize,
                             uint32_t num_chunks,
                             header_t header,
                             Kokkos::View<uint8_t*>& buffer_d,
                             Kokkos::View<uint8_t*>::HostMirror& buffer_h
                             ) {
    // Main checkpoint
    file.open(chkpt_files[file_idx], std::ifstream::in | std::ifstream::binary);
    file.read((char*)(buffer_h.data()), filesize);
    file.close();
    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();
    Kokkos::deep_copy(buffer_d, buffer_h);
    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c2 = std::chrono::high_resolution_clock::now();
    Kokkos::View<NodeID*> node_list("List of NodeIDs", num_chunks);
    Kokkos::deep_copy(node_list, NodeID());
    uint32_t ref_id = header.ref_id;
    uint32_t cur_id = header.chkpt_id;

    STDOUT_PRINT("Ref ID:           %u\n",  header.ref_id);
    STDOUT_PRINT("Chkpt ID:         %u\n",  header.chkpt_id);
    STDOUT_PRINT("Datalen:          %lu\n", header.datalen);
    STDOUT_PRINT("Chunk size:       %u\n",  header.chunk_size);
    STDOUT_PRINT("Window size:      %u\n",  header.window_size);
    STDOUT_PRINT("Distinct size:    %u\n",  header.distinct_size);
    STDOUT_PRINT("Curr repeat size: %u\n",  header.curr_repeat_size);
    STDOUT_PRINT("Prev repeat size: %u\n",  header.prev_repeat_size);
    STDOUT_PRINT("Num prior chkpts: %u\n",  header.num_prior_chkpts);

    size_t curr_repeat_offset = sizeof(header_t) + header.distinct_size*sizeof(uint32_t);
    size_t prev_repeat_offset = header.num_prior_chkpts*2*sizeof(uint32_t) + curr_repeat_offset + header.curr_repeat_size*2*sizeof(uint32_t);
    size_t data_offset = prev_repeat_offset + header.prev_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
    auto curr_repeat   = Kokkos::subview(buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    auto prev_repeat   = Kokkos::subview(buffer_d, std::make_pair(prev_repeat_offset, data_offset));
    auto distinct      = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
    auto data_subview  = Kokkos::subview(buffer_d, std::make_pair(data_offset, filesize));
    STDOUT_PRINT("Checkpoint %u\n", header.chkpt_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", filesize);
    STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
    STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
    STDOUT_PRINT("Data offset: %lu\n", data_offset);

    uint32_t chunk_size = header.chunk_size;
    size_t datalen = header.datalen;
    Kokkos::UnorderedMap<NodeID, size_t> distinct_map(header.distinct_size);
    Kokkos::parallel_for("Restart Hashlist distinct", Kokkos::RangePolicy<>(0, header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, distinct.data() + i*(sizeof(uint32_t)),  sizeof(uint32_t));
      distinct_map.insert(NodeID(node,cur_id),  i*chunk_size);
      node_list(node) = NodeID(node, cur_id);
      uint32_t datasize = chunk_size;
      if(node == num_chunks-1)
        datasize = datalen - node*chunk_size;
      memcpy(data.data()+chunk_size*node, data_subview.data()+i*chunk_size, datasize);
    });

    uint32_t num_prior_chkpts = header.num_prior_chkpts;
    Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
    auto repeat_region_sizes_h = Kokkos::create_mirror_view(repeat_region_sizes);
    Kokkos::deep_copy(repeat_region_sizes, 0);
    // Read map of repeats for each checkpoint
    Kokkos::parallel_for("Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t chkpt;
      memcpy(&chkpt, buffer_d.data()+curr_repeat_offset+i*2*sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&repeat_region_sizes(chkpt), buffer_d.data()+curr_repeat_offset+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
      STDOUT_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
    });
    Kokkos::deep_copy(repeat_region_sizes_h, repeat_region_sizes);
    // Perform exclusive scan to determine where regions start/stop
    Kokkos::parallel_scan("Repeat offsets", cur_id+1, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      partial_sum += repeat_region_sizes(i);
      if(is_final) repeat_region_sizes(i) = partial_sum;
    });
    STDOUT_PRINT("Num repeats: %u\n", header.curr_repeat_size+header.prev_repeat_size);
    // Load repeat entries and fill in metadata for chunks
    Kokkos::parallel_for("Restart Hash tree repeats main checkpoint", Kokkos::RangePolicy<>(0, header.curr_repeat_size+header.prev_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      uint32_t tree = 0;
      memcpy(&node, curr_repeat.data()+(num_prior_chkpts)*2*sizeof(uint32_t)+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, curr_repeat.data()+(num_prior_chkpts)*2*sizeof(uint32_t)+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
      // Determine ID 
      for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
        if(i < repeat_region_sizes(j)) {
          tree = j;
        }
      }
      uint32_t idx = distinct_map.find(NodeID(prev, tree));
      size_t offset = distinct_map.value_at(distinct_map.find(NodeID(prev, tree)));
      node_list(node) = NodeID(prev, tree);
      if(tree == cur_id) {
        uint32_t copysize = chunk_size;
        if(node == num_chunks-1)
          copysize = data.size() - chunk_size*node;
        memcpy(data.data()+chunk_size*node, data_subview.data()+offset, copysize);
      }
    });

    Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      NodeID entry = node_list(i);
      if(entry.node == UINT_MAX) {
        node_list(i) = NodeID(i, cur_id-1);
      }
    });
    Kokkos::fence();

    for(int idx=static_cast<int>(file_idx)-1; idx>static_cast<int>(ref_id); idx--) {
      STDOUT_PRINT("Processing checkpoint %u\n", idx);
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

      STDOUT_PRINT("Ref ID:           %u\n",  chkpt_header.ref_id);
      STDOUT_PRINT("Chkpt ID:         %u\n",  chkpt_header.chkpt_id);
      STDOUT_PRINT("Datalen:          %lu\n", chkpt_header.datalen);
      STDOUT_PRINT("Chunk size:       %u\n",  chkpt_header.chunk_size);
      STDOUT_PRINT("Window size:      %u\n",  chkpt_header.window_size);
      STDOUT_PRINT("Distinct size:    %u\n",  chkpt_header.distinct_size);
      STDOUT_PRINT("Curr repeat size: %u\n",  chkpt_header.curr_repeat_size);
      STDOUT_PRINT("Prev repeat size: %u\n",  chkpt_header.prev_repeat_size);
      STDOUT_PRINT("Num prior chkpts: %u\n",  chkpt_header.num_prior_chkpts);
      
      curr_repeat_offset = sizeof(header_t) + chkpt_header.distinct_size*sizeof(uint32_t);
      prev_repeat_offset = chkpt_header.num_prior_chkpts*2*sizeof(uint32_t) + curr_repeat_offset + chkpt_header.curr_repeat_size*2*sizeof(uint32_t);
      data_offset = prev_repeat_offset + chkpt_header.prev_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
      curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
      prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, data_offset));
      distinct      = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
      data_subview  = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, chkpt_size));
      STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
      STDOUT_PRINT("Checkpoint size: %lu\n", chkpt_size);
      STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
      STDOUT_PRINT("Curr repeat offset: %lu\n", curr_repeat_offset);
      STDOUT_PRINT("Prev repeat offset: %lu\n", prev_repeat_offset);
      STDOUT_PRINT("Data offset: %lu\n", data_offset);

      distinct_map.clear();
      distinct_map.rehash(chkpt_header.distinct_size);
      Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
      Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
        memcpy(&node, distinct.data()+i*sizeof(uint32_t), sizeof(uint32_t));
        distinct_map.insert(NodeID(node,cur_id), i*chunk_size);
      });
      num_prior_chkpts = chkpt_header.num_prior_chkpts;
      Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
      Kokkos::parallel_for("Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t chkpt;
        memcpy(&chkpt, chkpt_buffer_d.data()+curr_repeat_offset+i*2*sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&repeat_region_sizes(chkpt), chkpt_buffer_d.data()+curr_repeat_offset+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
        STDOUT_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
      });
      Kokkos::parallel_scan("Repeat offsets", cur_id+1, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
        partial_sum += repeat_region_sizes(i);
        if(is_final) repeat_region_sizes(i) = partial_sum;
      });

      STDOUT_PRINT("Num repeats: %u\n", chkpt_header.curr_repeat_size+chkpt_header.prev_repeat_size);
  
      uint32_t num_repeat_entries = chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size;

      Kokkos::parallel_for("Restart Hash tree repeats middle chkpts", Kokkos::RangePolicy<>(0,num_repeat_entries), KOKKOS_LAMBDA(const uint32_t i) { 
        uint32_t node, prev, tree=0;
        memcpy(&node, curr_repeat.data()+(num_prior_chkpts)*2*sizeof(uint32_t)+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
        memcpy(&prev, curr_repeat.data()+(num_prior_chkpts)*2*sizeof(uint32_t)+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
        for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
          if(i < repeat_region_sizes(j)) {
            tree = j;
          }
        }
        auto result = repeat_map.insert(node, NodeID(prev,tree));
        if(result.failed())
          STDOUT_PRINT("Failed to insert previous repeat %u: (%u,%u) into repeat map\n", node, prev, tree);
      });

      Kokkos::parallel_for("Fill data middle chkpts", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
        if(node_list(i).tree == current_id) {
          NodeID id = node_list(i);
          if(distinct_map.exists(id)) {
            size_t offset = distinct_map.value_at(distinct_map.find(id));
            uint32_t writesize = chunk_size;
            if(i*chunk_size+writesize > datalen) 
              writesize = datalen-i*chunk_size;
            memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
          } else if(repeat_map.exists(id.node)) {
            NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
            DEBUG_PRINT("Repaeat value: %u: (%u,%u)\n", id.node, prev.node, prev.tree);
            if(prev.tree == current_id) {
              if(!repeat_map.exists(id.node))
                printf("Failed to find repeat chunk %u\n", id.node);
              size_t offset = distinct_map.value_at(distinct_map.find(prev));
              uint32_t writesize = chunk_size;
              if(i*chunk_size+writesize > datalen) 
                writesize = datalen-i*chunk_size;
              memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
            } else {
              node_list(i) = prev;
            }
          } else {
            node_list(i) = NodeID(node_list(i).node, current_id-1);
          }
        }
      });
    }

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
    curr_repeat_offset = sizeof(header_t) + chkpt_header.distinct_size*sizeof(uint32_t);
    prev_repeat_offset = curr_repeat_offset + chkpt_header.curr_repeat_size*2*sizeof(uint32_t);
    data_offset = prev_repeat_offset + chkpt_header.prev_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
    curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, data_offset));
    distinct = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
    data_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, chkpt_size));
    
    distinct_map.clear();
    distinct_map.rehash(chkpt_header.distinct_size);
    Kokkos::fence();
    Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
    Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, distinct.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      distinct_map.insert(NodeID(node, current_id), i*chunk_size);
    });
    uint32_t num_repeat = chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size;
    Kokkos::parallel_for("Fill repeat map", Kokkos::RangePolicy<>(0, num_repeat), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      memcpy(&node, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
      repeat_map.insert(node, NodeID(prev, ref_id));
    });
    Kokkos::parallel_for("Fill list reference data", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      if(node_list(i).tree == current_id) {
        NodeID id = node_list(i);
        if(distinct_map.exists(id)) {
          size_t offset = distinct_map.value_at(distinct_map.find(id));
          uint32_t writesize = chunk_size;
          if(id.node == num_chunks-1)
            writesize = datalen-id.node*chunk_size;
          memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
        } else if(repeat_map.exists(id.node)) {
          NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
          if(prev.tree == current_id) {
            size_t offset = distinct_map.value_at(distinct_map.find(prev));
            uint32_t writesize = chunk_size;
            if(i == num_chunks-1)
              writesize = datalen-i*chunk_size;
            memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize); 
          } else {
            node_list(i) = prev;
          }
        } else {
          node_list(i) = NodeID(node_list(i).node, current_id-1);
        }
      }
    });
    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c3 = std::chrono::high_resolution_clock::now();
    double copy_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
    double restart_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c3-c2).count());
    return std::make_pair(copy_time, restart_time);
}

std::pair<double,double>
restart_incr_chkpt_naivehashlist( std::vector<std::vector<uint8_t>>& incr_chkpts,
                             const int idx, 
                             Kokkos::View<uint8_t*>& data) {
  size_t filesize = incr_chkpts[idx].size();

  DEBUG_PRINT("File size: %zd\n", filesize);
  header_t header;
  memcpy(&header, incr_chkpts[idx].data(), sizeof(header_t));
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

  uint32_t num_chunks = header.datalen / header.chunk_size;
  if(num_chunks*header.chunk_size < header.datalen) {
    num_chunks += 1;
  }
  Kokkos::resize(data, header.datalen);

  std::pair<double,double> times;
  times = restart_chkpt_naive(incr_chkpts, idx, data, filesize, num_chunks, header, buffer_d, buffer_h);
  Kokkos::fence();
  STDOUT_PRINT("Restarted checkpoint\n");
  return times;
}

std::pair<double,double>
restart_incr_chkpt_naivehashlist( std::vector<std::string>& chkpt_files,
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
  Kokkos::resize(data, header.datalen);

  std::pair<double,double> times;
  times = restart_chkpt_naive(chkpt_files, file_idx, file, data, filesize, num_chunks, header, buffer_d, buffer_h);
  Kokkos::fence();
  STDOUT_PRINT("Restarted checkpoint\n");
  return times;
}

std::pair<double,double>
restart_incr_chkpt_hashlist( std::vector<std::vector<uint8_t>>& incr_chkpts,
                             const int idx, 
                             Kokkos::View<uint8_t*>& data) {
  // Read main incremental checkpoint header
  size_t filesize = incr_chkpts[idx].size();

  DEBUG_PRINT("File size: %zd\n", filesize);
  header_t header;
  memcpy(&header, incr_chkpts[idx].data(), sizeof(header_t));
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

  uint32_t num_chunks = header.datalen / header.chunk_size;
  if(num_chunks*header.chunk_size < header.datalen) {
    num_chunks += 1;
  }
  Kokkos::resize(data, header.datalen);

  std::pair<double,double> times;
  if(header.window_size == 0) {
    times = restart_chkpt_local(incr_chkpts, idx, data, filesize, num_chunks, header, buffer_d, buffer_h);
  } else {
    times = restart_chkpt_global(incr_chkpts, idx, data, filesize, num_chunks, header, buffer_d, buffer_h);
  }
  Kokkos::fence();
  STDOUT_PRINT("Restarted checkpoint\n");
  return times;
}

std::pair<double,double>
restart_incr_chkpt_hashlist( std::vector<std::string>& chkpt_files,
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
  Kokkos::resize(data, header.datalen);

  std::pair<double,double> times;
  if(header.window_size == 0) {
    times = restart_chkpt_local(chkpt_files, file_idx, file, data, filesize, num_chunks, header, buffer_d, buffer_h);
  } else {
    times = restart_chkpt_global(chkpt_files, file_idx, file, data, filesize, num_chunks, header, buffer_d, buffer_h);
  }
  Kokkos::fence();
  STDOUT_PRINT("Restarted checkpoint\n");
  return times;
}

std::pair<uint64_t,uint64_t> 
write_incr_chkpt_hashlist_naive( const Kokkos::View<uint8_t*>& data, 
                           Kokkos::View<uint8_t*>& buffer_d, 
                           uint32_t chunk_size, 
                           Kokkos::Bitset<Kokkos::DefaultExecutionSpace>& changes,
                           uint32_t prior_chkpt_id,
                           uint32_t chkpt_id,
                           header_t& header) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size()) {
    num_chunks += 1;
  }

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
  uint64_t buffer_size = sizeof(header_t);
  buffer_size += changes.count()*(sizeof(uint32_t) + chunk_size);
  size_t data_offset = changes.count()*sizeof(uint32_t);

  DEBUG_PRINT("Changes: %u\n", changes.count());
  DEBUG_PRINT("Buffer size: %lu\n", buffer_size);
  buffer_d = Kokkos::View<uint8_t*>("Buffer", buffer_size);

  Kokkos::parallel_for("Make incremental checkpoint", Kokkos::RangePolicy<>(0, changes.size()), KOKKOS_LAMBDA(const uint32_t i) {
    if(changes.test(i)) {
      Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
      Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
      size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t));
      memcpy(buffer_d.data()+sizeof(header_t) + pos, &i, sizeof(uint32_t));
      uint32_t writesize = chunk_size;
      if((i+1)*chunk_size > data.size()) {
        writesize = data.size()-i*chunk_size;
      }
      memcpy(buffer_d.data()+sizeof(header_t) + data_offset+(pos/sizeof(uint32_t))*chunk_size, data.data()+chunk_size*i, writesize);
    }
  });
  Kokkos::fence();
  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
  Kokkos::fence();
  header.ref_id = prior_chkpt_id;
  header.chkpt_id = chkpt_id;
  header.datalen = data.size();
  header.chunk_size = chunk_size;
  header.window_size = 0;
  header.distinct_size = changes.count();
  header.curr_repeat_size = 0;
  header.prev_repeat_size = 0;
  header.num_prior_chkpts = 0;
  DEBUG_PRINT("Ref ID: %u\n"          , header.ref_id);
  DEBUG_PRINT("Chkpt ID: %u\n"        , header.chkpt_id);
  DEBUG_PRINT("Data len: %lu\n"       , header.datalen);
  DEBUG_PRINT("Chunk size: %u\n"      , header.chunk_size);
  DEBUG_PRINT("Window size: %u\n"     , header.window_size);
  DEBUG_PRINT("Distinct size: %u\n"   , header.distinct_size);
  DEBUG_PRINT("Curr repeat size: %u\n", header.curr_repeat_size);
  DEBUG_PRINT("prev repeat size: %u\n", header.prev_repeat_size);
  STDOUT_PRINT("Number of bytes written for incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_data_h(0) + num_bytes_metadata_h(0));
  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));

  DEBUG_PRINT("Trying to close file\n");
  DEBUG_PRINT("Closed file\n");
  return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
}

std::pair<uint64_t,uint64_t> 
write_incr_chkpt_hashlist_local( 
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
  uint32_t buffer_size = 0;
  buffer_size += sizeof(uint32_t)*2*shared.size();
  buffer_size += distinct.size()*(sizeof(uint32_t) + chunk_size);
  size_t data_offset = distinct.size()*sizeof(uint32_t)+2*shared.size()*sizeof(uint32_t);

  DEBUG_PRINT("Buffer size: %u\n", buffer_size);
  Kokkos::resize(buffer_d, buffer_size);
  DEBUG_PRINT("Distinct entries: %u\n", distinct.size());
  DEBUG_PRINT("Repeat entries: %u\n", shared.size());
  Kokkos::parallel_for("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      auto info = distinct.value_at(i);
      Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
      Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
      size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t));
      memcpy(buffer_d.data()+pos, &info.node, sizeof(uint32_t));
      uint32_t writesize = chunk_size;
      if(info.node == num_chunks-1) {
        writesize = data.size()-info.node*chunk_size;
      }
      memcpy(buffer_d.data()+data_offset+(pos/sizeof(uint32_t))*chunk_size, data.data()+chunk_size*info.node, writesize);
    }
  });
  Kokkos::parallel_for("Count curr repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      uint32_t k = shared.key_at(i);
      NodeID v = shared.value_at(i);
      if(v.tree == chkpt_id) {
        Kokkos::atomic_add(&num_curr_repeat_d(0), 1);
        Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
        uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
        memcpy(buffer_d.data()+pos, &k, sizeof(uint32_t));
        memcpy(buffer_d.data()+pos+sizeof(uint32_t), &v.node, sizeof(uint32_t));
      }
    }
  });
  Kokkos::parallel_for("Count prior repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      uint32_t k = shared.key_at(i);
      NodeID v = shared.value_at(i);
      if(v.tree != chkpt_id) {
        Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
        uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
        memcpy(buffer_d.data()+pos, &k, sizeof(uint32_t));
        memcpy(buffer_d.data()+pos+sizeof(uint32_t), &v.node, sizeof(uint32_t));
      }
    }
  });
  Kokkos::fence();
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
  if(prior_chkpt_id == chkpt_id) {
    header.num_prior_chkpts = 1;
  } else {
    header.num_prior_chkpts = 2;
  }
  DEBUG_PRINT("Ref ID: %u\n"          , header.ref_id);
  DEBUG_PRINT("Chkpt ID: %u\n"        , header.chkpt_id);
  DEBUG_PRINT("Data len: %lu\n"       , header.datalen);
  DEBUG_PRINT("Chunk size: %u\n"      , header.chunk_size);
  DEBUG_PRINT("Window size: %u\n"     , header.window_size);
  DEBUG_PRINT("Distinct size: %u\n"   , header.distinct_size);
  DEBUG_PRINT("Curr repeat size: %u\n", header.curr_repeat_size);
  DEBUG_PRINT("prev repeat size: %u\n", header.prev_repeat_size);
  STDOUT_PRINT("Number of bytes written for incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_data_h(0) + num_bytes_metadata_h(0));
  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));

  DEBUG_PRINT("Trying to close file\n");
  DEBUG_PRINT("Closed file\n");
  return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
}

std::pair<uint64_t,uint64_t> 
write_incr_chkpt_hashlist_global(
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

  Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
  Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
  Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
  Kokkos::View<uint64_t[1]> num_curr_repeat_d("Number of curr repeat entries");
  Kokkos::View<uint64_t[1]>::HostMirror num_curr_repeat_h = Kokkos::create_mirror_view(num_curr_repeat_d);
  Kokkos::View<uint64_t*> prior_counter_d("Counter for prior repeats", chkpt_id+1);
  Kokkos::View<uint64_t*>::HostMirror prior_counter_h = Kokkos::create_mirror_view(prior_counter_d);
  Kokkos::Experimental::ScatterView<uint64_t*> prior_counter_sv(prior_counter_d);
  Kokkos::deep_copy(prior_counter_d, 0);
  Kokkos::deep_copy(num_curr_repeat_d, 0);
  Kokkos::deep_copy(num_bytes_d, 0);
  Kokkos::deep_copy(num_bytes_data_d, 0);
  Kokkos::deep_copy(num_bytes_metadata_d, 0);
  // Reference checkpoint
  std::pair<uint64_t,uint64_t> num_written;
  if(prior_chkpt_id == chkpt_id) {
    num_written = write_incr_chkpt_hashlist_local(data, buffer_d, chunk_size, distinct, shared, prior_chkpt_id, chkpt_id, header);
  } else {
    // Subsequent checkpoints using a reference checkpoint as the baseline
    Kokkos::View<uint32_t[1]> distinct_counter_d("Num distinct");
    auto distinct_counter_h = Kokkos::create_mirror_view(distinct_counter_d);
    Kokkos::deep_copy(distinct_counter_d, 0);
    Kokkos::View<uint32_t[1]> distinct_repeat_counter_d("Num distinct_repeat");
    auto distinct_repeat_counter_h = Kokkos::create_mirror_view(distinct_repeat_counter_d);
    Kokkos::deep_copy(distinct_repeat_counter_d, 0);
    Kokkos::parallel_for("Count actual distinct", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
      if(distinct.valid_at(i)) {
        auto info = distinct.value_at(i);
        if(chkpt_id == info.tree) {
          Kokkos::atomic_add(&distinct_counter_d(0), 1);
        }
      }
    });
    // Small bitset to record which checkpoints are necessary for restart
    Kokkos::Bitset<Kokkos::DefaultExecutionSpace> chkpts_needed(chkpt_id+1);
    chkpts_needed.reset();
    // Count how many repeats belong to each checkpoint
    Kokkos::parallel_for("Count repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
      auto prior_counter_sa = prior_counter_sv.access();
      if(shared.valid_at(i)) {
        uint32_t k = shared.key_at(i);
        NodeID v = shared.value_at(i);
        chkpts_needed.set(v.tree);
        prior_counter_sa(v.tree) += 1; 
        if(v.tree == chkpt_id) {
          Kokkos::atomic_add(&distinct_repeat_counter_d(0), 1);
        }
      }
    });
    Kokkos::Experimental::contribute(prior_counter_d, prior_counter_sv);
    prior_counter_sv.reset_except(prior_counter_d);
    Kokkos::deep_copy(distinct_counter_h, distinct_counter_d);
    Kokkos::deep_copy(distinct_repeat_counter_h, distinct_repeat_counter_d);
    size_t data_offset = distinct_counter_h(0)*sizeof(uint32_t)+chkpts_needed.count()*2*sizeof(uint32_t) + shared.size()*2*sizeof(uint32_t);
    uint32_t buffer_size = 0;
    buffer_size += distinct_counter_h(0)*(sizeof(uint32_t)+chunk_size);
    buffer_size += chkpts_needed.count()*2*sizeof(uint32_t);
    buffer_size += shared.size()*2*sizeof(uint32_t);
    DEBUG_PRINT("Distinct size: %u\n", distinct_counter_h(0));
    DEBUG_PRINT("Current Repeat size: %u\n", distinct_repeat_counter_h(0));
    DEBUG_PRINT("Prior Repeat size: %u\n", shared.size()-distinct_repeat_counter_h(0));
    DEBUG_PRINT("Buffer size: %u\n", buffer_size);
    Kokkos::resize(buffer_d, buffer_size);
    Kokkos::View<uint64_t[1]> num_distinct_d("Number of distinct");
    auto num_distinct_h = Kokkos::create_mirror_view(num_distinct_d);
    Kokkos::deep_copy(num_distinct_d, 0);
    Kokkos::parallel_for("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
      if(distinct.valid_at(i)) {
        auto info = distinct.value_at(i);
        if(chkpt_id == info.tree) {
          Kokkos::atomic_add(&num_distinct_d(0), 1);
          Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
          Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
          size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t));
          memcpy(buffer_d.data()+pos, &info, sizeof(uint32_t));
          uint32_t writesize = chunk_size;
          if(info.node == num_chunks-1) {
            writesize = data.size()-info.node*chunk_size;
          }
          memcpy(buffer_d.data()+data_offset+(pos/sizeof(uint32_t))*chunk_size, data.data()+chunk_size*info.node, writesize);
        }
      }
    });
    // Write Repeat map for recording how many entries per checkpoint
    // (Checkpoint ID, # of entries)
    Kokkos::parallel_for("Write size map", prior_counter_d.size(), KOKKOS_LAMBDA(const uint32_t i) {
      if(prior_counter_d(i) > 0) {
        uint32_t num_repeats_i = static_cast<uint32_t>(prior_counter_d(i));
        Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
        memcpy(buffer_d.data()+pos, &i, sizeof(uint32_t));
        memcpy(buffer_d.data()+pos+sizeof(uint32_t), &num_repeats_i, sizeof(uint32_t));
        DEBUG_PRINT("Wrote table entry (%u,%u) at offset %lu\n", i, num_repeats_i, pos);
      }
    });
    // Calculate repeat indices so that we can separate entries by source ID
    Kokkos::parallel_scan("Calc repeat end indices", prior_counter_d.size(), KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      partial_sum += prior_counter_d(i);
      if(is_final) prior_counter_d(i) = partial_sum;
    });

    size_t prior_start = distinct_counter_h(0)*sizeof(uint32_t)+chkpts_needed.count()*2*sizeof(uint32_t);
    DEBUG_PRINT("Prior start offset: %lu\n", prior_start);

    // Write repeat entries
    Kokkos::parallel_for("Write repeat bytes", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
      if(shared.valid_at(i)) {
        uint32_t node = shared.key_at(i);
        NodeID prev = shared.value_at(i);
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(uint32_t));
        Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
        size_t pos = Kokkos::atomic_sub_fetch(&prior_counter_d(prev.tree), 1);
        memcpy(buffer_d.data()+prior_start+pos*2*sizeof(uint32_t), &node, sizeof(uint32_t));
        memcpy(buffer_d.data()+prior_start+pos*2*sizeof(uint32_t)+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
      }
    });
    Kokkos::fence();
    Kokkos::deep_copy(num_bytes_h, num_bytes_d);
    Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
    Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
    Kokkos::deep_copy(num_curr_repeat_h, num_curr_repeat_d);
    Kokkos::deep_copy(num_distinct_h, num_distinct_d);
    Kokkos::fence();
    header.ref_id = prior_chkpt_id;
    header.chkpt_id = chkpt_id;
    header.datalen = data.size();
    header.chunk_size = chunk_size;
    header.window_size = UINT32_MAX;
    header.distinct_size = num_distinct_h(0);
    header.curr_repeat_size = num_curr_repeat_h(0);
    header.prev_repeat_size = shared.size() - num_curr_repeat_h(0);
    header.num_prior_chkpts = chkpts_needed.count();
    DEBUG_PRINT("Dumping header: \n");
    DEBUG_PRINT("Ref ID:           %u\n",  header.ref_id);
    DEBUG_PRINT("Chkpt ID:         %u\n",  header.chkpt_id);
    DEBUG_PRINT("Datalen:          %lu\n", header.datalen);
    DEBUG_PRINT("Chunk size:       %u\n",  header.chunk_size);
    DEBUG_PRINT("Window size:      %u\n",  header.window_size);
    DEBUG_PRINT("Distinct size:    %u\n",  header.distinct_size);
    DEBUG_PRINT("Curr repeat size: %u\n",  header.curr_repeat_size);
    DEBUG_PRINT("Prev repeat size: %u\n",  header.prev_repeat_size);
    STDOUT_PRINT("Number of bytes written for incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_h(0));
    STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
    STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));
  }
  DEBUG_PRINT("Trying to close file\n");
  DEBUG_PRINT("Closed file\n");
  if(prior_chkpt_id == chkpt_id) {
    return num_written;
  } else {
    return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
  }
}

#endif

