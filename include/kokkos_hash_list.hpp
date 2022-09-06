#ifndef KOKKOS_HASH_LIST_HPP
#define KOKKOS_HASH_LIST_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <climits>
#include "hash_functions.hpp"
#include "map_helpers.hpp"
#include "utils.hpp"

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

//void find_distinct_chunks(const HashList& list, const uint32_t list_id, DistinctMap& distinct_map, SharedMap& shared_map, DistinctMap& prior_map) {
//  Kokkos::parallel_for("Find distinct chunks", Kokkos::RangePolicy<>(0,list.list_d.extent(0)), KOKKOS_LAMBDA(const uint32_t i) {
//    HashDigest digest = list.list_d(i);
//    NodeID info(i, list_id);
//    auto result = distinct_map.insert(digest, info);
//    if(result.failed()) 
//      printf("Warning: Failed to insert (%u,%u) into map for hashlist.\n", info.node, info.tree);
//
////    if(prior_map.exists(digest)) {
////      auto old_res = prior_map.find(digest);
////      NodeInfo old_node = prior_map.value_at(old_res);
////      NodeInfo new_node(i, old_node.src, old_node.tree);
////      auto distinct_res = distinct_map.insert(digest, new_node);
////      if(distinct_res.failed())
////        printf("Failed to insert entry into distinct map\n");
////    } else {
////      auto result = distinct_map.insert(digest, info);
////      if(result.existing()) {
////        NodeInfo old_info = distinct_map.value_at(result.index());
////        auto shared_res = shared_map.insert(i, old_info.node);
////        if(shared_res.failed())
////          printf("Failed to insert chunk into either distinct or shared maps\n");
////      }
////    }
//  });
//  printf("Number of comparisons (Hash List)  : %lu\n", list.list_d.extent(0));
//  Kokkos::fence();
//}

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
  printf("Number of chunks: %u\n", num_chunks);
  printf("Number of new chunks: %u\n", num_new_h(0));
  printf("Number of same chunks: %u\n", num_same_h(0));
  printf("Number of shift chunks: %u\n", num_shift_h(0));
  printf("Number of comp nodes: %u\n", num_comp_h(0));
  printf("Number of dupl nodes: %u\n", num_dupl_h(0));
#endif
}

template<class Hasher>
void compare_lists_global( Hasher& hasher, 
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
    NodeID info(i, list_id);
    uint32_t old_idx = prior_distinct_map.find(digest); // Search for digest in prior distinct map
    if(!prior_distinct_map.valid_at(old_idx)) { // Chunk not in prior chkpt
      auto result = distinct_map.insert(digest, info);
      if(result.existing()) {
        NodeID old = distinct_map.value_at(result.index());
        auto shared_result = shared_map.insert(i, old);
//printf("Inserted %u: (%u,%u)\n", i, old.node, old.tree);
        if(shared_result.existing()) {
          printf("Update SharedMap: Node %u already in shared map.\n", i);
        } else if(shared_result.failed()) {
          printf("Update SharedMap: Failed to insert %u into the shared map.\n", i);
        }
#ifdef STATS
        Kokkos::atomic_add(&num_dupl(0), static_cast<uint32_t>(1));
#endif
      } else if(result.failed())  {
        printf("Warning: Failed to insert (%u,%u) into map for hashlist.\n", info.node, info.tree);
#ifdef STATS
      } else if(result.success()) {
        Kokkos::atomic_add(&num_new(0), static_cast<uint32_t>(1));
#endif
      }
    } else { // Chunk is in prior chkpt
      NodeID old_distinct = prior_distinct_map.value_at(old_idx);
      if(i != old_distinct.node) { // Existing chunk is at a different offset
        uint32_t old_shared_idx = prior_shared_map.find(i);
        if(prior_shared_map.valid_at(old_shared_idx)) {
          NodeID old_shared = prior_shared_map.value_at(old_shared_idx);
          if(old_distinct.node != old_shared.node || old_distinct.tree != old_shared.tree) {
//printf("Inserted %u: (%u,%u)\n", i, old_distinct.node, old_distinct.tree);
            shared_map.insert(i, old_distinct);
#ifdef STATS
            Kokkos::atomic_add(&num_shift(0), static_cast<uint32_t>(1));
          } else {
            Kokkos::atomic_add(&num_same(0), static_cast<uint32_t>(1));
#endif
          }
        } else {
//printf("Inserted %u: (%u,%u)\n", i, old_distinct.node, old_distinct.tree);
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
  printf("Number of chunks: %u\n", num_chunks);
  printf("Number of new chunks: %u\n", num_new_h(0));
  printf("Number of same chunks: %u\n", num_same_h(0));
  printf("Number of shift chunks: %u\n", num_shift_h(0));
  printf("Number of comp nodes: %u\n", num_comp_h(0));
  printf("Number of dupl nodes: %u\n", num_dupl_h(0));
#endif
}

int 
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
//    size_t prev_repeat_offset = filesize - header.prev_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
//    size_t curr_repeat_offset = prev_repeat_offset - header.curr_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
//    auto curr_repeat   = Kokkos::subview(buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
//    auto prev_repeat   = Kokkos::subview(buffer_d, std::make_pair(prev_repeat_offset, filesize));
//    auto distinct = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));

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
//      memcpy(&node, distinct.data() + i*(sizeof(uint32_t)+chunk_size),  sizeof(uint32_t));
//      distinct_map.insert(NodeID(node,cur_id),  i*(sizeof(uint32_t)+chunk_size) + sizeof(uint32_t));
      memcpy(&node, distinct.data() + i*sizeof(uint32_t),  sizeof(uint32_t));
      distinct_map.insert(NodeID(node,cur_id),  i*chunk_size);
      node_list(node) = NodeID(node,cur_id);
      uint32_t datasize = chunk_size;
      if(node == num_chunks-1)
        datasize = datalen - node*chunk_size;
//      memcpy(data.data()+chunk_size*node, distinct.data()+i*(sizeof(uint32_t)+chunk_size)+sizeof(uint32_t), datasize);
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
//      memcpy(data.data()+chunk_size*node, distinct.data()+offset, copysize);
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
//      prev_repeat_offset = chkpt_size - chkpt_header.prev_repeat_size*(2*sizeof(uint32_t));
//      curr_repeat_offset = prev_repeat_offset - chkpt_header.curr_repeat_size*(2*sizeof(uint32_t));
//      curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
//      prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, chkpt_size));
//      distinct = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));

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
//        memcpy(&node, distinct.data()+i*(sizeof(uint32_t)+chunk_size), sizeof(uint32_t));
//        distinct_map.insert(NodeID(node, current_id), i*(sizeof(uint32_t)+chunk_size)+sizeof(uint32_t));
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
//            memcpy(data.data()+chunk_size*i, distinct.data()+offset, writesize);
          } else if(repeat_map.exists(id.node)) {
            NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
            if(prev.tree == current_id) {
              size_t offset = distinct_map.value_at(distinct_map.find(prev));
              uint32_t writesize = chunk_size;
              if(i == num_chunks-1)
                writesize = datalen-i*chunk_size;
              memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize); 
//              memcpy(data.data()+chunk_size*i, distinct.data()+offset, writesize); 
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
  } else {
    // Main checkpoint
    file.open(chkpt_files[file_idx], std::ifstream::in | std::ifstream::binary);
    file.read((char*)(buffer_h.data()), filesize);
    file.close();
    Kokkos::deep_copy(buffer_d, buffer_h);
    Kokkos::View<NodeID*> node_list("List of NodeIDs", num_chunks);
    Kokkos::deep_copy(node_list, NodeID());
    uint32_t ref_id = header.ref_id;
    uint32_t cur_id = header.chkpt_id;
//    size_t prev_repeat_offset = filesize - header.prev_repeat_size*(sizeof(NodeID)+sizeof(uint32_t));
//    size_t curr_repeat_offset = prev_repeat_offset - header.curr_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
//    auto curr_repeat   = Kokkos::subview(buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
//    auto prev_repeat   = Kokkos::subview(buffer_d, std::make_pair(prev_repeat_offset, filesize));
//    auto distinct = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));

    size_t curr_repeat_offset = sizeof(header_t) + header.distinct_size*sizeof(uint32_t);
    size_t prev_repeat_offset = curr_repeat_offset + header.curr_repeat_size*2*sizeof(uint32_t);
    size_t data_offset = prev_repeat_offset + header.prev_repeat_size*(sizeof(uint32_t)+sizeof(NodeID));
    auto curr_repeat   = Kokkos::subview(buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
    auto prev_repeat   = Kokkos::subview(buffer_d, std::make_pair(prev_repeat_offset, data_offset));
    auto distinct = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
    auto data_subview = Kokkos::subview(buffer_d, std::make_pair(data_offset, filesize));

    uint32_t chunk_size = header.chunk_size;
    size_t datalen = header.datalen;
    Kokkos::UnorderedMap<NodeID, size_t> distinct_map(header.distinct_size);
    Kokkos::parallel_for("Restart Hashlist distinct", Kokkos::RangePolicy<>(0, header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
//      memcpy(&node, distinct.data() + i*(sizeof(uint32_t)+chunk_size),  sizeof(uint32_t));
//      distinct_map.insert(NodeID(node,cur_id),  i*(sizeof(uint32_t)+chunk_size) + sizeof(uint32_t));
      memcpy(&node, distinct.data() + i*(sizeof(uint32_t)),  sizeof(uint32_t));
      distinct_map.insert(NodeID(node,cur_id),  i*chunk_size);
      node_list(node) = NodeID(node, cur_id);
      uint32_t datasize = chunk_size;
      if(node == num_chunks-1)
        datasize = datalen - node*chunk_size;
//      memcpy(data.data()+chunk_size*node, distinct.data()+i*(sizeof(uint32_t)+chunk_size)+sizeof(uint32_t), datasize);
      memcpy(data.data()+chunk_size*node, data_subview.data()+i*chunk_size, datasize);
    });
    Kokkos::parallel_for("Restart Hashlist current repeat", Kokkos::RangePolicy<>(0, header.curr_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      uint32_t prev;
      memcpy(&node, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
      memcpy(&prev, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
      node_list(node) = NodeID(prev,cur_id);
      size_t offset = distinct_map.value_at(distinct_map.find(NodeID(prev,cur_id)));
      uint32_t copysize = chunk_size;
      if(node == num_chunks-1)
        copysize = data.size() - chunk_size*(num_chunks-1);
//      memcpy(data.data()+chunk_size*node, distinct.data()+offset, copysize);
      memcpy(data.data()+chunk_size*node, data_subview.data()+offset, copysize);
    });
    Kokkos::parallel_for("Restart Hashlist previous repeat", Kokkos::RangePolicy<>(0, header.prev_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      NodeID prev;
      memcpy(&node, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(NodeID)), sizeof(uint32_t));
      memcpy(&prev, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(NodeID)) +sizeof(uint32_t), sizeof(NodeID));
      node_list(node) = prev;
    });
    Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
      NodeID entry = node_list(i);
      if(entry.node == UINT_MAX) {
        node_list(i) = NodeID(i, cur_id-1);
      }
    });
    Kokkos::fence();

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
//      prev_repeat_offset = chkpt_size - chkpt_header.prev_repeat_size*(sizeof(uint32_t)+sizeof(NodeID));
//      curr_repeat_offset = prev_repeat_offset - chkpt_header.curr_repeat_size*(sizeof(uint32_t)+sizeof(uint32_t));
//      curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
//      prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, chkpt_size));
//      distinct = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
      curr_repeat_offset = sizeof(header_t) + chkpt_header.distinct_size*sizeof(uint32_t);
      prev_repeat_offset = curr_repeat_offset + chkpt_header.curr_repeat_size*2*sizeof(uint32_t);
      data_offset = prev_repeat_offset + chkpt_header.prev_repeat_size*(sizeof(uint32_t)+sizeof(NodeID));
      curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
      prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, data_offset));
      distinct = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
      data_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, chkpt_size));
      
      distinct_map.clear();
      distinct_map.rehash(chkpt_header.distinct_size);
      Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size);
      Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.distinct_size), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
//        memcpy(&node, distinct.data()+i*(sizeof(uint32_t)+chunk_size), sizeof(uint32_t));
//        distinct_map.insert(NodeID(node,cur_id), i*(sizeof(uint32_t)+chunk_size)+sizeof(uint32_t));
        memcpy(&node, distinct.data()+i*sizeof(uint32_t), sizeof(uint32_t));
        distinct_map.insert(NodeID(node,cur_id), i*chunk_size);
      });
      uint32_t num_repeat = chkpt_header.curr_repeat_size + chkpt_header.prev_repeat_size;
      Kokkos::parallel_for("Fill current repeat map", Kokkos::RangePolicy<>(0, chkpt_header.curr_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
        uint32_t prev;
        memcpy(&node, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
        memcpy(&prev, curr_repeat.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
        repeat_map.insert(node, NodeID(prev,cur_id));
      });
      Kokkos::parallel_for("Fill previous repeat map", Kokkos::RangePolicy<>(0, chkpt_header.prev_repeat_size), KOKKOS_LAMBDA(const uint32_t i) {
        uint32_t node;
        NodeID prev;
        memcpy(&node, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(NodeID)), sizeof(uint32_t));
        memcpy(&prev, prev_repeat.data()+i*(sizeof(uint32_t)+sizeof(NodeID))+sizeof(uint32_t), sizeof(NodeID));
        repeat_map.insert(node, prev);
      });
      Kokkos::parallel_for("Fill data", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
        if(node_list(i).tree == current_id) {
          NodeID id = node_list(i);
          if(distinct_map.exists(id)) {
            size_t offset = distinct_map.value_at(distinct_map.find(id));
            uint32_t writesize = chunk_size;
            if(id.node == num_chunks-1)
              writesize = datalen-id.node*chunk_size;
//            memcpy(data.data()+chunk_size*i, distinct.data()+offset, writesize);
            memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
          } else if(repeat_map.exists(id.node)) {
            NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
            if(prev.tree == current_id) {
              size_t offset = distinct_map.value_at(distinct_map.find(prev));
              uint32_t writesize = chunk_size;
              if(i == num_chunks-1)
                writesize = datalen-i*chunk_size;
//              memcpy(data.data()+chunk_size*i, distinct.data()+offset, writesize); 
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
//    prev_repeat_offset = chkpt_size - chkpt_header.prev_repeat_size*(2*sizeof(uint32_t));
//    curr_repeat_offset = prev_repeat_offset - chkpt_header.curr_repeat_size*(2*sizeof(uint32_t));
//    curr_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(curr_repeat_offset, prev_repeat_offset));
//    prev_repeat   = Kokkos::subview(chkpt_buffer_d, std::make_pair(prev_repeat_offset, chkpt_size));
//    distinct = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), curr_repeat_offset));
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
//      memcpy(&node, distinct.data()+i*(sizeof(uint32_t)+chunk_size), sizeof(uint32_t));
//      distinct_map.insert(NodeID(node, current_id), i*(sizeof(uint32_t)+chunk_size)+sizeof(uint32_t));
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
//          memcpy(data.data()+chunk_size*i, distinct.data()+offset, writesize);
          memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
        } else if(repeat_map.exists(id.node)) {
          NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
          if(prev.tree == current_id) {
            size_t offset = distinct_map.value_at(distinct_map.find(prev));
            uint32_t writesize = chunk_size;
            if(i == num_chunks-1)
              writesize = datalen-i*chunk_size;
//            memcpy(data.data()+chunk_size*i, distinct.data()+offset, writesize); 
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
  }
  Kokkos::fence();
  STDOUT_PRINT("Restarted checkpoint\n");
  return 0;
}

std::pair<uint64_t,uint64_t> 
write_incr_chkpt_hashlist_local( const std::string& filename, 
                           const Kokkos::View<uint8_t*>& data, 
                           Kokkos::View<uint8_t*>& buffer_d, 
                           uint32_t chunk_size, 
                           const DistinctNodeIDMap& distinct, 
                           const SharedNodeIDMap& shared,
                           uint32_t prior_chkpt_id,
                           uint32_t chkpt_id,
                           header_t& header) {
//  std::ofstream file;
//  file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
//  file.open(filename, std::ofstream::out | std::ofstream::binary);

  uint32_t num_chunks = data.size()/chunk_size;
  if(num_chunks*chunk_size < data.size()) {
    num_chunks += 1;
  }

  // Write whether we are storing the hashes, length full checkpoint, chunk size, number of repeat chunks, number of distinct chunks
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

  STDOUT_PRINT("Buffer size: %u\n", buffer_size);
  buffer_d = Kokkos::View<uint8_t*>("Buffer", buffer_size);
STDOUT_PRINT("Distinct entries: %u\n", distinct.size());
STDOUT_PRINT("Repeat entries: %u\n", shared.size());
  Kokkos::parallel_for("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      auto info = distinct.value_at(i);
      Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
      Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
//      size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + chunk_size);
      size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t));
      memcpy(buffer_d.data()+pos, &info.node, sizeof(uint32_t));
      uint32_t writesize = chunk_size;
      if(info.node == num_chunks-1) {
        writesize = data.size()-info.node*chunk_size;
      }
//      memcpy(buffer_d.data()+pos+sizeof(uint32_t), data.data()+chunk_size*(info.node), writesize);
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
  STDOUT_PRINT("Ref ID: %u\n"          , header.ref_id);
  STDOUT_PRINT("Chkpt ID: %u\n"        , header.chkpt_id);
  STDOUT_PRINT("Data len: %lu\n"       , header.datalen);
  STDOUT_PRINT("Chunk size: %u\n"      , header.chunk_size);
  STDOUT_PRINT("Window size: %u\n"     , header.window_size);
  STDOUT_PRINT("Distinct size: %u\n"   , header.distinct_size);
  STDOUT_PRINT("Curr repeat size: %u\n", header.curr_repeat_size);
  STDOUT_PRINT("prev repeat size: %u\n", header.prev_repeat_size);
  STDOUT_PRINT("Number of bytes written for incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_data_h(0) + num_bytes_metadata_h(0));
  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));

  DEBUG_PRINT("Trying to close file\n");
  DEBUG_PRINT("Closed file\n");
  return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
}

std::pair<uint64_t,uint64_t> 
write_incr_chkpt_hashlist_global( const std::string& filename, 
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
  if(prior_chkpt_id == chkpt_id) {
    DEBUG_PRINT("Correct branch\n");
//    uint64_t buffer_size = 0;
//    buffer_size += 2*sizeof(uint32_t)*shared.size();
//    buffer_size += distinct.size()*(sizeof(uint32_t) + chunk_size);
//    buffer_d = Kokkos::View<uint8_t*>("Buffer", buffer_size);
//    size_t data_offset = distinct.size()*sizeof(uint32_t)+2*shared.size()*sizeof(uint32_t);
//    DEBUG_PRINT("Number of distinct entries: %u\n", distinct.size());
//    Kokkos::parallel_for("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
//      if(distinct.valid_at(i)) {
//        auto info = distinct.value_at(i);
////        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + chunk_size);
//        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t));
//        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
//        Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
//        memcpy(buffer_d.data()+pos, &info, sizeof(NodeID));
//        uint32_t writesize = chunk_size;
//        if(info.node == num_chunks-1) {
//          writesize = data.size()-info.node*chunk_size;
//        }
////        memcpy(buffer_d.data()+pos+sizeof(uint32_t), data.data()+chunk_size*info.node, writesize);
//        memcpy(buffer_d.data()+data_offset+(pos/sizeof(uint32_t))*chunk_size, data.data()+chunk_size*info.node, writesize);
//      }
//    });
//    DEBUG_PRINT("Wrote distinct map\n");
//    DEBUG_PRINT("Shared capacity: %u\tShared size: %u\n", shared.capacity(), shared.size());
//    Kokkos::parallel_for("Count curr repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
//      if(shared.valid_at(i)) {
//        uint32_t node = shared.key_at(i);
//        NodeID prev = shared.value_at(i);
//        if(prev.tree == chkpt_id) {
//          Kokkos::atomic_add(&num_curr_repeat_d(0), 1);
//          Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
//          size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
//          memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
//          memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
//        }
//      }
//    });
//    Kokkos::parallel_for("Count prior repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
//      if(shared.valid_at(i)) {
//        uint32_t node = shared.key_at(i);
//        NodeID prev = shared.value_at(i);
//        if(prev.tree != chkpt_id) {
//          Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
//          size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
//          memcpy(buffer_d.data()+pos, &node, sizeof(uint32_t));
//          memcpy(buffer_d.data()+pos+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
//        }
//      }
//    });
  uint32_t buffer_size = 0;
  buffer_size += sizeof(uint32_t)*2*shared.size();
  buffer_size += distinct.size()*(sizeof(uint32_t) + chunk_size);
  size_t data_offset = distinct.size()*sizeof(uint32_t)+2*shared.size()*sizeof(uint32_t);

  STDOUT_PRINT("Buffer size: %u\n", buffer_size);
  buffer_d = Kokkos::View<uint8_t*>("Buffer", buffer_size);
STDOUT_PRINT("Distinct entries: %u\n", distinct.size());
STDOUT_PRINT("Repeat entries: %u\n", shared.size());
  Kokkos::parallel_for("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      auto info = distinct.value_at(i);
      Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
      Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
//      size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + chunk_size);
      size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t));
      memcpy(buffer_d.data()+pos, &info.node, sizeof(uint32_t));
      uint32_t writesize = chunk_size;
      if(info.node == num_chunks-1) {
        writesize = data.size()-info.node*chunk_size;
      }
//      memcpy(buffer_d.data()+pos+sizeof(uint32_t), data.data()+chunk_size*(info.node), writesize);
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
    DEBUG_PRINT("Wrote shared map\n");
    Kokkos::deep_copy(num_bytes_h, num_bytes_d);
    Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
    Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
    Kokkos::deep_copy(num_curr_repeat_h, num_curr_repeat_d);
    header.ref_id = prior_chkpt_id;
    header.chkpt_id = chkpt_id;
    header.datalen = data.size();
    header.chunk_size = chunk_size;
    header.window_size = 0;
    header.distinct_size = distinct.size();
    header.curr_repeat_size = num_curr_repeat_h(0);
    header.prev_repeat_size = shared.size() - num_curr_repeat_h(0);
    DEBUG_PRINT("Copied counters to host\n");
    STDOUT_PRINT("Number of bytes written for incremental checkpoint: %lu\n", 8*sizeof(uint32_t) + num_bytes_h(0));
    STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
    STDOUT_PRINT("Number of bytes written for metadata: %lu\n", 8*sizeof(uint32_t) + num_bytes_metadata_h(0));
  } else {
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
    Kokkos::parallel_for("Count curr repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
      if(shared.valid_at(i)) {
        uint32_t k = shared.key_at(i);
        NodeID v = shared.value_at(i);
        if(v.tree == chkpt_id) {
          Kokkos::atomic_add(&distinct_repeat_counter_d(0), 1);
        }
      }
    });
    Kokkos::deep_copy(distinct_counter_h, distinct_counter_d);
    Kokkos::deep_copy(distinct_repeat_counter_h, distinct_repeat_counter_d);
    size_t data_offset = distinct_counter_h(0)*sizeof(uint32_t)+distinct_repeat_counter_h(0)*2*sizeof(uint32_t) + (shared.size()-distinct_repeat_counter_h(0))*(sizeof(uint32_t)+sizeof(NodeID));
    uint32_t buffer_size = 0;
    buffer_size += distinct_counter_h(0)*(sizeof(uint32_t)+chunk_size);
    buffer_size += distinct_repeat_counter_h(0)*2*sizeof(uint32_t);
    buffer_size += (shared.size()-distinct_repeat_counter_h(0))*(sizeof(uint32_t)+sizeof(NodeID));
STDOUT_PRINT("Distinct size: %u\n", distinct_counter_h(0));
STDOUT_PRINT("Current Repeat size: %u\n", distinct_repeat_counter_h(0));
STDOUT_PRINT("Prior Repeat size: %u\n", shared.size()-distinct_repeat_counter_h(0));
    STDOUT_PRINT("Buffer size: %u\n", buffer_size);
    buffer_d = Kokkos::View<uint8_t*>("Buffer", buffer_size);
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
//          size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t) + chunk_size);
          size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t));
          memcpy(buffer_d.data()+pos, &info, sizeof(uint32_t));
          uint32_t writesize = chunk_size;
          if(info.node == num_chunks-1) {
            writesize = data.size()-info.node*chunk_size;
          }
//          memcpy(buffer_d.data()+pos+sizeof(uint32_t), data.data()+chunk_size*info.node, writesize);
          memcpy(buffer_d.data()+data_offset+(pos/sizeof(uint32_t))*chunk_size, data.data()+chunk_size*info.node, writesize);
        }
      }
    });
    Kokkos::parallel_for("Count curr repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
      if(shared.valid_at(i)) {
        uint32_t k = shared.key_at(i);
        NodeID v = shared.value_at(i);
        if(v.tree == chkpt_id) {
          Kokkos::atomic_add(&num_curr_repeat_d(0), 1);
          Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(uint32_t));
          uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
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
          Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(NodeID));
          uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(NodeID)+sizeof(uint32_t));
          memcpy(buffer_d.data()+pos, &k, sizeof(uint32_t));
          memcpy(buffer_d.data()+pos+sizeof(uint32_t), &v, sizeof(NodeID));
        }
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
printf("Dumping header: \n");
printf("Ref ID:           %u\n",  header.ref_id);
printf("Chkpt ID:         %u\n",  header.chkpt_id);
printf("Datalen:          %lu\n", header.datalen);
printf("Chunk size:       %u\n",  header.chunk_size);
printf("Window size:      %u\n",  header.window_size);
printf("Distinct size:    %u\n",  header.distinct_size);
printf("Curr repeat size: %u\n",  header.curr_repeat_size);
printf("Prev repeat size: %u\n",  header.prev_repeat_size);
    STDOUT_PRINT("Number of bytes written for incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_h(0));
    STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
    STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));
  }
  DEBUG_PRINT("Trying to close file\n");
  DEBUG_PRINT("Closed file\n");
  return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
}

#endif

