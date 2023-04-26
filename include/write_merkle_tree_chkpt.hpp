#ifndef WRITE_MERKLE_TREE_CHKPT_HPP
#define WRITE_MERKLE_TREE_CHKPT_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_ScatterView.hpp>
#include <climits>
#include "hash_functions.hpp"
#include "map_helpers.hpp"
//#include "kokkos_queue.hpp"
#include "kokkos_merkle_tree.hpp"
#include <iostream>
#include "utils.hpp"

//template<typename DataView>
//std::pair<uint64_t,uint64_t> 
//write_incr_chkpt_hashtree_local_mode(  
//                                const DataView& data, 
//                                Kokkos::View<uint8_t*>& buffer_d,
//                                uint32_t chunk_size, 
//                                const DistinctNodeIDMap& distinct, 
//                                const SharedNodeIDMap& shared,
//                                uint32_t prior_chkpt_id,
//                                uint32_t chkpt_id,
//                                header_t& header) {
//  uint32_t num_chunks = data.size()/chunk_size;
//  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(chunk_size) < data.size()) {
//    num_chunks += 1;
//  }
//  uint32_t num_nodes = 2*num_chunks-1;
//
//  uint32_t distinct_size = 0;
//  uint32_t repeat_size = 0;
//  Kokkos::RangePolicy<> distinct_policy(0, distinct.capacity());
//  Kokkos::RangePolicy<> shared_policy(0, shared.capacity());
//
//  // Calculate the number of bytes for first occurrences
//  Kokkos::parallel_reduce("Count distinct updates", distinct_policy, KOKKOS_LAMBDA(const uint32_t i, uint32_t& sum) {
//    if(distinct.valid_at(i)) {
//      auto info = distinct.value_at(i);
//      if(info.node >= num_chunks-1) {
//        sum += sizeof(uint32_t)+chunk_size;
//      }
//    }
//  }, distinct_size);
//
//  // Calculate the number of bytes for shifted duplicates
//  Kokkos::parallel_reduce("Count shared updates", shared_policy, KOKKOS_LAMBDA(const uint32_t i, uint32_t& sum) {
//    if(shared.valid_at(i)) {
//      auto node = shared.key_at(i);
//      if(node >= num_chunks-1) {
//        sum += sizeof(uint32_t)*2;
//      }
//    }
//  }, repeat_size);
//  Kokkos::fence();
//
//  buffer_d = Kokkos::View<uint8_t*>("Buffer", sizeof(header_t) + repeat_size + distinct_size);
//  Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
//  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
//  Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
//  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
//  Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
//  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
//  Kokkos::View<uint64_t[1]> num_distinct_d("Number of distinct entries");
//  Kokkos::View<uint64_t[1]>::HostMirror num_distinct_h = Kokkos::create_mirror_view(num_distinct_d);
//  Kokkos::deep_copy(num_distinct_d, 0);
//  Kokkos::View<uint64_t[1]> num_curr_repeat_d("Number of curr repeat entries");
//  Kokkos::View<uint64_t[1]>::HostMirror num_curr_repeat_h = Kokkos::create_mirror_view(num_curr_repeat_d);
//  Kokkos::deep_copy(num_curr_repeat_d, 0);
//  Kokkos::View<uint64_t[1]> num_prev_repeat_d("Number of prev repeat entries");
//  Kokkos::View<uint64_t[1]>::HostMirror num_prev_repeat_h = Kokkos::create_mirror_view(num_prev_repeat_d);
//  Kokkos::deep_copy(num_prev_repeat_d, 0);
//  Kokkos::deep_copy(num_bytes_d, 0);
//  Kokkos::deep_copy(num_bytes_data_d, 0);
//  Kokkos::deep_copy(num_bytes_metadata_d, 0);
//  STDOUT_PRINT("Setup counters and buffers\n");
//  STDOUT_PRINT("Distinct capacity: %u, size: %u\n", distinct.capacity(), distinct_size);
//  STDOUT_PRINT("Repeat capacity: %u, size: %u\n", shared.capacity(), repeat_size);
//
//  // Cacluate offset for where data chunks are written to
//  size_t data_offset = static_cast<uint64_t>(distinct_size/(sizeof(uint32_t)+chunk_size))*sizeof(uint32_t) + repeat_size;
//  STDOUT_PRINT("Data offset: %lu\n", data_offset);
//  Kokkos::parallel_for("Count distinct updates", distinct_policy, KOKKOS_LAMBDA(const uint32_t i) {
//    if(distinct.valid_at(i)) {
//      auto info = distinct.value_at(i);
//      if(info.node >= num_chunks-1) {
//        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
//        Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
//        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t));
//        memcpy(buffer_d.data()+sizeof(header_t)+pos, &info.node, sizeof(uint32_t));
//        uint32_t writesize = chunk_size;
//        uint64_t src_offset = static_cast<uint64_t>(info.node-num_chunks+1)*static_cast<uint64_t>(chunk_size);
//        uint64_t dst_offset = (pos/sizeof(uint32_t))*static_cast<uint64_t>(chunk_size);
//        if(info.node == num_nodes-1) {
//          writesize = data.size()-src_offset;
//        }
//        memcpy( buffer_d.data()+sizeof(header_t)+data_offset+dst_offset, 
//                data.data()+src_offset, 
//                writesize);
////        DEBUG_PRINT("Writing region %u at %lu with offset %lu\n", info.node, pos, data_offset+(pos/sizeof(uint32_t))*chunk_size);
//        Kokkos::atomic_add(&num_distinct_d(0), 1);
//      }
//    }
//  });
//
//  Kokkos::parallel_for("Count curr repeat updates", shared_policy, KOKKOS_LAMBDA(const uint32_t i) {
//    if(shared.valid_at(i)) {
//      uint32_t k = shared.key_at(i);
//      if(k >= num_chunks-1) {
//        NodeID v = shared.value_at(i);
//        if(v.tree == chkpt_id) {
//          Kokkos::atomic_add(&num_curr_repeat_d(0), 1);
//          Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
//          uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
//          memcpy(buffer_d.data()+sizeof(header_t)+pos, &k, sizeof(uint32_t));
//          memcpy(buffer_d.data()+sizeof(header_t)+pos+sizeof(uint32_t), &v.node, sizeof(uint32_t));
//        }
//      }
//    }
//  });
//  Kokkos::parallel_for("Count prior repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
//    if(shared.valid_at(i)) {
//      uint32_t k = shared.key_at(i);
//      if(k >= num_chunks-1) {
//        NodeID v = shared.value_at(i);
//        if(v.tree != chkpt_id) {
//          Kokkos::atomic_add(&num_prev_repeat_d(0), 1);
//          Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
//          uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
//          memcpy(buffer_d.data()+sizeof(header_t)+pos, &k, sizeof(uint32_t));
//          memcpy(buffer_d.data()+sizeof(header_t)+pos+sizeof(uint32_t), &v.node, sizeof(uint32_t));
//        }
//      }
//    }
//  });
//  Kokkos::fence();
//  Kokkos::deep_copy(num_distinct_h, num_distinct_d);
//  Kokkos::deep_copy(num_curr_repeat_h, num_curr_repeat_d);
//  Kokkos::deep_copy(num_prev_repeat_h, num_prev_repeat_d);
//  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
//  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
//  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
//  Kokkos::fence();
//  header.ref_id = prior_chkpt_id;
//  header.chkpt_id = chkpt_id;
//  header.datalen = data.size();
//  header.chunk_size = chunk_size;
//  header.num_first_ocur = num_distinct_h(0);
//  header.num_shift_dupl = num_curr_repeat_h(0)+num_prev_repeat_h(0);
//  header.num_prior_chkpts = 2;
//  DEBUG_PRINT("Ref ID: %u\n"          , header.ref_id);
//  DEBUG_PRINT("Chkpt ID: %u\n"        , header.chkpt_id);
//  DEBUG_PRINT("Data len: %lu\n"       , header.datalen);
//  DEBUG_PRINT("Chunk size: %u\n"      , header.chunk_size);
//  DEBUG_PRINT("Num first ocur: %u\n"  , header.num_first_ocur);
//  DEBUG_PRINT("Num shift dupl: %u\n"  , header.num_shift_dupl);
//  DEBUG_PRINT("Num prior chkpts: %u\n", header.num_prior_chkpts);
//  DEBUG_PRINT("Copied data to host\n");
//  STDOUT_PRINT("Number of bytes written for compact incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_h(0));
//  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
//  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));
////  return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
//  uint64_t size_metadata = buffer_d.size() - distinct_size;
//  return std::make_pair(distinct_size, size_metadata);
//}
//
//std::pair<uint64_t,uint64_t> 
//write_incr_chkpt_hashtree_local_mode(  
//                                const Kokkos::View<uint8_t*>& data, 
//                                Kokkos::View<uint8_t*>& buffer_d,
//                                uint32_t chunk_size, 
//                                const DistinctNodeIDMap& distinct, 
//                                const NodeMap& shared,
//                                uint32_t prior_chkpt_id,
//                                uint32_t chkpt_id,
//                                header_t& header) {
//  uint32_t num_chunks = data.size()/chunk_size;
//  if(num_chunks*chunk_size < data.size()) {
//    num_chunks += 1;
//  }
//  uint32_t num_nodes = 2*num_chunks-1;
//
//  // Write whether we are storing the hashes, length full checkpoint, chunk size, number of repeat chunks, number of distinct chunks
//  DEBUG_PRINT("Wrote header\n");
//  uint32_t distinct_size = 0;
//  Kokkos::parallel_reduce("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i, uint32_t& sum) {
//    if(distinct.valid_at(i)) {
//      auto info = distinct.value_at(i);
//      if(info.node >= num_chunks-1) {
//        sum += sizeof(uint32_t)+chunk_size;
//      }
//    }
//  }, distinct_size);
//  uint32_t repeat_size = 0;
//  Kokkos::parallel_reduce("Count shared updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i, uint32_t& sum) {
//    if(shared.valid_at(i) && (shared.value_at(i).nodetype == Repeat)) {
//      auto node = shared.key_at(i);
//      if(node >= num_chunks-1) {
//        sum += sizeof(uint32_t)*2;
//      }
//    }
//  }, repeat_size);
//  Kokkos::fence();
//  buffer_d = Kokkos::View<uint8_t*>("Buffer", sizeof(header_t) + repeat_size + distinct_size);
//  Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
//  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
//  Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
//  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
//  Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
//  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
//  Kokkos::View<uint64_t[1]> num_distinct_d("Number of distinct entries");
//  Kokkos::View<uint64_t[1]>::HostMirror num_distinct_h = Kokkos::create_mirror_view(num_distinct_d);
//  Kokkos::deep_copy(num_distinct_d, 0);
//  Kokkos::View<uint64_t[1]> num_curr_repeat_d("Number of curr repeat entries");
//  Kokkos::View<uint64_t[1]>::HostMirror num_curr_repeat_h = Kokkos::create_mirror_view(num_curr_repeat_d);
//  Kokkos::deep_copy(num_curr_repeat_d, 0);
//  Kokkos::View<uint64_t[1]> num_prev_repeat_d("Number of prev repeat entries");
//  Kokkos::View<uint64_t[1]>::HostMirror num_prev_repeat_h = Kokkos::create_mirror_view(num_prev_repeat_d);
//  Kokkos::deep_copy(num_prev_repeat_d, 0);
//  Kokkos::deep_copy(num_bytes_d, 0);
//  Kokkos::deep_copy(num_bytes_data_d, 0);
//  Kokkos::deep_copy(num_bytes_metadata_d, 0);
//  STDOUT_PRINT("Setup counters and buffers\n");
//  STDOUT_PRINT("Distinct capacity: %u, size: %u\n", distinct.capacity(), distinct_size);
//  STDOUT_PRINT("Repeat capacity: %u, size: %u\n", shared.capacity(), repeat_size);
//
//  size_t data_offset = (distinct_size/(sizeof(uint32_t)+chunk_size))*sizeof(uint32_t) + repeat_size;
//  STDOUT_PRINT("Data offset: %lu\n", data_offset);
//  Kokkos::parallel_for("Count distinct updates", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
//    if(distinct.valid_at(i)) {
//      auto info = distinct.value_at(i);
//      if(info.node >= num_chunks-1) {
//        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
//        Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
//        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t));
//        memcpy(buffer_d.data()+sizeof(header_t)+pos, &info.node, sizeof(uint32_t));
//        uint32_t writesize = chunk_size;
//        if(info.node == num_nodes-1) {
//          writesize = data.size()-(info.node-num_chunks+1)*chunk_size;
//        }
//        memcpy(buffer_d.data()+sizeof(header_t)+data_offset+(pos/sizeof(uint32_t))*chunk_size, data.data()+chunk_size*(info.node-num_chunks+1), writesize);
////        DEBUG_PRINT("Writing region %u at %lu with offset %lu\n", info.node, pos, data_offset+(pos/sizeof(uint32_t))*chunk_size);
//        Kokkos::atomic_add(&num_distinct_d(0), 1);
//      }
//    }
//  });
//Kokkos::View<uint32_t[1]> counter_d("Counter");
//Kokkos::deep_copy(counter_d, 0);
//auto counter_h = Kokkos::create_mirror_view(counter_d);
//  Kokkos::parallel_for("Count curr repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
//    if(shared.valid_at(i) && (shared.value_at(i).nodetype == Repeat)) {
//      uint32_t k = shared.key_at(i);
//      if(k >= num_chunks-1) {
//Kokkos::atomic_add(&counter_d(0), 1);
////        NodeID v = shared.value_at(i);
//        Node v = shared.value_at(i);
//        if(v.tree == chkpt_id) {
//          Kokkos::atomic_add(&num_curr_repeat_d(0), 1);
//          Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
//          uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
//          memcpy(buffer_d.data()+sizeof(header_t)+pos, &k, sizeof(uint32_t));
//          memcpy(buffer_d.data()+sizeof(header_t)+pos+sizeof(uint32_t), &v.node, sizeof(uint32_t));
////Kokkos::atomic_add(&counter_d(0), 1);
////printf("Writing current repeat chunk: %u:%u at %lu\n", k, v.node, pos);
//        }
//      }
//    }
//  });
//Kokkos::deep_copy(counter_h, counter_d);
//STDOUT_PRINT("Number of current repeats: %u\n", counter_h(0));
//  Kokkos::parallel_for("Count prior repeat updates", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
//    if(shared.valid_at(i) && (shared.value_at(i).nodetype == Repeat)) {
//      uint32_t k = shared.key_at(i);
//      if(k >= num_chunks-1) {
////        NodeID v = shared.value_at(i);
//        Node v = shared.value_at(i);
//        if(v.tree != chkpt_id) {
//          Kokkos::atomic_add(&num_prev_repeat_d(0), 1);
//          Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
//          uint64_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
//          memcpy(buffer_d.data()+sizeof(header_t)+pos, &k, sizeof(uint32_t));
//          memcpy(buffer_d.data()+sizeof(header_t)+pos+sizeof(uint32_t), &v.node, sizeof(uint32_t));
//        }
//      }
//    }
//  });
//  Kokkos::fence();
//  Kokkos::deep_copy(num_distinct_h, num_distinct_d);
//  Kokkos::deep_copy(num_curr_repeat_h, num_curr_repeat_d);
//  Kokkos::deep_copy(num_prev_repeat_h, num_prev_repeat_d);
//  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
//  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
//  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
//  Kokkos::fence();
//  header.ref_id = prior_chkpt_id;
//  header.chkpt_id = chkpt_id;
//  header.datalen = data.size();
//  header.chunk_size = chunk_size;
//  header.num_first_ocur = num_distinct_h(0);
//  header.num_shift_dupl = num_curr_repeat_h(0) + num_prev_repeat_h(0);
//  header.num_prior_chkpts = 2;
//  STDOUT_PRINT("Buffer size: %lu\n", buffer_d.size());
//  STDOUT_PRINT("Ref ID: %u\n"         , header.ref_id);
//  STDOUT_PRINT("Chkpt ID: %u\n"       , header.chkpt_id);
//  STDOUT_PRINT("Data len: %lu\n"      , header.datalen);
//  STDOUT_PRINT("Chunk size: %u\n"     , header.chunk_size);
//  STDOUT_PRINT("First ocur size: %u\n", header.num_first_ocur);
//  STDOUT_PRINT("Shift dupl size: %u\n", header.num_shift_dupl);
//  DEBUG_PRINT("Copied data to host\n");
//  STDOUT_PRINT("Number of bytes written for compact incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_h(0));
//  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
//  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));
////  return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
//  uint64_t size_metadata = buffer_d.size() - distinct_size;
//  return std::make_pair(distinct_size, size_metadata);
//}
//
//std::pair<uint64_t,uint64_t> 
//write_incr_chkpt_hashtree_local_mode(  
//                                const Kokkos::View<uint8_t*>& data, 
//                                Kokkos::View<uint8_t*>& buffer_d, 
//                                uint32_t chunk_size, 
//                                const CompactTable& distinct, 
//                                const CompactTable& shared,
//                                uint32_t prior_chkpt_id,
//                                uint32_t chkpt_id,
//                                header_t& header) {
//  
//  uint32_t num_chunks = data.size()/chunk_size;
//  if(num_chunks*chunk_size < data.size()) {
//    num_chunks += 1;
//  }
//  uint32_t num_nodes = 2*num_chunks-1;
//
//  Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
//  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
//  Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
//  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
//  Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
//  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
//  Kokkos::View<uint64_t[1]> num_curr_repeat_d("Number of curr repeat entries");
//  Kokkos::View<uint64_t[1]>::HostMirror num_curr_repeat_h = Kokkos::create_mirror_view(num_curr_repeat_d);
//  Kokkos::deep_copy(num_curr_repeat_d, 0);
//  Kokkos::deep_copy(num_bytes_d, 0);
//  Kokkos::deep_copy(num_bytes_data_d, 0);
//  Kokkos::deep_copy(num_bytes_metadata_d, 0);
//  DEBUG_PRINT("Setup Views\n");
//
//  DEBUG_PRINT("Wrote shared metadata\n");
//  Kokkos::View<uint32_t[1]> max_reg("Max region size");
//  Kokkos::View<uint32_t[1]>::HostMirror max_reg_h = Kokkos::create_mirror_view(max_reg);
//  max_reg_h(0) = 0;
//  Kokkos::deep_copy(max_reg, max_reg_h);
//  Kokkos::View<uint32_t*> region_nodes("Region Nodes", distinct.size());
//  Kokkos::View<uint32_t*> region_len("Region lengths", distinct.size());
//  Kokkos::View<uint32_t[1]> counter_d("Counter");
//  Kokkos::deep_copy(counter_d, 0);
//  Kokkos::parallel_for("Count distinct bytes", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
//    if(distinct.valid_at(i)) {
//      uint32_t node = distinct.key_at(i);
//      NodeID prev = distinct.value_at(i);
//      if(node == prev.node && chkpt_id == prev.tree) {
//        uint32_t size = num_leaf_descendents(node, num_nodes);
//        Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t) + size*chunk_size);
//        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
//        Kokkos::atomic_max(&max_reg(0), size);
//        
//        uint32_t idx = Kokkos::atomic_fetch_add(&counter_d(0), 1);
//        region_nodes(idx) = node;
//        region_len(idx) = size;
//      }
//    }
//  });
//
//  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
//  uint32_t first_ocur_size = num_bytes_h(0);
//  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
//  num_bytes_h(0) += shared.size()*(sizeof(uint32_t)+sizeof(uint32_t));
//  Kokkos::deep_copy(max_reg_h, max_reg);
//  size_t data_offset = num_bytes_metadata_h(0)+shared.size()*(2*sizeof(uint32_t));
//  STDOUT_PRINT("Offset for data: %lu\n", data_offset);
//  uint32_t num_distinct = num_bytes_metadata_h(0)/sizeof(uint32_t);
//  Kokkos::parallel_scan("Calc offsets", num_distinct, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
//    const uint32_t len = region_len(i);
//    if(is_final) region_len(i) = partial_sum;
//    partial_sum += len;
//  });
////  auto region_len_h = Kokkos::create_mirror_view(region_len);
////  Kokkos::deep_copy(region_len_h, region_len);
////  auto region_nodes_h = Kokkos::create_mirror_view(region_nodes);
////  Kokkos::deep_copy(region_nodes_h, region_nodes);
//
//  STDOUT_PRINT("Length of buffer: %lu\n", num_bytes_h(0));
//  buffer_d = Kokkos::View<uint8_t*>("Buffer", sizeof(header_t)+num_bytes_h(0));
//
//  Kokkos::deep_copy(num_bytes_d, 0);
//  Kokkos::deep_copy(num_bytes_metadata_d, 0);
//
//  DEBUG_PRINT("Largest region: %u\n", max_reg_h(0));
//  if(max_reg_h(0) < 2048) {
//    Kokkos::parallel_for("Write distinct bytes", Kokkos::RangePolicy<>(0, num_distinct), KOKKOS_LAMBDA(const uint32_t i) {
//      uint32_t node = region_nodes(i);
//      uint32_t size = num_leaf_descendents(node, num_nodes);
//      uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
//      Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
//      Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(size*chunk_size));
//      size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t));
//      memcpy(buffer_d.data()+sizeof(header_t)+i*sizeof(uint32_t), &node, sizeof(uint32_t));
////      DEBUG_PRINT("Writing region %u at %lu with offset %lu\n", node, pos, data_offset+region_len(i)*chunk_size);
//      uint32_t writesize = chunk_size*size;
//      if(start*chunk_size+writesize > data.size())
//        writesize = data.size()-start*chunk_size;
//      memcpy(buffer_d.data()+sizeof(header_t)+data_offset+chunk_size*region_len(i), data.data()+start*chunk_size, writesize);
//    });
//  } else {
//    DEBUG_PRINT("Using explicit copy\n");
//    Kokkos::parallel_for("Write distinct bytes", Kokkos::TeamPolicy<>(num_distinct, Kokkos::AUTO) , 
//                         KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
//      uint32_t i=team_member.league_rank();
//      uint32_t node = region_nodes(i);
//      uint32_t size = num_leaf_descendents(node, num_nodes);
//      uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
//      if(team_member.team_rank() == 0) {
//        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
//        Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(size*chunk_size));
//        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t));
//        memcpy(buffer_d.data()+sizeof(header_t)+i*sizeof(uint32_t), &node, sizeof(uint32_t));
//      }
//      team_member.team_barrier();
//      uint32_t writesize = chunk_size*size;
//      if(start*chunk_size+writesize > data.size())
//        writesize = data.size()-start*chunk_size;
//      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, writesize), [&] (const uint64_t& j) {
//        buffer_d(sizeof(header_t)+data_offset+chunk_size*region_len(i)+j) = data(start*chunk_size+j);
//      });
//    });
//  }
//  Kokkos::fence();
//  DEBUG_PRINT("Done writing distinct bytes\n");
//  STDOUT_PRINT("Start writing shared metadata (%u total entries)\n", shared.size());
//  Kokkos::parallel_for("Write curr repeat bytes", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
//    if(shared.valid_at(i)) {
//      uint32_t node = shared.key_at(i);
//      NodeID prev = shared.value_at(i);
//      if(prev.tree == chkpt_id) {
//        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(uint32_t));
//        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
//        DEBUG_PRINT("Trying to write 8 bytes starting at %lu. Max size: %lu\n", pos, buffer_d.size());
//        memcpy(buffer_d.data()+sizeof(header_t)+pos, &node, sizeof(uint32_t));
//        memcpy(buffer_d.data()+sizeof(header_t)+pos+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
//        DEBUG_PRINT("Write current repeat: %u: (%u,%u)\n", node, prev.node, prev.tree);
//        Kokkos::atomic_add(&num_curr_repeat_d(0), 1);
//      }
//    }
//  });
//  Kokkos::fence();
//  DEBUG_PRINT("Done writing current repeat bytes\n");
//  Kokkos::parallel_for("Write prior repeat bytes", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
//    if(shared.valid_at(i)) {
//      uint32_t node = shared.key_at(i);
//      NodeID prev = shared.value_at(i);
//      if(prev.tree != chkpt_id) {
//        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(uint32_t));
//        size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
//        memcpy(buffer_d.data()+sizeof(header_t)+pos, &node, sizeof(uint32_t));
//        memcpy(buffer_d.data()+sizeof(header_t)+pos+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
//      }
//    }
//  });
//  Kokkos::fence();
//  DEBUG_PRINT("Done writing previous repeat bytes\n");
//  DEBUG_PRINT("Wrote shared metadata\n");
//  Kokkos::fence();
//  DEBUG_PRINT("Finished collecting data\n");
//  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
//  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
//  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
//  Kokkos::deep_copy(num_curr_repeat_h, num_curr_repeat_d);
//  Kokkos::fence();
//  header.ref_id = prior_chkpt_id;
//  header.chkpt_id = chkpt_id;
//  header.datalen = data.size();
//  header.chunk_size = chunk_size;
//  header.num_first_ocur = distinct.size();
//  header.num_shift_dupl = shared.size();
//  header.num_prior_chkpts = 2;
//  STDOUT_PRINT("Ref ID: %u\n"         , header.ref_id);
//  STDOUT_PRINT("Chkpt ID: %u\n"       , header.chkpt_id);
//  STDOUT_PRINT("Data len: %lu\n"      , header.datalen);
//  STDOUT_PRINT("Chunk size: %u\n"     , header.chunk_size);
//  STDOUT_PRINT("First ocur size: %u\n", header.num_first_ocur);
//  STDOUT_PRINT("Shift dupl size: %u\n", header.num_shift_dupl);
//  STDOUT_PRINT("Number of bytes written for compact incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_data_h(0) + num_bytes_metadata_h(0));
//  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
//  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));
//  DEBUG_PRINT("Closed file\n");
////  return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
//  uint64_t size_metadata = buffer_d.size() - first_ocur_size;
//  return std::make_pair(first_ocur_size, size_metadata);
//}

template<typename DataView>
std::pair<uint64_t,uint64_t> 
write_incr_chkpt_hashtree_global_mode( 
                                const DataView& data, 
                                Kokkos::View<uint8_t*>& buffer_d, 
                                uint32_t chunk_size, 
                                const CompactTable& distinct, 
                                const CompactTable& shared,
                                uint32_t prior_chkpt_id,
                                uint32_t chkpt_id,
                                header_t& header) {
  uint32_t num_chunks = data.size()/chunk_size;
  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(chunk_size) < data.size()) {
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

  Kokkos::View<uint32_t[1]> max_reg("Max region size");
  Kokkos::View<uint32_t[1]>::HostMirror max_reg_h = Kokkos::create_mirror_view(max_reg);
  Kokkos::deep_copy(max_reg, 0);
  Kokkos::View<uint32_t*> region_leaves("Region leaves", num_chunks);
  Kokkos::View<uint32_t*> region_nodes("Region Nodes", distinct.size());
  Kokkos::View<uint32_t*> region_len("Region lengths", distinct.size());
  Kokkos::View<uint32_t[1]> counter_d("Counter");
  Kokkos::View<uint32_t[1]>::HostMirror counter_h = Kokkos::create_mirror_view(counter_d);
  Kokkos::deep_copy(counter_d, 0);
  Kokkos::View<uint32_t[1]> chunk_counter_d("Counter");
  Kokkos::View<uint32_t[1]>::HostMirror chunk_counter_h = Kokkos::create_mirror_view(chunk_counter_d);
  Kokkos::deep_copy(chunk_counter_d, 0);
  Kokkos::View<uint64_t*> prior_counter_d("Counter for prior repeats", chkpt_id+1);
  Kokkos::View<uint64_t*>::HostMirror prior_counter_h = Kokkos::create_mirror_view(prior_counter_d);
  Kokkos::deep_copy(prior_counter_d, 0);
  Kokkos::Experimental::ScatterView<uint64_t*> prior_counter_sv(prior_counter_d);

//  Kokkos::fence();
  DEBUG_PRINT("Setup counters\n");

  // Filter and count space used for distinct entries
  // Calculate number of chunks each entry maps to
  Kokkos::parallel_for("Count distinct bytes", Kokkos::RangePolicy<>(0, distinct.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
    if(distinct.valid_at(i)) {
      uint32_t node = distinct.key_at(i);
      NodeID prev = distinct.value_at(i);
      if(node == prev.node && chkpt_id == prev.tree) {
        uint32_t size = num_leaf_descendents(node, num_nodes);
        uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
        Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t) + static_cast<uint64_t>(size)*static_cast<uint64_t>(chunk_size));
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
        Kokkos::atomic_max(&max_reg(0), size);
        uint32_t idx = Kokkos::atomic_fetch_add(&counter_d(0), 1);
        uint32_t chunk_idx = Kokkos::atomic_fetch_add(&chunk_counter_d(0), size);
//        for(uint32_t j=0; j<size; j++) {
//          region_leaves(chunk_idx+j) = start+j;
//        }
        region_nodes(idx) = node;
        region_len(idx) = size;
      } else {
        printf("Distinct node with different node/tree. Shouldn't happen.\n");
      }
    }
  });
//  Kokkos::fence();

  DEBUG_PRINT("Count distinct bytes\n");

  // Small bitset to record which checkpoints are necessary for restart
  Kokkos::Bitset<Kokkos::DefaultExecutionSpace> chkpts_needed(chkpt_id+1);
  chkpts_needed.reset();
  
//  Kokkos::fence();
  DEBUG_PRINT("Setup chkpt bitset\n");

  // Calculate space needed for repeat entries and number of entries per checkpoint
  Kokkos::RangePolicy<> shared_range_policy(0, shared.capacity());
  Kokkos::parallel_for("Count repeat bytes", shared_range_policy, KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      uint32_t node = shared.key_at(i);
      NodeID prev = shared.value_at(i);
      auto prior_counter_sa = prior_counter_sv.access();
      if(prev.tree == chkpt_id) {
        Kokkos::atomic_add(&num_curr_repeat_d(0), 1);
      }
      Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
      Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
      prior_counter_sa(prev.tree) += 1;
      chkpts_needed.set(prev.tree);
    }
  });
//  Kokkos::fence();
  DEBUG_PRINT("Count repeat bytes\n");
  Kokkos::Experimental::contribute(prior_counter_d, prior_counter_sv);
  prior_counter_sv.reset_except(prior_counter_d);

//  Kokkos::fence();
  DEBUG_PRINT("Collect prior counter\n");

  uint32_t num_prior_chkpts = chkpts_needed.count();

//  Kokkos::fence();
  DEBUG_PRINT("Number of checkpoints needed: %u\n", num_prior_chkpts);

  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
  Kokkos::deep_copy(max_reg_h, max_reg);
  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
  // Calculate offset for where chunks are written to in the buffer
  size_t data_offset = num_bytes_metadata_h(0) + static_cast<uint64_t>(num_prior_chkpts)*2*sizeof(uint32_t);
  DEBUG_PRINT("Offset for data: %lu\n", data_offset);
  Kokkos::deep_copy(counter_h, counter_d);
  uint32_t num_distinct = counter_h(0);
  STDOUT_PRINT("Number of distinct regions: %u\n", num_distinct);
  STDOUT_PRINT("Number of distinct chunks: %lu\n", num_bytes_h(0)/chunk_size);
  // Dividers for distinct chunks. Number of chunks per region varies.
  // Need offsets for each region so that writes can be done in parallel
  Kokkos::parallel_scan("Calc offsets", num_distinct, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
    const uint32_t len = region_len(i);
    if(is_final) region_len(i) = partial_sum;
    partial_sum += len;
  });

  Kokkos::parallel_for(num_distinct, KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t offset = region_len(i);
    uint32_t node = region_nodes(i);
    uint32_t size = num_leaf_descendents(node, num_nodes);
    uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
    Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size)*static_cast<uint64_t>(size));
//printf("%u: Writing %u chunks starting at %u to %u\n", i, size, start, offset);
    for(uint32_t j=0; j<size; j++) {
      region_leaves(offset+j) = start+j;
    }
  });

  DEBUG_PRINT("Length of buffer: %lu\n", num_bytes_h(0)+2*sizeof(uint32_t)*num_prior_chkpts);
  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
  uint32_t first_ocur_size = num_bytes_data_h(0);
//  uint64_t buffer_len = sizeof(header_t)+num_bytes_h(0)+2*sizeof(uint32_t)*chkpts_needed.count()+sizeof(uint32_t);
  uint64_t buffer_len = sizeof(header_t)+num_bytes_h(0)+2*sizeof(uint32_t)*static_cast<uint64_t>(chkpts_needed.count());
  buffer_d = Kokkos::View<uint8_t*>("Buffer", buffer_len);

  Kokkos::deep_copy(num_bytes_d, 0);
  Kokkos::deep_copy(num_bytes_metadata_d, 0);

  Kokkos::deep_copy(chunk_counter_h, chunk_counter_d);

  Kokkos::parallel_for("Copy first occur metadata", Kokkos::RangePolicy<>(0,num_distinct), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t node = region_nodes(i);
    Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
    Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t));
    memcpy(buffer_d.data()+sizeof(header_t)+static_cast<uint64_t>(i)*sizeof(uint32_t), &node, sizeof(uint32_t));
  });

  Kokkos::parallel_for("Copy data", Kokkos::TeamPolicy<>(chunk_counter_h(0), Kokkos::AUTO), 
                         KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
    uint32_t i = team_member.league_rank();
    uint32_t chunk = region_leaves(i);
    uint32_t writesize = chunk_size;
    uint64_t dst_offset = sizeof(header_t)+data_offset+static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);
    uint64_t src_offset = static_cast<uint64_t>(chunk)*static_cast<uint64_t>(chunk_size);
    if(chunk == num_chunks-1) {
      writesize = data.size()-src_offset;
    }
    if(team_member.team_rank() == 0) {
      Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
    }
//    memcpy(buffer_d.data()+sizeof(header_t)+data_offset+i*chunk_size, data.data()+chunk_size*chunk, writesize);
    uint32_t* buffer_u32 = (uint32_t*)(buffer_d.data()+dst_offset);
    uint32_t* data_u32 = (uint32_t*)(data.data()+src_offset);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, writesize/4), [&] (const uint64_t& j) {
      buffer_u32[j] = data_u32[j];
    });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, writesize%4), [&] (const uint64_t& j) {
      buffer_d(dst_offset+((writesize/4)*4)+j) = data(src_offset+((writesize/4)*4)+j);
    });
  });

  uint32_t num_prior = chkpts_needed.count();

  // Write Repeat map for recording how many entries per checkpoint
  // (Checkpoint ID, # of entries)
  Kokkos::parallel_for("Write size map", prior_counter_d.size(), KOKKOS_LAMBDA(const uint32_t i) {
    if(prior_counter_d(i) > 0) {
      uint32_t num_repeats_i = static_cast<uint32_t>(prior_counter_d(i));
      Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
      size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
      memcpy(buffer_d.data()+sizeof(header_t)+pos, &i, sizeof(uint32_t));
      memcpy(buffer_d.data()+sizeof(header_t)+pos+sizeof(uint32_t), &num_repeats_i, sizeof(uint32_t));
      DEBUG_PRINT("Wrote table entry (%u,%u) at offset %lu\n", i, num_repeats_i, pos);
    }
  });

  // Calculate repeat indices so that we can separate entries by source ID
  Kokkos::parallel_scan("Calc repeat end indices", prior_counter_d.size(), KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
    partial_sum += prior_counter_d(i);
    if(is_final) prior_counter_d(i) = partial_sum;
  });

  size_t prior_start = static_cast<uint64_t>(num_distinct)*sizeof(uint32_t)+static_cast<uint64_t>(num_prior)*2*sizeof(uint32_t);
  DEBUG_PRINT("Prior start offset: %lu\n", prior_start);

  Vector<uint32_t> shift_dupl_vec(shared.size());
  Kokkos::parallel_for(shared.capacity(), KOKKOS_LAMBDA(const uint32_t i) {
    if(shared.valid_at(i)) {
      shift_dupl_vec.push(shared.key_at(i));
    }
  });
  Kokkos::View<uint32_t*> chkpt_id_keys("Source checkpoint IDs", shift_dupl_vec.size());
  Kokkos::parallel_for(shift_dupl_vec.size(), KOKKOS_LAMBDA(const uint32_t i) {
    NodeID info = shared.value_at(shared.find(shift_dupl_vec(i)));
    chkpt_id_keys(i) = info.tree;
  });
  auto keys = chkpt_id_keys;
  using key_type = decltype(keys);
  using Comparator = Kokkos::BinOp1D<key_type>;
  Comparator comp(chkpt_id, 0, chkpt_id);
  Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, shift_dupl_vec.size(), comp, 0);
  bin_sort.create_permute_vector();
  bin_sort.sort(shift_dupl_vec.vector_d);
  bin_sort.sort(chkpt_id_keys);

  // Write repeat entries
  Kokkos::parallel_for("Write repeat bytes", Kokkos::RangePolicy<>(0, shift_dupl_vec.size()), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t node = shift_dupl_vec(i);
    NodeID prev = shared.value_at(shared.find(shift_dupl_vec(i)));
    memcpy(buffer_d.data()+sizeof(header_t)+prior_start+static_cast<uint64_t>(i)*2*sizeof(uint32_t), &node, sizeof(uint32_t));
    memcpy(buffer_d.data()+sizeof(header_t)+prior_start+static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
  });

//  // Write repeat entries
//  Kokkos::parallel_for("Write prior repeat bytes", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
//    if(shared.valid_at(i)) {
//      uint32_t node = shared.key_at(i);
//      NodeID prev = shared.value_at(i);
//      Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(uint32_t));
//      Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
//      size_t pos = Kokkos::atomic_sub_fetch(&prior_counter_d(prev.tree), 1);
//      memcpy(buffer_d.data()+sizeof(header_t)+prior_start+pos*2*sizeof(uint32_t), &node, sizeof(uint32_t));
//      memcpy(buffer_d.data()+sizeof(header_t)+prior_start+pos*2*sizeof(uint32_t)+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
//    }
//  });

  DEBUG_PRINT("Wrote shared metadata\n");
//  Kokkos::fence();
  DEBUG_PRINT("Finished collecting data\n");
  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
  Kokkos::deep_copy(num_curr_repeat_h, num_curr_repeat_d);
//  Kokkos::fence();
  header.ref_id = prior_chkpt_id;
  header.chkpt_id = chkpt_id;
  header.datalen = data.size();
  header.chunk_size = chunk_size;
  header.num_first_ocur = distinct.size();
  header.num_shift_dupl = shared.size();
  header.num_prior_chkpts = chkpts_needed.count();
  STDOUT_PRINT("Ref ID: %u\n"          , header.ref_id);
  STDOUT_PRINT("Chkpt ID: %u\n"        , header.chkpt_id);
  STDOUT_PRINT("Data len: %lu\n"       , header.datalen);
  STDOUT_PRINT("Chunk size: %u\n"      , header.chunk_size);
  STDOUT_PRINT("Num first ocur: %u\n"  , header.num_first_ocur);
  STDOUT_PRINT("Num shift dupl: %u\n"  , header.num_shift_dupl);
  STDOUT_PRINT("Num prior chkpts: %u\n", header.num_prior_chkpts);
  STDOUT_PRINT("Number of bytes written for compact incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_data_h(0) + num_bytes_metadata_h(0));
  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));
  DEBUG_PRINT("Closed file\n");
//  return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
  uint64_t size_metadata = buffer_d.size() - first_ocur_size;
  return std::make_pair(first_ocur_size, size_metadata);
}

template<typename DataView>
std::pair<uint64_t,uint64_t> 
write_incr_chkpt_hashtree_global_mode( 
                                const DataView& data, 
                                Kokkos::View<uint8_t*>& buffer_d, 
                                uint32_t chunk_size, 
                                MerkleTree& curr_tree, 
                                DistinctNodeIDMap& first_occur_d, 
                                const Vector<uint32_t>& first_ocur_vec, 
                                const Vector<uint32_t>& shift_dupl_vec,
                                uint32_t prior_chkpt_id,
                                uint32_t chkpt_id,
                                header_t& header) {
  std::string setup_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Setup");
  Kokkos::Profiling::pushRegion(setup_label);

  uint32_t num_chunks = data.size()/chunk_size;
  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(chunk_size) < data.size()) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;

//  Kokkos::View<uint64_t[1]> num_bytes_d("Number of bytes written");
//  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_h = Kokkos::create_mirror_view(num_bytes_d);
//  Kokkos::View<uint64_t[1]> num_bytes_data_d("Number of bytes written for checkpoint data");
//  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_data_h = Kokkos::create_mirror_view(num_bytes_data_d);
//  Kokkos::View<uint64_t[1]> num_bytes_metadata_d("Number of bytes written for checkpoint metadata");
//  Kokkos::View<uint64_t[1]>::HostMirror num_bytes_metadata_h = Kokkos::create_mirror_view(num_bytes_metadata_d);
//  Kokkos::deep_copy(num_bytes_d, 0);
//  Kokkos::deep_copy(num_bytes_data_d, 0);
//  Kokkos::deep_copy(num_bytes_metadata_d, 0);

//  Kokkos::View<uint32_t[1]> max_reg("Max region size");
//  Kokkos::View<uint32_t[1]>::HostMirror max_reg_h = Kokkos::create_mirror_view(max_reg);
//  Kokkos::deep_copy(max_reg, 0);
  Kokkos::View<uint32_t*> region_leaves("Region leaves", num_chunks);
  Kokkos::View<uint32_t*> region_nodes("Region Nodes", first_ocur_vec.size());
  Kokkos::View<uint32_t*> region_len("Region lengths", first_ocur_vec.size());
  Kokkos::View<uint32_t[1]> counter_d("Counter");
  Kokkos::View<uint32_t[1]>::HostMirror counter_h = Kokkos::create_mirror_view(counter_d);
  Kokkos::deep_copy(counter_d, 0);
  Kokkos::View<uint32_t[1]> chunk_counter_d("Counter");
  Kokkos::View<uint32_t[1]>::HostMirror chunk_counter_h = Kokkos::create_mirror_view(chunk_counter_d);
  Kokkos::deep_copy(chunk_counter_d, 0);
  Kokkos::View<uint64_t*> prior_counter_d("Counter for prior repeats", chkpt_id+1);
  Kokkos::View<uint64_t*>::HostMirror prior_counter_h = Kokkos::create_mirror_view(prior_counter_d);
  Kokkos::deep_copy(prior_counter_d, 0);
  Kokkos::Experimental::ScatterView<uint64_t*> prior_counter_sv(prior_counter_d);

//  Kokkos::fence();
  DEBUG_PRINT("Setup counters\n");

  Kokkos::Profiling::popRegion();

  // Filter and count space used for distinct entries
  // Calculate number of chunks each entry maps to
  std::string count_first_ocur_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Count first ocur bytes");
  Kokkos::parallel_for(count_first_ocur_label, Kokkos::RangePolicy<>(0, first_ocur_vec.size()), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node = first_ocur_vec(i);
      NodeID prev = first_occur_d.value_at(first_occur_d.find(curr_tree(node)));
      if(node == prev.node && chkpt_id == prev.tree) {
        uint32_t size = num_leaf_descendents(node, num_nodes);
        uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
//        Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t) + static_cast<uint64_t>(size)*static_cast<uint64_t>(chunk_size));
//        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
//        Kokkos::atomic_max(&max_reg(0), size);
        uint32_t idx = Kokkos::atomic_fetch_add(&counter_d(0), 1);
        Kokkos::atomic_add(&chunk_counter_d(0), size);
        region_nodes(idx) = node;
        region_len(idx) = size;
      } else {
        printf("Distinct node with different node/tree. Shouldn't happen.\n");
      }
  });
//  Kokkos::fence();
  std::string alloc_bitset_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Allocate bitset");
  Kokkos::Profiling::pushRegion(alloc_bitset_label);

  DEBUG_PRINT("Count distinct bytes\n");

  // Small bitset to record which checkpoints are necessary for restart
  Kokkos::Bitset<Kokkos::DefaultExecutionSpace> chkpts_needed(chkpt_id+1);
  chkpts_needed.reset();
  
//  Kokkos::fence();
  DEBUG_PRINT("Setup chkpt bitset\n");
  Kokkos::Profiling::popRegion();

  // Calculate space needed for repeat entries and number of entries per checkpoint
  Kokkos::RangePolicy<> shared_range_policy(0, shift_dupl_vec.size());
  std::string count_shift_dupl_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Count shift dupl bytes");
  Kokkos::parallel_for(count_shift_dupl_label, shared_range_policy, KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node = shift_dupl_vec(i);
      NodeID prev = first_occur_d.value_at(first_occur_d.find(curr_tree(node)));
      auto prior_counter_sa = prior_counter_sv.access();
//      Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
//      Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
      prior_counter_sa(prev.tree) += 1;
      chkpts_needed.set(prev.tree);
  });
//  Kokkos::fence();
  std::string contrib_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Contribute shift dupl");
  Kokkos::Profiling::pushRegion(contrib_label);
  DEBUG_PRINT("Count repeat bytes\n");
  Kokkos::Experimental::contribute(prior_counter_d, prior_counter_sv);
  prior_counter_sv.reset_except(prior_counter_d);

//  Kokkos::fence();
  DEBUG_PRINT("Collect prior counter\n");

  uint32_t num_prior_chkpts = chkpts_needed.count();

//  Kokkos::fence();
  DEBUG_PRINT("Number of checkpoints needed: %u\n", num_prior_chkpts);

//  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
//  Kokkos::deep_copy(max_reg_h, max_reg);
//  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
  // Calculate offset for where chunks are written to in the buffer
//  size_t data_offset = num_bytes_metadata_h(0) + static_cast<uint64_t>(num_prior_chkpts)*2*sizeof(uint32_t);
  size_t data_offset = first_ocur_vec.size()*sizeof(uint32_t) + shift_dupl_vec.size()*2*sizeof(uint32_t) + static_cast<uint64_t>(num_prior_chkpts)*2*sizeof(uint32_t);
  DEBUG_PRINT("Offset for data: %lu\n", data_offset);
  Kokkos::deep_copy(counter_h, counter_d);
  uint32_t num_distinct = counter_h(0);
  STDOUT_PRINT("Number of distinct regions: %u\n", num_distinct);
  Kokkos::Profiling::popRegion();
//  STDOUT_PRINT("Number of distinct chunks: %lu\n", num_bytes_h(0)/chunk_size);
  // Dividers for distinct chunks. Number of chunks per region varies.
  // Need offsets for each region so that writes can be done in parallel
  std::string calc_offsets_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Calculate offsets");
  Kokkos::parallel_scan(calc_offsets_label, num_distinct, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
    const uint32_t len = region_len(i);
    if(is_final) region_len(i) = partial_sum;
    partial_sum += len;
  });

  std::string find_region_leaves_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Find region leaves");
  Kokkos::parallel_for(find_region_leaves_label, Kokkos::RangePolicy<>(0,num_distinct), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t offset = region_len(i);
    uint32_t node = region_nodes(i);
    uint32_t size = num_leaf_descendents(node, num_nodes);
    uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
//    Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size)*static_cast<uint64_t>(size));
//printf("%u: Writing %u chunks starting at %u to %u\n", i, size, start, offset);
    for(uint32_t j=0; j<size; j++) {
      region_leaves(offset+j) = start+j;
    }
  });

  std::string alloc_buffer_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Allocate buffer");
  Kokkos::Profiling::pushRegion(alloc_buffer_label);
//  DEBUG_PRINT("Length of buffer: %lu\n", num_bytes_h(0)+2*sizeof(uint32_t)*num_prior_chkpts);
//  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
//  uint32_t first_ocur_size = num_bytes_data_h(0);
  Kokkos::deep_copy(chunk_counter_h, chunk_counter_d);
//  uint64_t buffer_len = sizeof(header_t)+num_bytes_h(0)+2*sizeof(uint32_t)*static_cast<uint64_t>(chkpts_needed.count());
  uint64_t buffer_len = sizeof(header_t)+first_ocur_vec.size()*sizeof(uint32_t)+2*sizeof(uint32_t)*static_cast<uint64_t>(chkpts_needed.count())+shift_dupl_vec.size()*2*sizeof(uint32_t)+chunk_counter_h(0)*static_cast<uint64_t>(chunk_size);
//  buffer_d = Kokkos::View<uint8_t*>("Buffer", buffer_len);
  Kokkos::resize(buffer_d, buffer_len);

  Kokkos::deep_copy(counter_d, sizeof(uint32_t)*num_distinct);
//  Kokkos::deep_copy(num_bytes_d, sizeof(uint32_t)*num_distinct);
//  Kokkos::deep_copy(num_bytes_metadata_d, 0);


  Kokkos::Profiling::popRegion();

  std::string copy_fo_metadata_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Copy first ocur metadata");
  Kokkos::parallel_for(copy_fo_metadata_label, Kokkos::RangePolicy<>(0,num_distinct), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t node = region_nodes(i);
//    Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
//    Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t));
    memcpy(buffer_d.data()+sizeof(header_t)+static_cast<uint64_t>(i)*sizeof(uint32_t), &node, sizeof(uint32_t));
  });

  std::string copy_data_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Copy data");
  Kokkos::parallel_for(copy_data_label, Kokkos::TeamPolicy<>(chunk_counter_h(0), Kokkos::AUTO), 
                         KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
    uint32_t i = team_member.league_rank();
    uint32_t chunk = region_leaves(i);
    uint32_t writesize = chunk_size;
    uint64_t dst_offset = sizeof(header_t)+data_offset+static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);
    uint64_t src_offset = static_cast<uint64_t>(chunk)*static_cast<uint64_t>(chunk_size);
    if(chunk == num_chunks-1) {
      writesize = data.size()-src_offset;
    }
//    if(team_member.team_rank() == 0) {
//      Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
//    }
//    memcpy(buffer_d.data()+sizeof(header_t)+data_offset+i*chunk_size, data.data()+chunk_size*chunk, writesize);
    uint32_t* buffer_u32 = (uint32_t*)(buffer_d.data()+dst_offset);
    uint32_t* data_u32 = (uint32_t*)(data.data()+src_offset);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, writesize/4), [&] (const uint64_t& j) {
      buffer_u32[j] = data_u32[j];
    });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, writesize%4), [&] (const uint64_t& j) {
      buffer_d(dst_offset+((writesize/4)*4)+j) = data(src_offset+((writesize/4)*4)+j);
    });
  });

  uint32_t num_prior = chkpts_needed.count();

  // Write Repeat map for recording how many entries per checkpoint
  // (Checkpoint ID, # of entries)
  std::string write_repeat_count_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Write repeat count");
  Kokkos::parallel_for(write_repeat_count_label, prior_counter_d.size(), KOKKOS_LAMBDA(const uint32_t i) {
    if(prior_counter_d(i) > 0) {
      uint32_t num_repeats_i = static_cast<uint32_t>(prior_counter_d(i));
//      Kokkos::atomic_add(&num_bytes_metadata_d(0), 2*sizeof(uint32_t));
//      size_t pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), 2*sizeof(uint32_t));
      size_t pos = Kokkos::atomic_fetch_add(&counter_d(0), 2*sizeof(uint32_t));
      memcpy(buffer_d.data()+sizeof(header_t)+pos, &i, sizeof(uint32_t));
      memcpy(buffer_d.data()+sizeof(header_t)+pos+sizeof(uint32_t), &num_repeats_i, sizeof(uint32_t));
      DEBUG_PRINT("Wrote table entry (%u,%u) at offset %lu\n", i, num_repeats_i, pos);
    }
  });

  // Calculate repeat indices so that we can separate entries by source ID
//  std::string calc_repeat_end_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Calculate repeat end indices");
//  Kokkos::parallel_scan(calc_repeat_end_label, prior_counter_d.size(), KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
//    partial_sum += prior_counter_d(i);
//    if(is_final) prior_counter_d(i) = partial_sum;
//  });

  size_t prior_start = static_cast<uint64_t>(num_distinct)*sizeof(uint32_t)+static_cast<uint64_t>(num_prior)*2*sizeof(uint32_t);
  DEBUG_PRINT("Prior start offset: %lu\n", prior_start);

//  Vector<uint32_t> shift_dupl_vec(shared.size());
//  Kokkos::parallel_for(shared.capacity(), KOKKOS_LAMBDA(const uint32_t i) {
//    if(shared.valid_at(i)) {
//      shift_dupl_vec.push(shared.key_at(i));
//    }
//  });
  Kokkos::View<uint32_t*> chkpt_id_keys("Source checkpoint IDs", shift_dupl_vec.size());
  std::string create_keys_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Create chkpt_id_keys");
  Kokkos::parallel_for(create_keys_label, Kokkos::RangePolicy<>(0,shift_dupl_vec.size()), KOKKOS_LAMBDA(const uint32_t i) {
    NodeID info = first_occur_d.value_at(first_occur_d.find(curr_tree(shift_dupl_vec(i))));
    chkpt_id_keys(i) = info.tree;
  });

  std::string sort_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Sort");
  Kokkos::Profiling::pushRegion(sort_label);
  auto keys = chkpt_id_keys;
  using key_type = decltype(keys);
  using Comparator = Kokkos::BinOp1D<key_type>;
  Comparator comp(chkpt_id, 0, chkpt_id);
  Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, shift_dupl_vec.size(), comp, 0);
  bin_sort.create_permute_vector();
  bin_sort.sort(shift_dupl_vec.vector_d);
  bin_sort.sort(chkpt_id_keys);
  Kokkos::Profiling::popRegion();

  // Write repeat entries
  std::string copy_metadata_label = std::string("Checkpoint ") + std::to_string(chkpt_id) + std::string(": Gather: Write repeat metadata");
  Kokkos::parallel_for(copy_metadata_label, Kokkos::RangePolicy<>(0, shift_dupl_vec.size()), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t node = shift_dupl_vec(i);
    NodeID prev = first_occur_d.value_at(first_occur_d.find(curr_tree(shift_dupl_vec(i))));
    memcpy(buffer_d.data()+sizeof(header_t)+prior_start+static_cast<uint64_t>(i)*2*sizeof(uint32_t), &node, sizeof(uint32_t));
    memcpy(buffer_d.data()+sizeof(header_t)+prior_start+static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
  });

//  // Write repeat entries
//  Kokkos::parallel_for("Write prior repeat bytes", Kokkos::RangePolicy<>(0, shared.capacity()), KOKKOS_LAMBDA(const uint32_t i) {
//    if(shared.valid_at(i)) {
//      uint32_t node = shared.key_at(i);
//      NodeID prev = shared.value_at(i);
//      Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t)+sizeof(uint32_t));
//      Kokkos::atomic_add(&num_bytes_d(0), sizeof(uint32_t)+sizeof(uint32_t));
//      size_t pos = Kokkos::atomic_sub_fetch(&prior_counter_d(prev.tree), 1);
//      memcpy(buffer_d.data()+sizeof(header_t)+prior_start+pos*2*sizeof(uint32_t), &node, sizeof(uint32_t));
//      memcpy(buffer_d.data()+sizeof(header_t)+prior_start+pos*2*sizeof(uint32_t)+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
//    }
//  });

  DEBUG_PRINT("Wrote shared metadata\n");
//  Kokkos::fence();
  DEBUG_PRINT("Finished collecting data\n");
//  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
//  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
//  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
//  Kokkos::fence();
  header.ref_id = prior_chkpt_id;
  header.chkpt_id = chkpt_id;
  header.datalen = data.size();
  header.chunk_size = chunk_size;
  header.num_first_ocur = first_ocur_vec.size();
  header.num_shift_dupl = shift_dupl_vec.size();
  header.num_prior_chkpts = chkpts_needed.count();
//  STDOUT_PRINT("Ref ID: %u\n"          , header.ref_id);
//  STDOUT_PRINT("Chkpt ID: %u\n"        , header.chkpt_id);
//  STDOUT_PRINT("Data len: %lu\n"       , header.datalen);
//  STDOUT_PRINT("Chunk size: %u\n"      , header.chunk_size);
//  STDOUT_PRINT("Num first ocur: %u\n"  , header.num_first_ocur);
//  STDOUT_PRINT("Num shift dupl: %u\n"  , header.num_shift_dupl);
//  STDOUT_PRINT("Num prior chkpts: %u\n", header.num_prior_chkpts);
//  STDOUT_PRINT("Number of bytes written for compact incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_data_h(0) + num_bytes_metadata_h(0));
//  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
//  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));
//  DEBUG_PRINT("Closed file\n");
//  return std::make_pair(num_bytes_data_h(0), sizeof(header_t) + num_bytes_metadata_h(0));
//  uint64_t size_metadata = buffer_len - first_ocur_size;
  uint64_t size_metadata = first_ocur_vec.size()*sizeof(uint32_t)+static_cast<uint64_t>(num_prior)*2*sizeof(uint32_t)+shift_dupl_vec.size()*2*sizeof(uint32_t);
  uint64_t size_data = buffer_len - size_metadata;
  first_ocur_vec.clear();
  shift_dupl_vec.clear();
  return std::make_pair(size_data, size_metadata);
}

#endif // WRITE_MERKLE_TREE_CHKPT_HPP


