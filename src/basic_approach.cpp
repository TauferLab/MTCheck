#include "basic_approach.hpp"

BasicDeduplicator::BasicDeduplicator() {}

BasicDeduplicator::BasicDeduplicator(uint32_t bytes_per_chunk) {
  chunk_size = bytes_per_chunk;
  current_id = 0;
  baseline_id = 0;
}

BasicDeduplicator::~BasicDeduplicator() {}

void 
BasicDeduplicator::dedup_data(const uint8_t* data_ptr, const size_t len) {
  // Calculate useful constants
  data_len = len;
  num_chunks = data_len/chunk_size;
  if(num_chunks*chunk_size < data_len)
    num_chunks += 1;
  // Reset bitset so all chunks are assumed unchanged
  changes_bitset.reset();
  // Parallelization policy. Split chunks amoung teams of threads
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>((num_chunks/TEAM_SIZE)+1, TEAM_SIZE);
  Kokkos::parallel_for("Dedup chunks", team_policy, KOKKOS_CLASS_LAMBDA(member_type team_member) {
    uint32_t i=team_member.league_rank();
    uint32_t j=team_member.team_rank();
    uint32_t idx = i*team_member.team_size()+j;
    if(idx < num_chunks) {
      uint32_t num_bytes = chunk_size;
      uint64_t offset = static_cast<uint64_t>(idx)*static_cast<uint64_t>(chunk_size);
      if(idx == num_chunks-1)
        num_bytes = data_len-offset;
      HashDigest new_hash;
      hash(data_ptr+offset, num_bytes, new_hash.digest);
      if(current_id > 0) {
        if(!digests_same(list(idx), new_hash)) {
          list(idx) = new_hash;
          changes_bitset.set(idx);
        }
      } else {
        changes_bitset.set(idx);
        list(idx) = new_hash;
      }
    }
  });
  Kokkos::fence();
  STDOUT_PRINT("(Bitset): Number of changed chunks: %u\n", changes_bitset.count());
}

std::pair<uint64_t,uint64_t> 
BasicDeduplicator::collect_diff( const uint8_t* data_ptr, 
              const size_t len,
              Kokkos::View<uint8_t*>& buffer_d, 
              header_t& header) {
  // Allocate counters for logging data use
  // TODO Remove all but num_bytes_d. The rest are unnecessary now
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

  // Calculate buffer size and resize buffer
  uint64_t buffer_size = sizeof(header_t);
  buffer_size += static_cast<uint64_t>(changes_bitset.count())*static_cast<uint64_t>(sizeof(uint32_t) + chunk_size);
  Kokkos::resize(buffer_d, buffer_size);

  // Get offset for start of data section
  size_t data_offset = static_cast<size_t>(changes_bitset.count())*sizeof(uint32_t);

  STDOUT_PRINT("Changes: %u\n", changes_bitset.count());
  STDOUT_PRINT("Buffer size: %lu\n", buffer_size);

  // Gather chunks and form diff in parallel
  auto policy = Kokkos::TeamPolicy<>(changes_bitset.size(), Kokkos::AUTO());
  Kokkos::parallel_for("Make incremental checkpoint", policy, 
                       KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
    uint32_t i = team_member.league_rank();
    // Test if chunk has changed
    if(changes_bitset.test(i)) { 
      // Calculate position in diff
      size_t pos = 0;
      if(team_member.team_rank() == 0) { 
        Kokkos::atomic_add(&num_bytes_metadata_d(0), sizeof(uint32_t));
        Kokkos::atomic_add(&num_bytes_data_d(0), static_cast<uint64_t>(chunk_size));
        pos = Kokkos::atomic_fetch_add(&num_bytes_d(0), sizeof(uint32_t));
        // Mark chunk in metadata
        memcpy(buffer_d.data()+sizeof(header_t) + pos, &i, sizeof(uint32_t));
      }
      // Update all threads in team
      team_member.team_broadcast(pos, 0); 
      // Check if chunk is last in list (size may be different)
      uint32_t writesize = chunk_size;
      if(i==num_chunks-1) { 
        writesize = data_len-static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);
      }
      // Calculate offsets
      uint64_t dst_offset = data_offset+(pos/sizeof(uint32_t))*static_cast<uint64_t>(chunk_size);
      uint64_t src_offset = static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);

      // Copy chunk
      uint8_t* src = (uint8_t*)(data_ptr+src_offset);
      uint8_t* dst = (uint8_t*)(buffer_d.data()+sizeof(header_t)+dst_offset);
      team_memcpy(dst, src, writesize, team_member);
    }
  });
  Kokkos::fence();
  // Update host
  Kokkos::deep_copy(num_bytes_h, num_bytes_d);
  Kokkos::deep_copy(num_bytes_data_h, num_bytes_data_d);
  Kokkos::deep_copy(num_bytes_metadata_h, num_bytes_metadata_d);
  Kokkos::fence();
  // Update header fields
  header.ref_id = baseline_id;
  header.chkpt_id = current_id;
  header.datalen = data_len;
  header.chunk_size = chunk_size;
  header.num_first_ocur = changes_bitset.count();
  header.num_shift_dupl = 0;
  header.num_prior_chkpts = 0;
  STDOUT_PRINT("Ref ID: %u\n"          , header.ref_id);
  STDOUT_PRINT("Chkpt ID: %u\n"        , header.chkpt_id);
  STDOUT_PRINT("Data len: %lu\n"       , header.datalen);
  STDOUT_PRINT("Chunk size: %u\n"      , header.chunk_size);
  STDOUT_PRINT("Num first ocur: %u\n"  , header.num_first_ocur);
  STDOUT_PRINT("Num shift dupl: %u\n"  , header.num_shift_dupl);
  STDOUT_PRINT("Num prior chkpts: %u\n", header.num_prior_chkpts);

  STDOUT_PRINT("Number of bytes written for incremental checkpoint: %lu\n", sizeof(header_t) + num_bytes_data_h(0) + num_bytes_metadata_h(0));
  STDOUT_PRINT("Number of bytes written for data: %lu\n", num_bytes_data_h(0));
  STDOUT_PRINT("Number of bytes written for metadata: %lu\n", sizeof(header_t) + num_bytes_metadata_h(0));

  uint64_t size_metadata = buffer_d.size() - header.num_first_ocur*chunk_size;
  return std::make_pair(header.num_first_ocur*chunk_size, size_metadata);
}

/**
 * Restart data from incremental checkpoint.
 *
 * \param incr_chkpts Vector of Host Views containing the diffs
 * \param chkpt_idx   ID of which checkpoint to restart
 * \param data        View for restarting the checkpoint to
 *
 * \return Time spent copying incremental checkpoints from host to device and restarting data
 */
std::pair<double,double>
BasicDeduplicator::restart_chkpt( std::vector<Kokkos::View<uint8_t*>::HostMirror>& incr_chkpts,
                                  const int chkpt_idx, 
                                  Kokkos::View<uint8_t*>& data) {
  // Get size of desired checkpoint
  size_t size = incr_chkpts[chkpt_idx].size();

  // Read header
  header_t header;
  memcpy(&header, incr_chkpts[chkpt_idx].data(), sizeof(header_t));
  STDOUT_PRINT("Ref ID: %u\n",               header.ref_id);
  STDOUT_PRINT("Chkpt ID: %u\n",             header.chkpt_id);
  STDOUT_PRINT("Data len: %lu\n",            header.datalen);
  STDOUT_PRINT("Chunk size: %u\n",           header.chunk_size);
  STDOUT_PRINT("Num first ocur: %u\n",       header.num_first_ocur);
  STDOUT_PRINT("Num prior chkpts: %u\n",     header.num_prior_chkpts);
  STDOUT_PRINT("Num shift dupl: %u\n",       header.num_shift_dupl);

  // Allocate buffer and initialize
  Kokkos::View<uint8_t*> buffer_d("Buffer", size);
  Kokkos::deep_copy(buffer_d, 0);

  // Calculate chunk sizes
  uint32_t num_chunks = header.datalen / header.chunk_size;
  if(num_chunks*header.chunk_size < header.datalen) {
    num_chunks += 1;
  }

  // Resize data
  Kokkos::resize(data, header.datalen);

  // Main checkpoint
  auto& checkpoint_h = incr_chkpts[chkpt_idx];

  // Copy incremental checkpoint to device View
  std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();
  Kokkos::deep_copy(buffer_d, checkpoint_h);
  Kokkos::fence();
  std::chrono::high_resolution_clock::time_point c2 = std::chrono::high_resolution_clock::now();

  // Allocate nodelist for tracking which chunks have been restarted
  Kokkos::View<NodeID*> node_list("List of NodeIDs", num_chunks);
  Kokkos::deep_copy(node_list, NodeID());

  // Load info from header
  uint32_t ref_id = header.ref_id;
  uint32_t cur_id = header.chkpt_id;
  uint32_t num_first_ocur = header.num_first_ocur;
  uint32_t chunk_size = header.chunk_size;
  size_t datalen = header.datalen;

  // Calculate offsets and create subviews for data and metadata sections
  size_t data_offset    = sizeof(header_t)+num_first_ocur*sizeof(uint32_t);
  auto metadata_subview = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t), data_offset));
  auto data_subview     = Kokkos::subview(buffer_d, std::make_pair(data_offset, size));

  // Allocate map for tracking which chunks have been seen
  Kokkos::UnorderedMap<NodeID, size_t> distinct_map(num_first_ocur);
  Kokkos::fence();

  // Load any first occurrences
  Kokkos::parallel_for("Restart Hashlist first occurrence", Kokkos::TeamPolicy<>(num_first_ocur, Kokkos::AUTO()), KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
    uint32_t i = team_member.league_rank();
    uint64_t src_offset = static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);
    uint32_t node=0;
    // Identify node, load node in map, and mark entry in nodelist
    if(team_member.team_rank() == 0) {
      memcpy(&node, metadata_subview.data() + i*sizeof(uint32_t), sizeof(uint32_t));
      distinct_map.insert(NodeID(node, cur_id), src_offset);
      node_list(node) = NodeID(node, cur_id);
    }
    team_member.team_broadcast(node, 0);
    uint64_t dst_offset = static_cast<uint64_t>(node)*static_cast<uint64_t>(chunk_size);
    uint32_t datasize = chunk_size;
    if(node == num_chunks-1)
      datasize = datalen - dst_offset;

    // Restart chunk for the node
    uint8_t* dst = (uint8_t*)(data.data()+dst_offset);
    uint8_t* src = (uint8_t*)(data_subview.data()+src_offset);
    team_memcpy(dst, src, datasize, team_member);
  });

  // Mark any untouched entries as unchanged
  Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    NodeID entry = node_list(i);
    if(entry.node == UINT_MAX) {
      node_list(i) = NodeID(i, cur_id-1);
    }
  });
  Kokkos::fence();

  // Go through checkpoint in reverse order 
  for(int idx=static_cast<int>(chkpt_idx)-1; idx>=static_cast<int>(ref_id); idx--) {
    // Load checkpoint and copy to device View
    size_t chkpt_size = incr_chkpts[idx].size();
    STDOUT_PRINT("Processing checkpoint %u\n", idx);
    STDOUT_PRINT("Checkpoint size: %lu\n", chkpt_size);
    Kokkos::View<uint8_t*> chkpt_buffer_d("Checkpoint buffer", chkpt_size);
    auto chkpt_buffer_h = Kokkos::create_mirror_view(chkpt_buffer_d);
    memcpy(chkpt_buffer_h.data(), incr_chkpts[idx].data(), chkpt_size);
    // Load header 
    header_t chkpt_header;
    memcpy(&chkpt_header, chkpt_buffer_h.data(), sizeof(header_t));
    datalen = chkpt_header.datalen;
    chunk_size = chkpt_header.chunk_size;
    Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
    ref_id = chkpt_header.ref_id;
    cur_id = chkpt_header.chkpt_id;
    num_first_ocur = chkpt_header.num_first_ocur;

    STDOUT_PRINT("Ref ID:           %u\n",  chkpt_header.ref_id);
    STDOUT_PRINT("Chkpt ID:         %u\n",  chkpt_header.chkpt_id);
    STDOUT_PRINT("Datalen:          %lu\n", chkpt_header.datalen);
    STDOUT_PRINT("Chunk size:       %u\n",  chkpt_header.chunk_size);
    STDOUT_PRINT("Distinct size:    %u\n",  chkpt_header.num_first_ocur);
    STDOUT_PRINT("Num prior chkpts: %u\n",  chkpt_header.num_prior_chkpts);
    
    // Create subviews for metadata and data sections
    data_offset      = sizeof(header_t)+static_cast<uint64_t>(num_first_ocur)*sizeof(uint32_t);
    metadata_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), data_offset));
    data_subview     = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, chkpt_size));
    STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", chkpt_size);
    STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Data offset: %lu\n", data_offset);

    // Insert all first occurrences into map
    distinct_map.clear();
    distinct_map.rehash(chkpt_header.num_first_ocur);
    Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.num_first_ocur), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, metadata_subview.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      distinct_map.insert(NodeID(node,cur_id), static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size));
    });

    // Restart first occurrences
    using TeamMember = Kokkos::TeamPolicy<>::member_type;
    auto chunk_policy = Kokkos::TeamPolicy<>(num_chunks, Kokkos::AUTO());
    Kokkos::parallel_for("Restart Hashlist first occurrence", chunk_policy, KOKKOS_CLASS_LAMBDA(const TeamMember& team_member) {
      uint32_t i = team_member.league_rank();
      if(node_list(i).tree == cur_id) {
        NodeID id = node_list(i);
        // If entry corresponds to this checkpoint
        if(distinct_map.exists(id)) {
          size_t offset = distinct_map.value_at(distinct_map.find(id));
          uint32_t writesize = chunk_size;
          if(i == num_chunks-1) 
            writesize = datalen-static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);

          // Restart chunk
          uint8_t* src = (uint8_t*)(data_subview.data()+offset);
          uint8_t* dst = (uint8_t*)(data.data()+static_cast<uint64_t>(chunk_size)*static_cast<uint64_t>(i));
          team_memcpy(dst, src, writesize, team_member);
        } else if(team_member.team_rank() == 0) {
          node_list(i) = NodeID(i, cur_id-1);
        }
      }
    });
  }
  Kokkos::fence();
  std::chrono::high_resolution_clock::time_point c3 = std::chrono::high_resolution_clock::now();

  // Return time spent copying or restarting data
  double copy_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
  double restart_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c3-c2).count());
  return std::make_pair(copy_time, restart_time);
}

std::pair<double,double>
BasicDeduplicator::restart_chkpt( std::vector<std::string>& chkpt_files,
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
  STDOUT_PRINT("Num first ocur: %u\n",        header.num_first_ocur);
  STDOUT_PRINT("Num prior chkpts: %u\n",      header.num_prior_chkpts);
  STDOUT_PRINT("Num shift dupl: %u\n",        header.num_shift_dupl);

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
  uint32_t chunk_size = header.chunk_size;
  size_t datalen = header.datalen;
  uint32_t num_first_ocur = header.num_first_ocur;

  size_t data_offset = sizeof(header_t) + num_first_ocur*sizeof(uint32_t);
  auto distinct = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t), data_offset));
  auto data_subview = Kokkos::subview(buffer_d, std::make_pair(data_offset, filesize));

  STDOUT_PRINT("Checkpoint %u\n", cur_id);
  STDOUT_PRINT("Checkpoint size: %lu\n", filesize);
  STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
  STDOUT_PRINT("Data offset: %lu\n", data_offset);

  Kokkos::UnorderedMap<NodeID, size_t> distinct_map(num_first_ocur);
  Kokkos::parallel_for("Restart Hashlist distinct", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    uint32_t node;
    memcpy(&node, distinct.data() + i*sizeof(uint32_t),  sizeof(uint32_t));
    node_list(node) = NodeID(node,cur_id);
    uint32_t datasize = chunk_size;
    if(node == num_chunks-1)
      datasize = datalen - node*chunk_size;
    memcpy(data.data()+chunk_size*node, data_subview.data()+i*chunk_size, datasize);
  });
  Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    NodeID entry = node_list(i);
    if(entry.node == UINT_MAX) {
      node_list(i) = NodeID(i, cur_id-1);
    }
  });
  Kokkos::fence();

  for(int idx=static_cast<int>(file_idx)-1; idx>=static_cast<int>(ref_id); idx--) {
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
    datalen = chkpt_header.datalen;
    chunk_size = chkpt_header.chunk_size;
    Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);
    ref_id = chkpt_header.ref_id;
    cur_id = chkpt_header.chkpt_id;

    STDOUT_PRINT("Ref ID:           %u\n",  chkpt_header.ref_id);
    STDOUT_PRINT("Chkpt ID:         %u\n",  chkpt_header.chkpt_id);
    STDOUT_PRINT("Datalen:          %lu\n", chkpt_header.datalen);
    STDOUT_PRINT("Chunk size:       %u\n",  chkpt_header.chunk_size);
    STDOUT_PRINT("Distinct size:    %u\n",  chkpt_header.num_first_ocur);
    STDOUT_PRINT("Num prior chkpts: %u\n",  chkpt_header.num_prior_chkpts);
    
    data_offset = sizeof(header_t)+chkpt_header.num_first_ocur*sizeof(uint32_t);
    distinct      = Kokkos::subview(chkpt_buffer_d, std::make_pair(sizeof(header_t), data_offset));
    data_subview  = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, chkpt_size));
    STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", chkpt_size);
    STDOUT_PRINT("Distinct offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Data offset: %lu\n", data_offset);

    distinct_map.clear();
    distinct_map.rehash(chkpt_header.num_first_ocur);
    Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, chkpt_header.num_first_ocur), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, distinct.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      distinct_map.insert(NodeID(node,cur_id), i*chunk_size);
    });

    Kokkos::parallel_for("Fill data middle chkpts", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      if(node_list(i).tree == cur_id) {
        NodeID id = node_list(i);
        if(distinct_map.exists(id)) {
          size_t offset = distinct_map.value_at(distinct_map.find(id));
          uint32_t writesize = chunk_size;
          if(i == num_chunks-1) 
            writesize = datalen-i*chunk_size;
          memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
        } else {
          node_list(i) = NodeID(i, cur_id-1);
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

void 
BasicDeduplicator::checkpoint(header_t& header, 
                              uint8_t* data_ptr, 
                              size_t len,
                              Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                              bool make_baseline) {
  using Timer = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double>;
  // ==========================================================================================
  // Setup
  // ==========================================================================================
  Timer::time_point beg_chkpt = Timer::now();
  std::string setup_region_name = std::string("Deduplication chkpt ") + 
                                  std::to_string(current_id) + std::string(": Setup");
  Kokkos::Profiling::pushRegion(setup_region_name.c_str());

  // Set important values
  data_len = len;
  num_chunks = data_len/chunk_size;
  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(chunk_size) < data_len)
    num_chunks += 1;

  // Allocate or resize necessary variables for each approach
  if(make_baseline) {
    list = HashList(num_chunks);
    changes_bitset = Kokkos::Bitset<Kokkos::DefaultExecutionSpace>(num_chunks);
  }
  if(list.list_d.size() < num_chunks) {
    Kokkos::resize(list.list_d, num_chunks);
    Kokkos::resize(list.list_h, num_chunks);
  }
  Kokkos::Profiling::popRegion();

  // ==========================================================================================
  // Deduplicate data
  // ==========================================================================================
  std::string dedup_region_name = std::string("Deduplication chkpt ") + 
                                  std::to_string(current_id);
  Timer::time_point start_create_tree0 = Timer::now();
  timers[0] = std::chrono::duration_cast<Duration>(start_create_tree0 - beg_chkpt).count();
  Kokkos::Profiling::pushRegion(dedup_region_name.c_str());

  // Deduplicate data and identify nodes and chunks needed for the incremental checkpoint
  if((current_id == 0) || make_baseline) {
    baseline_id = current_id;
  }
  dedup_data(data_ptr, len);

  Kokkos::Profiling::popRegion();
  Timer::time_point end_create_tree0 = Timer::now();
  timers[1] = std::chrono::duration_cast<Duration>(end_create_tree0 - start_create_tree0).count();

  // ==========================================================================================
  // Create Diff
  // ==========================================================================================
  Kokkos::View<uint8_t*> diff;
  std::string collect_region_name = std::string("Start writing incremental checkpoint ") 
                            + std::to_string(current_id);
  Timer::time_point start_collect = Timer::now();
  Kokkos::Profiling::pushRegion(collect_region_name.c_str());

  datasizes = collect_diff(data_ptr, data_len, diff, header);

  Kokkos::Profiling::popRegion();
  Timer::time_point end_collect = Timer::now();
  timers[2] = std::chrono::duration_cast<Duration>(end_collect - start_collect).count();

  // ==========================================================================================
  // Copy diff to host 
  // ==========================================================================================
  Timer::time_point start_write = Timer::now();
  Kokkos::resize(diff_h, diff.size());
  std::string write_region_name = std::string("Copy diff to host ") 
                                  + std::to_string(current_id);
  Kokkos::Profiling::pushRegion(write_region_name.c_str());

  Kokkos::deep_copy(diff_h, diff);
  memcpy(diff_h.data(), &header, sizeof(header_t));

  Kokkos::Profiling::popRegion();
  Timer::time_point end_write = Timer::now();
  timers[3] = std::chrono::duration_cast<Duration>(end_write - start_write).count();
}

void 
BasicDeduplicator::checkpoint(uint8_t* data_ptr, 
                              size_t len, 
                              std::string& filename, 
                              std::string& logname, 
                              bool make_baseline) {
  Kokkos::View<uint8_t*>::HostMirror diff_h;
  header_t header;
  checkpoint(header, data_ptr, len, diff_h, make_baseline);
  write_chkpt_log(header, diff_h, logname);
  // Write checkpoint to file
  std::ofstream file;
  file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  file.open(filename, std::ofstream::out | std::ofstream::binary);
  file.write((const char*)(diff_h.data()), diff_h.size());
  file.flush();
  file.close();
  current_id += 1;
}

void 
BasicDeduplicator::checkpoint(uint8_t* data_ptr, 
                              size_t len, 
                              Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                              bool make_baseline) {
  header_t header;
  checkpoint(header, data_ptr, len, diff_h, make_baseline);
  current_id += 1;
}

void 
BasicDeduplicator::checkpoint(uint8_t* data_ptr, 
                              size_t len, 
                              Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                              std::string& logname, 
                              bool make_baseline) {
  header_t header;
  checkpoint(header, data_ptr, len, diff_h, make_baseline);
  write_chkpt_log(header, diff_h, logname);
  current_id += 1;
}
                   
void 
BasicDeduplicator::restart(Kokkos::View<uint8_t*> data, 
                           std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                           std::string& logname, 
                           uint32_t chkpt_id) {
  auto basic_list_times = restart_chkpt(chkpts, chkpt_id, data);
  restart_timers[0] = basic_list_times.first;
  restart_timers[1] = basic_list_times.second;
  std::string restart_logname = logname + ".chunk_size." + std::to_string(chunk_size) +
                                ".restart_timing.csv";
  write_restart_log(chkpt_id, restart_logname);
}

void 
BasicDeduplicator::restart(uint8_t* data_ptr, 
                           size_t len, 
                           std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                           std::string& logname, 
                           uint32_t chkpt_id) {
  Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
  restart(data, chkpts, logname, chkpt_id);
}

void 
BasicDeduplicator::restart(Kokkos::View<uint8_t*> data, 
                           std::vector<std::string>& chkpt_filenames, 
                           std::string& logname, 
                           uint32_t chkpt_id) {
  std::vector<std::string> basiclist_chkpt_files;
  for(uint32_t i=0; i<chkpt_filenames.size(); i++) {
    basiclist_chkpt_files.push_back(chkpt_filenames[i]+".basic.incr_chkpt");
  }
  auto basic_list_times = restart_chkpt(basiclist_chkpt_files, chkpt_id, data);
  restart_timers[0] = basic_list_times.first;
  restart_timers[1] = basic_list_times.second;
  write_restart_log(chkpt_id, logname);
}

void 
BasicDeduplicator::write_chkpt_log(header_t& header, 
                                   Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                                   std::string& logname) {
  std::fstream result_data, timing_file, size_file;
  std::string result_logname = logname+".chunk_size."+std::to_string(chunk_size)+".csv";
  std::string size_logname = logname+".chunk_size."+std::to_string(chunk_size)+".size.csv";
  std::string timing_logname = logname+".chunk_size."+std::to_string(chunk_size)+".timing.csv";
  result_data.open(result_logname, std::fstream::ate | std::fstream::out | std::fstream::app);
  size_file.open(size_logname, std::fstream::out | std::fstream::app);
  timing_file.open(timing_logname, std::fstream::out | std::fstream::app);
  if(result_data.tellp() == 0) {
    result_data << "Approach,Chkpt ID,Chunk Size,Uncompressed Size,Compressed Size,Data Size,Metadata Size,Setup Time,Comparison Time,Gather Time,Write Time" << std::endl;
  }

  // TODO make this more generic 
//  uint32_t num_chkpts = 10;

  result_data << "Basic" << "," // Approach
              << current_id << "," // Chkpt ID
              << chunk_size << "," // Chunk size
              << data_len << "," // Uncompressed size
              << datasizes.first+datasizes.second << "," // Compressed size
              << datasizes.first << "," // Compressed data size
              << datasizes.second << "," // Compressed metadata size
              << timers[0] << "," // Compression setup time
              << timers[1] << "," // Compression comparison time
              << timers[2] << "," // Compression gather chunks time
              << timers[3] << std::endl; // Compression copy diff to host
  timing_file << "Basic" << ","      // Approach
              << current_id << ","          // Checkpoint ID
              << chunk_size << ","   // Chunk size      
              << timers[1] << "," // Comparison time 
              << timers[2] << "," // Collection time 
              << timers[3]          // Write time
              << std::endl;
  size_file << "Basic" << ","           // Approach
            << current_id << ","               // Checkpoint ID
            << chunk_size << ","        // Chunk size
            << datasizes.first << ","   // Size of data
            << datasizes.second << ","; // Size of metadata
  write_metadata_breakdown(size_file, DedupMode::Basic, header, diff_h, current_id-baseline_id+1);
}

/**
 * Function for writing the restart log.
 *
 * \param select_chkpt Which checkpoint to write the log
 * \param logname      Filename for writing log
 */
void 
BasicDeduplicator::write_restart_log(uint32_t select_chkpt, std::string& logname) {
  std::fstream timing_file;
  timing_file.open(logname, std::fstream::out | std::fstream::app);
  timing_file << "Basic" << ","; 
  timing_file << select_chkpt << "," 
              << chunk_size << "," 
              << restart_timers[0] << "," 
              << restart_timers[1] << std::endl;
  timing_file.close();
}
