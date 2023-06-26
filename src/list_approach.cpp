#include "list_approach.hpp"

ListDeduplicator::ListDeduplicator() {}

ListDeduplicator::ListDeduplicator(uint32_t bytes_per_chunk) {
  chunk_size = bytes_per_chunk;
  current_id = 0;
  baseline_id = 0;
}

ListDeduplicator::~ListDeduplicator() {}

/**
 * Deduplicate provided data view using the list incremental checkpoint approach. 
 * Split data into chunks and compute hashes for each chunk. Compare each hash with 
 * the hash at the same offset. If the hash has never been seen, save the chunk.
 * If the hash has been seen before, mark it as a shifted duplicate and save metadata.
 *
 * \param data_ptr      Pointer to data for deduplicating
 * \param len           Length of data to deduplicate
 */
void 
ListDeduplicator::dedup_data(const uint8_t* data_ptr, 
                             const size_t len) {
  // Calculate useful constants
  data_len = len;
  num_chunks = len/chunk_size;
  if(num_chunks*chunk_size < len)
    num_chunks += 1;

  // Clear Vectors
  shift_dupl_vec.clear();
  first_ocur_vec.clear();

  // Parallelization policy. Split chunks amoung teams of threads
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>((num_chunks/TEAM_SIZE)+1, TEAM_SIZE);
  Kokkos::parallel_for("Dedup chunks", team_policy, 
  KOKKOS_CLASS_LAMBDA(member_type team_member) {
    uint32_t i=team_member.league_rank();
    uint32_t j=team_member.team_rank();
    uint32_t block_idx = i*team_member.team_size()+j;
    if(block_idx < num_chunks) {
      uint32_t num_bytes = chunk_size;
      uint64_t offset = static_cast<uint64_t>(block_idx)*static_cast<uint64_t>(chunk_size);
      if(block_idx == num_chunks-1)
        num_bytes = len-offset;
      HashDigest new_hash;
      hash(data_ptr+offset, num_bytes, new_hash.digest); /// Compute hash
      if(!digests_same(list(block_idx), new_hash)) { /// Test if hash is different
        NodeID info(block_idx, current_id);
        auto result = first_ocur_d.insert(new_hash, info);
        if(result.success()) { // New hash (first occurrence)
          first_ocur_vec.push(block_idx);
        } else if(result.existing()) { // Hash exists (shifted duplicate)
          shift_dupl_vec.push(block_idx);
        }
        list(block_idx) = new_hash;
      }
    }
  });
  Kokkos::fence();
  STDOUT_PRINT("Comparing Lists\n");
  STDOUT_PRINT("Number of first occurrences: %u\n", first_ocur_vec.size());
  STDOUT_PRINT("Number of shifted duplicates: %u\n", shift_dupl_vec.size());
}

/**
 * Gather the scattered chunks for the diff and write the checkpoint to a contiguous buffer.
 *
 * \param data_ptr       Pointer to data for deduplicate
 * \param len            Length of data to deduplicate in bytes
 * \param buffer_d       View to store the diff in
 * \param header         Incremental checkpoint header
 *
 * \return Pair containing amount of data and metadata in the checkpoint
 */
std::pair<uint64_t,uint64_t> 
ListDeduplicator::collect_diff( const uint8_t* data_ptr, 
                                const size_t len,
                                Kokkos::View<uint8_t*>& buffer_d, 
                                header_t& header) {
  // Calculate number of chunks
  num_chunks = data_len/chunk_size;
  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(chunk_size) < data_len) {
    num_chunks += 1;
  }
  STDOUT_PRINT("Number of first occurrence digests: %u\n", first_ocur_d.size());
  STDOUT_PRINT("Num first occurrences: %u\n", first_ocur_vec.size());
  STDOUT_PRINT("Num shifted duplicates: %u\n", shift_dupl_vec.size());

  // Allocate counters for shifted duplicates
  Kokkos::View<uint64_t*> prior_counter_d("Counter for prior repeats", current_id+1);
  Kokkos::View<uint64_t*>::HostMirror prior_counter_h = Kokkos::create_mirror_view(prior_counter_d);
  Kokkos::Experimental::ScatterView<uint64_t*> prior_counter_sv(prior_counter_d);

  // Small bitset to record which checkpoints are necessary for restart
  Kokkos::Bitset<Kokkos::DefaultExecutionSpace> chkpts_needed(current_id+1);
  chkpts_needed.reset();
  STDOUT_PRINT("Reset bitset\n");

  // Count how many duplicates belong to each checkpoint
  auto shift_dupl_policy = Kokkos::RangePolicy<>(0, shift_dupl_vec.size());
  Kokkos::parallel_for("Count shifted dupl", shift_dupl_policy, KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    auto prior_counter_sa = prior_counter_sv.access();
    NodeID entry = first_ocur_d.value_at(first_ocur_d.find(list.list_d(shift_dupl_vec(i))));
    chkpts_needed.set(entry.tree);
    prior_counter_sa(entry.tree) += 1;
  });
  Kokkos::Experimental::contribute(prior_counter_d, prior_counter_sv);
  prior_counter_sv.reset_except(prior_counter_d);
  Kokkos::deep_copy(prior_counter_h, prior_counter_d);

  Kokkos::fence();
  STDOUT_PRINT("Counted shifted duplicates\n");

  // Calculate offsets for writing the diff
  uint64_t num_first_ocur = static_cast<uint64_t>(first_ocur_vec.size());
  uint64_t num_chkpts = static_cast<uint64_t>(chkpts_needed.count());
  uint64_t num_shift_dupl = static_cast<uint64_t>(shift_dupl_vec.size());
  size_t first_ocur_offset = sizeof(header_t);
  size_t shift_dupl_count_offset = first_ocur_offset + num_first_ocur*sizeof(uint32_t);
  size_t shift_dupl_offset = shift_dupl_count_offset + num_chkpts*2*sizeof(uint32_t);
  size_t data_offset = shift_dupl_offset + num_shift_dupl*2*sizeof(uint32_t);
  // Calculate size of diff
  uint64_t buffer_size = sizeof(header_t);
  buffer_size += num_first_ocur*(sizeof(uint32_t)+static_cast<uint64_t>(chunk_size)); // First occurrence metadata
  buffer_size += num_chkpts*2*sizeof(uint32_t); // Shifted duplicate counts metadata
  buffer_size += num_shift_dupl*2*sizeof(uint32_t); // Shifted duplicate metadata
  Kokkos::resize(buffer_d, buffer_size);
  STDOUT_PRINT("Resized buffer\n");
  Kokkos::View<uint64_t[1]> dupl_map_offset_d("Dupl map offset");
  Kokkos::deep_copy(dupl_map_offset_d, shift_dupl_count_offset);
  STDOUT_PRINT("Created duplicate map counter\n");

  STDOUT_PRINT("Buffer length: %lu\n", buffer_size);
  STDOUT_PRINT("Data offset: %zu\n", data_offset);
  STDOUT_PRINT("Duplicate map offset: %zu\n", shift_dupl_count_offset);

  // Copy first occurrence metadata
  using ByteView = Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  ByteView first_ocur_bytes((uint8_t*)(first_ocur_vec.data()), num_first_ocur*sizeof(uint32_t));
  size_t metadata_end = sizeof(header_t)+num_first_ocur*sizeof(uint32_t);
  auto metadata_subview = Kokkos::subview(buffer_d, std::make_pair(sizeof(header_t),metadata_end));
  Kokkos::deep_copy(metadata_subview, first_ocur_bytes);

  // Write Repeat map for recording how many entries per checkpoint
  // (Checkpoint ID, # of entries)
  Kokkos::parallel_for("Write size map", prior_counter_d.size(), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    if(prior_counter_d(i) > 0) {
      uint32_t num_repeats_i = static_cast<uint32_t>(prior_counter_d(i));
      size_t pos = Kokkos::atomic_fetch_add(&dupl_map_offset_d(0), 2*sizeof(uint32_t));
      memcpy(buffer_d.data()+pos, &i, sizeof(uint32_t));
      memcpy(buffer_d.data()+pos+sizeof(uint32_t), &num_repeats_i, sizeof(uint32_t));
    }
  });
  STDOUT_PRINT("Write duplicate counts\n");
  // Calculate repeat indices so that we can separate entries by source ID
  Kokkos::parallel_scan("Calc repeat end indices", prior_counter_d.size(), 
    KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
    partial_sum += prior_counter_d(i);
    if(is_final) prior_counter_d(i) = partial_sum;
  });
  STDOUT_PRINT("Calculated duplicate offsets\n");

  // Sort shifted duplicates by the list ID
  // Prepare keys
  Kokkos::View<uint32_t*> chkpt_id_keys("Source checkpoint IDs", shift_dupl_vec.size());
  Kokkos::deep_copy(chkpt_id_keys, 0);
  Kokkos::parallel_for(shift_dupl_vec.size(), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    if(!first_ocur_d.valid_at(first_ocur_d.find(list(shift_dupl_vec(i)))))
      DEBUG_PRINT("Invalid index!\n");
    NodeID prev = first_ocur_d.value_at(first_ocur_d.find(list(shift_dupl_vec(i))));
    chkpt_id_keys(i) = prev.tree;
  });
  // Find max key for sorting
  uint32_t max_key = 0;
  Kokkos::parallel_reduce("Get max key", shift_dupl_policy, 
  KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& max) {
    if(chkpt_id_keys(i) > max)
      max = chkpt_id_keys(i);
  }, Kokkos::Max<uint32_t>(max_key));
  DEBUG_PRINT("Updated chkpt ID keys for sorting\n");
  // Sort duplicates
  using key_type = decltype(chkpt_id_keys);
  using Comparator = Kokkos::BinOp1D<key_type>;
  Comparator comp(shift_dupl_vec.size(), 0, max_key);
  DEBUG_PRINT("Created comparator\n");
  Kokkos::BinSort<key_type, Comparator> bin_sort(chkpt_id_keys, 0, shift_dupl_vec.size(), comp, 0);
  DEBUG_PRINT("Created BinSort\n");
  bin_sort.create_permute_vector();
  DEBUG_PRINT("Created permute vector\n");
  bin_sort.sort(shift_dupl_vec.vector_d);
  DEBUG_PRINT("Sorted duplicate offsets\n");
  bin_sort.sort(chkpt_id_keys);
  DEBUG_PRINT("Sorted chkpt id keys\n");

  // Write repeat entries
  Kokkos::parallel_for("Write repeat bytes", shift_dupl_policy, KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    uint32_t node = shift_dupl_vec(i);
    NodeID prev = first_ocur_d.value_at(first_ocur_d.find(list(node)));
    uint64_t dupl_offset = static_cast<uint64_t>(i)*2*sizeof(uint32_t);
    memcpy(buffer_d.data()+shift_dupl_offset+dupl_offset, &node, sizeof(uint32_t));
    memcpy(buffer_d.data()+shift_dupl_offset+dupl_offset+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
  });
  DEBUG_PRINT("Wrote duplicates\n");

  // Write data
  Kokkos::parallel_for("Copy data", Kokkos::TeamPolicy<>(first_ocur_vec.size(), Kokkos::AUTO()), 
                         KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
    uint32_t i = team_member.league_rank();
    uint32_t chunk = first_ocur_vec(i);
    uint32_t writesize = chunk_size;
    uint64_t src_offset = static_cast<uint64_t>(chunk_size)*static_cast<uint64_t>(chunk);
    uint64_t dst_offset = data_offset+static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);
    if(chunk == num_chunks-1) {
      writesize = data_len-src_offset;
    }

    uint8_t* src = (uint8_t*)(data_ptr+src_offset);
    uint8_t* dst = (uint8_t*)(buffer_d.data()+dst_offset);
    team_memcpy(dst, src, writesize, team_member);
  });
  Kokkos::fence();

  // Update header
  header.ref_id = baseline_id;
  header.chkpt_id = current_id;
  header.datalen = data_len;
  header.chunk_size = chunk_size;
  header.num_first_ocur = first_ocur_vec.size();
  header.num_shift_dupl = shift_dupl_vec.size();
  header.num_prior_chkpts = chkpts_needed.count();
  DEBUG_PRINT("Ref ID: %u\n"          , header.ref_id);
  DEBUG_PRINT("Chkpt ID: %u\n"        , header.chkpt_id);
  DEBUG_PRINT("Data len: %lu\n"       , header.datalen);
  DEBUG_PRINT("Chunk size: %u\n"      , header.chunk_size);
  DEBUG_PRINT("Num first ocur: %u\n"  , header.num_first_ocur);
  DEBUG_PRINT("Num shift dupl: %u\n"  , header.num_shift_dupl);
  DEBUG_PRINT("Num prior chkpts: %u\n", header.num_prior_chkpts);
  size_t data_size = static_cast<size_t>(chunk_size)*num_first_ocur;
  STDOUT_PRINT("Number of bytes written for data: %lu\n", data_size);
  DEBUG_PRINT("Trying to close file\n");
  DEBUG_PRINT("Closed file\n");
  uint64_t size_metadata = buffer_size - data_size;
  return std::make_pair(data_size, size_metadata);
}

/**
 * Restart data from incremental checkpoints stored in Kokkos Views on the host.
 *
 * \param incr_chkpts Vector of Host Views containing the diffs
 * \param chkpt_idx   ID of which checkpoint to restart
 * \param data        View for restarting the checkpoint to
 *
 * \return Time spent copying incremental checkpoints from host to device and restarting data
 */
std::pair<double,double>
ListDeduplicator::restart_chkpt( std::vector<Kokkos::View<uint8_t*>::HostMirror >& incr_chkpts,
                                 const int chkpt_idx, 
                                 Kokkos::View<uint8_t*>& data) {
  // Load checkpoint to restart and the header
  size_t chkpt_size = incr_chkpts[chkpt_idx].size();
  header_t header;
  memcpy(&header, incr_chkpts[chkpt_idx].data(), sizeof(header_t));

  // Allocate buffer for chkpt on the device
  Kokkos::View<uint8_t*> buffer_d("Buffer", chkpt_size);
  Kokkos::deep_copy(buffer_d, 0);

  // Calculate number of chunks
  uint32_t num_chunks = header.datalen / static_cast<uint64_t>(header.chunk_size);
  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(header.chunk_size) < header.datalen) {
    num_chunks += 1;
  }
  Kokkos::resize(data, header.datalen);

  auto& checkpoint_h = incr_chkpts[chkpt_idx];
  Kokkos::fence();
  // Copy checkpoint to the device
  std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();
  Kokkos::deep_copy(buffer_d, checkpoint_h);
  Kokkos::fence();
  std::chrono::high_resolution_clock::time_point c2 = std::chrono::high_resolution_clock::now();

  // Allocate node list to track which chunks have been restarted
  Kokkos::View<NodeID*> node_list("List of NodeIDs", num_chunks);
  Kokkos::deep_copy(node_list, NodeID());

  // Load header values
  uint32_t ref_id = header.ref_id;
  uint32_t cur_id = header.chkpt_id;
  size_t datalen = header.datalen;
  uint32_t chunk_size = header.chunk_size;
  uint32_t num_first_ocur = header.num_first_ocur;
  uint32_t num_prior_chkpts = header.num_prior_chkpts;
  uint32_t num_shift_dupl = header.num_shift_dupl;

  STDOUT_PRINT("Ref ID:           %u\n",  header.ref_id);
  STDOUT_PRINT("Chkpt ID:         %u\n",  header.chkpt_id);
  STDOUT_PRINT("Datalen:          %lu\n", header.datalen);
  STDOUT_PRINT("Chunk size:       %u\n",  header.chunk_size);
  STDOUT_PRINT("Distinct size:    %u\n",  header.num_first_ocur);
  STDOUT_PRINT("Num prior chkpts: %u\n",  header.num_prior_chkpts);
  STDOUT_PRINT("Num shift dupl:   %u\n",  header. num_shift_dupl);

  // Calculate offsets and subviews for reading sections of the checkpoint
  size_t first_ocur_offset = sizeof(header_t);
  size_t dupl_count_offset = first_ocur_offset + static_cast<size_t>(num_first_ocur)*sizeof(uint32_t);
  size_t dupl_map_offset   = dupl_count_offset + static_cast<size_t>(num_prior_chkpts)*2*sizeof(uint32_t);
  size_t data_offset       = dupl_map_offset   + static_cast<size_t>(num_shift_dupl)*2*sizeof(uint32_t);
  auto first_ocur_subview    = Kokkos::subview(buffer_d, std::make_pair(first_ocur_offset, dupl_count_offset));
  auto dupl_count_subview    = Kokkos::subview(buffer_d, std::make_pair(dupl_count_offset, dupl_map_offset));
  auto shift_dupl_subview    = Kokkos::subview(buffer_d, std::make_pair(dupl_map_offset, data_offset));
  auto data_subview  = Kokkos::subview(buffer_d, std::make_pair(data_offset, chkpt_size));
  STDOUT_PRINT("Checkpoint %u\n", header.chkpt_id);
  STDOUT_PRINT("Checkpoint size: %lu\n", chkpt_size);
  STDOUT_PRINT("First ocur offset: %lu\n", sizeof(header_t));
  STDOUT_PRINT("Dupl count offset: %lu\n", dupl_count_offset);
  STDOUT_PRINT("Dupl map offset: %lu\n", dupl_map_offset);
  STDOUT_PRINT("Data offset: %lu\n", data_offset);

  // Create map for tracking which chunks exist in the checkpoint
  Kokkos::UnorderedMap<NodeID, size_t> first_occur_map(num_first_ocur);
  // Load map and mark any first occurrences
  Kokkos::parallel_for("Restart Hashlist first occurrence", Kokkos::TeamPolicy<>(num_first_ocur, Kokkos::AUTO()), KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
    uint32_t i = team_member.league_rank();
    uint32_t node=0;
    // Identify chunk and mark entry in node list
    if(team_member.team_rank() == 0) {
      memcpy(&node, first_ocur_subview.data() + static_cast<uint64_t>(i)*sizeof(uint32_t), sizeof(uint32_t));
      first_occur_map.insert(NodeID(node, cur_id), static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size));
      node_list(node) = NodeID(node, cur_id);
    }
    team_member.team_broadcast(node, 0); /// Share node ID with other threads

    // Calculate offsets for coying chunk data
    uint64_t src_offset = static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);
    uint64_t dst_offset = static_cast<uint64_t>(node)*static_cast<uint64_t>(chunk_size);
    uint32_t datasize = chunk_size;
    if(node == num_chunks-1)
      datasize = datalen - dst_offset;

    // Cooperative team copy of chunk data
    uint8_t* src = (uint8_t*)(data_subview.data()+src_offset);
    uint8_t* dst = (uint8_t*)(data.data()+dst_offset);
    team_memcpy(dst, src, datasize, team_member);
  });

  Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
  auto repeat_region_sizes_h = Kokkos::create_mirror_view(repeat_region_sizes);
  Kokkos::deep_copy(repeat_region_sizes, 0);

  // Read map of repeats for each checkpoint
  Kokkos::parallel_for("Load repeat counts", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_CLASS_LAMBDA(const uint64_t i) {
    uint32_t chkpt;
    memcpy(&chkpt, dupl_count_subview.data()+i*2*sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&repeat_region_sizes(chkpt), 
           dupl_count_subview.data()+i*2*sizeof(uint32_t)+sizeof(uint32_t), 
           sizeof(uint32_t));
    STDOUT_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
  });
  Kokkos::deep_copy(repeat_region_sizes_h, repeat_region_sizes);

  // Perform exclusive scan to determine where regions of shifted duplicates start/stop
  Kokkos::parallel_scan("Repeat offsets", cur_id+1, KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
    partial_sum += repeat_region_sizes(i);
    if(is_final) repeat_region_sizes(i) = partial_sum;
  });
  STDOUT_PRINT("Num repeats: %u\n", num_shift_dupl);
  Kokkos::fence();
  STDOUT_PRINT("Done determining where things belong\n");
  STDOUT_PRINT("Size of hash table: %u\n", first_occur_map.size());

  // Load repeat entries and fill in metadata for chunks
  Kokkos::parallel_for("Restart Hashlist shifted duplicates", Kokkos::TeamPolicy<>(num_shift_dupl, Kokkos::AUTO()), KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
    uint32_t i = team_member.league_rank();
    uint32_t node=0, prev, tree=0;
    size_t src_offset = 0;
    // Load metadata and copy to all threads
    if(team_member.team_rank() == 0) {
      memcpy(&node, shift_dupl_subview.data() + static_cast<uint64_t>(i)*2*sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&prev, shift_dupl_subview.data() + static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
      for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
        if(i < repeat_region_sizes(j)) {
          tree = j;
        }
      }
      uint32_t idx = first_occur_map.find(NodeID(prev, tree));
      if(first_occur_map.valid_at(idx)) {
        src_offset = first_occur_map.value_at(idx);
      }
      node_list(node) = NodeID(prev, tree);
    }
    team_member.team_broadcast(node, 0);
    team_member.team_broadcast(prev, 0);
    team_member.team_broadcast(tree, 0);
    team_member.team_broadcast(src_offset, 0);
    // Check if the first occurrence of the thunk is for this checkpoint
    if(tree == cur_id) {
      uint32_t datasize = chunk_size;
      size_t dst_offset = static_cast<uint64_t>(node) * static_cast<uint64_t>(chunk_size);
      if(node == num_chunks-1)
        datasize = datalen - dst_offset;

      // Cooperative Team copy with threads
      uint8_t* src = (uint8_t*)(data_subview.data()+src_offset);
      uint8_t* dst = (uint8_t*)(data.data()+dst_offset);
      team_memcpy(dst, src, datasize, team_member);
    }
  });

  Kokkos::fence();
  STDOUT_PRINT("Restarted first occurrences\n");

  // Mark any entries untouched so far as fixed duplicates
  Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    NodeID entry = node_list(i);
    if(entry.node == UINT_MAX) {
      node_list(i) = NodeID(i, cur_id-1);
    }
  });
  Kokkos::fence();

  STDOUT_PRINT("Filled remaining entries\n");

  // Iterate through checkpoints in reverse. Fill in rest of chunks
  for(int idx=static_cast<int>(chkpt_idx)-1; idx>=static_cast<int>(ref_id) && idx < chkpt_idx; idx--) {
    STDOUT_PRINT("Processing checkpoint %u\n", idx);
    // Load checkpoint and read header
    chkpt_size = incr_chkpts[idx].size();
    Kokkos::View<uint8_t*> chkpt_buffer_d("Checkpoint buffer", chkpt_size);
    auto& chkpt_buffer_h = incr_chkpts[idx];
    header_t chkpt_header;
    memcpy(&chkpt_header, chkpt_buffer_h.data(), sizeof(header_t));
    uint32_t current_id = chkpt_header.chkpt_id;
    datalen = chkpt_header.datalen;
    chunk_size = chkpt_header.chunk_size;
    // Copy checkpoint to device
    Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);

    // Update header values
    ref_id = chkpt_header.ref_id;
    cur_id = chkpt_header.chkpt_id;
    datalen = chkpt_header.datalen;
    chunk_size = chkpt_header.chunk_size;
    num_first_ocur = chkpt_header.num_first_ocur;
    num_prior_chkpts = chkpt_header.num_prior_chkpts;
    num_shift_dupl = chkpt_header.num_shift_dupl;

    STDOUT_PRINT("Ref ID:           %u\n",  chkpt_header.ref_id);
    STDOUT_PRINT("Chkpt ID:         %u\n",  chkpt_header.chkpt_id);
    STDOUT_PRINT("Datalen:          %lu\n", chkpt_header.datalen);
    STDOUT_PRINT("Chunk size:       %u\n",  chkpt_header.chunk_size);
    STDOUT_PRINT("Distinct size:    %u\n",  chkpt_header.num_first_ocur);
    STDOUT_PRINT("Num prior chkpts: %u\n",  chkpt_header.num_prior_chkpts);
    STDOUT_PRINT("Num shift dupl:   %u\n",  chkpt_header. num_shift_dupl);

    // Calculate and create subviews for each section of the incremental checkpoint
    first_ocur_offset = sizeof(header_t);
    dupl_count_offset = first_ocur_offset + static_cast<uint64_t>(num_first_ocur)*sizeof(uint32_t);
    dupl_map_offset   = dupl_count_offset + static_cast<uint64_t>(num_prior_chkpts)*2*sizeof(uint32_t);
    data_offset       = dupl_map_offset   + static_cast<uint64_t>(num_shift_dupl)*2*sizeof(uint32_t);
    first_ocur_subview    = Kokkos::subview(chkpt_buffer_d, std::make_pair(first_ocur_offset, dupl_count_offset));
    dupl_count_subview    = Kokkos::subview(chkpt_buffer_d, std::make_pair(dupl_count_offset, dupl_map_offset));
    shift_dupl_subview    = Kokkos::subview(chkpt_buffer_d, std::make_pair(dupl_map_offset, data_offset));
    data_subview  = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, chkpt_size));
    STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", chkpt_size);
    STDOUT_PRINT("First ocur offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Dupl count offset: %lu\n", dupl_count_offset);
    STDOUT_PRINT("Dupl map offset: %lu\n", dupl_map_offset);
    STDOUT_PRINT("Data offset: %lu\n", data_offset);

    // Clear first occurrence map 
    first_occur_map.clear();
    first_occur_map.rehash(num_first_ocur);
    // Create map for repeats
    Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(num_shift_dupl);
    
    // Fill in first occurrences
    Kokkos::parallel_for("Fill first_ocur map", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, first_ocur_subview.data()+static_cast<uint64_t>(i)*sizeof(uint32_t), sizeof(uint32_t));
      first_occur_map.insert(NodeID(node,cur_id), static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size));
    });
    // Read # of duplicates for each prior checkpoint
    Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
    Kokkos::parallel_for("Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      uint32_t chkpt;
      memcpy(&chkpt, chkpt_buffer_d.data()+dupl_count_offset+static_cast<uint64_t>(i)*2*sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&repeat_region_sizes(chkpt), chkpt_buffer_d.data()+dupl_count_offset+static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
      STDOUT_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
    });

    // Perform prefix scan to get the offsets for each prior checkpoint
    Kokkos::parallel_scan("Repeat offsets", cur_id+1, KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      partial_sum += repeat_region_sizes(i);
      if(is_final) repeat_region_sizes(i) = partial_sum;
    });

    STDOUT_PRINT("Num repeats: %u\n", num_shift_dupl);
  
    // Load map of shifted duplicates
    Kokkos::parallel_for("Restart Hash tree repeats middle chkpts", Kokkos::RangePolicy<>(0,num_shift_dupl), KOKKOS_CLASS_LAMBDA(const uint32_t i) { 
      uint32_t node, prev, tree=0;
      memcpy(&node, shift_dupl_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&prev, shift_dupl_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
      for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
        if(i < repeat_region_sizes(j)) {
          tree = j;
        }
      }
      auto result = repeat_map.insert(node, NodeID(prev,tree));
      if(result.failed())
        STDOUT_PRINT("Failed to insert previous repeat %u: (%u,%u) into repeat map\n", node, prev, tree);
    });

    // Restart data
    Kokkos::parallel_for("Restart Hashlist first occurrence", Kokkos::TeamPolicy<>(num_chunks, Kokkos::AUTO()), KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
      uint32_t i = team_member.league_rank();
      if(node_list(i).tree == current_id) { /// Check if entry is for the current checkpoint
        NodeID id = node_list(i);
        if(first_occur_map.exists(id)) { /// Restart first occurrences
          size_t src_offset = 0;
          if(team_member.team_rank() == 0) {
            src_offset = first_occur_map.value_at(first_occur_map.find(id));
          }
          team_member.team_broadcast(src_offset, 0);
          size_t dst_offset = static_cast<size_t>(i)*static_cast<size_t>(chunk_size);
          uint32_t writesize = chunk_size;
          if(dst_offset+writesize > datalen) 
            writesize = datalen-dst_offset;

          // Copy data with cooperative team copy
          uint8_t* src = (uint8_t*)(data_subview.data()+src_offset);
          uint8_t* dst = (uint8_t*)(data.data()+dst_offset);
          team_memcpy(dst, src, writesize, team_member);
        } else if(repeat_map.exists(id.node)) { /// Restart shifted duplicates
          NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
          DEBUG_PRINT("Repaeat value: %u: (%u,%u)\n", id.node, prev.node, prev.tree);
          if(prev.tree == current_id) { /// Check if the first occurrence is for this checkpoint
            if(!repeat_map.exists(id.node))
              printf("Failed to find repeat chunk %u\n", id.node);
            // Calculate the source and destination offsets
            size_t src_offset;
            if(team_member.team_rank() == 0) {
              src_offset = first_occur_map.value_at(first_occur_map.find(prev));
            }
            team_member.team_broadcast(src_offset, 0);
            size_t dst_offset = static_cast<size_t>(i)*static_cast<size_t>(chunk_size);
            uint32_t writesize = chunk_size;
            if(dst_offset+writesize > datalen) 
              writesize = datalen-dst_offset;

            // Copy data with cooperative team copy
            uint8_t* src = (uint8_t*)(data_subview.data()+src_offset);
            uint8_t* dst = (uint8_t*)(data.data()+dst_offset);
            team_memcpy(dst, src, writesize, team_member);
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

/**
 * Restart data from incremental checkpoints stored in files.
 *
 * \param incr_chkpts Vector of filenames containing the diffs
 * \param chkpt_idx   ID of which checkpoint to restart
 * \param data        View for restarting the checkpoint to
 *
 * \return Time spent copying incremental checkpoints from host to device and restarting data
 */
std::pair<double,double>
ListDeduplicator::restart_chkpt( std::vector<std::string>& chkpt_files,
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

  uint32_t num_chunks = static_cast<uint32_t>(header.datalen / static_cast<uint64_t>(header.chunk_size));
  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(header.chunk_size) < header.datalen) {
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
  size_t datalen = header.datalen;
  uint32_t chunk_size = header.chunk_size;
  uint32_t num_first_ocur = header.num_first_ocur;
  uint32_t num_prior_chkpts = header.num_prior_chkpts;
  uint32_t num_shift_dupl = header.num_shift_dupl;

  STDOUT_PRINT("Ref ID:           %u\n",  header.ref_id);
  STDOUT_PRINT("Chkpt ID:         %u\n",  header.chkpt_id);
  STDOUT_PRINT("Datalen:          %lu\n", header.datalen);
  STDOUT_PRINT("Chunk size:       %u\n",  header.chunk_size);
  STDOUT_PRINT("Distinct size:    %u\n",  header.num_first_ocur);
  STDOUT_PRINT("Num prior chkpts: %u\n",  header.num_prior_chkpts);
  STDOUT_PRINT("Num shift dupl:   %u\n",  header. num_shift_dupl);

  size_t first_ocur_offset = sizeof(header_t);
  size_t dupl_count_offset = first_ocur_offset + num_first_ocur*sizeof(uint32_t);
  size_t dupl_map_offset = dupl_count_offset + num_prior_chkpts*2*sizeof(uint32_t);
  size_t data_offset = dupl_map_offset + num_shift_dupl*2*sizeof(uint32_t);
  auto first_ocur_subview    = Kokkos::subview(buffer_d, std::make_pair(first_ocur_offset, dupl_count_offset));
  auto dupl_count_subview    = Kokkos::subview(buffer_d, std::make_pair(dupl_count_offset, dupl_map_offset));
  auto shift_dupl_subview    = Kokkos::subview(buffer_d, std::make_pair(dupl_map_offset, data_offset));
  auto data_subview  = Kokkos::subview(buffer_d, std::make_pair(data_offset, filesize));
  STDOUT_PRINT("Checkpoint %u\n", header.chkpt_id);
  STDOUT_PRINT("Checkpoint size: %lu\n", filesize);
  STDOUT_PRINT("First ocur offset: %lu\n", sizeof(header_t));
  STDOUT_PRINT("Dupl count offset: %lu\n", dupl_count_offset);
  STDOUT_PRINT("Dupl map offset: %lu\n", dupl_map_offset);
  STDOUT_PRINT("Data offset: %lu\n", data_offset);

  Kokkos::UnorderedMap<NodeID, size_t> first_ocur_map(num_first_ocur);
  Kokkos::parallel_for("Restart Hashlist distinct", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    uint32_t node;
    memcpy(&node, first_ocur_subview.data() + i*(sizeof(uint32_t)),  sizeof(uint32_t));
    first_ocur_map.insert(NodeID(node,cur_id),  i*chunk_size);
    node_list(node) = NodeID(node, cur_id);
    uint32_t datasize = chunk_size;
    if(node == num_chunks-1)
      datasize = datalen - node*chunk_size;
    memcpy(data.data()+chunk_size*node, data_subview.data()+i*chunk_size, datasize);
  });

  Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
  auto repeat_region_sizes_h = Kokkos::create_mirror_view(repeat_region_sizes);
  Kokkos::deep_copy(repeat_region_sizes, 0);
  // Read map of repeats for each checkpoint
  Kokkos::parallel_for("Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    uint32_t chkpt;
    memcpy(&chkpt, dupl_count_subview.data()+i*2*sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&repeat_region_sizes(chkpt), dupl_count_subview.data()+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
    STDOUT_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
  });
  Kokkos::deep_copy(repeat_region_sizes_h, repeat_region_sizes);
  // Perform exclusive scan to determine where regions start/stop
  Kokkos::parallel_scan("Repeat offsets", cur_id+1, KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
    partial_sum += repeat_region_sizes(i);
    if(is_final) repeat_region_sizes(i) = partial_sum;
  });
  STDOUT_PRINT("Num repeats: %u\n", num_shift_dupl);
  // Load repeat entries and fill in metadata for chunks
  Kokkos::parallel_for("Restart Hash tree repeats main checkpoint", Kokkos::RangePolicy<>(0, num_shift_dupl), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    uint32_t node;
    uint32_t prev;
    uint32_t tree = 0;
    memcpy(&node, shift_dupl_subview.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
    memcpy(&prev, shift_dupl_subview.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
    // Determine ID 
    for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
      if(i < repeat_region_sizes(j)) {
        tree = j;
      }
    }
    size_t offset = first_ocur_map.value_at(first_ocur_map.find(NodeID(prev, tree)));
    node_list(node) = NodeID(prev, tree);
    if(tree == cur_id) {
      uint32_t copysize = chunk_size;
      if(node == num_chunks-1)
        copysize = data.size() - chunk_size*node;
      memcpy(data.data()+chunk_size*node, data_subview.data()+offset, copysize);
    }
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
    uint32_t current_id = chkpt_header.chkpt_id;
    datalen = chkpt_header.datalen;
    chunk_size = chkpt_header.chunk_size;
    Kokkos::deep_copy(chkpt_buffer_d, chkpt_buffer_h);

    ref_id = chkpt_header.ref_id;
    cur_id = chkpt_header.chkpt_id;
    datalen = chkpt_header.datalen;
    chunk_size = chkpt_header.chunk_size;
    num_first_ocur = chkpt_header.num_first_ocur;
    num_prior_chkpts = chkpt_header.num_prior_chkpts;
    num_shift_dupl = chkpt_header.num_shift_dupl;

    STDOUT_PRINT("Ref ID:           %u\n",  chkpt_header.ref_id);
    STDOUT_PRINT("Chkpt ID:         %u\n",  chkpt_header.chkpt_id);
    STDOUT_PRINT("Datalen:          %lu\n", chkpt_header.datalen);
    STDOUT_PRINT("Chunk size:       %u\n",  chkpt_header.chunk_size);
    STDOUT_PRINT("Distinct size:    %u\n",  chkpt_header.num_first_ocur);
    STDOUT_PRINT("Num prior chkpts: %u\n",  chkpt_header.num_prior_chkpts);
    STDOUT_PRINT("Num shift dupl:   %u\n",  chkpt_header. num_shift_dupl);

    first_ocur_offset = sizeof(header_t);
    dupl_count_offset = first_ocur_offset + num_first_ocur*sizeof(uint32_t);
    dupl_map_offset = dupl_count_offset + num_prior_chkpts*2*sizeof(uint32_t);
    data_offset = dupl_map_offset + num_shift_dupl*2*sizeof(uint32_t);
    first_ocur_subview = Kokkos::subview(buffer_d,std::make_pair(first_ocur_offset,dupl_count_offset));
    dupl_count_subview = Kokkos::subview(buffer_d,std::make_pair(dupl_count_offset,dupl_map_offset));
    shift_dupl_subview = Kokkos::subview(buffer_d,std::make_pair(dupl_map_offset, data_offset));
    data_subview  = Kokkos::subview(buffer_d, std::make_pair(data_offset, filesize));
    STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", filesize);
    STDOUT_PRINT("First ocur offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Dupl count offset: %lu\n", dupl_count_offset);
    STDOUT_PRINT("Dupl map offset: %lu\n", dupl_map_offset);
    STDOUT_PRINT("Data offset: %lu\n", data_offset);

    first_ocur_map.clear();
    first_ocur_map.rehash(num_first_ocur);
    Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(num_shift_dupl);
    Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, first_ocur_subview.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      first_ocur_map.insert(NodeID(node,cur_id), i*chunk_size);
    });
    Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
    Kokkos::parallel_for("Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      uint32_t chkpt;
      memcpy(&chkpt, dupl_count_subview.data()+i*2*sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&repeat_region_sizes(chkpt), dupl_count_subview.data()+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
      STDOUT_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
    });
    Kokkos::parallel_scan("Repeat offsets", cur_id+1, KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      partial_sum += repeat_region_sizes(i);
      if(is_final) repeat_region_sizes(i) = partial_sum;
    });

    STDOUT_PRINT("Num repeats: %u\n", num_shift_dupl);
  
    Kokkos::parallel_for("Restart Hash tree repeats middle chkpts", Kokkos::RangePolicy<>(0,num_shift_dupl), KOKKOS_CLASS_LAMBDA(const uint32_t i) { 
      uint32_t node, prev, tree=0;
      memcpy(&node, shift_dupl_subview.data()+i*2*sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&prev, shift_dupl_subview.data()+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
      for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
        if(i < repeat_region_sizes(j)) {
          tree = j;
        }
      }
      auto result = repeat_map.insert(node, NodeID(prev,tree));
      if(result.failed())
        STDOUT_PRINT("Failed to insert previous repeat %u: (%u,%u) into repeat map\n", node, prev, tree);
    });

    Kokkos::parallel_for("Fill data middle chkpts", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      if(node_list(i).tree == current_id) {
        NodeID id = node_list(i);
        if(first_ocur_map.exists(id)) {
          size_t offset = first_ocur_map.value_at(first_ocur_map.find(id));
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
            size_t offset = first_ocur_map.value_at(first_ocur_map.find(prev));
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

  Kokkos::fence();
  std::chrono::high_resolution_clock::time_point c3 = std::chrono::high_resolution_clock::now();
  double copy_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
  double restart_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c3-c2).count());
  return std::make_pair(copy_time, restart_time);
}

void 
ListDeduplicator::checkpoint( header_t& header, 
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
    first_ocur_d = DigestNodeIDDeviceMap(num_chunks);
    first_ocur_vec = Vector<uint32_t>(num_chunks);
    shift_dupl_vec = Vector<uint32_t>(num_chunks);
  }
  first_ocur_vec.clear();
  shift_dupl_vec.clear();
  if(list.list_d.size() < num_chunks) {
    Kokkos::resize(list.list_d, num_chunks);
    Kokkos::resize(list.list_h, num_chunks);
  }
  if(first_ocur_d.capacity() < first_ocur_d.size()+num_chunks)
    first_ocur_d.rehash(first_ocur_d.size()+num_chunks);

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
ListDeduplicator::checkpoint(uint8_t* data_ptr, 
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
ListDeduplicator::checkpoint(uint8_t* data_ptr, 
                              size_t len, 
                              Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                              bool make_baseline) {
  header_t header;
  checkpoint(header, data_ptr, len, diff_h, make_baseline);
  current_id += 1;
}

void 
ListDeduplicator::checkpoint(uint8_t* data_ptr, 
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
ListDeduplicator::restart(Kokkos::View<uint8_t*> data, 
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
ListDeduplicator::restart(uint8_t* data_ptr, 
                           size_t len, 
                           std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                           std::string& logname, 
                           uint32_t chkpt_id) {
  Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
  restart(data, chkpts, logname, chkpt_id);
}

void 
ListDeduplicator::restart(Kokkos::View<uint8_t*> data, 
                           std::vector<std::string>& chkpt_filenames, 
                           std::string& logname, 
                           uint32_t chkpt_id) {
  std::vector<std::string> hashlist_chkpt_files;
  for(uint32_t i=0; i<chkpt_filenames.size(); i++) {
    hashlist_chkpt_files.push_back(chkpt_filenames[i]+".hashlist.incr_chkpt");
  }
  auto list_times = restart_chkpt(hashlist_chkpt_files, chkpt_id, data);
  restart_timers[0] = list_times.first;
  restart_timers[1] = list_times.second;
  write_restart_log(chkpt_id, logname);
}

void 
ListDeduplicator::write_chkpt_log(header_t& header, 
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

  result_data << "List" << "," // Approach
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
  timing_file << "List" << "," 
              << current_id << "," 
              << chunk_size << "," 
              << timers[1] << "," // Comparison time 
              << timers[2] << "," // Collection time 
              << timers[3]          // Write time
              << std::endl;
  size_file << "List" << "," 
            << current_id << "," 
            << chunk_size << "," 
            << datasizes.first << "," 
            << datasizes.second << ",";
  write_metadata_breakdown(size_file, DedupMode::List, header, diff_h, current_id);
}

/**
 * Function for writing the restart log.
 *
 * \param select_chkpt Which checkpoint to write the log
 * \param logname      Filename for writing log
 */
void 
ListDeduplicator::write_restart_log(uint32_t select_chkpt, std::string& logname) {
  std::fstream timing_file;
  timing_file.open(logname, std::fstream::out | std::fstream::app);
  timing_file << "List" << ","; 
  timing_file << select_chkpt << "," 
              << chunk_size << "," 
              << restart_timers[0] << "," 
              << restart_timers[1] << std::endl;
  timing_file.close();
}
