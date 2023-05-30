#ifndef LIST_APPROACH_HPP
#define LIST_APPROACH_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_Bitset.hpp>
#include <climits>
#include <chrono>
#include <fstream>
#include <vector>
#include <utility>
#include "kokkos_hash_list.hpp"
#include "hash_functions.hpp"
#include "map_helpers.hpp"
#include "utils.hpp"

class ListDeduplicator : public BaseDeduplicator {
  private:
    HashList list;
    DigestNodeIDDeviceMap first_ocur_d; // Map of first occurrences
    Vector<uint32_t> first_ocur_vec; // First occurrence root offsets
    Vector<uint32_t> shift_dupl_vec; // Shifted duplicate root offsets
    uint32_t num_chunks;

    void dedup_data(const uint8_t* data_ptr, 
                    const size_t len);

    std::pair<uint64_t,uint64_t> 
    collect_diff( const uint8_t* data_ptr, 
                  const size_t len,
                  Kokkos::View<uint8_t*>& buffer_d, 
                  header_t& header);

    std::pair<double,double>
    restart_chkpt( std::vector<Kokkos::View<uint8_t*>::HostMirror>& incr_chkpts,
                   const int chkpt_idx, 
                   Kokkos::View<uint8_t*>& data);

    std::pair<double,double>
    restart_chkpt( std::vector<std::string>& chkpt_files,
                   const int file_idx, 
                   Kokkos::View<uint8_t*>& data);
  public:
    ListDeduplicator();

    ListDeduplicator(uint32_t bytes_per_chunk);

    ~ListDeduplicator() override;

    /**
     * Main checkpointing function. Given a Kokkos View, create an incremental checkpoint using 
     * the chosen checkpoint strategy. The deduplication mode can be one of the following:
     *   - Full: No deduplication
     *   - Basic: Remove chunks that have not changed since the previous checkpoint
     *   - List: Save a single copy of each unique chunk and use metadata to handle duplicates
     *   - Tree: Save minimal set of chunks and use a compact metadata representations
     *
     * \param header        The checkpoint header
     * \param data_ptr      Data to be checkpointed
     * \param data_len      Length of data in bytes
     * \param diff_h        The output incremental checkpoint on the Host
     * \param make_baseline Flag determining whether to make a baseline checkpoint
     */
    void checkpoint(header_t& header, 
                    uint8_t* data_ptr, 
                    size_t data_len,
                    Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                    bool make_baseline) override ;

    /**
     * Main checkpointing function. Given a raw device pointer, create an incremental checkpoint 
     * using the chosen checkpoint strategy. The deduplication mode can be one of the following:
     *   - Full: No deduplication
     *   - Basic: Remove chunks that have not changed since the previous checkpoint
     *   - List: Save a single copy of each unique chunk and use metadata to handle duplicates
     *   - Tree: Save minimal set of chunks and use a compact metadata representations
     *
     * \param data_ptr      Raw data pointer that needs to be deduplicated
     * \param len           Length of data
     * \param filename      Filename to save checkpoint
     * \param logname       Base filename for logs
     * \param make_baseline Flag determining whether to make a baseline checkpoint
     */
    void checkpoint(uint8_t* data_ptr, 
                    size_t len, 
                    std::string& filename, 
                    std::string& logname, 
                    bool make_baseline) override;

    /**
     * Main checkpointing function. Given a raw device pointer, create an incremental checkpoint 
     * using the chosen checkpoint strategy. Save checkpoint to host view. 
     * The deduplication mode can be one of the following:
     *   - Full: No deduplication
     *   - Basic: Remove chunks that have not changed since the previous checkpoint
     *   - List: Save a single copy of each unique chunk and use metadata to handle duplicates
     *   - Tree: Save minimal set of chunks and use a compact metadata representations
     *
     * \param data_ptr      Raw data pointer that needs to be deduplicated
     * \param len           Length of data
     * \param diff_h        Host View to store incremental checkpoint
     * \param make_baseline Flag determining whether to make a baseline checkpoint
     */
    void checkpoint(uint8_t* data_ptr, 
                    size_t len, 
                    Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                    bool make_baseline) override;

    /**
     * Main checkpointing function. Given a raw device pointer, create an incremental checkpoint 
     * using the chosen checkpoint strategy. Save checkpoint to host view and write logs.
     * The deduplication mode can be one of the following:
     *   - Full: No deduplication
     *   - Basic: Remove chunks that have not changed since the previous checkpoint
     *   - List: Save a single copy of each unique chunk and use metadata to handle duplicates
     *   - Tree: Save minimal set of chunks and use a compact metadata representations
     *
     * \param data_ptr      Raw data pointer that needs to be deduplicated
     * \param len           Length of data
     * \param diff_h        Host View to store incremental checkpoint
     * \param logname       Base filename for logs
     * \param make_baseline Flag determining whether to make a baseline checkpoint
     */
    void checkpoint(uint8_t* data_ptr, 
                    size_t len, 
                    Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                    std::string& logname, 
                    bool make_baseline) override;

    /**
     * Restart checkpoint from vector of incremental checkpoints loaded on the Host.
     *
     * \param data       Data View to restart checkpoint into
     * \param chkpts     Vector of prior incremental checkpoints stored on the Host
     * \param logname    Filename for restart logs
     * \param chkpt_id   ID of checkpoint to restart
     */
    void restart(Kokkos::View<uint8_t*> data, 
                 std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                 std::string& logname, 
                 uint32_t chkpt_id) override;

    /**
     * Restart checkpoint from vector of incremental checkpoints loaded on the Host. 
     * Store result into raw device pointer.
     *
     * \param data_ptr   Device pointer to save checkpoint in
     * \param len        Length of data
     * \param chkpts     Vector of prior incremental checkpoints stored on the Host
     * \param logname    Filename for restart logs
     * \param chkpt_id   ID of checkpoint to restart
     */
    void restart(uint8_t* data_ptr, 
                 size_t len, 
                 std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                 std::string& logname, 
                 uint32_t chkpt_id) override;

    /**
     * Restart checkpoint from checkpoint files
     *
     * \param data       Data View to restart checkpoint into
     * \param filenames  Vector of prior incremental checkpoints stored in files
     * \param logname    Filename for restart logs
     * \param chkpt_id   ID of checkpoint to restart
     */
    void restart(Kokkos::View<uint8_t*> data, 
                 std::vector<std::string>& chkpt_filenames, 
                 std::string& logname, 
                 uint32_t chkpt_id) override;

    /**
     * Write logs for the checkpoint metadata/data breakdown, runtimes, and the overall summary.
     * The data breakdown log shows the proportion of data and metadata as well as how much 
     * metadata corresponds to each prior checkpoint.
     * The timing log contains the time spent comparing chunks, gathering scattered chunks,
     * and the time spent copying the resulting checkpoint from the device to host.
     *
     * \param header  The checkpoint header
     * \param diff_h  The incremental checkpoint
     * \param logname Base filename for the logs
     */
    void write_chkpt_log(header_t& header, 
                         Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                         std::string& logname) override;
    /**
     * Function for writing the restart log.
     *
     * \param select_chkpt Which checkpoint to write the log
     * \param logname      Filename for writing log
     */
    void write_restart_log(uint32_t select_chkpt, 
                           std::string& logname) override;
};


/**
 * Deduplicate provided data view using the list incremental checkpoint approach. 
 * Split data into chunks and compute hashes for each chunk. Compare each hash with 
 * the hash at the same offset. If the hash has never been seen, save the chunk.
 * If the hash has been seen before, mark it as a shifted duplicate and save metadata.
 *
 * \param list          List of hashes for identifying differences
 * \param list_id       ID of current checkpoint
 * \param data          View of data for deduplicating
 * \param chunk_size    Size in bytes for splitting data into chunks
 * \param first_occur_d Map for tracking first occurrence hashes
 * \param first_ocur    Vector of first occurrence chunks
 * \param shift_dupl    Vector of shifted duplicate chunks
 */
void dedup_data_list( const HashList& list, 
                      const uint32_t list_id, 
                      const uint8_t* data_ptr, 
                      const size_t data_size, 
                      const uint32_t chunk_size, 
                      DigestNodeIDDeviceMap& first_occur_d,
                      Vector<uint32_t> first_ocur,
                      Vector<uint32_t> shift_dupl);

/**
 * Gather the scattered chunks for the diff and write the checkpoint to a contiguous buffer.
 *
 * \param data           Data View containing data to deduplicate
 * \param buffer_d       View to store the diff in
 * \param chunk_size     Size of chunks in bytes
 * \param list           List of hash digests
 * \param first_occur_d  Map for tracking first occurrence hashes
 * \param first_ocur     Vector of first occurrence chunks
 * \param shift_dupl     Vector of shifted duplicate chunks
 * \param prior_chkpt_id ID of the last checkpoint
 * \param chkpt_id       ID for the current checkpoint
 * \param header         Incremental checkpoint header
 *
 * \return Pair containing amount of data and metadata in the checkpoint
 */
std::pair<uint64_t,uint64_t> 
write_diff_list(
                const uint8_t* data_ptr, 
                const size_t data_size, 
                Kokkos::View<uint8_t*>& buffer_d, 
                uint32_t chunk_size, 
                const HashList& list, 
                const DigestNodeIDDeviceMap& first_occur_d, 
                Vector<uint32_t>& first_ocur,
                Vector<uint32_t>& shift_dupl,
                uint32_t prior_chkpt_id,
                uint32_t chkpt_id,
                header_t& header);

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
restart_chkpt_list( std::vector<Kokkos::View<uint8_t*>::HostMirror >& incr_chkpts,
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
  Kokkos::parallel_for("Restart Hashlist first occurrence", Kokkos::TeamPolicy<>(num_first_ocur, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
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
  Kokkos::parallel_for("Load repeat counts", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_LAMBDA(const uint64_t i) {
    uint32_t chkpt;
    memcpy(&chkpt, dupl_count_subview.data()+i*2*sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&repeat_region_sizes(chkpt), 
           dupl_count_subview.data()+i*2*sizeof(uint32_t)+sizeof(uint32_t), 
           sizeof(uint32_t));
    STDOUT_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
  });
  Kokkos::deep_copy(repeat_region_sizes_h, repeat_region_sizes);

  // Perform exclusive scan to determine where regions of shifted duplicates start/stop
  Kokkos::parallel_scan("Repeat offsets", cur_id+1, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
    partial_sum += repeat_region_sizes(i);
    if(is_final) repeat_region_sizes(i) = partial_sum;
  });
  STDOUT_PRINT("Num repeats: %u\n", num_shift_dupl);
  Kokkos::fence();
  STDOUT_PRINT("Done determining where things belong\n");
  STDOUT_PRINT("Size of hash table: %u\n", first_occur_map.size());

  // Load repeat entries and fill in metadata for chunks
  Kokkos::parallel_for("Restart Hashlist shifted duplicates", Kokkos::TeamPolicy<>(num_shift_dupl, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
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
  Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
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
    Kokkos::parallel_for("Fill first_ocur map", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, first_ocur_subview.data()+static_cast<uint64_t>(i)*sizeof(uint32_t), sizeof(uint32_t));
      first_occur_map.insert(NodeID(node,cur_id), static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size));
    });
    // Read # of duplicates for each prior checkpoint
    Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
    Kokkos::parallel_for("Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t chkpt;
      memcpy(&chkpt, chkpt_buffer_d.data()+dupl_count_offset+static_cast<uint64_t>(i)*2*sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&repeat_region_sizes(chkpt), chkpt_buffer_d.data()+dupl_count_offset+static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
      STDOUT_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
    });

    // Perform prefix scan to get the offsets for each prior checkpoint
    Kokkos::parallel_scan("Repeat offsets", cur_id+1, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      partial_sum += repeat_region_sizes(i);
      if(is_final) repeat_region_sizes(i) = partial_sum;
    });

    STDOUT_PRINT("Num repeats: %u\n", num_shift_dupl);
  
    // Load map of shifted duplicates
    Kokkos::parallel_for("Restart Hash tree repeats middle chkpts", Kokkos::RangePolicy<>(0,num_shift_dupl), KOKKOS_LAMBDA(const uint32_t i) { 
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
    Kokkos::parallel_for("Restart Hashlist first occurrence", Kokkos::TeamPolicy<>(num_chunks, Kokkos::AUTO()), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
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
restart_chkpt_list( std::vector<std::string>& chkpt_files,
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
  Kokkos::parallel_for("Restart Hashlist distinct", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_LAMBDA(const uint32_t i) {
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
  Kokkos::parallel_for("Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_LAMBDA(const uint32_t i) {
    uint32_t chkpt;
    memcpy(&chkpt, dupl_count_subview.data()+i*2*sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&repeat_region_sizes(chkpt), dupl_count_subview.data()+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
    STDOUT_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
  });
  Kokkos::deep_copy(repeat_region_sizes_h, repeat_region_sizes);
  // Perform exclusive scan to determine where regions start/stop
  Kokkos::parallel_scan("Repeat offsets", cur_id+1, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
    partial_sum += repeat_region_sizes(i);
    if(is_final) repeat_region_sizes(i) = partial_sum;
  });
  STDOUT_PRINT("Num repeats: %u\n", num_shift_dupl);
  // Load repeat entries and fill in metadata for chunks
  Kokkos::parallel_for("Restart Hash tree repeats main checkpoint", Kokkos::RangePolicy<>(0, num_shift_dupl), KOKKOS_LAMBDA(const uint32_t i) {
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

  Kokkos::parallel_for("Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
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
    Kokkos::parallel_for("Fill distinct map", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, first_ocur_subview.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      first_ocur_map.insert(NodeID(node,cur_id), i*chunk_size);
    });
    Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
    Kokkos::parallel_for("Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_LAMBDA(const uint32_t i) {
      uint32_t chkpt;
      memcpy(&chkpt, dupl_count_subview.data()+i*2*sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&repeat_region_sizes(chkpt), dupl_count_subview.data()+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
      STDOUT_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
    });
    Kokkos::parallel_scan("Repeat offsets", cur_id+1, KOKKOS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      partial_sum += repeat_region_sizes(i);
      if(is_final) repeat_region_sizes(i) = partial_sum;
    });

    STDOUT_PRINT("Num repeats: %u\n", num_shift_dupl);
  
    Kokkos::parallel_for("Restart Hash tree repeats middle chkpts", Kokkos::RangePolicy<>(0,num_shift_dupl), KOKKOS_LAMBDA(const uint32_t i) { 
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

    Kokkos::parallel_for("Fill data middle chkpts", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_LAMBDA(const uint32_t i) {
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

#endif // LIST_APPROACH_HPP
