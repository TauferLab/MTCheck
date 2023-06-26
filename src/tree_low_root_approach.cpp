#include "tree_approach.hpp"

//=================================================================================================
// Tree Low Root
//=================================================================================================
TreeLowRootDeduplicator::TreeLowRootDeduplicator() {}

TreeLowRootDeduplicator::TreeLowRootDeduplicator(uint32_t bytes_per_chunk) {
  chunk_size = bytes_per_chunk;
  current_id = 0;
  baseline_id = 0;
}

TreeLowRootDeduplicator::~TreeLowRootDeduplicator() {}

void 
TreeLowRootDeduplicator::dedup_data_low_root(const uint8_t* data_ptr, 
                                    const size_t data_size) {
  Kokkos::View<uint64_t[4]> chunk_counters("Chunk counters");
  Kokkos::View<uint64_t[4]> region_counters("Region counters");
  Kokkos::deep_copy(chunk_counters, 0);
  Kokkos::deep_copy(region_counters, 0);
  auto chunk_counters_h  = Kokkos::create_mirror_view(chunk_counters);
  auto region_counters_h = Kokkos::create_mirror_view(region_counters);
  Kokkos::Experimental::ScatterView<uint64_t[4]> chunk_counters_sv(chunk_counters);
  Kokkos::Experimental::ScatterView<uint64_t[4]> region_counters_sv(region_counters);

  num_chunks = (tree.tree_h.extent(0)+1)/2;
  num_nodes = tree.tree_h.extent(0);
  STDOUT_PRINT("Num chunks: %u\n", num_chunks);
  STDOUT_PRINT("Num nodes: %u\n", num_nodes);

  DigestIdxDeviceMap first_occurrences(num_nodes);
  Kokkos::View<uint32_t*> duplicates("Duplicate nodes", num_nodes);
  Kokkos::View<uint32_t*> dupl_keys("Duplicate keys", num_nodes);
  Kokkos::View<uint32_t[1]> num_dupl_hash("Num duplicates");
  Kokkos::View<uint32_t[1]> hash_id_counter("Counter");
  Kokkos::UnorderedMap<uint32_t,uint32_t> id_map(num_nodes);
  Kokkos::deep_copy(duplicates, UINT_MAX);
  Kokkos::deep_copy(dupl_keys, UINT_MAX);
  Kokkos::deep_copy(num_dupl_hash, 0);
  Kokkos::deep_copy(hash_id_counter, 0);

  uint32_t level_beg = 0;
  uint32_t level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  Kokkos::View<char*> labels("Labels", num_nodes);
  Kokkos::deep_copy(labels, DONE);
  Vector<uint32_t> tree_roots(num_chunks);

  // Process leaves first
  Kokkos::parallel_for("Leaves", Kokkos::RangePolicy<>(num_chunks-1, num_nodes), KOKKOS_CLASS_LAMBDA(const uint32_t leaf) {
    auto chunk_counters_sa = chunk_counters_sv.access();
    uint32_t num_bytes = chunk_size;
    uint64_t offset = static_cast<uint64_t>(leaf-(num_chunks-1))*static_cast<uint64_t>(chunk_size);
    if(leaf == num_nodes-1) // Calculate how much data to hash
      num_bytes = data_size-offset;
    // Hash chunk
    HashDigest digest;
    hash(data_ptr+offset, num_bytes, digest.digest);
    if(digests_same(tree(leaf), digest)) {
      labels(leaf) = FIXED_DUPL;
      chunk_counters_sa(labels(leaf)) += 1;
    } else if(first_ocur_d.exists(digest)) {
      labels(leaf) = SHIFT_DUPL;
      chunk_counters_sa(labels(leaf)) += 1;
    } else {
      labels(leaf) = FIRST_DUPL;
      chunk_counters_sa(labels(leaf)) += 1;

      uint32_t id = leaf+num_nodes;
      auto result = first_occurrences.insert(digest, id);
      id = first_occurrences.value_at(result.index());
      uint32_t offset = Kokkos::atomic_fetch_add(&num_dupl_hash(0), 1);
      duplicates(offset) = leaf;
      dupl_keys(offset) = id;
    }
    tree(leaf) = digest;
  });

  // Build up forest of Merkle Trees
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  while(level_beg <= num_nodes) { // Intensional unsigned integer underflow
    Kokkos::parallel_for("Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_CLASS_LAMBDA(const uint32_t node) {
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) == FIRST_DUPL && labels(child_r) == FIRST_DUPL) {
          hash((uint8_t*)&tree(child_l), 2*sizeof(HashDigest), tree(node).digest);

          labels(node) = FIRST_DUPL;

          uint32_t id = node+num_nodes;
          auto result = first_occurrences.insert(tree(node), id);
          if(result.existing()) {
            id = first_occurrences.value_at(result.index());
          }
          uint32_t offset = Kokkos::atomic_fetch_add(&num_dupl_hash(0), 1);
          duplicates(offset) = node;
          dupl_keys(offset) = id;
        }
        if(node == 0 && labels(0) == FIRST_DUPL) {
          tree_roots.push(0);
        }
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }
  DEBUG_PRINT("Processed leaves and trees\n");

  auto hash_id_counter_h = Kokkos::create_mirror_view(hash_id_counter);
  auto num_dupl_hash_h = Kokkos::create_mirror_view(num_dupl_hash);
  Kokkos::deep_copy(num_dupl_hash_h, num_dupl_hash);
  uint32_t num_first_occur = num_dupl_hash_h(0);

  Kokkos::View<uint32_t*> num_duplicates("Number of duplicates", first_occurrences.size()+1);
  Kokkos::deep_copy(num_duplicates, 0);
  Kokkos::deep_copy(hash_id_counter, 0);
  Kokkos::parallel_for("Create id map", Kokkos::RangePolicy<>(0, first_occurrences.capacity()), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    if(first_occurrences.valid_at(i)) {
      uint32_t& old_id = first_occurrences.value_at(i);
      uint32_t new_id = Kokkos::atomic_fetch_add(&hash_id_counter(0), static_cast<uint32_t>(1));
      id_map.insert(old_id, new_id);
      old_id = new_id;
    }
  });
  Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<>(0, num_first_occur), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    uint32_t old_id = dupl_keys(i);
    uint32_t new_id = id_map.value_at(id_map.find(old_id));
    dupl_keys(i) = new_id;
    Kokkos::atomic_add(&num_duplicates(dupl_keys(i)), 1);
  });

  Kokkos::deep_copy(hash_id_counter_h, hash_id_counter);

  auto keys = dupl_keys;
  using key_type = decltype(keys);
  using Comparator = Kokkos::BinOp1D<key_type>;
  Comparator comp(hash_id_counter_h(0), 0, hash_id_counter_h(0));
  Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, num_dupl_hash_h(0), comp, 0);
  bin_sort.create_permute_vector();
  bin_sort.sort(duplicates);

  uint32_t total_duplicates = 0;
  Kokkos::parallel_scan("Find vector offsets", Kokkos::RangePolicy<>(0,num_duplicates.size()), KOKKOS_CLASS_LAMBDA(uint32_t i, uint32_t& partial_sum, bool is_final) {
    uint32_t num = num_duplicates(i);
    if(is_final) num_duplicates(i) = partial_sum;
    partial_sum += num;
  }, total_duplicates);

  Kokkos::parallel_for("Remove roots with duplicate leaves", Kokkos::RangePolicy<>(0, first_occurrences.capacity()), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    if(first_occurrences.valid_at(i)) {
      uint32_t id = first_occurrences.value_at(i);
      if(num_duplicates(id+1)-num_duplicates(id) > 1) {
        uint32_t root = num_nodes;
        bool found_dup = true;
        while(found_dup) {
          found_dup = false;
          root = num_nodes;
          for(uint32_t idx=0; idx<num_duplicates(id+1)-num_duplicates(id); idx++) {
            uint32_t u = duplicates(num_duplicates(id)+idx);
            uint32_t possible_root = u;
            while((num_nodes < possible_root) && (possible_root > 0) && first_occurrences.exists(tree((possible_root-1)/2))) {
              possible_root = (possible_root-1)/2;
            }
            if(possible_root < root) {
              root = possible_root;
            } else if(possible_root == root) {
              first_occurrences.erase(tree(root));
              found_dup = true;
              break;
            }
          }
        }
      }
    }
  });

  Kokkos::parallel_for("Select first occurrence leaves", Kokkos::RangePolicy<>(num_chunks-1, num_nodes), KOKKOS_CLASS_LAMBDA(const uint32_t node) {
    if(labels(node) == FIRST_DUPL) {
      auto chunk_counters_sa = chunk_counters_sv.access();
      uint32_t id = first_occurrences.value_at(first_occurrences.find(tree(node)));
      uint32_t select = duplicates(num_duplicates(id));
      uint32_t root = select;
      for(uint32_t idx=0; idx<num_duplicates(id+1)-num_duplicates(id); idx++) {
        uint32_t u = duplicates(num_duplicates(id)+idx);
        uint32_t possible_root = u;
        while(possible_root > 0 && first_occurrences.exists(tree((possible_root-1)/2))) {
          possible_root = (possible_root-1)/2;
        }
        if(possible_root < root) {
          root = possible_root;
          select = u;
        }
      }
      for(uint32_t idx=0; idx<num_duplicates(id+1)-num_duplicates(id); idx++) {
        uint32_t u = duplicates(num_duplicates(id)+idx);
        labels(u) = SHIFT_DUPL;
        chunk_counters_sa(labels(u)) += 1;
      }
      labels(select) = FIRST_OCUR;
      chunk_counters_sa(FIRST_OCUR) += 1;
      first_ocur_d.insert(tree(select), NodeID(select, current_id));
    }
  });

  // Build up forest of Merkle Trees
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  while(level_beg <= num_nodes) { // Intensional unsigned integer underflow
    Kokkos::parallel_for("Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_CLASS_LAMBDA(const uint32_t node) {
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) == FIRST_OCUR && labels(child_r) == FIRST_OCUR) {
          labels(node) = FIRST_OCUR;
          hash((uint8_t*)&tree(child_l), 2*sizeof(HashDigest), tree(node).digest);
          first_ocur_d.insert(tree(node), NodeID(node, current_id));
        }
        if(node == 0 && labels(0) == FIRST_OCUR)
          tree_roots.push(0);
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  while(level_beg <= num_nodes) { // unsigned integer underflow
    Kokkos::parallel_for("Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_CLASS_LAMBDA(const uint32_t node) {
      auto region_counters_sa = region_counters_sv.access();
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) != labels(child_r)) { // Children have different labels
          labels(node) = DONE;
          if((labels(child_l) != FIXED_DUPL) && (labels(child_l) != DONE)) {
            tree_roots.push(child_l);
            region_counters_sa(labels(child_l)) += 1;
          }
          if((labels(child_r) != FIXED_DUPL) && (labels(child_r) != DONE)) {
            tree_roots.push(child_r);
            region_counters_sa(labels(child_r)) += 1;
          }
        } else if(labels(child_l) == FIXED_DUPL) { // Children are both fixed duplicates
          labels(node) = FIXED_DUPL;
        } else if(labels(child_l) == SHIFT_DUPL) { // Children are both shifted duplicates
          hash((uint8_t*)&tree(child_l), 2*sizeof(HashDigest), tree(node).digest);
          if(first_ocur_d.exists(tree(node))) { // This node is also a shifted duplicate
            labels(node) = SHIFT_DUPL;
          } else { // Node is not a shifted duplicate. Save child trees
            labels(node) = DONE; // Add children to tree root maps
            tree_roots.push(child_l);
            tree_roots.push(child_r);
            region_counters_sa(SHIFT_DUPL) += 2;
          }
        }
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }

  // Count regions
  Kokkos::parallel_for("Count regions", Kokkos::RangePolicy<>(0,tree_roots.size()), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    auto chunk_counters_sa = chunk_counters_sv.access();
    auto region_counters_sa = region_counters_sv.access();
    uint32_t root = tree_roots.vector_d(i);
    if(labels(root) != DONE) {
      if(labels(root) == FIRST_OCUR) {
        first_ocur_vec.push(root);
      } else if(labels(root) == SHIFT_DUPL) {
        shift_dupl_vec.push(root);
      }
    }
  });

  Kokkos::Experimental::contribute(region_counters, region_counters_sv);
  Kokkos::deep_copy(chunk_counters_h, chunk_counters);
  Kokkos::deep_copy(region_counters_h, region_counters);

  STDOUT_PRINT("Checkpoint %u\n", current_id);
  STDOUT_PRINT("Number of first occurrence chunks:  %lu\n", chunk_counters_h(FIRST_OCUR));
  STDOUT_PRINT("Number of fixed duplicate chunks:   %lu\n", chunk_counters_h(FIXED_DUPL));
  STDOUT_PRINT("Number of shifted duplicate chunks: %lu\n", chunk_counters_h(SHIFT_DUPL));
  STDOUT_PRINT("Number of first occurrence regions:  %lu\n", region_counters_h(FIRST_OCUR));
  STDOUT_PRINT("Number of fixed duplicate regions:   %lu\n", region_counters_h(FIXED_DUPL));
  STDOUT_PRINT("Number of shifted duplicate regions: %lu\n", region_counters_h(SHIFT_DUPL));
  return;
}

/**
 * Main checkpointing function. Given a Kokkos View, create an incremental checkpoint using 
 * the chosen checkpoint strategy. The deduplication mode can be one of the following:
 *   - Full: No deduplication
 *   - Basic: Remove chunks that have not changed since the previous checkpoint
 *   - Tree: Save a single copy of each unique chunk and use metadata to handle duplicates
 *   - Tree: Save minimal set of chunks and use a compact metadata representations
 *
 * \param header        The checkpoint header
 * \param data_ptr      Data to be checkpointed
 * \param data_len      Length of data in bytes
 * \param diff_h        The output incremental checkpoint on the Host
 * \param make_baseline Flag determining whether to make a baseline checkpoint
 */
void 
TreeLowRootDeduplicator::checkpoint(header_t& header, 
                             uint8_t* data_ptr, 
                             size_t data_size,
                             Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                             bool make_baseline) {
  using Timer = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double>;
  // ==========================================================================================
  // Deduplicate data
  // ==========================================================================================
  Timer::time_point beg_chkpt = Timer::now();
  std::string setup_region_name = std::string("Deduplication chkpt ") + 
                                  std::to_string(current_id) + std::string(": Setup");
  Kokkos::Profiling::pushRegion(setup_region_name.c_str());

  // Set important values
  data_len = data_size;
  num_chunks = data_len/chunk_size;
  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(chunk_size) < data_len)
    num_chunks += 1;
  num_nodes = 2*num_chunks-1;

  // Allocate or resize necessary variables for each approach
  if(make_baseline) {
    tree = MerkleTree(num_chunks);
    first_ocur_d = DigestNodeIDDeviceMap(num_nodes);
    first_ocur_vec = Vector<uint32_t>(num_chunks);
    shift_dupl_vec = Vector<uint32_t>(num_chunks);
  }
  first_ocur_vec.clear();
  shift_dupl_vec.clear();
  std::string resize_tree_label = std::string("Deduplication chkpt ") + 
                                  std::to_string(current_id) + 
                                  std::string(": Setup: Resize Tree");
  Kokkos::Profiling::pushRegion(resize_tree_label.c_str());
  if(tree.tree_d.size() < num_nodes) {
    Kokkos::resize(tree.tree_d, num_nodes);
    Kokkos::resize(tree.tree_h, num_nodes);
  }
  Kokkos::Profiling::popRegion();
  std::string resize_map_label = std::string("Deduplication chkpt ") + 
                                 std::to_string(current_id) + 
                                 std::string(": Setup: Resize First Ocur Map");
  Kokkos::Profiling::pushRegion(resize_map_label.c_str());
  if(first_ocur_d.capacity() < first_ocur_d.size()+num_nodes)
    first_ocur_d.rehash(first_ocur_d.size()+num_nodes);
  Kokkos::Profiling::popRegion();
  std::string resize_updates_label = std::string("Deduplication chkpt ") + 
                                     std::to_string(current_id) + 
                                     std::string(": Setup: Resize Update Map");
  Kokkos::Profiling::pushRegion(resize_updates_label.c_str());
  Kokkos::Profiling::popRegion();
  std::string clear_updates_label = std::string("Deduplication chkpt ") + 
                                    std::to_string(current_id) + 
                                    std::string(": Setup: Clear Update Map");
  Kokkos::Profiling::pushRegion(clear_updates_label.c_str());
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();

  std::string dedup_region_name = std::string("Deduplication chkpt ") + 
                                  std::to_string(current_id);
  Timer::time_point start_create_tree0 = Timer::now();
  timers[0] = std::chrono::duration_cast<Duration>(start_create_tree0 - beg_chkpt).count();
  Kokkos::Profiling::pushRegion(dedup_region_name.c_str());

  // Deduplicate data and identify nodes and chunks needed for the incremental checkpoint
  if((current_id == 0) || make_baseline) {
    baseline_id = current_id;
  }
  if((current_id == 0) || make_baseline) {
    dedup_data_baseline(data_ptr, data_size);
    baseline_id = current_id;
  } else {
    // Use the lowest offset to determine which node is the first occurrence
    dedup_data(data_ptr, data_size);
  }

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

  datasizes = collect_diff(data_ptr, data_size, diff, header);

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

/**
 * Main checkpointing function. Given a raw device pointer, create an incremental checkpoint 
 * using the chosen checkpoint strategy. The deduplication mode can be one of the following:
 *   - Full: No deduplication
 *   - Basic: Remove chunks that have not changed since the previous checkpoint
 *   - Tree: Save a single copy of each unique chunk and use metadata to handle duplicates
 *   - Tree: Save minimal set of chunks and use a compact metadata representations
 *
 * \param data_ptr      Raw data pointer that needs to be deduplicated
 * \param len           Length of data
 * \param filename      Filename to save checkpoint
 * \param logname       Base filename for logs
 * \param make_baseline Flag determining whether to make a baseline checkpoint
 */
void 
TreeLowRootDeduplicator::checkpoint(uint8_t* data_ptr, 
                             size_t len, 
                             std::string& filename, 
                             std::string& logname, 
                             bool make_baseline) {
  Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
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

/**
 * Main checkpointing function. Given a raw device pointer, create an incremental checkpoint 
 * using the chosen checkpoint strategy. Save checkpoint to host view. 
 * The deduplication mode can be one of the following:
 *   - Full: No deduplication
 *   - Basic: Remove chunks that have not changed since the previous checkpoint
 *   - Tree: Save a single copy of each unique chunk and use metadata to handle duplicates
 *   - Tree: Save minimal set of chunks and use a compact metadata representations
 *
 * \param data_ptr      Raw data pointer that needs to be deduplicated
 * \param len           Length of data
 * \param diff_h        Host View to store incremental checkpoint
 * \param make_baseline Flag determining whether to make a baseline checkpoint
 */
void 
TreeLowRootDeduplicator::checkpoint(uint8_t* data_ptr, 
                             size_t len, 
                             Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                             bool make_baseline) {
  Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
  header_t header;
  checkpoint(header, data_ptr, len, diff_h, make_baseline);
  current_id += 1;
}

/**
 * Main checkpointing function. Given a raw device pointer, create an incremental checkpoint 
 * using the chosen checkpoint strategy. Save checkpoint to host view and write logs.
 * The deduplication mode can be one of the following:
 *   - Full: No deduplication
 *   - Basic: Remove chunks that have not changed since the previous checkpoint
 *   - Tree: Save a single copy of each unique chunk and use metadata to handle duplicates
 *   - Tree: Save minimal set of chunks and use a compact metadata representations
 *
 * \param data_ptr      Raw data pointer that needs to be deduplicated
 * \param len           Length of data
 * \param diff_h        Host View to store incremental checkpoint
 * \param logname       Base filename for logs
 * \param make_baseline Flag determining whether to make a baseline checkpoint
 */
void 
TreeLowRootDeduplicator::checkpoint(uint8_t* data_ptr, 
                             size_t len, 
                             Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                             std::string& logname, 
                             bool make_baseline) {
  Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
  header_t header;
  checkpoint(header, data_ptr, len, diff_h, make_baseline);
  write_chkpt_log(header, diff_h, logname);
  current_id += 1;
}

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
void 
TreeLowRootDeduplicator::write_chkpt_log(header_t& header, 
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

  std::string approach("TreeLowRoot");

  result_data << approach << "," // Approach
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
  timing_file << approach << "," 
              << current_id << "," 
              << chunk_size << "," 
              << timers[1] << "," 
              << timers[2] << "," 
              << timers[3] << std::endl;
  size_file << approach << "," 
            << current_id << "," 
            << chunk_size << "," 
            << datasizes.first << "," 
            << datasizes.second << ",";
  write_metadata_breakdown(size_file, DedupMode::Tree, header, diff_h, current_id);
}

/**
 * Function for writing the restart log.
 *
 * \param select_chkpt Which checkpoint to write the log
 * \param logname      Filename for writing log
 */
void 
TreeLowRootDeduplicator::write_restart_log(uint32_t select_chkpt, 
                                    std::string& logname) {
  std::fstream timing_file;
  timing_file.open(logname, std::fstream::out | std::fstream::app);
  timing_file << "TreeLowRoot" << ","; 
  timing_file << select_chkpt << "," 
              << chunk_size << "," 
              << restart_timers[0] << "," 
              << restart_timers[1] << std::endl;
  timing_file.close();
}
