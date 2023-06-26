#include "tree_approach.hpp"

TreeDeduplicator::TreeDeduplicator() {}

TreeDeduplicator::TreeDeduplicator(uint32_t bytes_per_chunk) {
  chunk_size = bytes_per_chunk;
  current_id = 0;
  baseline_id = 0;
}

TreeDeduplicator::~TreeDeduplicator() {}

void 
TreeDeduplicator::dedup_data_baseline(const uint8_t* data_ptr, 
                                      const size_t data_size) {
  // Get number of chunks and nodes
  num_chunks = (tree.tree_h.extent(0)+1)/2;
  num_nodes = tree.tree_h.extent(0);
  STDOUT_PRINT("Num chunks: %u\n", num_chunks);
  STDOUT_PRINT("Num nodes: %u\n", num_nodes);

  // Stats for debugging or analysis
#ifdef STATS
  Kokkos::View<uint64_t[3]> chunk_counters("Chunk counters");
  Kokkos::View<uint64_t[3]> region_counters("Region counters");
  Kokkos::View<uint64_t*> first_region_sizes("Num first regions per size", num_chunks+1);
  Kokkos::View<uint64_t*> shift_region_sizes("Num shift regions per size", num_chunks+1);
  Kokkos::deep_copy(chunk_counters, 0);
  Kokkos::deep_copy(region_counters, 0);
  Kokkos::deep_copy(first_region_sizes, 0);
  Kokkos::deep_copy(shift_region_sizes, 0);
  auto chunk_counters_h  = Kokkos::create_mirror_view(chunk_counters);
  auto region_counters_h = Kokkos::create_mirror_view(region_counters);
  auto first_region_sizes_h = Kokkos::create_mirror_view(first_region_sizes);
  auto shift_region_sizes_h = Kokkos::create_mirror_view(shift_region_sizes);
  Kokkos::Experimental::ScatterView<uint64_t[3]> chunk_counters_sv(chunk_counters);
  Kokkos::Experimental::ScatterView<uint64_t[3]> region_counters_sv(region_counters);
  Kokkos::Experimental::ScatterView<uint64_t*> first_region_sizes_sv(first_region_sizes);
  Kokkos::Experimental::ScatterView<uint64_t*> shift_region_sizes_sv(shift_region_sizes);
#endif

  // Setup markers for beginning and end of tree level
  uint32_t level_beg = 0;
  uint32_t level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }

  // Create labels
  Kokkos::View<char*> labels("Labels", num_nodes);
  Kokkos::deep_copy(labels, DONE);

  // Process leaves first
  using member_type = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type;
  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>(((num_nodes-num_chunks+1)/TEAM_SIZE)+1, TEAM_SIZE);
  Kokkos::parallel_for("Baseline: Leaves", team_policy, 
  KOKKOS_CLASS_LAMBDA(member_type team_member) {
    uint64_t i=team_member.league_rank();
    uint64_t j=team_member.team_rank();
    uint64_t leaf = num_chunks-1+i*team_member.team_size()+j;
    if(leaf < num_nodes) {
#ifdef STATS
      auto chunk_counters_sa = chunk_counters_sv.access();
      auto region_counters_sa = region_counters_sv.access();
#endif
      uint32_t num_bytes = chunk_size;
      uint64_t offset = static_cast<uint64_t>(leaf-num_chunks+1)*static_cast<uint64_t>(chunk_size);
      if(leaf == num_nodes-1) // Calculate how much data to hash
        num_bytes = data_size-offset;
      // Hash chunk
      HashDigest digest;
      hash(data_ptr+offset, num_bytes, digest.digest);
      // Insert into table
      auto result = first_ocur_d.insert(digest, NodeID(leaf, current_id)); 
      if(digests_same(digest, tree(leaf))) { // Fixed duplicate chunk
        labels(leaf) = FIXED_DUPL;
      } else if(result.success()) { // First occurrence chunk
        labels(leaf) = FIRST_OCUR;
      } else if(result.existing()) { // Shifted duplicate chunk
        auto& info = first_ocur_d.value_at(result.index());
        if(info.tree == current_id) {
          Kokkos::atomic_min(&info.node, leaf);
          labels(leaf) = FIRST_OCUR;
        } else {
          labels(leaf) = SHIFT_DUPL;
        }
      }
      tree(leaf) = digest; /// Update tree
#ifdef STATS
      chunk_counters_sa(labels(leaf)) += 1;
#endif
    }
  });

  /**
   * Identify first occurrences for leaves. In the case of duplicate hash, 
   * select the chunk with the lowest index.
   */
  Kokkos::parallel_for("Baseline: Leaves: Choose first occurrences", Kokkos::RangePolicy<>(num_chunks-1, num_nodes), KOKKOS_CLASS_LAMBDA(const uint32_t leaf) {
#ifdef STATS
    auto chunk_counters_sa = chunk_counters_sv.access();
    auto region_counters_sa = region_counters_sv.access();
#endif
    if(labels(leaf) == FIRST_OCUR) {
      auto info = first_ocur_d.value_at(first_ocur_d.find(tree(leaf))); 
      if((info.tree == current_id) && (leaf != info.node)) {
        labels(leaf) = SHIFT_DUPL;
#ifdef STATS
        chunk_counters_sa(labels(leaf)) += 1;
#endif
      }
    }
  });

  // Build up forest of Merkle Trees
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  // Iterate through each level of tree and build First occurrence trees
  while(level_beg <= num_nodes) { // Intentional unsigned integer underflow
    Kokkos::parallel_for("Baseline: Build First Occurrence Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_CLASS_LAMBDA(const uint32_t node) {
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) == FIRST_OCUR && labels(child_r) == FIRST_OCUR) {
          labels(node) = FIRST_OCUR;
          hash((uint8_t*)&tree(child_l), 2*sizeof(HashDigest), tree(node).digest);
          first_ocur_d.insert(tree(node), NodeID(node, current_id));
        }
        if(node == 0 && labels(0) == FIRST_OCUR) {
          first_ocur_vec.push(node);
#ifdef STATS
          auto first_region_sizes_sa = first_region_sizes_sv.access();
          first_region_sizes_sa(num_chunks) += 1;
#endif
        }
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }

  // Build up forest of trees
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  // Iterate through each level of tree and build shifted duplicate trees
  while(level_beg <= num_nodes) { // unsigned integer underflow
    Kokkos::parallel_for("Baseline: Build Forest", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_CLASS_LAMBDA(const uint32_t node) {
#ifdef STATS
      auto region_counters_sa = region_counters_sv.access();
      auto first_region_sizes_sa = first_region_sizes_sv.access();
      auto shift_region_sizes_sa = shift_region_sizes_sv.access();
#endif
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) != labels(child_r)) { // Children have different labels
          labels(node) = DONE;
          if((labels(child_l) != FIXED_DUPL) && (labels(child_l) != DONE)) {
            if(labels(child_l) == SHIFT_DUPL) {
              shift_dupl_vec.push(child_l);
#ifdef STATS
              shift_region_sizes_sa(num_leaf_descendents(child_l, num_nodes)) += 1;
#endif
            } else {
              first_ocur_vec.push(child_l);
#ifdef STATS
              first_region_sizes_sa(num_leaf_descendents(child_l, num_nodes)) += 1;
#endif
            }
#ifdef STATS
            region_counters_sa(labels(child_l)) += 1;
#endif
          }
          if((labels(child_r) != FIXED_DUPL) && (labels(child_r) != DONE)) {
            if(labels(child_r) == SHIFT_DUPL) {
              shift_dupl_vec.push(child_r);
#ifdef STATS
              shift_region_sizes_sa(num_leaf_descendents(child_r, num_nodes)) += 1;
#endif
            } else {
              first_ocur_vec.push(child_r);
#ifdef STATS
              first_region_sizes_sa(num_leaf_descendents(child_r, num_nodes)) += 1;
#endif
            }
#ifdef STATS
            region_counters_sa(labels(child_r)) += 1;
#endif
          }
        } else if(labels(child_l) == FIXED_DUPL) { // Children are both fixed duplicates
          labels(node) = FIXED_DUPL;
        } else if(labels(child_l) == SHIFT_DUPL) { // Children are both shifted duplicates
          if(first_ocur_d.exists(tree(node))) { // This node is also a shifted duplicate
            labels(node) = SHIFT_DUPL;
          } else { // Node is not a shifted duplicate. Save child trees
            labels(node) = DONE; // Add children to tree root maps
            shift_dupl_vec.push(child_l);
            shift_dupl_vec.push(child_r);
#ifdef STATS
            region_counters_sa(SHIFT_DUPL) += 2;
            shift_region_sizes_sa(num_leaf_descendents(child_l, num_nodes)) += 1;
            shift_region_sizes_sa(num_leaf_descendents(child_r, num_nodes)) += 1;
#endif
          }
        }
      }
    });
    // Insert digests into map
    Kokkos::parallel_for("Baseline: Build Forest: Insert entries", Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_CLASS_LAMBDA(const uint32_t node) {
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        hash((uint8_t*)&tree(child_l), 2*sizeof(HashDigest), tree(node).digest);
        first_ocur_d.insert(tree(node), NodeID(node, current_id));
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }

#ifdef STATS
  Kokkos::Experimental::contribute(region_counters, region_counters_sv);
  Kokkos::Experimental::contribute(first_region_sizes, first_region_sizes_sv);
  Kokkos::Experimental::contribute(shift_region_sizes, shift_region_sizes_sv);
  Kokkos::fence();
  Kokkos::deep_copy(chunk_counters_h, chunk_counters);
  Kokkos::deep_copy(region_counters_h, region_counters);
  Kokkos::deep_copy(first_region_sizes_h, first_region_sizes);
  Kokkos::deep_copy(shift_region_sizes_h, shift_region_sizes);

  STDOUT_PRINT("Checkpoint %u\n", current_id);
  STDOUT_PRINT("Number of first occurrence chunks:  %lu\n", chunk_counters_h(FIRST_OCUR));
  STDOUT_PRINT("Number of fixed duplicate chunks:   %lu\n", chunk_counters_h(FIXED_DUPL));
  STDOUT_PRINT("Number of shifted duplicate chunks: %lu\n", chunk_counters_h(SHIFT_DUPL));
  STDOUT_PRINT("Number of first occurrence regions:  %lu\n", region_counters_h(FIRST_OCUR));
  STDOUT_PRINT("Number of fixed duplicate regions:   %lu\n", region_counters_h(FIXED_DUPL));
  STDOUT_PRINT("Number of shifted duplicate regions: %lu\n", region_counters_h(SHIFT_DUPL));
  for(uint32_t i=0; i<num_chunks+1; i++) {
    if(first_region_sizes_h(i) > 0) {
      printf("First Occurrence: Num regions of size %u: %lu\n", i, first_region_sizes_h(i));
    }
  }
  for(uint32_t i=0; i<num_chunks+1; i++) {
    if(shift_region_sizes_h(i) > 0) {
      printf("Shift Occurrence: Num regions of size %u: %lu\n", i, shift_region_sizes_h(i));
    }
  }
#endif
  Kokkos::fence();
  return;
}

void 
TreeDeduplicator::dedup_data(const uint8_t* data_ptr, 
                             const size_t data_size) {
  // Get number of chunks and nodes
  std::string setup_label = std::string("Deduplicate Checkpoint ") + std::to_string(current_id) + std::string(": Setup");
  Kokkos::Profiling::pushRegion(setup_label);
  num_chunks = (tree.tree_h.extent(0)+1)/2;
  num_nodes = tree.tree_h.extent(0);
  STDOUT_PRINT("Num chunks: %u\n", num_chunks);
  STDOUT_PRINT("Num nodes: %u\n", num_nodes);

  // Stats for debugging or analysis
#ifdef STATS
  Kokkos::View<uint64_t[3]> chunk_counters("Chunk counters");
  Kokkos::View<uint64_t[3]> region_counters("Region counters");
  Kokkos::View<uint64_t*> first_region_sizes("Num first regions per size", num_chunks+1);
  Kokkos::View<uint64_t*> shift_region_sizes("Num shift regions per size", num_chunks+1);
  Kokkos::deep_copy(chunk_counters, 0);
  Kokkos::deep_copy(region_counters, 0);
  Kokkos::deep_copy(first_region_sizes, 0);
  Kokkos::deep_copy(shift_region_sizes, 0);
  auto chunk_counters_h  = Kokkos::create_mirror_view(chunk_counters);
  auto region_counters_h = Kokkos::create_mirror_view(region_counters);
  auto first_region_sizes_h = Kokkos::create_mirror_view(first_region_sizes);
  auto shift_region_sizes_h = Kokkos::create_mirror_view(shift_region_sizes);
  Kokkos::Experimental::ScatterView<uint64_t[3]> chunk_counters_sv(chunk_counters);
  Kokkos::Experimental::ScatterView<uint64_t[3]> region_counters_sv(region_counters);
  Kokkos::Experimental::ScatterView<uint64_t*> first_region_sizes_sv(first_region_sizes);
  Kokkos::Experimental::ScatterView<uint64_t*> shift_region_sizes_sv(shift_region_sizes);
#endif

  // Setup markers for beginning and end of tree level
  uint32_t level_beg = 0;
  uint32_t level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }

  // Create labels
  Kokkos::View<char*> labels("Labels", num_nodes);
  Kokkos::deep_copy(labels, DONE);
  Kokkos::Profiling::popRegion();

  // Process leaves first
  std::string leaves_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Leaves");
  using member_type = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type;
  auto team_policy = Kokkos::TeamPolicy<>(((num_nodes-num_chunks+1)/TEAM_SIZE)+1, TEAM_SIZE);
  Kokkos::parallel_for(leaves_label, team_policy, KOKKOS_CLASS_LAMBDA(member_type team_member) {
    uint64_t i=team_member.league_rank();
    uint64_t j=team_member.team_rank();
#ifdef STATS
    auto chunk_counters_sa = chunk_counters_sv.access();
#endif
    uint64_t leaf = num_chunks-1+i*team_member.team_size()+j;
    if(leaf < num_nodes) {
      uint32_t num_bytes = chunk_size;
      uint64_t offset = static_cast<uint64_t>(leaf-(num_chunks-1))*static_cast<uint64_t>(chunk_size);
      if(leaf == num_nodes-1) // Calculate how much data to hash
        num_bytes = data_size-offset;
      // Hash chunk
      HashDigest digest;
      hash(data_ptr+offset, num_bytes, digest.digest);
      // Insert into table
      auto result = first_ocur_d.insert(digest, NodeID(leaf, current_id)); 
      if(digests_same(digest, tree(leaf))) { // Fixed duplicate chunk
        labels(leaf) = FIXED_DUPL;
#ifdef STATS
      chunk_counters_sa(labels(leaf)) += 1;
#endif
      } else if(result.success()) { // First occurrence chunk
        labels(leaf) = FIRST_OCUR;
        tree(leaf) = digest;
#ifdef STATS
      chunk_counters_sa(labels(leaf)) += 1;
#endif
      } else if(result.existing()) { // Shifted duplicate chunk
        auto& info = first_ocur_d.value_at(result.index());
        if(info.tree == current_id) {
          Kokkos::atomic_min(&info.node, leaf);
          labels(leaf) = FIRST_OCUR;
        } else {
          labels(leaf) = SHIFT_DUPL;
#ifdef STATS
      chunk_counters_sa(labels(leaf)) += 1;
#endif
        }
        tree(leaf) = digest;
      }
    }
  });

  // TODO May not be necessary
  // Ensure any duplicate first occurrences are labeled correctly
  std::string leaves_first_ocur_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Leaves: Choose first occurrences");
  Kokkos::parallel_for(leaves_first_ocur_label, Kokkos::RangePolicy<>(num_chunks-1, num_nodes), KOKKOS_CLASS_LAMBDA(const uint32_t leaf) {
    if(labels(leaf) == FIRST_OCUR) {
#ifdef STATS
      auto chunk_counters_sa = chunk_counters_sv.access();
#endif
      auto info = first_ocur_d.value_at(first_ocur_d.find(tree(leaf))); 
      if((info.tree == current_id) && (leaf != info.node)) {
        labels(leaf) = SHIFT_DUPL;
      }
#ifdef STATS
      chunk_counters_sa(labels(leaf)) += 1;
#endif
    }
  });

  // Build up forest of Merkle Trees for First occurrences
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  while(level_beg <= num_nodes) { // Intensional unsigned integer underflow
    std::string first_ocur_forest_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Build First Occurrence Forest");
    Kokkos::parallel_for(first_ocur_forest_label, Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_CLASS_LAMBDA(const uint32_t node) {
#ifdef STATS
      auto region_counters_sa = region_counters_sv.access();
#endif
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) == FIRST_OCUR && labels(child_r) == FIRST_OCUR) {
          labels(node) = FIRST_OCUR;
          hash((uint8_t*)&tree(child_l), 2*sizeof(HashDigest), tree(node).digest);
          first_ocur_d.insert(tree(node), NodeID(node, current_id));
        }
        if(node == 0 && labels(0) == FIRST_OCUR) { // Handle case where all chunks are new
          first_ocur_vec.push(node);
#ifdef STATS
          auto first_region_sizes_sa = first_region_sizes_sv.access();
          first_region_sizes_sa(num_chunks) += 1;
          region_counters_sa(FIRST_OCUR) += 1;
#endif
        }
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }

  // Build up forest of Merkle trees for duplicates
  level_beg = 0;
  level_end = 0;
  while(level_end < num_nodes) {
    level_beg = 2*level_beg + 1;
    level_end = 2*level_end + 2;
  }
  while(level_beg <= num_nodes) { // unsigned integer underflow
    std::string forest_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Build Forest");
    Kokkos::parallel_for(forest_label, Kokkos::RangePolicy<>(level_beg, level_end+1), KOKKOS_CLASS_LAMBDA(const uint32_t node) {
#ifdef STATS
      auto region_counters_sa = region_counters_sv.access();
      auto first_region_sizes_sa = first_region_sizes_sv.access();
      auto shift_region_sizes_sa = shift_region_sizes_sv.access();
#endif
      if(node < num_chunks-1) {
        uint32_t child_l = 2*node+1;
        uint32_t child_r = 2*node+2;
        if(labels(child_l) != labels(child_r)) { // Children have different labels
          labels(node) = DONE;
          if((labels(child_l) != FIXED_DUPL) && (labels(child_l) != DONE)) {
            if(labels(child_l) == SHIFT_DUPL) {
              shift_dupl_vec.push(child_l);
#ifdef STATS
              region_counters_sa(SHIFT_DUPL) += 1;
              shift_region_sizes_sa(num_leaf_descendents(child_l, num_nodes)) += 1;
#endif
            } else {
              first_ocur_vec.push(child_l);
#ifdef STATS
              region_counters_sa(FIRST_OCUR) += 1;
              first_region_sizes_sa(num_leaf_descendents(child_l, num_nodes)) += 1;
#endif
            }
          }
          if((labels(child_r) != FIXED_DUPL) && (labels(child_r) != DONE)) {
            if(labels(child_r) == SHIFT_DUPL) {
              shift_dupl_vec.push(child_r);
#ifdef STATS
              region_counters_sa(SHIFT_DUPL) += 1;
              shift_region_sizes_sa(num_leaf_descendents(child_r, num_nodes)) += 1;
#endif
            } else {
              first_ocur_vec.push(child_r);
#ifdef STATS
              region_counters_sa(FIRST_OCUR) += 1;
              first_region_sizes_sa(num_leaf_descendents(child_r, num_nodes)) += 1;
#endif
            }
          }
        } else if(labels(child_l) == FIXED_DUPL) { // Children are both fixed duplicates
          labels(node) = FIXED_DUPL;
        } else if(labels(child_l) == SHIFT_DUPL) { // Children are both shifted duplicates
          hash((uint8_t*)&tree(child_l), 2*sizeof(HashDigest), tree(node).digest);
          if(first_ocur_d.exists(tree(node))) { // This node is also a shifted duplicate
            labels(node) = SHIFT_DUPL;
          } else { // Node is not a shifted duplicate. Save child trees
            labels(node) = DONE; // Add children to tree root maps
            shift_dupl_vec.push(child_l);
            shift_dupl_vec.push(child_r);
#ifdef STATS
            region_counters_sa(SHIFT_DUPL) += 2;
            shift_region_sizes_sa(num_leaf_descendents(child_l, num_nodes)) += 1;
            shift_region_sizes_sa(num_leaf_descendents(child_r, num_nodes)) += 1;
#endif
          }
        }
      }
    });
    level_beg = (level_beg-1)/2;
    level_end = (level_end-2)/2;
  }

#ifdef STATS
  Kokkos::Experimental::contribute(region_counters, region_counters_sv);
  Kokkos::Experimental::contribute(first_region_sizes, first_region_sizes_sv);
  Kokkos::Experimental::contribute(shift_region_sizes, shift_region_sizes_sv);
  Kokkos::fence();
  Kokkos::deep_copy(chunk_counters_h, chunk_counters);
  Kokkos::deep_copy(region_counters_h, region_counters);
  Kokkos::deep_copy(first_region_sizes_h, first_region_sizes);
  Kokkos::deep_copy(shift_region_sizes_h, shift_region_sizes);

  STDOUT_PRINT("Checkpoint %u\n", current_id);
  STDOUT_PRINT("Number of first occurrence chunks:  %lu\n", chunk_counters_h(FIRST_OCUR));
  STDOUT_PRINT("Number of fixed duplicate chunks:   %lu\n", chunk_counters_h(FIXED_DUPL));
  STDOUT_PRINT("Number of shifted duplicate chunks: %lu\n", chunk_counters_h(SHIFT_DUPL));
  STDOUT_PRINT("Number of first occurrence regions:  %lu\n", region_counters_h(FIRST_OCUR));
  STDOUT_PRINT("Number of fixed duplicate regions:   %lu\n", region_counters_h(FIXED_DUPL));
  STDOUT_PRINT("Number of shifted duplicate regions: %lu\n", region_counters_h(SHIFT_DUPL));
  for(uint32_t i=0; i<num_chunks+1; i++) {
    if(first_region_sizes_h(i) > 0) {
      printf("First Occurrence: Num regions of size %u: %lu\n", i, first_region_sizes_h(i));
    }
  }
  for(uint32_t i=0; i<num_chunks+1; i++) {
    if(shift_region_sizes_h(i) > 0) {
      printf("Shift Occurrence: Num regions of size %u: %lu\n", i, shift_region_sizes_h(i));
    }
  }
#endif
  Kokkos::fence();
  return;
}

std::pair<uint64_t,uint64_t> 
TreeDeduplicator::collect_diff( const uint8_t* data_ptr, 
                                const size_t data_size,
                                Kokkos::View<uint8_t*>& buffer_d, 
                                header_t& header) {
  std::string setup_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Setup");
  Kokkos::Profiling::pushRegion(setup_label);

  uint32_t num_chunks = data_size/chunk_size;
  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(chunk_size) < data_size) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;

  Kokkos::View<uint32_t*> region_leaves("Region leaves", num_chunks);
  Kokkos::View<uint32_t*> region_nodes("Region Nodes", first_ocur_vec.size());
  Kokkos::View<uint32_t*> region_len("Region lengths", first_ocur_vec.size());
  Kokkos::View<uint32_t[1]> counter_d("Counter");
  Kokkos::View<uint32_t[1]>::HostMirror counter_h = Kokkos::create_mirror_view(counter_d);
  Kokkos::deep_copy(counter_d, 0);
  Kokkos::View<uint32_t[1]> chunk_counter_d("Counter");
  Kokkos::View<uint32_t[1]>::HostMirror chunk_counter_h = Kokkos::create_mirror_view(chunk_counter_d);
  Kokkos::deep_copy(chunk_counter_d, 0);
  Kokkos::View<uint64_t*> prior_counter_d("Counter for prior repeats", current_id+1);
  Kokkos::View<uint64_t*>::HostMirror prior_counter_h = Kokkos::create_mirror_view(prior_counter_d);
  Kokkos::deep_copy(prior_counter_d, 0);
  Kokkos::Experimental::ScatterView<uint64_t*> prior_counter_sv(prior_counter_d);

  DEBUG_PRINT("Setup counters\n");

  Kokkos::Profiling::popRegion();

  // Filter and count space used for distinct entries
  // Calculate number of chunks each entry maps to
  std::string count_first_ocur_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Count first ocur bytes");
  Kokkos::parallel_for(count_first_ocur_label, Kokkos::RangePolicy<>(0, first_ocur_vec.size()), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      uint32_t node = first_ocur_vec(i);
      NodeID prev = first_ocur_d.value_at(first_ocur_d.find(tree(node)));
      if(node == prev.node && current_id == prev.tree) {
        uint32_t size = num_leaf_descendents(node, num_nodes);
        uint32_t idx = Kokkos::atomic_fetch_add(&counter_d(0), 1);
        Kokkos::atomic_add(&chunk_counter_d(0), size);
        region_nodes(idx) = node;
        region_len(idx) = size;
      } else {
        printf("Distinct node with different node/tree. Shouldn't happen.\n");
      }
  });
  std::string alloc_bitset_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Allocate bitset");
  Kokkos::Profiling::pushRegion(alloc_bitset_label);

  DEBUG_PRINT("Count distinct bytes\n");

  // Small bitset to record which checkpoints are necessary for restart
  Kokkos::Bitset<Kokkos::DefaultExecutionSpace> chkpts_needed(current_id+1);
  chkpts_needed.reset();
  
  DEBUG_PRINT("Setup chkpt bitset\n");
  Kokkos::Profiling::popRegion();

  // Calculate space needed for repeat entries and number of entries per checkpoint
  Kokkos::RangePolicy<> shared_range_policy(0, shift_dupl_vec.size());
  std::string count_shift_dupl_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Count shift dupl bytes");
  Kokkos::parallel_for(count_shift_dupl_label, shared_range_policy, KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      uint32_t node = shift_dupl_vec(i);
      NodeID prev = first_ocur_d.value_at(first_ocur_d.find(tree(node)));
      auto prior_counter_sa = prior_counter_sv.access();
      prior_counter_sa(prev.tree) += 1;
      chkpts_needed.set(prev.tree);
  });
  std::string contrib_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Contribute shift dupl");
  Kokkos::Profiling::pushRegion(contrib_label);
  DEBUG_PRINT("Count repeat bytes\n");
  Kokkos::Experimental::contribute(prior_counter_d, prior_counter_sv);
  prior_counter_sv.reset_except(prior_counter_d);

  DEBUG_PRINT("Collect prior counter\n");

  uint32_t num_prior_chkpts = chkpts_needed.count();

  DEBUG_PRINT("Number of checkpoints needed: %u\n", num_prior_chkpts);

  size_t data_offset = first_ocur_vec.size()*sizeof(uint32_t) + shift_dupl_vec.size()*2*sizeof(uint32_t) + static_cast<uint64_t>(num_prior_chkpts)*2*sizeof(uint32_t);
  DEBUG_PRINT("Offset for data: %lu\n", data_offset);
  Kokkos::deep_copy(counter_h, counter_d);
  uint32_t num_distinct = counter_h(0);
  STDOUT_PRINT("Number of distinct regions: %u\n", num_distinct);
  Kokkos::Profiling::popRegion();
  // Dividers for distinct chunks. Number of chunks per region varies.
  // Need offsets for each region so that writes can be done in parallel
  std::string calc_offsets_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Calculate offsets");
  Kokkos::parallel_scan(calc_offsets_label, num_distinct, KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
    const uint32_t len = region_len(i);
    if(is_final) region_len(i) = partial_sum;
    partial_sum += len;
  });

  std::string find_region_leaves_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Find region leaves");
  Kokkos::parallel_for(find_region_leaves_label, Kokkos::RangePolicy<>(0,num_distinct), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    uint32_t offset = region_len(i);
    uint32_t node = region_nodes(i);
    uint32_t size = num_leaf_descendents(node, num_nodes);
    uint32_t start = leftmost_leaf(node, num_nodes) - (num_chunks-1);
    for(uint32_t j=0; j<size; j++) {
      region_leaves(offset+j) = start+j;
    }
  });

  std::string alloc_buffer_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Allocate buffer");
  Kokkos::Profiling::pushRegion(alloc_buffer_label);
  Kokkos::deep_copy(chunk_counter_h, chunk_counter_d);
  uint64_t buffer_len = sizeof(header_t)+first_ocur_vec.size()*sizeof(uint32_t)+2*sizeof(uint32_t)*static_cast<uint64_t>(chkpts_needed.count())+shift_dupl_vec.size()*2*sizeof(uint32_t)+chunk_counter_h(0)*static_cast<uint64_t>(chunk_size);
  Kokkos::resize(buffer_d, buffer_len);

  Kokkos::deep_copy(counter_d, sizeof(uint32_t)*num_distinct);

  Kokkos::Profiling::popRegion();

  std::string copy_fo_metadata_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Copy first ocur metadata");
  Kokkos::parallel_for(copy_fo_metadata_label, Kokkos::RangePolicy<>(0,num_distinct), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    uint32_t node = region_nodes(i);
    memcpy(buffer_d.data()+sizeof(header_t)+static_cast<uint64_t>(i)*sizeof(uint32_t), &node, sizeof(uint32_t));
  });

  std::string copy_data_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Copy data");
  Kokkos::parallel_for(copy_data_label, Kokkos::TeamPolicy<>(chunk_counter_h(0), Kokkos::AUTO), 
                         KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
    uint32_t i = team_member.league_rank();
    uint32_t chunk = region_leaves(i);
    uint32_t writesize = chunk_size;
    uint64_t dst_offset = sizeof(header_t)+data_offset+static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);
    uint64_t src_offset = static_cast<uint64_t>(chunk)*static_cast<uint64_t>(chunk_size);
    if(chunk == num_chunks-1) {
      writesize = data_size-src_offset;
    }

    uint8_t* dst = (uint8_t*)(buffer_d.data()+dst_offset);
    uint8_t* src = (uint8_t*)(data_ptr+src_offset);
    team_memcpy(dst, src, writesize, team_member);
  });

  uint32_t num_prior = chkpts_needed.count();

  // Write Repeat map for recording how many entries per checkpoint
  // (Checkpoint ID, # of entries)
  std::string write_repeat_count_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Write repeat count");
  Kokkos::parallel_for(write_repeat_count_label, prior_counter_d.size(), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    if(prior_counter_d(i) > 0) {
      uint32_t num_repeats_i = static_cast<uint32_t>(prior_counter_d(i));
      size_t pos = Kokkos::atomic_fetch_add(&counter_d(0), 2*sizeof(uint32_t));
      memcpy(buffer_d.data()+sizeof(header_t)+pos, &i, sizeof(uint32_t));
      memcpy(buffer_d.data()+sizeof(header_t)+pos+sizeof(uint32_t), &num_repeats_i, sizeof(uint32_t));
      DEBUG_PRINT("Wrote table entry (%u,%u) at offset %lu\n", i, num_repeats_i, pos);
    }
  });

  size_t prior_start = static_cast<uint64_t>(num_distinct)*sizeof(uint32_t)+static_cast<uint64_t>(num_prior)*2*sizeof(uint32_t);
  DEBUG_PRINT("Prior start offset: %lu\n", prior_start);

  Kokkos::View<uint32_t*> current_id_keys("Source checkpoint IDs", shift_dupl_vec.size());
  std::string create_keys_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Create current_id_keys");
  Kokkos::parallel_for(create_keys_label, Kokkos::RangePolicy<>(0,shift_dupl_vec.size()), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    NodeID info = first_ocur_d.value_at(first_ocur_d.find(tree(shift_dupl_vec(i))));
    current_id_keys(i) = info.tree;
  });

  std::string sort_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Sort");
  Kokkos::Profiling::pushRegion(sort_label);
  auto keys = current_id_keys;
  using key_type = decltype(keys);
  using Comparator = Kokkos::BinOp1D<key_type>;
  Comparator comp(current_id, 0, current_id);
  Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, shift_dupl_vec.size(), comp, 0);
  bin_sort.create_permute_vector();
  bin_sort.sort(shift_dupl_vec.vector_d);
  bin_sort.sort(current_id_keys);
  Kokkos::Profiling::popRegion();

  // Write repeat entries
  std::string copy_metadata_label = std::string("Checkpoint ") + std::to_string(current_id) + std::string(": Gather: Write repeat metadata");
  Kokkos::parallel_for(copy_metadata_label, Kokkos::RangePolicy<>(0, shift_dupl_vec.size()), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
    uint32_t node = shift_dupl_vec(i);
    NodeID prev = first_ocur_d.value_at(first_ocur_d.find(tree(shift_dupl_vec(i))));
    memcpy(buffer_d.data()+sizeof(header_t)+prior_start+static_cast<uint64_t>(i)*2*sizeof(uint32_t), &node, sizeof(uint32_t));
    memcpy(buffer_d.data()+sizeof(header_t)+prior_start+static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), &prev.node, sizeof(uint32_t));
  });

  DEBUG_PRINT("Wrote shared metadata\n");
  DEBUG_PRINT("Finished collecting data\n");
  header.ref_id = baseline_id;
  header.chkpt_id = current_id;
  header.datalen = data_size;
  header.chunk_size = chunk_size;
  header.num_first_ocur = first_ocur_vec.size();
  header.num_shift_dupl = shift_dupl_vec.size();
  header.num_prior_chkpts = chkpts_needed.count();
  uint64_t size_metadata = first_ocur_vec.size()*sizeof(uint32_t)+static_cast<uint64_t>(num_prior)*2*sizeof(uint32_t)+shift_dupl_vec.size()*2*sizeof(uint32_t);
  uint64_t size_data = buffer_len - size_metadata;
  return std::make_pair(size_data, size_metadata);
}

std::pair<double,double>
TreeDeduplicator::restart_chkpt( std::vector<Kokkos::View<uint8_t*>::HostMirror>& incr_chkpts,
                                 const int chkpt_idx, 
                                 Kokkos::View<uint8_t*>& data) {
  // Read main incremental checkpoint header
  size_t size = incr_chkpts[chkpt_idx].size();

  header_t header;
  memcpy(&header, incr_chkpts[chkpt_idx].data(), sizeof(header_t));
  STDOUT_PRINT("Ref ID: %u\n",               header.ref_id);
  STDOUT_PRINT("Chkpt ID: %u\n",             header.chkpt_id);
  STDOUT_PRINT("Data len: %lu\n",            header.datalen);
  STDOUT_PRINT("Chunk size: %u\n",           header.chunk_size);
  STDOUT_PRINT("Num first ocur: %u\n",        header.num_first_ocur);
  STDOUT_PRINT("Num prior chkpts: %u\n",      header.num_prior_chkpts);
  STDOUT_PRINT("Num shift dupl: %u\n",        header.num_shift_dupl);

  Kokkos::View<uint8_t*> buffer_d("Buffer", size);
  Kokkos::deep_copy(buffer_d, 0);
//  auto buffer_h = Kokkos::create_mirror_view(buffer_d);
//  Kokkos::deep_copy(buffer_h, 0);

  uint32_t num_chunks = header.datalen / header.chunk_size;
  if(static_cast<uint64_t>(num_chunks)*static_cast<uint64_t>(header.chunk_size) < header.datalen) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;
  Kokkos::resize(data, header.datalen);

  std::pair<double,double> times;

    // Main checkpoint
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(chkpt_idx));
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(chkpt_idx)+":Read checkpoint");
    DEBUG_PRINT("Global checkpoint\n");
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    Kokkos::resize(buffer_d, size);
    auto& buffer_h = incr_chkpts[chkpt_idx];
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    Kokkos::fence();
Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(chkpt_idx)+" Setup");
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
    size_t dupl_count_offset = first_ocur_offset + static_cast<uint64_t>(num_first_ocur)*sizeof(uint32_t);
    size_t dupl_map_offset   = dupl_count_offset + static_cast<uint64_t>(num_prior_chkpts)*2*sizeof(uint32_t);
    size_t data_offset       = dupl_map_offset   + static_cast<uint64_t>(num_shift_dupl)*2*sizeof(uint32_t);
    auto first_ocur_subview    = Kokkos::subview(buffer_d, std::make_pair(first_ocur_offset, dupl_count_offset));
    auto dupl_count_subview    = Kokkos::subview(buffer_d, std::make_pair(dupl_count_offset, dupl_map_offset));
    auto shift_dupl_subview    = Kokkos::subview(buffer_d, std::make_pair(dupl_map_offset, data_offset));
    auto data_subview  = Kokkos::subview(buffer_d, std::make_pair(data_offset, size));
    STDOUT_PRINT("Checkpoint %u\n", header.chkpt_id);
    STDOUT_PRINT("Checkpoint size: %lu\n", size);
    STDOUT_PRINT("First ocur offset: %lu\n", sizeof(header_t));
    STDOUT_PRINT("Dupl count offset: %lu\n", dupl_count_offset);
    STDOUT_PRINT("Dupl map offset: %lu\n", dupl_map_offset);
    STDOUT_PRINT("Data offset: %lu\n", data_offset);

    Kokkos::View<uint64_t[1]> counter_d("Write counter");
    auto counter_h = Kokkos::create_mirror_view(counter_d);
    Kokkos::deep_copy(counter_d, 0);

    Kokkos::UnorderedMap<NodeID, size_t> distinct_map(num_nodes);
    Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(2*num_nodes-1);
    Kokkos::View<uint32_t*> distinct_nodes("Nodes", num_first_ocur);
    Kokkos::View<uint32_t*> chunk_len("Num chunks for node", num_first_ocur);
Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(chkpt_idx)+" Restart distinct");
    // Calculate sizes of each distinct region
    Kokkos::parallel_for("Tree:Main:Calculate num chunks", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, first_ocur_subview.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      uint32_t len = num_leaf_descendents(node, num_nodes);
      distinct_nodes(i) = node;
      chunk_len(i) = len;
    });

    // Perform exclusive prefix scan to determine where to write chunks for each region
    Kokkos::parallel_scan("Tree:Main:Calc offsets", num_first_ocur, KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      const uint32_t len = chunk_len(i);
      if(is_final) chunk_len(i) = partial_sum;
      partial_sum += len;
    });

    Kokkos::View<uint32_t[1]> total_region_size("Total region size");
    Kokkos::View<uint32_t[1]>::HostMirror total_region_size_h = Kokkos::create_mirror_view(total_region_size);
    Kokkos::deep_copy(total_region_size, 0);

STDOUT_PRINT("Calculated offsets\n");

    // Restart distinct entries by reading and inserting full tree into distinct map
    Kokkos::parallel_for("Tree:Main:Restart Hash tree repeats main checkpoint", Kokkos::TeamPolicy<>(num_first_ocur, Kokkos::AUTO()), KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
      const uint32_t i = team_member.league_rank();
      uint32_t node = distinct_nodes(i);
      if(team_member.team_rank() == 0)
        distinct_map.insert(NodeID(node, cur_id), static_cast<uint64_t>(chunk_len(i))*static_cast<uint64_t>(chunk_size));
      uint32_t start = leftmost_leaf(node, num_nodes);
      uint32_t len = num_leaf_descendents(node, num_nodes);
      uint32_t end = start+len-1;
      uint32_t left = 2*node+1;
      uint32_t right = 2*node+2;
      while(left < num_nodes) {
        if(right >= num_nodes)
          right = num_nodes;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, right-left+1), [&] (const uint32_t j) {
          uint32_t u = left+j;
          uint32_t leaf = leftmost_leaf(u, num_nodes);
          auto result = distinct_map.insert(NodeID(u, cur_id), static_cast<uint64_t>(chunk_len(i)+leaf-start)*static_cast<uint64_t>(chunk_size));
          if(result.failed())
            printf("Failed to insert (%u,%u): %lu\n", u, cur_id, static_cast<uint64_t>(chunk_len(i)+(leaf-start))*static_cast<uint64_t>(chunk_size));
        });
        team_member.team_barrier();
        left = 2*left+1;
        right = 2*right+2;
      }
      // Update chunk metadata list
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, len), [&] (const uint32_t j) {
        node_list(start-num_chunks+1+j) = NodeID(start-num_chunks+1+j, cur_id);
      });
if(team_member.team_rank() == 0) {
Kokkos::atomic_add(&total_region_size(0), len);
}
      uint64_t src_offset = static_cast<uint64_t>(chunk_len(i))*static_cast<uint64_t>(chunk_size);
      uint64_t dst_offset = static_cast<uint64_t>(start-num_chunks+1)*static_cast<uint64_t>(chunk_size);
      uint64_t datasize = static_cast<uint64_t>(len)*static_cast<uint64_t>(chunk_size);
      if(end == num_nodes-1)
        datasize = datalen - dst_offset;

      uint8_t* dst = (uint8_t*)(data.data()+dst_offset);
      uint8_t* src = (uint8_t*)(data_subview.data()+src_offset);
      team_memcpy(dst, src, datasize, team_member);
    });
Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(chkpt_idx)+" Restart repeats");
    Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
    auto repeat_region_sizes_h = Kokkos::create_mirror_view(repeat_region_sizes);
    Kokkos::deep_copy(repeat_region_sizes, 0);
    // Read map of repeats for each checkpoint
    Kokkos::parallel_for("Tree:Main:Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      uint32_t chkpt;
      memcpy(&chkpt, dupl_count_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&repeat_region_sizes(chkpt), dupl_count_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
      DEBUG_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
    });
    Kokkos::deep_copy(repeat_region_sizes_h, repeat_region_sizes);
    // Perform exclusive scan to determine where regions start/stop
    Kokkos::parallel_scan("Tree:Main:Repeat offsets", cur_id+1, KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      partial_sum += repeat_region_sizes(i);
      if(is_final) repeat_region_sizes(i) = partial_sum;
    });

    STDOUT_PRINT("Num repeats: %u\n", num_shift_dupl);
    // Load repeat entries and fill in metadata for chunks
    Kokkos::parallel_for("Tree:Main:Restart Hash tree repeats main checkpoint", Kokkos::TeamPolicy<>(num_shift_dupl, Kokkos::AUTO()), KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
      uint32_t i = team_member.league_rank();
      uint32_t node=0, prev=0, tree=0;
      size_t offset = 0;
      if(team_member.team_rank() == 0) {
        memcpy(&node, shift_dupl_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&prev, shift_dupl_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
        for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
          if(i < repeat_region_sizes(j)) {
            tree = j;
          }
        }
        uint32_t idx = distinct_map.find(NodeID(prev, tree));
        if(distinct_map.valid_at(idx)) {
          offset = distinct_map.value_at(idx);
        }
      }
      team_member.team_broadcast(node, 0);
      team_member.team_broadcast(prev, 0);
      team_member.team_broadcast(tree, 0);
      team_member.team_broadcast(offset, 0);
      uint32_t node_start = leftmost_leaf(node, num_nodes);
      uint32_t prev_start = leftmost_leaf(prev, num_nodes);
      uint32_t len = num_leaf_descendents(prev, num_nodes);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, len), [&] (const uint64_t j) {
        node_list(node_start+j-num_chunks+1) = NodeID(prev_start+j, tree);
      });
      if(tree == cur_id) {
Kokkos::atomic_add(&total_region_size(0), len);
        uint64_t dst_offset = static_cast<uint64_t>(chunk_size)*static_cast<uint64_t>(node_start-num_chunks+1);
        uint64_t datasize = static_cast<uint64_t>(chunk_size)*static_cast<uint64_t>(len);
        if(node_start+len-1 == num_nodes-1)
          datasize = data.size() - dst_offset;

        uint8_t* dst = (uint8_t*)(data.data()+dst_offset);
        uint8_t* src = (uint8_t*)(data_subview.data()+offset);
        team_memcpy(dst, src, datasize, team_member);
      }
    });
Kokkos::deep_copy(total_region_size_h, total_region_size);
DEBUG_PRINT("Chkpt %u: total region size: %u\n", cur_id, total_region_size_h(0));

Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(chkpt_idx)+" Fill repeats");
    // All remaining entries are identical 
    Kokkos::parallel_for("Tree:Main:Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      NodeID entry = node_list(i);
      if(entry.node == UINT_MAX) {
        node_list(i) = NodeID(i+num_chunks-1, cur_id-1);
      }
    });
    Kokkos::fence();
Kokkos::Profiling::popRegion();
Kokkos::Profiling::popRegion();

DEBUG_PRINT("Start: %u, end: %u\n", chkpt_idx-1, ref_id);
    for(int idx=static_cast<int>(chkpt_idx)-1; idx>=static_cast<int>(ref_id); idx--) {
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx));
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+":Read checkpoint");
      DEBUG_PRINT("Processing checkpoint %u\n", idx);
      t1 = std::chrono::high_resolution_clock::now();
      size_t chkpt_size = incr_chkpts[idx].size();
      auto chkpt_buffer_d = buffer_d;
      auto chkpt_buffer_h = buffer_h;
      Kokkos::resize(chkpt_buffer_d, chkpt_size);
      Kokkos::resize(chkpt_buffer_h, chkpt_size);
      chkpt_buffer_h = incr_chkpts[idx];
      t2 = std::chrono::high_resolution_clock::now();
      STDOUT_PRINT("Time spent reading checkpoint %d from file: %f\n", idx, (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()));
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+" Setup");
      header_t chkpt_header;
      memcpy(&chkpt_header, chkpt_buffer_h.data(), sizeof(header_t));
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
      dupl_count_offset = first_ocur_offset + static_cast<uint64_t>(num_first_ocur)*sizeof(uint32_t);
      dupl_map_offset   = dupl_count_offset + static_cast<uint64_t>(num_prior_chkpts)*2*sizeof(uint32_t);
      data_offset       = dupl_map_offset   + static_cast<uint64_t>(num_shift_dupl)*2*sizeof(uint32_t);
      first_ocur_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(first_ocur_offset, dupl_count_offset));
      dupl_count_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(dupl_count_offset, dupl_map_offset));
      shift_dupl_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(dupl_map_offset, data_offset));
      data_subview       = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, size));
      STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
      STDOUT_PRINT("Checkpoint size: %lu\n", size);
      STDOUT_PRINT("First ocur offset: %lu\n", sizeof(header_t));
      STDOUT_PRINT("Dupl count offset: %lu\n", dupl_count_offset);
      STDOUT_PRINT("Dupl map offset: %lu\n", dupl_map_offset);
      STDOUT_PRINT("Data offset: %lu\n", data_offset);

      distinct_map.clear();
      repeat_map.clear();
      
      Kokkos::View<uint64_t[1]> counter_d("Write counter");
      auto counter_h = Kokkos::create_mirror_view(counter_d);
      Kokkos::deep_copy(counter_d, 0);
  
      Kokkos::resize(distinct_nodes, num_first_ocur);
      Kokkos::resize(chunk_len, num_first_ocur);
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+" Load maps");
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Calculate num chunks", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
        uint32_t node;
        memcpy(&node, first_ocur_subview.data()+static_cast<uint64_t>(i)*sizeof(uint32_t), sizeof(uint32_t));
        uint32_t len = num_leaf_descendents(node, num_nodes);
        distinct_nodes(i) = node;
        chunk_len(i) = len;
      });
      Kokkos::parallel_scan("Tree:"+std::to_string(idx)+":Calc offsets", num_first_ocur, KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
        const uint32_t len = chunk_len(i);
        if(is_final) chunk_len(i) = partial_sum;
        partial_sum += len;
      });
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Restart Hashtree distinct", Kokkos::TeamPolicy<>(num_first_ocur, Kokkos::AUTO()), KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
        uint32_t i = team_member.league_rank();
        uint32_t node = distinct_nodes(i);
        uint64_t offset = static_cast<uint64_t>(chunk_len(i)) * static_cast<uint64_t>(chunk_size);
        if(team_member.team_rank() == 0)
          distinct_map.insert(NodeID(node, cur_id), offset);
        uint32_t start = leftmost_leaf(node, num_nodes);
        uint32_t left = 2*node+1;
        uint32_t right = 2*node+2;
        while(left < num_nodes) {
          if(right >= num_nodes)
            right = num_nodes;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, right-left+1), [&] (const uint64_t j) {
            uint32_t u=left+j;
            uint32_t leaf = leftmost_leaf(u, num_nodes);
            uint64_t leaf_offset = static_cast<uint64_t>(leaf-start)*static_cast<uint64_t>(chunk_size);
            auto result = distinct_map.insert(NodeID(u, cur_id), offset + leaf_offset);
            if(result.failed())
              printf("Failed to insert (%u,%u): %lu\n", u, cur_id, offset+leaf_offset);
          });
          team_member.team_barrier();
          left = 2*left+1;
          right = 2*right+2;
        }
      });
  
      Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
        uint32_t chkpt;
        memcpy(&chkpt, dupl_count_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&repeat_region_sizes(chkpt), dupl_count_subview.data()+static_cast<uint64_t>(i)*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
        DEBUG_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
      });
      Kokkos::parallel_scan("Tree:"+std::to_string(idx)+":Repeat offsets", cur_id+1, KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
        partial_sum += repeat_region_sizes(i);
        if(is_final) repeat_region_sizes(i) = partial_sum;
      });

      DEBUG_PRINT("Num repeats: %u\n", num_shift_dupl);
  
      Kokkos::TeamPolicy<> repeat_policy(num_shift_dupl, Kokkos::AUTO);
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Restart Hash tree repeats middle chkpts", repeat_policy, KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
        uint32_t i = team_member.league_rank();
        uint32_t node, prev, tree=0;
        memcpy(&node, shift_dupl_subview.data()+static_cast<uint64_t>(i)*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
        memcpy(&prev, shift_dupl_subview.data()+static_cast<uint64_t>(i)*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
        for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
          if(i < repeat_region_sizes(j)) {
            tree = j;
          }
        }
        if(team_member.team_rank() == 0) {
          auto result = repeat_map.insert(node, NodeID(prev,tree));
          if(result.failed())
            STDOUT_PRINT("Failed to insert previous repeat %u: (%u,%u) into repeat map\n", node, prev, tree);
        }
        uint32_t curr_start = leftmost_leaf(node, num_nodes);
        uint32_t prev_start = leftmost_leaf(prev, num_nodes);
        uint32_t len = num_leaf_descendents(node, num_nodes);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, len), [&] (const uint32_t u) {
          repeat_map.insert(curr_start+u, NodeID(prev_start+u, tree));
        });
      });
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+" Fill chunks");

      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Fill data middle chkpts", Kokkos::TeamPolicy<>(num_chunks, Kokkos::AUTO()), KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
        uint32_t i = team_member.league_rank();
        if(node_list(i).tree == cur_id) {
          NodeID id = node_list(i);
          if(distinct_map.exists(id)) {
            size_t src_offset = distinct_map.value_at(distinct_map.find(id));
            size_t dst_offset = static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);
            uint32_t writesize = chunk_size;
            if(dst_offset+writesize > datalen) 
              writesize = datalen-dst_offset;

            uint8_t* dst = (uint8_t*)(data.data()+dst_offset);
            uint8_t* src = (uint8_t*)(data_subview.data()+src_offset);
            team_memcpy(dst, src, writesize, team_member);
          } else if(repeat_map.exists(id.node)) {
            NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
            if(prev.tree == cur_id) {
              if(!repeat_map.exists(id.node))
                printf("Failed to find repeat chunk %u\n", id.node);
              size_t src_offset = distinct_map.value_at(distinct_map.find(prev));
              size_t dst_offset = static_cast<uint64_t>(i)*static_cast<uint64_t>(chunk_size);
              uint32_t writesize = chunk_size;
              if(dst_offset+writesize > datalen) 
                writesize = datalen-dst_offset;

              uint8_t* dst = (uint8_t*)(data.data()+dst_offset);
              uint8_t* src = (uint8_t*)(data_subview.data()+src_offset);
              team_memcpy(dst, src, writesize, team_member);
            } else {
              node_list(i) = prev;
            }
          } else {
            node_list(i) = NodeID(node_list(i).node, cur_id-1);
          }
        }
      });
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::popRegion();
    }

    Kokkos::fence();
    std::chrono::high_resolution_clock::time_point c3 = std::chrono::high_resolution_clock::now();
    double copy_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
    double restart_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c3-t0).count());
    return std::make_pair(copy_time, restart_time);
}

std::pair<double,double>
TreeDeduplicator::restart_chkpt( std::vector<std::string>& chkpt_files,
                                 const int file_idx, 
                                 Kokkos::View<uint8_t*>& data) {
  // Read main incremental checkpoint header
  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  file.open(chkpt_files[file_idx], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
  size_t filesize = file.tellg();
  file.seekg(0);

  header_t header;
  file.read((char*)&header, sizeof(header_t));
  file.close();
  STDOUT_PRINT("Ref ID: %u\n",               header.ref_id);
  STDOUT_PRINT("Chkpt ID: %u\n",             header.chkpt_id);
  STDOUT_PRINT("Data len: %lu\n",            header.datalen);
  STDOUT_PRINT("Chunk size: %u\n",           header.chunk_size);
  STDOUT_PRINT("Num first ocur: %u\n",        header.num_first_ocur);
  STDOUT_PRINT("Num prior chkpts: %u\n",      header.num_prior_chkpts);
  STDOUT_PRINT("Num shift dupl: %u\n",        header.num_shift_dupl);

  std::vector<Kokkos::View<uint8_t*>> chkpts_d;
  std::vector<Kokkos::View<uint8_t*>::HostMirror> chkpts_h;
  for(uint32_t i=0; i<chkpt_files.size(); i++) {
    file.open(chkpt_files[i], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
    size_t filesize = file.tellg();
    file.seekg(0);
    Kokkos::View<uint8_t*> chkpt_d("Checkpoint", filesize);
    auto chkpt_h = Kokkos::create_mirror_view(chkpt_d);;
    file.read((char*)(chkpt_h.data()), filesize);
    file.close();
    chkpts_d.push_back(chkpt_d);
    chkpts_h.push_back(chkpt_h);
  }

  Kokkos::View<uint8_t*> buffer_d("Buffer", filesize);
  Kokkos::deep_copy(buffer_d, 0);
  auto buffer_h = Kokkos::create_mirror_view(buffer_d);
  Kokkos::deep_copy(buffer_h, 0);

  uint32_t num_chunks = header.datalen / header.chunk_size;
  if(num_chunks*header.chunk_size < header.datalen) {
    num_chunks += 1;
  }
  uint32_t num_nodes = 2*num_chunks-1;
  Kokkos::resize(data, header.datalen);

  std::pair<double,double> times;
    
    // Main checkpoint
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(file_idx));
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(file_idx)+":Read checkpoint");
    DEBUG_PRINT("Global checkpoint\n");
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    Kokkos::resize(buffer_d, filesize);
    Kokkos::resize(buffer_h, filesize);
    file.open(chkpt_files[file_idx], std::ifstream::in | std::ifstream::binary);
    file.read((char*)(buffer_h.data()), filesize);
    file.close();
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    STDOUT_PRINT("Time spent reading checkpoint %u from file: %f\n", file_idx, (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()));
    Kokkos::fence();
Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(file_idx)+" Setup");
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

    Kokkos::View<uint64_t[1]> counter_d("Write counter");
    auto counter_h = Kokkos::create_mirror_view(counter_d);
    Kokkos::deep_copy(counter_d, 0);

    Kokkos::UnorderedMap<NodeID, size_t> distinct_map(num_nodes);
    Kokkos::UnorderedMap<uint32_t, NodeID> repeat_map(2*num_nodes-1);
    Kokkos::View<uint32_t*> distinct_nodes("Nodes", num_first_ocur);
    Kokkos::View<uint32_t*> chunk_len("Num chunks for node", num_first_ocur);
Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(file_idx)+" Restart distinct");
    // Calculate sizes of each distinct region
    Kokkos::parallel_for("Tree:Main:Calculate num chunks", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      uint32_t node;
      memcpy(&node, first_ocur_subview.data()+i*sizeof(uint32_t), sizeof(uint32_t));
      uint32_t len = num_leaf_descendents(node, num_nodes);
      distinct_nodes(i) = node;
      chunk_len(i) = len;
    });
    // Perform exclusive prefix scan to determine where to write chunks for each region
    Kokkos::parallel_scan("Tree:Main:Calc offsets", num_first_ocur, KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      const uint32_t len = chunk_len(i);
      if(is_final) chunk_len(i) = partial_sum;
      partial_sum += len;
    });

    Kokkos::View<uint32_t[1]> total_region_size("TOtal region size");
    Kokkos::View<uint32_t[1]>::HostMirror total_region_size_h = Kokkos::create_mirror_view(total_region_size);
    Kokkos::deep_copy(total_region_size, 0);

    // Restart distinct entries by reading and inserting full tree into distinct map
    Kokkos::parallel_for("Tree:Main:Restart Hashtree distinct", Kokkos::RangePolicy<>(0,num_first_ocur), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      uint32_t node = distinct_nodes(i);
      distinct_map.insert(NodeID(node, cur_id), chunk_len(i)*chunk_size);
      uint32_t start = leftmost_leaf(node, num_nodes);
      uint32_t len = num_leaf_descendents(node, num_nodes);
      uint32_t end = start+len-1;
      uint32_t left = 2*node+1;
      uint32_t right = 2*node+2;
      while(left < num_nodes) {
        if(right >= num_nodes)
          right = num_nodes;
        for(uint32_t u=left; u<=right; u++) {
          uint32_t leaf = leftmost_leaf(u, num_nodes);
          auto result = distinct_map.insert(NodeID(u, cur_id), chunk_len(i)*chunk_size + (leaf-start)*chunk_size);
          if(result.failed())
            printf("Failed to insert (%u,%u): %u\n", u, cur_id, chunk_len(i)*chunk_size+(leaf-start)*chunk_size);
        }
        left = 2*left+1;
        right = 2*right+2;
      }
      // Update chunk metadata list
      for(uint32_t j=0; j<len; j++) {
        node_list(start-num_chunks+1+j) = NodeID(start-num_chunks+1+j, cur_id);
      }
      uint32_t datasize = len*chunk_size;
      if(end == num_nodes-1)
        datasize = datalen - (start-num_chunks+1)*chunk_size;
      memcpy(data.data()+chunk_size*(start-num_chunks+1), data_subview.data()+chunk_len(i)*chunk_size, datasize);
Kokkos::atomic_add(&total_region_size(0), len);
    });

Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(file_idx)+" Restart repeats");
    Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
    auto repeat_region_sizes_h = Kokkos::create_mirror_view(repeat_region_sizes);
    Kokkos::deep_copy(repeat_region_sizes, 0);
    // Read map of repeats for each checkpoint
    Kokkos::parallel_for("Tree:Main:Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      uint32_t chkpt;
      memcpy(&chkpt, dupl_count_subview.data()+i*2*sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&repeat_region_sizes(chkpt), dupl_count_subview.data()+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
      DEBUG_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
    });
    Kokkos::deep_copy(repeat_region_sizes_h, repeat_region_sizes);
    // Perform exclusive scan to determine where regions start/stop
    Kokkos::parallel_scan("Tree:Main:Repeat offsets", cur_id+1, KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
      partial_sum += repeat_region_sizes(i);
      if(is_final) repeat_region_sizes(i) = partial_sum;
    });

    DEBUG_PRINT("Num repeats: %u\n", num_shift_dupl);
    // Load repeat entries and fill in metadata for chunks
    Kokkos::parallel_for("Tree:Main:Restart Hash tree repeats main checkpoint", Kokkos::RangePolicy<>(0, num_shift_dupl), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
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
      size_t offset = distinct_map.value_at(distinct_map.find(NodeID(prev, tree)));
      uint32_t node_start = leftmost_leaf(node, num_nodes);
      uint32_t prev_start = leftmost_leaf(prev, num_nodes);
      uint32_t len = num_leaf_descendents(prev, num_nodes);
      for(uint32_t j=0; j<len; j++) {
        node_list(node_start+j-num_chunks+1) = NodeID(prev_start+j, tree);
      }
      if(tree == cur_id) {
Kokkos::atomic_add(&total_region_size(0), len);
        uint32_t copysize = chunk_size*len;
        if(node_start+len-1 == num_nodes-1)
          copysize = data.size() - chunk_size*(node_start-num_chunks+1);
        memcpy(data.data()+chunk_size*(node_start-num_chunks+1), data_subview.data()+offset, copysize);
      }
    });
Kokkos::deep_copy(total_region_size_h, total_region_size);
DEBUG_PRINT("Chkpt %u: total region size: %u\n", cur_id, total_region_size_h(0));

Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(file_idx)+" Fill repeats");
    // All remaining entries are identical 
    Kokkos::parallel_for("Tree:Main:Fill same entries", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
      NodeID entry = node_list(i);
      if(entry.node == UINT_MAX) {
        node_list(i) = NodeID(i+num_chunks-1, cur_id-1);
      }
    });
    Kokkos::fence();
Kokkos::Profiling::popRegion();
Kokkos::Profiling::popRegion();

    for(int idx=static_cast<int>(file_idx)-1; idx>=static_cast<int>(ref_id); idx--) {
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx));
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+":Read checkpoint");
      DEBUG_PRINT("Processing checkpoint %u\n", idx);
      t1 = std::chrono::high_resolution_clock::now();
      file.open(chkpt_files[idx], std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
      size_t chkpt_size = file.tellg();
      file.seekg(0);
      auto chkpt_buffer_d = buffer_d;
      auto chkpt_buffer_h = buffer_h;
      Kokkos::resize(chkpt_buffer_d, chkpt_size);
      Kokkos::resize(chkpt_buffer_h, chkpt_size);
      file.read((char*)(chkpt_buffer_h.data()), chkpt_size);
      file.close();
      t2 = std::chrono::high_resolution_clock::now();
      STDOUT_PRINT("Time spent reading checkpoint %d from file: %f\n", idx, (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()));
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+" Setup");
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
      first_ocur_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(first_ocur_offset, dupl_count_offset));
      dupl_count_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(dupl_count_offset, dupl_map_offset));
      shift_dupl_subview = Kokkos::subview(chkpt_buffer_d, std::make_pair(dupl_map_offset, data_offset));
      data_subview  = Kokkos::subview(chkpt_buffer_d, std::make_pair(data_offset, filesize));
      STDOUT_PRINT("Checkpoint %u\n", chkpt_header.chkpt_id);
      STDOUT_PRINT("Checkpoint size: %lu\n", filesize);
      STDOUT_PRINT("First ocur offset: %lu\n", sizeof(header_t));
      STDOUT_PRINT("Dupl count offset: %lu\n", dupl_count_offset);
      STDOUT_PRINT("Dupl map offset: %lu\n", dupl_map_offset);
      STDOUT_PRINT("Data offset: %lu\n", data_offset);

      distinct_map.clear();
      repeat_map.clear();

      Kokkos::View<uint64_t[1]> counter_d("Write counter");
      auto counter_h = Kokkos::create_mirror_view(counter_d);
      Kokkos::deep_copy(counter_d, 0);
  
      Kokkos::resize(distinct_nodes, num_first_ocur);
      Kokkos::resize(chunk_len, num_first_ocur);
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+" Load maps");
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Calculate num chunks", Kokkos::RangePolicy<>(0, num_first_ocur), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
        uint32_t node;
        memcpy(&node, first_ocur_subview.data()+i*sizeof(uint32_t), sizeof(uint32_t));
        uint32_t len = num_leaf_descendents(node, num_nodes);
        distinct_nodes(i) = node;
        chunk_len(i) = len;
      });
      Kokkos::parallel_scan("Tree:"+std::to_string(idx)+":Calc offsets", num_first_ocur, KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
        const uint32_t len = chunk_len(i);
        if(is_final) chunk_len(i) = partial_sum;
        partial_sum += len;
      });
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Restart Hashtree distinct", Kokkos::RangePolicy<>(0,num_first_ocur), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
        uint32_t node = distinct_nodes(i);
        distinct_map.insert(NodeID(node, cur_id), chunk_len(i)*chunk_size);
        uint32_t start = leftmost_leaf(node, num_nodes);
        uint32_t left = 2*node+1;
        uint32_t right = 2*node+2;
        while(left < num_nodes) {
          if(right >= num_nodes)
            right = num_nodes;
          for(uint32_t u=left; u<=right; u++) {
            uint32_t leaf = leftmost_leaf(u, num_nodes);
            auto result = distinct_map.insert(NodeID(u, cur_id), chunk_len(i)*chunk_size + (leaf-start)*chunk_size);
            if(result.failed())
              printf("Failed to insert (%u,%u): %u\n", u, cur_id, chunk_len(i)*chunk_size+(leaf-start)*chunk_size);
          }
          left = 2*left+1;
          right = 2*right+2;
        }
      });
  
      Kokkos::View<uint32_t*> repeat_region_sizes("Repeat entires per chkpt", cur_id+1);
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Load repeat map", Kokkos::RangePolicy<>(0,num_prior_chkpts), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
        uint32_t chkpt;
        memcpy(&chkpt, dupl_count_subview.data()+i*2*sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&repeat_region_sizes(chkpt), dupl_count_subview.data()+i*2*sizeof(uint32_t)+sizeof(uint32_t), sizeof(uint32_t));
        DEBUG_PRINT("Chkpt: %u, region size: %u\n", chkpt, repeat_region_sizes(chkpt));
      });
      Kokkos::parallel_scan("Tree:"+std::to_string(idx)+":Repeat offsets", cur_id+1, KOKKOS_CLASS_LAMBDA(const uint32_t i, uint32_t& partial_sum, bool is_final) {
        partial_sum += repeat_region_sizes(i);
        if(is_final) repeat_region_sizes(i) = partial_sum;
      });

      DEBUG_PRINT("Num repeats: %u\n", num_shift_dupl);
  
      Kokkos::TeamPolicy<> repeat_policy(num_shift_dupl, Kokkos::AUTO);
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Restart Hash tree repeats middle chkpts", repeat_policy, 
                           KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
        uint32_t i = team_member.league_rank();
        uint32_t node, prev, tree=0;
        memcpy(&node, shift_dupl_subview.data()+i*(sizeof(uint32_t)+sizeof(uint32_t)), sizeof(uint32_t));
        memcpy(&prev, shift_dupl_subview.data()+i*(sizeof(uint32_t)+sizeof(uint32_t))+sizeof(uint32_t), sizeof(uint32_t));
        for(uint32_t j=repeat_region_sizes.size()-1; j<repeat_region_sizes.size(); j--) {
          if(i < repeat_region_sizes(j)) {
            tree = j;
          }
        }
        if(team_member.team_rank() == 0) {
          auto result = repeat_map.insert(node, NodeID(prev,tree));
          if(result.failed())
            STDOUT_PRINT("Failed to insert previous repeat %u: (%u,%u) into repeat map\n", node, prev, tree);
        }
        uint32_t curr_start = leftmost_leaf(node, num_nodes);
        uint32_t prev_start = leftmost_leaf(prev, num_nodes);
        uint32_t len = num_leaf_descendents(node, num_nodes);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, len), [&] (const uint32_t u) {
          repeat_map.insert(curr_start+u, NodeID(prev_start+u, tree));
        });
      });
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::pushRegion("Checkpoint "+std::to_string(idx)+" Fill chunks");

Kokkos::View<uint32_t[1]> curr_identical_counter("Num identical entries in current checkpoint");
Kokkos::View<uint32_t[1]>::HostMirror curr_identical_counter_h = Kokkos::create_mirror_view(curr_identical_counter);
Kokkos::deep_copy(curr_identical_counter, 0);
Kokkos::View<uint32_t[1]> prev_identical_counter("Num identical entries in previous checkpoint");
Kokkos::View<uint32_t[1]> base_identical_counter("Num identical entries in baseline checkpoint");
Kokkos::View<uint32_t[1]>::HostMirror prev_identical_counter_h = Kokkos::create_mirror_view(prev_identical_counter);;
Kokkos::View<uint32_t[1]>::HostMirror base_identical_counter_h = Kokkos::create_mirror_view(base_identical_counter);;
Kokkos::deep_copy(prev_identical_counter, 0);
Kokkos::deep_copy(base_identical_counter, 0);
Kokkos::deep_copy(total_region_size, 0);
Kokkos::View<uint32_t[1]> curr_chunks("Num identical entries in current checkpoint");
Kokkos::View<uint32_t[1]>::HostMirror curr_chunks_h = Kokkos::create_mirror_view(curr_chunks);
Kokkos::deep_copy(curr_chunks, 0);
      Kokkos::parallel_for("Tree:"+std::to_string(idx)+":Fill data middle chkpts", Kokkos::RangePolicy<>(0, num_chunks), KOKKOS_CLASS_LAMBDA(const uint32_t i) {
        if(node_list(i).tree == current_id) {
Kokkos::atomic_add(&curr_chunks(0), 1);
          NodeID id = node_list(i);
          if(distinct_map.exists(id)) {
            uint32_t len = num_leaf_descendents(id.node, num_nodes);
Kokkos::atomic_add(&total_region_size(0), len);
            size_t offset = distinct_map.value_at(distinct_map.find(id));
            uint32_t writesize = chunk_size;
            if(i*chunk_size+writesize > datalen) 
              writesize = datalen-i*chunk_size;
            memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
            Kokkos::atomic_add(&counter_d(0), writesize);
Kokkos::atomic_add(&curr_identical_counter(0), writesize);
          } else if(repeat_map.exists(id.node)) {
            NodeID prev = repeat_map.value_at(repeat_map.find(id.node));
            if(prev.tree == current_id) {
              if(!repeat_map.exists(id.node))
                printf("Failed to find repeat chunk %u\n", id.node);
              size_t offset = distinct_map.value_at(distinct_map.find(prev));
              uint32_t len = num_leaf_descendents(prev.node, num_nodes);
Kokkos::atomic_add(&total_region_size(0), len);
//              uint32_t start = leftmost_leaf(prev.node, num_nodes);
              uint32_t writesize = chunk_size;
              if(i*chunk_size+writesize > datalen) 
                writesize = datalen-i*chunk_size;
              memcpy(data.data()+chunk_size*i, data_subview.data()+offset, writesize);
              Kokkos::atomic_add(&counter_d(0), writesize);
Kokkos::atomic_add(&curr_identical_counter(0), writesize);
            } else {
              node_list(i) = prev;
            }
          } else {
Kokkos::atomic_add(&prev_identical_counter(0), 1);
            node_list(i) = NodeID(node_list(i).node, current_id-1);
          }
        } else if(node_list(i).tree < current_id) {
Kokkos::atomic_add(&base_identical_counter(0), 1);
        }
      });
      Kokkos::Profiling::popRegion();
      Kokkos::Profiling::popRegion();
    }

    Kokkos::fence();
Kokkos::Profiling::popRegion();
Kokkos::Profiling::popRegion();
    std::chrono::high_resolution_clock::time_point c3 = std::chrono::high_resolution_clock::now();
    double copy_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c2-c1).count());
    double restart_time = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(c3-t0).count());
    return std::make_pair(copy_time, restart_time);
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
TreeDeduplicator::checkpoint(header_t& header, 
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
TreeDeduplicator::checkpoint(uint8_t* data_ptr, 
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
TreeDeduplicator::checkpoint(uint8_t* data_ptr, 
                             size_t len, 
                             Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                             bool make_baseline) {
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
TreeDeduplicator::checkpoint(uint8_t* data_ptr, 
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
 * Restart checkpoint from vector of incremental checkpoints loaded on the Host.
 *
 * \param data       Data View to restart checkpoint into
 * \param chkpts     Vector of prior incremental checkpoints stored on the Host
 * \param logname    Filename for restart logs
 * \param chkpt_id   ID of checkpoint to restart
 */
void 
TreeDeduplicator::restart(Kokkos::View<uint8_t*> data, 
                          std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                          std::string& logname, 
                          uint32_t chkpt_id) {
  auto tree_times = restart_chkpt(chkpts, chkpt_id, data);
  restart_timers[0] = tree_times.first;
  restart_timers[1] = tree_times.second;
  std::string restart_logname = logname + ".chunk_size." + std::to_string(chunk_size) +
                                ".restart_timing.csv";
  write_restart_log(chkpt_id, restart_logname);
}

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
void 
TreeDeduplicator::restart(uint8_t* data_ptr, 
                          size_t len, 
                          std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                          std::string& logname, 
                          uint32_t chkpt_id) {
  Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
  restart(data, chkpts, logname, chkpt_id);
}

/**
 * Restart checkpoint from checkpoint files
 *
 * \param data       Data View to restart checkpoint into
 * \param filenames  Vector of prior incremental checkpoints stored in files
 * \param logname    Filename for restart logs
 * \param chkpt_id   ID of checkpoint to restart
 */
void 
TreeDeduplicator::restart(Kokkos::View<uint8_t*> data, 
                          std::vector<std::string>& chkpt_filenames, 
                          std::string& logname, 
                          uint32_t chkpt_id) {
  std::vector<std::string> hashtree_chkpt_files;
  for(uint32_t i=0; i<chkpt_filenames.size(); i++) {
    hashtree_chkpt_files.push_back(chkpt_filenames[i]+".hashtree.incr_chkpt");
  }
  auto tree_times = restart_chkpt(hashtree_chkpt_files, chkpt_id, data);
  restart_timers[0] = tree_times.first;
  restart_timers[1] = tree_times.second;
  write_restart_log(chkpt_id, logname);
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
TreeDeduplicator::write_chkpt_log(header_t& header, 
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

  std::string approach("Tree");

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
TreeDeduplicator::write_restart_log(uint32_t select_chkpt, 
                                    std::string& logname) {
  std::fstream timing_file;
  timing_file.open(logname, std::fstream::out | std::fstream::app);
  timing_file << "Tree" << ","; 
  timing_file << select_chkpt << "," 
              << chunk_size << "," 
              << restart_timers[0] << "," 
              << restart_timers[1] << std::endl;
  timing_file.close();
}

