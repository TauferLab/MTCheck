#include "full_approach.hpp"

FullDeduplicator::FullDeduplicator() {}

FullDeduplicator::FullDeduplicator(uint32_t bytes_per_chunk) {
  chunk_size = bytes_per_chunk;
  current_id = 0;
  baseline_id = 0;
}

FullDeduplicator::~FullDeduplicator() {}

void 
FullDeduplicator::checkpoint(header_t& header, 
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

  Kokkos::Profiling::popRegion();

  // ==========================================================================================
  // Deduplicate data
  // ==========================================================================================
  std::string dedup_region_name = std::string("Deduplication chkpt ") + 
                                  std::to_string(current_id);
  Timer::time_point start_create_tree0 = Timer::now();
  timers[0] = std::chrono::duration_cast<Duration>(start_create_tree0 - beg_chkpt).count();
  Kokkos::Profiling::pushRegion(dedup_region_name.c_str());

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

  datasizes = std::make_pair(data_len, 0);

  Kokkos::Profiling::popRegion();
  Timer::time_point end_collect = Timer::now();
  timers[2] = std::chrono::duration_cast<Duration>(end_collect - start_collect).count();

  // ==========================================================================================
  // Copy diff to host 
  // ==========================================================================================
  Timer::time_point start_write = Timer::now();
  Kokkos::resize(diff_h, data_len);
  std::string write_region_name = std::string("Copy diff to host ") 
                                  + std::to_string(current_id);
  Kokkos::Profiling::pushRegion(write_region_name.c_str());

  Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
  Kokkos::deep_copy(diff_h, data);

  Kokkos::Profiling::popRegion();
  Timer::time_point end_write = Timer::now();
  timers[3] = std::chrono::duration_cast<Duration>(end_write - start_write).count();
}

void 
FullDeduplicator::checkpoint(uint8_t* data_ptr, 
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
FullDeduplicator::checkpoint(uint8_t* data_ptr, 
                              size_t len, 
                              Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                              bool make_baseline) {
  header_t header;
  checkpoint(header, data_ptr, len, diff_h, make_baseline);
  current_id += 1;
}

void 
FullDeduplicator::checkpoint(uint8_t* data_ptr, 
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
FullDeduplicator::restart(Kokkos::View<uint8_t*> data, 
                           std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                           std::string& logname, 
                           uint32_t chkpt_id) {
  using Nanoseconds = std::chrono::nanoseconds;
  using Timer = std::chrono::high_resolution_clock;
  // Full checkpoint
  Kokkos::resize(data, chkpts[chkpt_id].size());
  // Total time
  // Copy checkpoint to GPU
  Timer::time_point c1 = Timer::now();
  Kokkos::deep_copy(data, chkpts[chkpt_id]);
  Timer::time_point c2 = Timer::now();
  // Update timers
  restart_timers[0] = (1e-9)*(std::chrono::duration_cast<Nanoseconds>(c2-c1).count());
  restart_timers[1] = 0.0;
  std::string restart_logname = logname + ".chunk_size." + std::to_string(chunk_size) +
                                ".restart_timing.csv";
  write_restart_log(chkpt_id, restart_logname);
}

void 
FullDeduplicator::restart(uint8_t* data_ptr, 
                           size_t len, 
                           std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                           std::string& logname, 
                           uint32_t chkpt_id) {
  Kokkos::View<uint8_t*, Kokkos::MemoryTraits<Kokkos::Unmanaged> > data(data_ptr, len);
  restart(data, chkpts, logname, chkpt_id);
}

void 
FullDeduplicator::restart(Kokkos::View<uint8_t*> data, 
                           std::vector<std::string>& chkpt_filenames, 
                           std::string& logname, 
                           uint32_t chkpt_id) {
  using Timer = std::chrono::high_resolution_clock;
  using Nanoseconds = std::chrono::nanoseconds;
  // Full checkpoint
  std::fstream file;
  auto fileflags = std::ifstream::in | std::ifstream::binary | std::ifstream::ate;
  file.open(chkpt_filenames[chkpt_id], fileflags);
  size_t filesize = file.tellg();
  file.seekg(0);
  Kokkos::resize(data, filesize);
  auto data_h = Kokkos::create_mirror_view(data);
  // Read checkpoint
  //Timer::time_point r1 = Timer::now();
  file.read((char*)(data_h.data()), filesize);
  file.close();
  //Timer::time_point r2 = Timer::now();
  // Total time
  //Timer::time_point t1 = Timer::now();
  // Copy checkpoint to GPU
  Timer::time_point c1 = Timer::now();
  Kokkos::deep_copy(data, data_h);
  Timer::time_point c2 = Timer::now();
  //Timer::time_point t2 = Timer::now();
  // Update timers
  restart_timers[0] = (1e-9)*(std::chrono::duration_cast<Nanoseconds>(c2-c1).count());
  restart_timers[1] = 0.0;
  write_restart_log(chkpt_id, logname);
}

void 
FullDeduplicator::write_chkpt_log(header_t& header, 
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

  result_data << "Full" << "," // Approach
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
  timing_file << "Full" << ","     // Approach
              << current_id << ","        // Checkpoint ID
              << chunk_size << "," // Chunk size
              << "0.0" << ","      // Comparison time
              << "0.0" << ","      // Collection time
              << timers[3]        // Write time
              << std::endl;
  size_file << "Full" << ","         // Approach
            << current_id << ","            // Checkpoint ID
            << chunk_size << ","     // Chunk size
            << datasizes.first << "," // Size of data
            << "0,"                  // Size of metadata
            << "0,"                  // Size of header
            << "0,"                  // Size of distinct metadata
            << "0";                  // Size of repeat map
  for(uint32_t j=0; j<current_id; j++) {
    size_file << ",0"; // Size of metadata for checkpoint j
  }
  size_file << std::endl;
}

/**
 * Function for writing the restart log.
 *
 * \param select_chkpt Which checkpoint to write the log
 * \param logname      Filename for writing log
 */
void 
FullDeduplicator::write_restart_log(uint32_t select_chkpt, std::string& logname) {
  std::fstream timing_file;
  timing_file.open(logname, std::fstream::out | std::fstream::app);
  timing_file << "Full" << ",";
  timing_file << select_chkpt << "," 
              << chunk_size << "," 
              << restart_timers[0] << "," 
              << restart_timers[1] << std::endl;
  timing_file.close();
}
