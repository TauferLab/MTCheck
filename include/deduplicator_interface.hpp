#ifndef DEDUPLICATOR_INTERFACE_HPP
#define DEDUPLICATOR_INTERFACE_HPP
#include <Kokkos_Core.hpp>
//#include <Kokkos_Random.hpp>
//#include <Kokkos_UnorderedMap.hpp>
//#include <Kokkos_Bitset.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <utility>
#include "stdio.h"

class BaseDeduplicator {
  protected:
    uint32_t chunk_size;
    uint32_t current_id;
    uint32_t baseline_id;
    uint64_t data_len;
    // timers and data sizes
    std::pair<uint64_t,uint64_t> datasizes;
    double timers[4];
    double restart_timers[2];

  public:
    /**
     * Constructor
     */
    BaseDeduplicator() {}

    BaseDeduplicator(uint32_t bytes_per_chunk) {}

    /**
     * Destructor
     */
    virtual ~BaseDeduplicator() {}

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
    virtual
    void checkpoint(header_t& header, 
                    uint8_t* data_ptr, 
                    size_t data_len,
                    Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                    bool make_baseline) = 0;

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
    virtual void checkpoint(uint8_t* data_ptr, 
                            size_t len, 
                            std::string& filename, 
                            std::string& logname, 
                            bool make_baseline) = 0;

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
    virtual void checkpoint(uint8_t* data_ptr, 
                            size_t len, 
                            Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                            bool make_baseline) = 0;

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
    virtual void checkpoint(uint8_t* data_ptr, 
                            size_t len, 
                            Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                            std::string& logname, 
                            bool make_baseline) = 0;

    /**
     * Restart checkpoint from vector of incremental checkpoints loaded on the Host.
     *
     * \param data       Data View to restart checkpoint into
     * \param chkpts     Vector of prior incremental checkpoints stored on the Host
     * \param logname    Filename for restart logs
     * \param chkpt_id   ID of checkpoint to restart
     */
    virtual void restart(Kokkos::View<uint8_t*> data, 
                         std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                         std::string& logname, 
                         uint32_t chkpt_id) = 0;

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
    virtual void restart(uint8_t* data_ptr, 
                         size_t len, 
                         std::vector<Kokkos::View<uint8_t*>::HostMirror>& chkpts, 
                         std::string& logname, 
                         uint32_t chkpt_id) = 0;

    /**
     * Restart checkpoint from checkpoint files
     *
     * \param data       Data View to restart checkpoint into
     * \param filenames  Vector of prior incremental checkpoints stored in files
     * \param logname    Filename for restart logs
     * \param chkpt_id   ID of checkpoint to restart
     */
    virtual void restart(Kokkos::View<uint8_t*> data, 
                         std::vector<std::string>& chkpt_filenames, 
                         std::string& logname, 
                         uint32_t chkpt_id) = 0;

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
    virtual void write_chkpt_log(header_t& header, 
                                 Kokkos::View<uint8_t*>::HostMirror& diff_h, 
                                 std::string& logname) = 0;

    /**
     * Function for writing the restart log.
     *
     * \param select_chkpt Which checkpoint to write the log
     * \param logname      Filename for writing log
     */
    virtual void write_restart_log(uint32_t select_chkpt, 
                                   std::string& logname) = 0;
};


#endif // DEDUPLICATOR_INTERFACE
