# Data deduplication #
Header only library for deduplicating memory using an array of methods with varying complexity and performance. Library uses Kokkos for portability across architectures. Includes a simple data generator for creating files or Views containing simple data patterns.

# Deduplication Approaches #
* **Full Approach**: Simple test that generates random data and performs an in-memory checkpoint by copying the full data from device to host.
* **Basic Approach**: Tests a more complex data deduplication method where code is broken into chunks and hashes are used to identify which chunks changed. Any changes are saved
* **List Approach**: Similar to the basic approach but reduces the amount of data stored by only saving a single instance of each chunk of data. Uses metadata to track duplicates of chunks at different offsets.
* **Our Approach**: Builds on the list approach by representing the metadata with a compact representation using Merkle Trees

# Installation #
## Dependencies ##
* [Kokkos](https://github.com/kokkos/kokkos)
* OpenSSL
* CMake

## Build ##
```
mkdir build
cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=$HOME/kokkos/bin/nvcc_wrapper \
  -DCMAKE_INSTALL_PREFIX=$HOME/Src_Deduplication_Module/build/install \
  -DKokkos_DIR=$HOME/kokkos/build/install/lib64/cmake/Kokkos \
  ..
```
Note that the exact install directory and Kokkos directory will vary depending on the system and where you install Kokkos.

# Tests #
## General tests ##
Each test accepts 2 arguments along with any Kokkos specific options
`[full|basic|list|tree]_chkpt_test SizeOfChunks NumberOfCheckpoints`
* `full_chkpt.cpp`: Test the full approach using randomly generated data
* `basic_chkpt.cpp`: Test the basic approach using randomly generated data
* `list_chkpt.cpp`: Test the list approach using randomly generated data
* `tree_chkpt.cpp`: Test the tree approach using randomly generated data

## Tree approach specific tests #
`test_case_[00-10]` test simple inputs with known optimal solutions. Success means that the code correctly identifies the optimal set of chunks to save and the minimal amount of metadata needed to reconstruct the full data.

## Run tests ##
`make test`

# Included Binaries #
The deduplication methods in the repository are header only but we have included a set of source files for additional testing.
* `data_generation`: A simple program for generating files/Arrays of randomly generated data with some simple patterns for testing. Measures breakdown of resulting incremental checkpoints as well as time spent deduplicating data.
* `dedup_chkpt_files`: Program that ingests files and deduplicates the contents. Outputs incremental checkpoints in the same directory as the input files with different file extensions (`.full_chkpt`, `.basic.incr_chkpt`, `.hashlist.incr_chkpt`, `.hashtree.incr_chkpt`) 
  * `dedup_chkpt_files chunk_size num_files [approach] [files]`
  * Possible approaches: (The tree approach has different variations dedicated to different implementations and methods for selecting which chunks are labeled first occurrences)
  *  `--run-full-chkpt`  :   Full approach 
  *  `--run-naive-chkpt` :   Basic approach
  *  `--run-list-chkpt`  :   List approach
  *  `--run-tree-chkpt`  :   Tree approach (defaults to current development)
  *  `--run-tree-low-offset-ref-chkpt`  :   Choose leaf with lowest offset as the first occurrence (Serial reference implementation) 
  *  `--run-tree-low-offset-chkpt`      :   Choose leaf with lowest offset as the first occurrence
  *  `--run-tree-low-root-ref-chkpt`    :   Choose leaf with lowest root offset as the first occurrence (Serial reference implementation)
  *  `--run-tree-low-root-chkpt`        :   Choose leaf with lowest root offset as the first occurrence (in-progress)
* `restart_chkpt_files`: Program that restarts data from incremental checkpoints produced by `dedup_chkpt_files`. Computes the hash of the restarted data for verification. Records runtime performance automatically in csv files.
  * `restart_chkpt_files chkpt_to_restart num_files num_iterations chunk_size [approach] [files]`
  * Input filenames should be the same as those supplied to `dedup_chkpt_files`. Binary will automatically add the necessary file extensions based on the supplied approach.
  * Possible approaches: (The tree approach has different variations dedicated to different implementations and methods for selecting which chunks are labeled first occurrences)
  *  `--run-full-chkpt`  :   Full approach 
  *  `--run-naive-chkpt` :   Basic approach
  *  `--run-list-chkpt`  :   List approach
  *  `--run-tree-chkpt`  :   Tree approach (applies to any variation)

