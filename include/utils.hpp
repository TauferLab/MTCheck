#ifndef UTILS_HPP
#define UTILS_HPP

//#define STDOUT
//#define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(...) do{ printf( __VA_ARGS__ ); } while( false )
#else
#define DEBUG_PRINT(...) do{ } while ( false )
#endif

#ifdef STDOUT
#define STDOUT_PRINT(...) do{ printf( __VA_ARGS__ ); } while( false )
#else
#define STDOUT_PRINT(...) do{ } while ( false )
#endif

#ifdef __CUDA_ARCH__
#define TEAM_SIZE 32
#else
#define TEAM_SIZE 2
#endif

using counter_t = Kokkos::View<uint64_t[1]>;

typedef struct header_t {
  uint32_t ref_id;           // ID of reference checkpoint
  uint32_t chkpt_id;         // ID of checkpoint
  uint64_t datalen;          // Length of memory region in bytes
  uint32_t chunk_size;       // Size of chunks
  uint32_t num_first_ocur;    // Number of first occurrence entries
  uint32_t num_prior_chkpts;
  uint32_t num_shift_dupl;      // Number of duplicate entries
} header_t;

enum DedupMode {
  Unknown,
  Full,
  Basic,
  List,
  Tree,
  TreeLowOffsetRef,
  TreeLowOffset,
  TreeLowRootRef,
  TreeLowRoot
};

enum Label : uint8_t {
  FIRST_OCUR = 0,
  FIXED_DUPL = 1,
  SHIFT_DUPL = 2,
  FIRST_DUPL = 3,
  DONE = 4
};

void print_mode_help() {
  printf("Modes: \n");
  printf("Full checkpoint:                                   --run-full-chkpt\n");
  printf("Basic checkpoint:                                  --run-basic-chkpt\n");
  printf("List checkpoint:                                   --run-list-chkpt\n");
  printf("Tree checkpoint:                                   --run-tree-chkpt\n");
  printf("Tree checkpoint lowest offset reference algorithm: --run-tree-low-offset-ref-chkpt\n");
  printf("Tree checkpoint lowest offset algorithm:           --run-tree-low-offset-chkpt\n");
  printf("Tree checkpoint lowest root reference algorithm:   --run-tree-low-root-ref-chkpt\n");
  printf("Tree checkpoint lowest root algorithm:             --run-tree-low-root-chkpt\n");
}

DedupMode get_mode(int argc, char** argv) {
  for(int i=0; i<argc; i++) {
    if((strcmp(argv[i], "--run-full-chkpt") == 0)) {
      return Full;
    } else if(strcmp(argv[i], "--run-basic-chkpt") == 0) {
      return Basic;
    } else if(strcmp(argv[i], "--run-list-chkpt") == 0) {
      return List;
    } else if(strcmp(argv[i], "--run-tree-chkpt") == 0) {
      return Tree;
    } else if(strcmp(argv[i], "--run-tree-low-offset-ref-chkpt") == 0) {
      return TreeLowOffsetRef;
    } else if(strcmp(argv[i], "--run-tree-low-offset-chkpt") == 0) {
      return TreeLowOffset;
    } else if(strcmp(argv[i], "--run-tree-low-root-ref-chkpt") == 0) {
      return TreeLowRootRef;
    } else if(strcmp(argv[i], "--run-tree-low-root-chkpt") == 0) {
      return TreeLowRoot;
    }
  }
  return Unknown;
}

template <typename TeamMember>
KOKKOS_FORCEINLINE_FUNCTION
void team_memcpy(uint8_t* dst, uint8_t* src, size_t len, TeamMember& team_member) {
  uint32_t* src_u32 = (uint32_t*)(src);
  uint32_t* dst_u32 = (uint32_t*)(dst);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, len/4), [&] (const uint64_t& j) {
    dst_u32[j] = src_u32[j];
  });
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, len%4), [&] (const uint64_t& j) {
    dst[((len/4)*4)+j] = src[((len/4)*4)+j];
  });
}

#endif
