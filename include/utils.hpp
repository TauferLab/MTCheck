#ifndef UTILS_HPP
#define UTILS_HPP

#define STDOUT
//#define DEBUG
//#define STATS
//#define GLOBAL_TABLE

#ifdef DEBUG
#define DEBUG_PRINT(...) do{ fprintf( stderr, __VA_ARGS__ ); } while( false )
#else
#define DEBUG_PRINT(...) do{ } while ( false )
#endif

#ifdef STDOUT
#define STDOUT_PRINT(...) do{ printf( __VA_ARGS__ ); } while( false )
#else
#define STDOUT_PRINT(...) do{ } while ( false )
#endif

typedef struct header_t {
  uint32_t ref_id;           // ID of reference checkpoint
  uint32_t chkpt_id;         // ID of checkpoint
  uint64_t datalen;          // Length of memory region in bytes
  uint32_t chunk_size;       // Size of chunks
  uint32_t window_size;      // Number of prior checkpoints to use 
  uint32_t distinct_size;    // Number of distinct entries
  uint32_t curr_repeat_size; // Number of repeat entries from current checkpoint
  uint32_t prev_repeat_size; // Number of repeat entries from prior checkpoints
  uint32_t pad;
} header_t;

#endif
