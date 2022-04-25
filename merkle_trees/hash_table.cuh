#ifndef __HASH_TABLE_CUH
#define __HASH_TABLE_CUH

#include "gpu_sha1.hpp"

template<class Key>
struct DefaultHash {
  inline STDGPU_HOST_DEVICE unsigned int
  operator()(const Key& key) const {
    unsigned int hash = 0;
    unsigned char digest[20];
    sha1_hash(&key, sizeof(Key), digest);
    const unsigned int* key_u32 = (const unsigned int*)(digest);
    hash ^= key_u32[0];
    hash ^= key_u32[1];
    hash ^= key_u32[2];
    hash ^= key_u32[3];
    hash ^= key_u32[4];
    return hash;
  }
};

template<class Key>
struct ReduceHash {
  inline STDGPU_HOST_DEVICE unsigned int
  operator()(const Key& key) const {
    unsigned int hash = 0;
    const unsigned int* key_u32 = (const unsigned int*)(key.ptr);
    hash ^= key_u32[0];
    hash ^= key_u32[1];
    hash ^= key_u32[2];
    hash ^= key_u32[3];
    hash ^= key_u32[4];
    return hash;
  }
};

template<typename Key>
__global__ static void init_keys(Key *m_keys_d, unsigned int len) {
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  for(unsigned int offset=idx; offset<len; offset+= blockDim.x) {
    m_keys_d[offset].ptr = NULL;
  }
}

template<typename Value>
__global__ static void init_vals(Value *m_vals_d, unsigned int len) {
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  for(unsigned int offset=idx; offset<len; offset+= blockDim.x) {
    m_vals_d[offset] = Value();
  }
}

template<typename Key, typename Value, class Hasher=ReduceHash<Key>>
struct HashTable {
  unsigned int m_capacity;
  unsigned int m_size;
  unsigned int *m_capacity_d;
  unsigned int *m_size_d;
  Key          *m_keys_d;
  Value        *m_values_d;
  unsigned int *m_available_indices_d;
  Hasher hasher;

  HashTable(unsigned int _size) {
    m_capacity = _size;
    m_size = 0;
    
    cudaMalloc(&m_capacity_d,          sizeof(unsigned int));
    cudaMalloc(&m_size_d,              sizeof(unsigned int));
    cudaMalloc(&m_keys_d,              sizeof(Key)*(_size+1));
    cudaMalloc(&m_values_d,            sizeof(Value)*(_size+1));
    cudaMalloc(&m_available_indices_d, sizeof(bool)*_size);
    cudaMemcpy(m_capacity_d, &m_capacity, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(m_size_d, 0x00,           sizeof(unsigned int));
    cudaMemset(m_keys_d, 0xff,           sizeof(Key)*(_size+1));
    cudaMemset(m_values_d, 0xff,         sizeof(Value)*(_size+1));
    cudaMemset(m_available_indices_d, 0, sizeof(unsigned int)*_size);
    init_keys<<<1,_size>>>(m_keys_d, _size);
    init_vals<<<1,_size>>>(m_values_d, _size);
  }

  unsigned int size() const {
    cudaMemcpy((unsigned int*)(&m_size), m_size_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return m_size;
  }

  unsigned int capacity() const {
    cudaMemcpy((unsigned int*)(&m_capacity), m_capacity_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return m_capacity;
  }

  __device__ bool compare_digests(const uint8_t* ptr_a, const uint8_t* ptr_b) {
    for(int i=0; i<digest_size(); i++) {
      if(ptr_a[i] != ptr_b[i])
        return false;
    }
    return true;
  }
 
  __device__
  Value* end() {
    return m_values_d + *m_capacity_d;
  }

  __device__
  bool insert(Key &key, Value &val) {
    unsigned int hash = hasher(key);
    unsigned int slot = hash % (*m_capacity_d);
    unsigned int start = slot;
    do {
//      unsigned int prev = atomicCAS(&m_available_indices_d[slot], 0, 1);
      unsigned int prev = atomicExch(&m_available_indices_d[slot], 1);
      if(prev == 0) {
        m_keys_d[slot] = key;
        m_values_d[slot] = val;
        m_available_indices_d[slot] = 1;
        atomicAdd(m_size_d, 1);
        return true;
      } else {
        if(key == m_keys_d[slot]) {
          return false;
        }
      }
      slot = (slot + 1) % (*m_capacity_d);
    } while(slot != start);
    return false;
  }

  __device__
  Value* find(Key& key) {
    unsigned int hash = hasher(key);
    unsigned int slot = hash % (*m_capacity_d);
    unsigned int start = slot;
    do {
      if(m_keys_d[slot].ptr != NULL && m_keys_d[slot] == key) {
//     if(m_available_indices_d[slot] == 1 && m_keys_d[slot] == key) {
//        if(m_keys_d[slot] == key) {
//          return m_values_d[slot];
          return &(m_values_d[slot]);
//        }
      }
      slot = (slot + 1) % (*m_capacity_d);
    } while(slot != start);
//    return Value();
    return m_values_d+(*m_capacity_d);
//    return NULL;
  }
  
  __device__
  void remove(Key const &key) {
#ifdef DEBUG
printf("\t\tTrying to remove key %p\n", key.ptr);
#endif
    unsigned int hash = hasher(key);
    unsigned int slot = hash % (*m_capacity_d);
    unsigned int start = slot;
    do {
#ifdef DEBUG
printf("\t\tChecking %p\n", m_keys_d[slot].ptr);
#endif
      if(m_keys_d[slot].ptr != NULL && m_keys_d[slot] == key) {
//      if( (m_available_indices_d[slot] == 1) && (m_keys_d[slot] == key) ) {
//        if(m_keys_d[slot].ptr == key.ptr && m_keys_d[slot] == key && compare_digests(m_keys_d[slot].ptr, key.ptr)) {
#ifdef DEBUG
printf("\t\tDeleting key %p with (%u,%u,%u)\n", m_keys_d[slot].ptr, m_values_d[slot].node, m_values_d[slot].src, m_values_d[slot].tree);
#endif
        m_keys_d[slot] = Key();
        m_values_d[slot] = Value();
        atomicSub(m_size_d, 1);
        atomicExch(&(m_available_indices_d[slot]), 0);
//        m_available_indices_d[slot] = 0;
        return;
      }
      slot = (slot + 1) % (*m_capacity_d);
    } while(slot != start);
  }
};

template<typename Keys, typename Values>
__global__
static void _print_hash_table(unsigned int *m_capacity_d, unsigned int *m_available_indices_d, Keys* keys, Values* vals) {
  unsigned int num_distinct = 0;
  for(unsigned int idx=0; idx<*m_capacity_d; idx++) {
//    if(m_available_indices_d[idx] == 1) {
    if(keys[idx].ptr != NULL) {
      num_distinct += 1;
      printf("Node %u is distinct: (%u,%u,%u)\n", vals[idx].node, vals[idx].node, vals[idx].src, vals[idx].tree);
    }
  }
  printf("%u distinct nodes\n", num_distinct);
}

template<typename Keys, typename Values>
static void print_hash_table(unsigned int *m_capacity_d, unsigned int *m_available_indices, Keys* keys, Values* vals) {
  _print_hash_table<<<1,1>>>(m_capacity_d, m_available_indices, keys, vals);
  cudaDeviceSynchronize();
}

#endif
