#ifndef KOKKOS_VECTOR_HPP
#define KOKKOS_VECTOR_HPP
#include <Kokkos_Core.hpp>

template<typename StoreType>
class Vector {
  public:
    Kokkos::View<StoreType*> vector_d;
    Kokkos::View<uint32_t[1]> len_d;
    typename Kokkos::View<StoreType*>::HostMirror vector_h;
    Kokkos::View<uint32_t[1]>::HostMirror len_h;
    
    Vector() {
      vector_d = Kokkos::View<StoreType*>("Vector", 1);
      len_d   = Kokkos::View<uint32_t[1]>("Vector length");
      vector_h = Kokkos::create_mirror_view(vector_d);
      len_h   = Kokkos::create_mirror_view(len_d);
      Kokkos::deep_copy(len_d, 0);
    }

    Vector(uint32_t capacity) {
      vector_d = Kokkos::View<StoreType*>("Vector", capacity);
      len_d   = Kokkos::View<uint32_t[1]>("Vector length");
      vector_h = Kokkos::create_mirror_view(vector_d);
      len_h   = Kokkos::create_mirror_view(len_d);
      Kokkos::deep_copy(len_d, 0);
    }

    KOKKOS_INLINE_FUNCTION StoreType* data() const {
      return vector_d.data();
    }

    KOKKOS_INLINE_FUNCTION StoreType& operator()(const uint32_t x) const {
      return vector_d(x);
    }

    KOKKOS_INLINE_FUNCTION void push(StoreType item) const {
      uint32_t len = Kokkos::atomic_fetch_add(&len_d(0), 1);
      vector_d(len) = item;
    }

    void host_push(StoreType item) const {
      uint32_t len = Kokkos::atomic_fetch_add(&len_h(0), 1);
      vector_h(len) = item;
      Kokkos::deep_copy(vector_d, vector_h);
      Kokkos::deep_copy(len_d, len_h);
    }

    KOKKOS_INLINE_FUNCTION
    uint32_t size() const {
      #ifdef __CUDA_ARCH__
        return len_d(0);
      #else
        Kokkos::deep_copy(len_h, len_d);
        Kokkos::fence();
        return len_h(0);
      #endif
    }

    KOKKOS_INLINE_FUNCTION uint32_t capacity() const {
      return vector_d.extent(0);
    }

    void clear() const {
      Kokkos::deep_copy(len_d, 0);
      Kokkos::deep_copy(len_h, 0);
    }
};

template<typename StoreType, uint32_t N>
class Array {
  public:
    Kokkos::View<StoreType[N]> array_d;
    Kokkos::View<uint32_t[1]> len_d;
    typename Kokkos::View<StoreType[N]>::HostMirror array_h;
    Kokkos::View<uint32_t[1]>::HostMirror len_h;
    
    Array() {
      array_d = Kokkos::View<StoreType[N]>("Array");
      len_d   = Kokkos::View<uint32_t[1]>("Array length");
      array_h = Kokkos::create_mirror_view(array_d);
      len_h   = Kokkos::create_mirror_view(len_d);
      Kokkos::deep_copy(len_d, 0);
    }

    Array(uint32_t capacity) {
      array_d = Kokkos::View<StoreType[N]>("Array", capacity);
      len_d   = Kokkos::View<uint32_t[1]>("Array length");
      array_h = Kokkos::create_mirror_view(array_d);
      len_h   = Kokkos::create_mirror_view(len_d);
      Kokkos::deep_copy(len_d, 0);
    }

    KOKKOS_INLINE_FUNCTION StoreType* data() const {
      return array_d.data();
    }

    KOKKOS_INLINE_FUNCTION StoreType& operator()(const uint32_t x) const {
      return array_d(x);
    }

    KOKKOS_INLINE_FUNCTION void push(StoreType item) const {
      uint32_t len = Kokkos::atomic_fetch_add(&len_d(0), 1);
      array_d(len) = item;
    }

    void host_push(StoreType item) const {
      uint32_t len = Kokkos::atomic_fetch_add(&len_h(0), 1);
      array_h(len) = item;
      Kokkos::deep_copy(array_d, array_h);
      Kokkos::deep_copy(len_d, len_h);
    }

    KOKKOS_INLINE_FUNCTION
    uint32_t size() const {
      #ifdef __CUDA_ARCH__
        return len_d(0);
      #else
        Kokkos::deep_copy(len_h, len_d);
        Kokkos::fence();
        return len_h(0);
      #endif
    }

    KOKKOS_INLINE_FUNCTION uint32_t capacity() const {
      return array_d.extent(0);
    }

    void clear() const {
      Kokkos::deep_copy(len_d, 0);
      Kokkos::deep_copy(len_h, 0);
    }
};

template<uint32_t N>
class FixedSizeVector {
  public:
    Kokkos::View<uint32_t[N]> array_d;
    Kokkos::View<uint32_t[1]> len_d;
    typename Kokkos::View<uint32_t[N]>::HostMirror array_h;
    Kokkos::View<uint32_t[1]>::HostMirror len_h;
    
    FixedSizeVector() {
      array_d = Kokkos::View<uint32_t[N]>("Vector");
      len_d   = Kokkos::View<uint32_t[1]>("Vector length");
      array_h = Kokkos::create_mirror_view(array_d);
      len_h   = Kokkos::create_mirror_view(len_d);
      Kokkos::deep_copy(len_d, 0);
    }

    KOKKOS_INLINE_FUNCTION void push(uint32_t item) const {
      uint32_t len = Kokkos::atomic_fetch_add(&len_d(0), 1);
      array_d(len) = item;
    }

    void host_push(uint32_t item) const {
      uint32_t len = Kokkos::atomic_fetch_add(&len_h(0), 1);
      array_h(len) = item;
      Kokkos::deep_copy(array_d, array_h);
      Kokkos::deep_copy(len_d, len_h);
    }

    KOKKOS_INLINE_FUNCTION
    uint32_t size() const {
      #ifdef __CUDA_ARCH__
        return len_d(0);
      #else
        Kokkos::deep_copy(len_h, len_d);
        Kokkos::fence();
        return len_h(0);
      #endif
    }

    KOKKOS_INLINE_FUNCTION uint32_t capacity() const {
      return array_d.extent(0);
    }

    void clear() const {
      Kokkos::deep_copy(array_d, 0);
      Kokkos::deep_copy(array_h, 0);
      Kokkos::deep_copy(len_d, 0);
      Kokkos::deep_copy(len_h, 0);
    }
};

//template<uint32_t N>
//class Array {
//  public:
//    uint32_t len;
//    uint32_t array[N];
//    
//    KOKKOS_INLINE_FUNCTION
//    Array() {
//      len = 0;
//    }
//
//    KOKKOS_INLINE_FUNCTION uint32_t operator()(int32_t i) const {
//      return array[i];
//    }
//
//    KOKKOS_INLINE_FUNCTION void push(uint32_t item) {
//      uint32_t end = Kokkos::atomic_fetch_add(&len, 1);
//      array[end] = item;
//    }
//
//    KOKKOS_INLINE_FUNCTION
//    uint32_t size() const {
//      return len;
//    }
//
//    KOKKOS_INLINE_FUNCTION uint32_t capacity() const {
//      return N;
//    }
//
//    KOKKOS_INLINE_FUNCTION
//    void clear() const {
//      len = 0;
//    }
//};
#endif // KOKKOS_VECTOR_HPP

