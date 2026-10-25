// Bounded single-producer / single-consumer lock-free ring buffer.
//
// Memory ordering, in one paragraph: the producer writes the slot, then
// publishes it with a release store of `tail_`. The consumer reads `tail_`
// with acquire, which makes the slot write visible, reads the slot, then
// frees it with a release store of `head_`; the producer's acquire load of
// `head_` makes sure it never overwrites a slot still being read. No CAS, no
// locks: each index has exactly one writer.
//
// Each side also keeps a cached copy of the other side's index so the common
// case touches only its own cache line. head_ and tail_ live on separate
// 64-byte lines to avoid false sharing between the two cores.
#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <type_traits>

#include "tickforge/memory.hpp"

namespace tickforge {

template <class T, std::size_t Capacity>
class SpscRing {
  static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0, "capacity must be a power of two");
  static_assert(std::is_trivially_copyable_v<T>);
  static constexpr std::size_t kMask = Capacity - 1;

 public:
  // Producer side.
  bool try_push(const T& v) noexcept {
    const std::size_t tail = tail_.load(std::memory_order_relaxed);
    if (tail - head_cache_ == Capacity) {
      head_cache_ = head_.load(std::memory_order_acquire);
      if (tail - head_cache_ == Capacity) return false;  // full
    }
    slots_[tail & kMask] = v;
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }

  // Consumer side.
  bool try_pop(T& out) noexcept {
    const std::size_t head = head_.load(std::memory_order_relaxed);
    if (head == tail_cache_) {
      tail_cache_ = tail_.load(std::memory_order_acquire);
      if (head == tail_cache_) return false;  // empty
    }
    out = slots_[head & kMask];
    head_.store(head + 1, std::memory_order_release);
    return true;
  }

  std::optional<T> try_pop() noexcept {
    T v;
    if (try_pop(v)) return v;
    return std::nullopt;
  }

  static constexpr std::size_t capacity() noexcept { return Capacity; }

 private:
  alignas(kCacheLine) std::atomic<std::size_t> head_{0};  // written by consumer
  alignas(kCacheLine) std::size_t tail_cache_ = 0;         // consumer's view of tail_
  alignas(kCacheLine) std::atomic<std::size_t> tail_{0};  // written by producer
  alignas(kCacheLine) std::size_t head_cache_ = 0;         // producer's view of head_
  alignas(kCacheLine) T slots_[Capacity];
};

}  // namespace tickforge
