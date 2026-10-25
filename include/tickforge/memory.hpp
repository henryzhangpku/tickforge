// Memory primitives for the hot path.
//
// - kCacheLine / AlignedBuffer: 64-byte aligned, fixed-size storage. One cache
//   line is 64 bytes on every x86-64 and most ARM server cores, so aligning
//   columns and per-thread state to it avoids split loads and false sharing.
// - Arena: a std::pmr::monotonic_buffer_resource over a caller-owned aligned
//   block. Allocation is a pointer bump, deallocation is a no-op, and the whole
//   arena is released at once when the backtest finishes.
// - CountingResource: a pass-through memory_resource that counts calls. Tests
//   wrap an arena's upstream in one to prove the steady state allocates nothing.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <new>
#include <span>
#include <type_traits>

namespace tickforge {

inline constexpr std::size_t kCacheLine = 64;

constexpr std::size_t align_up(std::size_t n, std::size_t a) noexcept {
  return (n + a - 1) & ~(a - 1);
}

// Owning, 64-byte aligned array of trivially-copyable T. Not resizable on
// purpose: size is decided once, outside the hot loop.
template <class T>
class AlignedBuffer {
  static_assert(std::is_trivially_copyable_v<T>);

 public:
  AlignedBuffer() = default;
  explicit AlignedBuffer(std::size_t n) : size_(n) {
    if (n == 0) return;
    const std::size_t bytes = align_up(n * sizeof(T), kCacheLine);
    data_.reset(static_cast<T*>(::operator new(bytes, std::align_val_t{kCacheLine})));
  }

  T* data() noexcept { return data_.get(); }
  const T* data() const noexcept { return data_.get(); }
  std::size_t size() const noexcept { return size_; }
  T& operator[](std::size_t i) noexcept { return data_.get()[i]; }
  const T& operator[](std::size_t i) const noexcept { return data_.get()[i]; }
  std::span<T> span() noexcept { return {data(), size_}; }
  std::span<const T> span() const noexcept { return {data(), size_}; }

 private:
  struct Free {
    void operator()(T* p) const noexcept { ::operator delete(p, std::align_val_t{kCacheLine}); }
  };
  std::unique_ptr<T, Free> data_;
  std::size_t size_ = 0;
};

// Counts allocate/deallocate calls and bytes, forwarding to an upstream.
class CountingResource final : public std::pmr::memory_resource {
 public:
  explicit CountingResource(std::pmr::memory_resource* upstream = std::pmr::get_default_resource())
      : upstream_(upstream) {}

  std::size_t allocations() const noexcept { return allocs_.load(std::memory_order_relaxed); }
  std::size_t deallocations() const noexcept { return deallocs_.load(std::memory_order_relaxed); }
  std::size_t bytes_allocated() const noexcept { return bytes_.load(std::memory_order_relaxed); }

 private:
  void* do_allocate(std::size_t bytes, std::size_t align) override {
    allocs_.fetch_add(1, std::memory_order_relaxed);
    bytes_.fetch_add(bytes, std::memory_order_relaxed);
    return upstream_->allocate(bytes, align);
  }
  void do_deallocate(void* p, std::size_t bytes, std::size_t align) override {
    deallocs_.fetch_add(1, std::memory_order_relaxed);
    upstream_->deallocate(p, bytes, align);
  }
  bool do_is_equal(const std::pmr::memory_resource& o) const noexcept override { return this == &o; }

  std::pmr::memory_resource* upstream_;
  std::atomic<std::size_t> allocs_{0};
  std::atomic<std::size_t> deallocs_{0};
  std::atomic<std::size_t> bytes_{0};
};

// Per-backtest scratch arena. The initial block is cache-line aligned and
// owned by the arena; if it runs out, the monotonic resource falls back to
// `upstream` (which tests replace with a CountingResource or
// null_memory_resource() to make overflow loud).
class Arena {
 public:
  explicit Arena(std::size_t bytes,
                 std::pmr::memory_resource* upstream = std::pmr::get_default_resource())
      : block_(align_up(bytes, kCacheLine)),
        resource_(block_.data(), block_.size(), upstream) {}

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  std::pmr::memory_resource* resource() noexcept { return &resource_; }
  std::size_t capacity() const noexcept { return block_.size(); }
  void release() noexcept { resource_.release(); }

 private:
  AlignedBuffer<std::byte> block_;
  std::pmr::monotonic_buffer_resource resource_;
};

}  // namespace tickforge
