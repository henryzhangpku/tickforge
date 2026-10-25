// Work-stealing thread pool.
//
// Each worker owns a deque. A worker pushes tasks it spawns onto the back of
// its own deque and pops from the back (LIFO: the freshest task, whose data is
// still in cache); idle workers steal from the *front* of a victim's deque
// (FIFO: the oldest, usually largest, piece of work). Tasks submitted from
// outside the pool are spread round-robin across the deques.
//
// Each deque has its own mutex, so workers only contend when one steals from
// another. A lock-free Chase-Lev deque would remove even that, at the cost of
// much subtler code; per-deque locks already remove the single hot lock that
// limits ThreadPool when tasks are tiny or spawn children (see bench_main.cpp).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "tickforge/memory.hpp"

namespace tickforge {

class WorkStealingPool {
 public:
  explicit WorkStealingPool(std::size_t threads = 0);  // 0 = hardware_concurrency
  ~WorkStealingPool();
  WorkStealingPool(const WorkStealingPool&) = delete;
  WorkStealingPool& operator=(const WorkStealingPool&) = delete;

  std::size_t size() const noexcept { return threads_.size(); }

  // Fire-and-forget. From inside a task, the new task goes to the caller's own deque.
  void post(std::function<void()> fn);

  template <class F>
  auto submit(F&& f) -> std::future<std::invoke_result_t<std::decay_t<F>>> {
    using R = std::invoke_result_t<std::decay_t<F>>;
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    std::future<R> fut = task->get_future();
    post([task] { (*task)(); });
    return fut;
  }

  // Number of tasks taken from another worker's deque so far (for tests/benchmarks).
  std::size_t steals() const noexcept { return steals_.load(std::memory_order_relaxed); }

 private:
  struct alignas(kCacheLine) Worker {
    std::mutex mu;
    std::deque<std::function<void()>> q;
  };

  bool try_take(std::size_t self, std::function<void()>& out);
  void worker_loop(const std::stop_token& st, std::size_t self);

  std::vector<std::unique_ptr<Worker>> workers_;
  std::atomic<std::size_t> queued_{0};  // tasks sitting in deques, not yet taken
  std::atomic<std::size_t> next_{0};    // round-robin cursor for external submits
  std::atomic<std::size_t> steals_{0};
  std::mutex sleep_mu_;
  std::condition_variable_any wake_;
  std::vector<std::jthread> threads_;
};

}  // namespace tickforge
