// A small fixed-size thread pool returning std::futures.
//
// One shared queue under a mutex, not work stealing. That is a deliberate
// choice for this workload: a sweep task is a whole backtest (milliseconds to
// seconds), so the queue is touched a few hundred times per sweep and lock
// contention is noise. Work stealing pays off when tasks are microseconds
// long or spawn sub-tasks; neither is true here. The README says so too.
#pragma once

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

namespace tickforge {

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t threads = 0);  // 0 = hardware_concurrency
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  std::size_t size() const noexcept { return workers_.size(); }

  template <class F>
  auto submit(F&& f) -> std::future<std::invoke_result_t<std::decay_t<F>>> {
    using R = std::invoke_result_t<std::decay_t<F>>;
    // std::function needs a copyable target; packaged_task is move-only.
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    std::future<R> fut = task->get_future();
    {
      std::lock_guard lk(mu_);
      queue_.emplace_back([task] { (*task)(); });
    }
    cv_.notify_one();
    return fut;
  }

 private:
  void worker_loop(std::stop_token st);

  std::mutex mu_;
  std::condition_variable_any cv_;
  std::deque<std::function<void()>> queue_;
  std::vector<std::jthread> workers_;
};

}  // namespace tickforge
