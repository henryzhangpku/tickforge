// A small fixed-size thread pool returning std::futures.
//
// One shared queue under a mutex. For coarse tasks such as whole backtests the
// queue is touched a few dozen times per sweep and the lock is noise, so this
// simple pool is as fast as anything. When tasks are microseconds long or
// spawn children, every worker fights over that one lock; WorkStealingPool
// (work_stealing_pool.hpp) exists for that case, and bench_main.cpp measures both.
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

  // Fire-and-forget.
  void post(std::function<void()> fn) {
    {
      std::lock_guard lk(mu_);
      queue_.push_back(std::move(fn));
    }
    cv_.notify_one();
  }

  template <class F>
  auto submit(F&& f) -> std::future<std::invoke_result_t<std::decay_t<F>>> {
    using R = std::invoke_result_t<std::decay_t<F>>;
    // std::function needs a copyable target; packaged_task is move-only.
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    std::future<R> fut = task->get_future();
    post([task] { (*task)(); });
    return fut;
  }

 private:
  void worker_loop(const std::stop_token& st);

  std::mutex mu_;
  std::condition_variable_any cv_;
  std::deque<std::function<void()>> queue_;
  std::vector<std::jthread> workers_;
};

}  // namespace tickforge
