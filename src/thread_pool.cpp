#include "tickforge/thread_pool.hpp"

#include <algorithm>

namespace tickforge {

ThreadPool::ThreadPool(std::size_t threads) {
  if (threads == 0) threads = std::max(1U, std::thread::hardware_concurrency());
  workers_.reserve(threads);
  for (std::size_t i = 0; i < threads; ++i)
    workers_.emplace_back([this](const std::stop_token& st) { worker_loop(st); });
}

ThreadPool::~ThreadPool() {
  for (auto& w : workers_) w.request_stop();
  cv_.notify_all();
  // jthread joins on destruction. Workers drain the queue before exiting, so
  // every future handed out is satisfied.
}

void ThreadPool::worker_loop(const std::stop_token& st) {
  for (;;) {
    std::function<void()> job;
    {
      std::unique_lock lk(mu_);
      cv_.wait(lk, st, [this] { return !queue_.empty(); });
      if (queue_.empty()) return;  // stop requested and nothing left to do
      job = std::move(queue_.front());
      queue_.pop_front();
    }
    job();
  }
}

}  // namespace tickforge
