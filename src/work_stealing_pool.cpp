#include "tickforge/work_stealing_pool.hpp"

#include <algorithm>

namespace tickforge {

namespace {
// Which pool (if any) the current thread works for, and its index there.
thread_local const WorkStealingPool* tl_pool = nullptr;
thread_local std::size_t tl_index = 0;
}  // namespace

WorkStealingPool::WorkStealingPool(std::size_t threads) {
  if (threads == 0) threads = std::max(1U, std::thread::hardware_concurrency());
  workers_.reserve(threads);
  for (std::size_t i = 0; i < threads; ++i) workers_.push_back(std::make_unique<Worker>());
  threads_.reserve(threads);
  for (std::size_t i = 0; i < threads; ++i)
    threads_.emplace_back([this, i](const std::stop_token& st) { worker_loop(st, i); });
}

WorkStealingPool::~WorkStealingPool() {
  for (auto& t : threads_) t.request_stop();
  {
    std::lock_guard lk(sleep_mu_);
  }
  wake_.notify_all();
  threads_.clear();  // join before the deques they read are destroyed
}

void WorkStealingPool::post(std::function<void()> fn) {
  const std::size_t n = workers_.size();
  const std::size_t idx = tl_pool == this ? tl_index : next_.fetch_add(1, std::memory_order_relaxed) % n;
  {
    std::lock_guard lk(workers_[idx]->mu);
    workers_[idx]->q.push_back(std::move(fn));
    // Counted under the deque lock, so no taker can decrement it first.
    queued_.fetch_add(1, std::memory_order_release);
  }
  // Take the sleep lock before notifying so a worker that has just checked
  // `queued_ == 0` and is about to wait cannot miss this wake-up.
  {
    std::lock_guard lk(sleep_mu_);
  }
  wake_.notify_one();
}

bool WorkStealingPool::try_take(std::size_t self, std::function<void()>& out) {
  {  // Own deque, newest first.
    Worker& w = *workers_[self];
    std::lock_guard lk(w.mu);
    if (!w.q.empty()) {
      out = std::move(w.q.back());
      w.q.pop_back();
      queued_.fetch_sub(1, std::memory_order_relaxed);
      return true;
    }
  }
  const std::size_t n = workers_.size();
  for (std::size_t k = 1; k < n; ++k) {  // Steal oldest from the others.
    Worker& v = *workers_[(self + k) % n];
    std::lock_guard lk(v.mu);
    if (!v.q.empty()) {
      out = std::move(v.q.front());
      v.q.pop_front();
      queued_.fetch_sub(1, std::memory_order_relaxed);
      steals_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }
  return false;
}

void WorkStealingPool::worker_loop(const std::stop_token& st, std::size_t self) {
  tl_pool = this;
  tl_index = self;
  std::function<void()> job;
  for (;;) {
    if (try_take(self, job)) {
      job();
      job = nullptr;
      continue;
    }
    std::unique_lock lk(sleep_mu_);
    wake_.wait(lk, st, [this] { return queued_.load(std::memory_order_acquire) > 0; });
    // Exit only when asked to and nothing is left, so every future is satisfied.
    if (st.stop_requested() && queued_.load(std::memory_order_acquire) == 0) return;
  }
}

}  // namespace tickforge
