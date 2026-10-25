// Deterministic synthetic tick generator.
//
// The random numbers come from a hand-rolled xoshiro256** + Box-Muller rather
// than <random> distributions: std::normal_distribution is implementation-
// defined, so libstdc++ and libc++ would produce different files from the same
// seed. Here the same seed gives byte-identical output everywhere.
//
// The price is an Ornstein-Uhlenbeck deviation around a slow random-walk
// anchor, so a mean-reversion toy has something to find. It is not a model of
// any real market.
#pragma once

#include <cstdint>

#include "tickforge/tick_store.hpp"

namespace tickforge {

class Xoshiro256 {
 public:
  explicit Xoshiro256(std::uint64_t seed) noexcept;
  std::uint64_t next() noexcept;
  double uniform() noexcept;  // [0, 1)
  double normal() noexcept;   // N(0, 1), Box-Muller

 private:
  std::uint64_t s_[4]{};
  double spare_ = 0.0;
  bool has_spare_ = false;
};

struct SynthParams {
  std::uint64_t n_ticks = 1'000'000;
  std::uint64_t seed = 42;
  double start_price = 100.0;
  double tick_size = 0.01;
  double anchor_vol = 0.00002;   // per-tick log vol of the slow anchor
  double ou_kappa = 0.002;       // mean-reversion speed of the deviation
  double ou_vol = 0.0004;        // per-tick log vol of the deviation
  double mean_gap_ns = 50'000;   // average inter-arrival, exponential
  std::int64_t start_ts = 1'700'000'000'000'000'000;
};

TickVectors generate_ticks(const SynthParams& p);

}  // namespace tickforge
