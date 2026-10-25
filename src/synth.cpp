#include "tickforge/synth.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace tickforge {

namespace {
std::uint64_t splitmix64(std::uint64_t& x) noexcept {
  std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept { return (x << k) | (x >> (64 - k)); }
}  // namespace

Xoshiro256::Xoshiro256(std::uint64_t seed) noexcept {
  for (auto& s : s_) s = splitmix64(seed);
}

std::uint64_t Xoshiro256::next() noexcept {
  const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
  const std::uint64_t t = s_[1] << 17;
  s_[2] ^= s_[0];
  s_[3] ^= s_[1];
  s_[1] ^= s_[2];
  s_[0] ^= s_[3];
  s_[2] ^= t;
  s_[3] = rotl(s_[3], 45);
  return result;
}

double Xoshiro256::uniform() noexcept { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

double Xoshiro256::normal() noexcept {
  if (has_spare_) {
    has_spare_ = false;
    return spare_;
  }
  double u1 = uniform();
  while (u1 <= 0.0) u1 = uniform();
  const double u2 = uniform();
  const double r = std::sqrt(-2.0 * std::log(u1));
  const double theta = 2.0 * std::numbers::pi * u2;
  spare_ = r * std::sin(theta);
  has_spare_ = true;
  return r * std::cos(theta);
}

TickVectors generate_ticks(const SynthParams& p) {
  Xoshiro256 rng(p.seed);
  TickVectors out;
  const auto n = static_cast<std::size_t>(p.n_ticks);
  out.ts.resize(n);
  out.price.resize(n);
  out.size.resize(n);
  out.side.resize(n);

  double log_anchor = std::log(p.start_price);
  double dev = 0.0;
  double ts = static_cast<double>(p.start_ts);
  double prev_price = p.start_price;
  for (std::size_t i = 0; i < n; ++i) {
    log_anchor += p.anchor_vol * rng.normal();
    dev += -p.ou_kappa * dev + p.ou_vol * rng.normal();
    const double raw = std::exp(log_anchor + dev);
    const double px = std::max(p.tick_size, std::round(raw / p.tick_size) * p.tick_size);

    ts += -std::log(1.0 - rng.uniform()) * p.mean_gap_ns;
    // Aggressor side leans with the price move (buyers lift offers on upticks),
    // with noise so order-flow imbalance is informative but not perfect.
    const double lean = px > prev_price ? 0.7 : (px < prev_price ? 0.3 : 0.5);
    const std::int8_t side = rng.uniform() < lean ? std::int8_t{1} : std::int8_t{-1};
    // Sizes: round lots with a heavy-ish tail.
    const auto lots = static_cast<std::uint32_t>(1 + std::floor(-std::log(1.0 - rng.uniform()) * 3.0));

    out.ts[i] = static_cast<std::int64_t>(ts);
    out.price[i] = px;
    out.size[i] = lots * 100;
    out.side[i] = side;
    prev_price = px;
  }
  return out;
}

}  // namespace tickforge
