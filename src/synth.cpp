#include "tickforge/synth.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>

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
  out.reserve(static_cast<std::size_t>(p.n_ticks));

  const double tick = p.tick_size;
  double log_anchor = std::log(p.start_price);
  double dev = 0.0;
  double ts = static_cast<double>(p.start_ts);
  double prev_fair = p.start_price;
  for (std::uint64_t i = 0; i < p.n_ticks; ++i) {
    log_anchor += p.anchor_vol * rng.normal();
    dev += -p.ou_kappa * dev + p.ou_vol * rng.normal();
    const double fair = std::exp(log_anchor + dev);

    // Quote: 1-3 ticks wide around fair value, on the price grid.
    const double u = rng.uniform();
    const int spread_ticks = u < 0.70 ? 1 : (u < 0.95 ? 2 : 3);
    const double bid = std::max(tick, std::floor((fair - 0.5 * spread_ticks * tick) / tick + 0.5) * tick);
    const double ask = bid + spread_ticks * tick;

    ts += -std::log(1.0 - rng.uniform()) * p.mean_gap_ns;
    // Aggressor side leans with the fair-value move (buyers lift offers on
    // upticks), with noise so order-flow imbalance is informative but not perfect.
    const double lean = fair > prev_fair ? 0.7 : (fair < prev_fair ? 0.3 : 0.5);
    const std::int8_t side = rng.uniform() < lean ? std::int8_t{1} : std::int8_t{-1};
    // Sizes: round lots with a heavy-ish tail.
    const auto lots = [&rng](double mean) {
      return static_cast<std::uint32_t>(1 + std::floor(-std::log(1.0 - rng.uniform()) * mean));
    };
    const std::uint32_t trade_lots = lots(3.0);

    Tick t{};
    t.ts = static_cast<std::int64_t>(ts);
    t.side = side;
    t.price = side > 0 ? ask : bid;  // trades print at the touch they hit
    t.size = trade_lots * 100;
    t.bid = bid;
    t.ask = ask;
    t.bid_size = lots(5.0) * 100;
    t.ask_size = lots(5.0) * 100;
    out.push_back(t);
    prev_fair = fair;
  }
  return out;
}

std::vector<std::filesystem::path> generate_universe(const std::filesystem::path& dir, std::size_t n_symbols,
                                                     const SynthParams& base) {
  std::filesystem::create_directories(dir);
  std::vector<std::filesystem::path> paths;
  paths.reserve(n_symbols);
  Xoshiro256 rng(base.seed ^ 0x5eedULL);
  for (std::size_t s = 0; s < n_symbols; ++s) {
    SynthParams p = base;
    p.seed = base.seed + 1000 * (s + 1);
    p.start_price = 20.0 + 180.0 * rng.uniform();
    p.ou_vol = base.ou_vol * (0.5 + rng.uniform());
    char name[32];
    std::snprintf(name, sizeof name, "SYN%03zu", s);
    const auto path = dir / (std::string(name) + ".tf");
    const auto t = generate_ticks(p);
    write_tick_file(path, t.view(), p.seed, name);
    paths.push_back(path);
  }
  return paths;
}

}  // namespace tickforge
