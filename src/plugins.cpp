#include "tickforge/plugins.hpp"

#include <algorithm>
#include <map>
#include <mutex>

namespace tickforge {

namespace {

struct Registry {
  std::mutex mu;
  std::map<std::string, StrategyFactory, std::less<>> factories;

  Registry() {
    factories.emplace("mean_reversion",
                      [](const StrategyParams& p) { return std::make_unique<VirtualMeanReversion>(p); });
  }
};

Registry& registry() {
  static Registry r;  // thread-safe initialisation (magic static)
  return r;
}

}  // namespace

bool register_strategy(std::string name, StrategyFactory factory) {
  auto& r = registry();
  std::lock_guard lk(r.mu);
  return r.factories.emplace(std::move(name), std::move(factory)).second;
}

std::unique_ptr<IStrategy> make_strategy(std::string_view name, const StrategyParams& p) {
  auto& r = registry();
  StrategyFactory f;
  {
    std::lock_guard lk(r.mu);
    const auto it = r.factories.find(name);
    if (it == r.factories.end()) return nullptr;
    f = it->second;
  }
  return f(p);
}

std::vector<std::string> strategy_names() {
  auto& r = registry();
  std::lock_guard lk(r.mu);
  std::vector<std::string> out;
  out.reserve(r.factories.size());
  for (const auto& kv : r.factories) out.push_back(kv.first);
  return out;
}

}  // namespace tickforge
