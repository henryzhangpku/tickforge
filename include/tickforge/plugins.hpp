// Runtime strategy registry: the "plugin" side of the CRTP-vs-virtual split.
//
// Strategies registered here are created by name and driven through
// IStrategy*, so new ones can be added (or loaded from a shared library)
// without recompiling the engine. The price is an indirect call per tick that
// the optimiser cannot inline; living in its own translation unit also stops
// the compiler from devirtualising it, so benchmarks see the real cost.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tickforge/strategy.hpp"

namespace tickforge {

using StrategyFactory = std::function<std::unique_ptr<IStrategy>(const StrategyParams&)>;

// Registers a factory under `name`; returns false if the name is taken.
bool register_strategy(std::string name, StrategyFactory factory);

// nullptr if no strategy has that name. "mean_reversion" is built in.
std::unique_ptr<IStrategy> make_strategy(std::string_view name, const StrategyParams& p);

std::vector<std::string> strategy_names();

}  // namespace tickforge
