#pragma once

#include <filesystem>
#include <string>

#include <unistd.h>

#include "tickforge/synth.hpp"

namespace tickforge::test {

inline TickVectors small_ticks(std::uint64_t n = 20'000, std::uint64_t seed = 7) {
  SynthParams p;
  p.n_ticks = n;
  p.seed = seed;
  return generate_ticks(p);
}

// A unique temp path that removes itself.
struct TempFile {
  std::filesystem::path path;
  explicit TempFile(const std::string& stem)
      : path(std::filesystem::temp_directory_path() / (stem + "_" + std::to_string(::getpid()) + ".tf")) {}
  ~TempFile() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

}  // namespace tickforge::test
