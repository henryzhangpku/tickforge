#include <gtest/gtest.h>

#include <cstdint>
#include <memory_resource>
#include <vector>

#include "tickforge/memory.hpp"

using namespace tickforge;

TEST(AlignedBuffer, IsCacheLineAligned) {
  for (std::size_t n : {1U, 3U, 64U, 1000U}) {
    AlignedBuffer<double> b(n);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b.data()) % kCacheLine, 0U);
    EXPECT_EQ(b.size(), n);
  }
}

TEST(Arena, ServesFromItsBlockWithoutTouchingUpstream) {
  CountingResource upstream(std::pmr::new_delete_resource());
  Arena arena(64 * 1024, &upstream);
  {
    std::pmr::vector<double> v(1000, 0.0, arena.resource());
    std::pmr::vector<int> w(1000, 0, arena.resource());
  }
  EXPECT_EQ(upstream.allocations(), 0U);
}

TEST(Arena, OverflowGoesUpstreamAndIsCounted) {
  CountingResource upstream(std::pmr::new_delete_resource());
  Arena arena(1024, &upstream);
  std::pmr::vector<double> v(10'000, 0.0, arena.resource());
  EXPECT_GE(upstream.allocations(), 1U);
}

TEST(Arena, NullUpstreamMakesOverflowLoud) {
  Arena arena(1024, std::pmr::null_memory_resource());
  EXPECT_THROW(std::pmr::vector<double>(10'000, 0.0, arena.resource()), std::bad_alloc);
}
