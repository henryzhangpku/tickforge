#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <thread>

#include "tickforge/spsc_ring.hpp"

using namespace tickforge;

TEST(SpscRing, FifoAndCapacity) {
  auto r = std::make_unique<SpscRing<int, 8>>();
  int out = 0;
  EXPECT_FALSE(r->try_pop(out));
  for (int i = 0; i < 8; ++i) EXPECT_TRUE(r->try_push(i));
  EXPECT_FALSE(r->try_push(99));  // full
  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(r->try_pop(out));
    EXPECT_EQ(out, i);
  }
  EXPECT_FALSE(r->try_pop().has_value());
}

TEST(SpscRing, WrapsAround) {
  auto r = std::make_unique<SpscRing<int, 4>>();
  for (int round = 0; round < 100; ++round) {
    ASSERT_TRUE(r->try_push(round));
    ASSERT_TRUE(r->try_push(round + 1000));
    EXPECT_EQ(r->try_pop().value(), round);
    EXPECT_EQ(r->try_pop().value(), round + 1000);
  }
}

// Two real threads; every value must arrive exactly once and in order.
// Run under TSan in CI to check the acquire/release pairing.
TEST(SpscRing, ConcurrentProducerConsumerPreservesOrder) {
  constexpr std::uint64_t kN = 200'000;
  auto r = std::make_unique<SpscRing<std::uint64_t, 1024>>();
  std::thread producer([&] {
    for (std::uint64_t i = 0; i < kN; ++i)
      while (!r->try_push(i)) std::this_thread::yield();
  });
  std::uint64_t expected = 0;
  std::uint64_t sum = 0;
  while (expected < kN) {
    std::uint64_t v = 0;
    if (r->try_pop(v)) {
      ASSERT_EQ(v, expected);
      sum += v;
      ++expected;
    } else {
      std::this_thread::yield();
    }
  }
  producer.join();
  EXPECT_EQ(sum, kN * (kN - 1) / 2);
}
