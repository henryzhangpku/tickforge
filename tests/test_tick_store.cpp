#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>

#include "fixtures.hpp"
#include "tickforge/memory.hpp"
#include "tickforge/tick_store.hpp"

using namespace tickforge;

TEST(Synth, SameSeedSameData) {
  const auto a = test::small_ticks(5000, 1);
  const auto b = test::small_ticks(5000, 1);
  const auto c = test::small_ticks(5000, 2);
  EXPECT_EQ(a.price, b.price);
  EXPECT_EQ(a.ts, b.ts);
  EXPECT_EQ(a.side, b.side);
  EXPECT_NE(a.price, c.price);
}

TEST(Synth, SaneValues) {
  const auto t = test::small_ticks(50'000);
  for (std::size_t i = 0; i < t.price.size(); ++i) {
    ASSERT_GT(t.price[i], 0.0);
    ASSERT_GT(t.size[i], 0U);
    ASSERT_TRUE(t.side[i] == 1 || t.side[i] == -1);
    if (i > 0) ASSERT_GE(t.ts[i], t.ts[i - 1]);
  }
}

TEST(TickStore, RoundTripThroughMmap) {
  const auto t = test::small_ticks(12'345);
  test::TempFile f("roundtrip");
  write_tick_file(f.path, t.view(), 7);

  const MappedTickFile m(f.path);
  ASSERT_EQ(m.size(), t.price.size());
  EXPECT_EQ(m.header().seed, 7U);
  const auto& c = m.columns();
  for (std::size_t i = 0; i < t.price.size(); ++i) {
    ASSERT_EQ(c.ts[i], t.ts[i]);
    ASSERT_EQ(c.price[i], t.price[i]);
    ASSERT_EQ(c.size[i], t.size[i]);
    ASSERT_EQ(c.side[i], t.side[i]);
  }
}

TEST(TickStore, ColumnsAreCacheLineAligned) {
  const auto t = test::small_ticks(1001);  // odd length so offsets need padding
  test::TempFile f("aligned");
  write_tick_file(f.path, t.view());
  const MappedTickFile m(f.path);
  const auto addr = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p); };
  EXPECT_EQ(addr(m.columns().ts.data()) % kCacheLine, 0U);
  EXPECT_EQ(addr(m.columns().price.data()) % kCacheLine, 0U);
  EXPECT_EQ(addr(m.columns().size.data()) % kCacheLine, 0U);
  EXPECT_EQ(addr(m.columns().side.data()) % kCacheLine, 0U);
}

TEST(TickStore, ViewsAreZeroCopy) {
  const auto t = test::small_ticks(1000);
  test::TempFile f("zerocopy");
  write_tick_file(f.path, t.view());
  const MappedTickFile m(f.path);
  // The price column must point inside the mapping, right after the header.
  const auto* base = reinterpret_cast<const std::byte*>(&m.header());
  const auto* price = reinterpret_cast<const std::byte*>(m.columns().price.data());
  EXPECT_EQ(price - base, static_cast<std::ptrdiff_t>(m.header().off_price));
}

TEST(TickStore, MoveTransfersMapping) {
  const auto t = test::small_ticks(100);
  test::TempFile f("move");
  write_tick_file(f.path, t.view());
  MappedTickFile a(f.path);
  const double first = a.columns().price[0];
  MappedTickFile b(std::move(a));
  EXPECT_EQ(a.size(), 0U);  // NOLINT(bugprone-use-after-move): checking moved-from state
  EXPECT_EQ(b.columns().price[0], first);
}

TEST(TickStore, RejectsGarbage) {
  test::TempFile f("garbage");
  {
    std::ofstream out(f.path, std::ios::binary);
    for (int i = 0; i < 512; ++i) out.put('x');
  }
  EXPECT_THROW(MappedTickFile{f.path}, std::runtime_error);
  EXPECT_THROW(MappedTickFile{"/nonexistent/tickforge.tf"}, std::system_error);
}
