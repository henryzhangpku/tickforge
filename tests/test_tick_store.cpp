#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "fixtures.hpp"
#include "tickforge/memory.hpp"
#include "tickforge/synth.hpp"
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
    ASSERT_GT(t.bid[i], 0.0);
    ASSERT_GT(t.ask[i], t.bid[i]);
    ASSERT_EQ(t.price[i], t.side[i] > 0 ? t.ask[i] : t.bid[i]);  // prints at the touch it hit
    ASSERT_GT(t.bid_size[i], 0U);
    ASSERT_GT(t.ask_size[i], 0U);
    ASSERT_GT(t.size[i], 0U);
    ASSERT_TRUE(t.side[i] == 1 || t.side[i] == -1);
    if (i > 0) ASSERT_GE(t.ts[i], t.ts[i - 1]);
  }
}

TEST(TickStore, RoundTripThroughMmap) {
  const auto t = test::small_ticks(12'345);
  test::TempFile f("roundtrip");
  write_tick_file(f.path, t.view(), 7, "SYN007");

  const MappedTickFile m(f.path);
  ASSERT_EQ(m.size(), t.price.size());
  EXPECT_EQ(m.header().seed, 7U);
  EXPECT_EQ(m.symbol(), "SYN007");
  const auto& c = m.columns();
  for (std::size_t i = 0; i < t.price.size(); ++i) {
    ASSERT_EQ(c.ts[i], t.ts[i]);
    ASSERT_EQ(c.price[i], t.price[i]);
    ASSERT_EQ(c.size[i], t.size[i]);
    ASSERT_EQ(c.side[i], t.side[i]);
    ASSERT_EQ(c.bid[i], t.bid[i]);
    ASSERT_EQ(c.ask[i], t.ask[i]);
    ASSERT_EQ(c.bid_size[i], t.bid_size[i]);
    ASSERT_EQ(c.ask_size[i], t.ask_size[i]);
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
  EXPECT_EQ(addr(m.columns().bid.data()) % kCacheLine, 0U);
  EXPECT_EQ(addr(m.columns().ask_size.data()) % kCacheLine, 0U);
}

TEST(TickStore, ViewsAreZeroCopy) {
  const auto t = test::small_ticks(1000);
  test::TempFile f("zerocopy");
  write_tick_file(f.path, t.view());
  const MappedTickFile m(f.path);
  // The price column must point inside the mapping, right after the header.
  const auto* base = reinterpret_cast<const std::byte*>(&m.header());
  const auto* price = reinterpret_cast<const std::byte*>(m.columns().price.data());
  EXPECT_EQ(price - base, static_cast<std::ptrdiff_t>(m.header().off[1]));
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

TEST(TickStore, RejectsOtherFormatVersion) {
  const auto t = test::small_ticks(100);
  test::TempFile f("version");
  write_tick_file(f.path, t.view());
  {
    std::fstream io(f.path, std::ios::binary | std::ios::in | std::ios::out);
    const std::uint32_t v1 = 1;
    io.seekp(8);
    io.write(reinterpret_cast<const char*>(&v1), sizeof v1);
  }
  try {
    MappedTickFile m(f.path);
    FAIL() << "expected a version error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("version 1"), std::string::npos) << e.what();
  }
}

TEST(TickStore, SymbolTooLongIsRejected) {
  const auto t = test::small_ticks(10);
  test::TempFile f("longsym");
  EXPECT_THROW(write_tick_file(f.path, t.view(), 0, "THIS_SYMBOL_IS_WAY_TOO_LONG"), std::invalid_argument);
}

TEST(Universe, GeneratesDistinctSymbols) {
  const auto dir = std::filesystem::temp_directory_path() / ("tf_universe_" + std::to_string(::getpid()));
  SynthParams p;
  p.n_ticks = 2000;
  const auto paths = generate_universe(dir, 3, p);
  ASSERT_EQ(paths.size(), 3U);
  std::vector<double> first;
  for (std::size_t i = 0; i < paths.size(); ++i) {
    const MappedTickFile m(paths[i]);
    EXPECT_EQ(m.size(), 2000U);
    EXPECT_EQ(m.symbol(), "SYN00" + std::to_string(i));
    first.push_back(m.columns().price[0]);
  }
  EXPECT_NE(first[0], first[1]);
  std::filesystem::remove_all(dir);
}

namespace {
void write_text(const std::filesystem::path& p, const std::string& s) {
  std::ofstream out(p);
  out << s;
}
}  // namespace

TEST(Csv, ImportsTradesAndQuotes) {
  test::TempFile f("csv_full");
  write_text(f.path,
             "ts,side,price,size,bid,ask,bid_size,ask_size\n"
             "100,1,10.02,300,10.00,10.02,500,400\n"
             "250,-1,10.00,100,10.00,10.03,200,700\r\n");
  const auto t = read_tick_csv(f.path);
  ASSERT_EQ(t.price.size(), 2U);
  EXPECT_EQ(t.ts[1], 250);
  EXPECT_EQ(t.side[1], -1);
  EXPECT_DOUBLE_EQ(t.ask[1], 10.03);
  EXPECT_EQ(t.ask_size[0], 400U);
}

TEST(Csv, TradesOnlyUseTradePriceAsQuote) {
  test::TempFile f("csv_trades");
  write_text(f.path, "ts,price,size,side\n1,5.5,100,1\n2,5.6,200,-1\n");
  const auto t = read_tick_csv(f.path);
  ASSERT_EQ(t.price.size(), 2U);
  EXPECT_DOUBLE_EQ(t.bid[1], 5.6);
  EXPECT_DOUBLE_EQ(t.ask[1], 5.6);
  EXPECT_EQ(t.bid_size[1], 200U);
}

TEST(Csv, ReportsBadLines) {
  test::TempFile f("csv_bad");
  write_text(f.path, "ts,price,size,side\n1,5.5,100,1\n2,abc,100,1\n");
  try {
    read_tick_csv(f.path);
    FAIL();
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("line 3"), std::string::npos) << e.what();
  }
  write_text(f.path, "ts,price,size,side\n5,5.5,100,1\n2,5.5,100,1\n");
  EXPECT_THROW(read_tick_csv(f.path), std::runtime_error);  // backwards in time
  write_text(f.path, "ts,price,size\n1,5.5,100\n");
  EXPECT_THROW(read_tick_csv(f.path), std::runtime_error);  // missing side
}
