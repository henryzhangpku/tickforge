// Columnar (structure-of-arrays) tick file and its memory-mapped reader.
//
// File layout (format v2), little-endian, every column starting on a 64-byte
// boundary:
//
//   [Header: 128 bytes]
//   [ts       : int64  x n]  ns since epoch, non-decreasing
//   [price    : double x n]  trade price
//   [size     : uint32 x n]  trade size, shares
//   [side     : int8   x n]  +1 buyer-initiated, -1 seller-initiated
//   [bid      : double x n]  best bid in force at the trade
//   [ask      : double x n]  best ask in force at the trade
//   [bid_size : uint32 x n]  shares displayed at the bid
//   [ask_size : uint32 x n]  shares displayed at the ask
//
// A backtest that only needs prices touches only the price pages; the kernel
// pages them in lazily and the views are std::span straight into the mapping,
// so opening a file costs one mmap call and zero copies.
//
// One file holds one instrument. A universe is a directory of such files,
// the usual partitioning for column stores (one file per symbol per day).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tickforge {

inline constexpr char kMagic[8] = {'T', 'I', 'C', 'K', 'F', 'R', 'G', '1'};
inline constexpr std::uint32_t kFormatVersion = 2;
inline constexpr std::size_t kNumColumns = 8;
inline constexpr std::size_t kSymbolBytes = 16;

struct FileHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint64_t n_ticks;
  std::uint64_t seed;                       // generator seed, for provenance (0 if imported)
  std::uint64_t file_bytes;
  std::uint64_t off[kNumColumns];           // column offsets, in the order listed above
  char symbol[kSymbolBytes];                // NUL-padded
  std::uint8_t reserved[8];
};
static_assert(sizeof(FileHeader) == 128);

// Plain value view of one tick, used when pushing through the pipeline.
struct Tick {
  std::int64_t ts;
  double price;
  std::uint32_t size;
  std::int8_t side;
  double bid;
  double ask;
  std::uint32_t bid_size;
  std::uint32_t ask_size;

  double mid() const noexcept { return 0.5 * (bid + ask); }
};

// Non-owning SoA view. Anything that reads ticks takes this, whether the
// memory is an mmap, a test vector, or a NumPy array.
struct TickColumns {
  std::span<const std::int64_t> ts;
  std::span<const double> price;
  std::span<const std::uint32_t> size;
  std::span<const std::int8_t> side;
  std::span<const double> bid;
  std::span<const double> ask;
  std::span<const std::uint32_t> bid_size;
  std::span<const std::uint32_t> ask_size;

  std::size_t size_ticks() const noexcept { return price.size(); }
  Tick operator[](std::size_t i) const noexcept {
    return {ts[i], price[i], size[i], side[i], bid[i], ask[i], bid_size[i], ask_size[i]};
  }
};

// Owning in-memory columns, used by the generator, the CSV importer and tests.
struct TickVectors {
  std::vector<std::int64_t> ts;
  std::vector<double> price;
  std::vector<std::uint32_t> size;
  std::vector<std::int8_t> side;
  std::vector<double> bid;
  std::vector<double> ask;
  std::vector<std::uint32_t> bid_size;
  std::vector<std::uint32_t> ask_size;

  TickColumns view() const noexcept { return {ts, price, size, side, bid, ask, bid_size, ask_size}; }
  void push_back(const Tick& t);
  void reserve(std::size_t n);
};

void write_tick_file(const std::filesystem::path& path, const TickColumns& cols, std::uint64_t seed = 0,
                     std::string_view symbol = {});

// RAII read-only mapping. Move-only; the spans it hands out live as long as it does.
class MappedTickFile {
 public:
  explicit MappedTickFile(const std::filesystem::path& path);
  ~MappedTickFile();
  MappedTickFile(MappedTickFile&& o) noexcept;
  MappedTickFile& operator=(MappedTickFile&& o) noexcept;
  MappedTickFile(const MappedTickFile&) = delete;
  MappedTickFile& operator=(const MappedTickFile&) = delete;

  const FileHeader& header() const noexcept { return *header_; }
  const TickColumns& columns() const noexcept { return cols_; }
  std::size_t size() const noexcept { return cols_.size_ticks(); }
  std::string symbol() const;

 private:
  void unmap() noexcept;
  void* base_ = nullptr;
  std::size_t bytes_ = 0;
  const FileHeader* header_ = nullptr;
  TickColumns cols_{};
};

// Reads a CSV with header `ts,price,size,side` and optionally
// `bid,ask,bid_size,ask_size` (any column order). Without quote columns the
// quote is set to the trade price with the trade size on both sides, so fills
// happen at the trade price. Throws with the line number on malformed input.
TickVectors read_tick_csv(const std::filesystem::path& path);

}  // namespace tickforge
