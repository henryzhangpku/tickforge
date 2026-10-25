// Columnar (structure-of-arrays) tick file and its memory-mapped reader.
//
// File layout, little-endian, every column starting on a 64-byte boundary:
//
//   [Header: 128 bytes]
//   [timestamps : int64  x n]  ns since epoch, non-decreasing
//   [prices     : double x n]
//   [sizes      : uint32 x n]
//   [sides      : int8   x n]  +1 buyer-initiated, -1 seller-initiated
//
// A backtest that only needs prices touches only the price pages; the kernel
// pages them in lazily and the views are std::span straight into the mapping,
// so opening a file costs one mmap call and zero copies.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace tickforge {

inline constexpr char kMagic[8] = {'T', 'I', 'C', 'K', 'F', 'R', 'G', '1'};
inline constexpr std::uint32_t kFormatVersion = 1;

struct FileHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint64_t n_ticks;
  std::uint64_t off_ts;
  std::uint64_t off_price;
  std::uint64_t off_size;
  std::uint64_t off_side;
  std::uint64_t file_bytes;
  std::uint64_t seed;          // generator seed, for provenance
  std::uint8_t reserved[56];
};
static_assert(sizeof(FileHeader) == 128);

// Plain value view of one tick, used when pushing through the pipeline.
struct Tick {
  std::int64_t ts;
  double price;
  std::uint32_t size;
  std::int8_t side;
};

// Non-owning SoA view. Anything that reads ticks takes this, whether the
// memory is an mmap, a test vector, or a NumPy array.
struct TickColumns {
  std::span<const std::int64_t> ts;
  std::span<const double> price;
  std::span<const std::uint32_t> size;
  std::span<const std::int8_t> side;

  std::size_t size_ticks() const noexcept { return price.size(); }
  Tick operator[](std::size_t i) const noexcept { return {ts[i], price[i], size[i], side[i]}; }
};

// Owning in-memory columns, used by the generator and in tests.
struct TickVectors {
  std::vector<std::int64_t> ts;
  std::vector<double> price;
  std::vector<std::uint32_t> size;
  std::vector<std::int8_t> side;
  TickColumns view() const noexcept { return {ts, price, size, side}; }
};

void write_tick_file(const std::filesystem::path& path, const TickColumns& cols, std::uint64_t seed = 0);

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

 private:
  void unmap() noexcept;
  void* base_ = nullptr;
  std::size_t bytes_ = 0;
  const FileHeader* header_ = nullptr;
  TickColumns cols_{};
};

}  // namespace tickforge
