#include "tickforge/tick_store.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#include "tickforge/memory.hpp"

namespace tickforge {

static_assert(std::endian::native == std::endian::little, "tick files are little-endian");

namespace {

// Element size of each column, in file order.
constexpr std::array<std::size_t, kNumColumns> kElemBytes = {8, 8, 4, 1, 8, 8, 4, 4};

[[noreturn]] void throw_errno(const std::string& what) {
  throw std::system_error(errno, std::generic_category(), what);
}

template <class T>
void write_column(std::ofstream& out, std::span<const T> col, std::uint64_t offset) {
  out.seekp(static_cast<std::streamoff>(offset));
  out.write(reinterpret_cast<const char*>(col.data()), static_cast<std::streamsize>(col.size_bytes()));
}

// The mapping holds bytes that were written as T by write_tick_file. C++23's
// std::start_lifetime_as is the precise tool here; for trivially-copyable
// arithmetic types every major compiler treats this cast as well-defined.
template <class T>
std::span<const T> column(const void* base, std::uint64_t offset, std::size_t n) {
  return {reinterpret_cast<const T*>(static_cast<const std::byte*>(base) + offset), n};
}

}  // namespace

void TickVectors::push_back(const Tick& t) {
  ts.push_back(t.ts);
  price.push_back(t.price);
  size.push_back(t.size);
  side.push_back(t.side);
  bid.push_back(t.bid);
  ask.push_back(t.ask);
  bid_size.push_back(t.bid_size);
  ask_size.push_back(t.ask_size);
}

void TickVectors::reserve(std::size_t n) {
  ts.reserve(n);
  price.reserve(n);
  size.reserve(n);
  side.reserve(n);
  bid.reserve(n);
  ask.reserve(n);
  bid_size.reserve(n);
  ask_size.reserve(n);
}

void write_tick_file(const std::filesystem::path& path, const TickColumns& cols, std::uint64_t seed,
                     std::string_view symbol) {
  const std::size_t n = cols.size_ticks();
  const std::array<std::size_t, kNumColumns> lens = {cols.ts.size(),  cols.price.size(), cols.size.size(),
                                                     cols.side.size(), cols.bid.size(),   cols.ask.size(),
                                                     cols.bid_size.size(), cols.ask_size.size()};
  if (std::any_of(lens.begin(), lens.end(), [n](std::size_t l) { return l != n; }))
    throw std::invalid_argument("write_tick_file: column lengths differ");
  if (symbol.size() > kSymbolBytes) throw std::invalid_argument("write_tick_file: symbol longer than 16 bytes");

  FileHeader h{};
  std::memcpy(h.magic, kMagic, sizeof kMagic);
  h.version = kFormatVersion;
  h.header_bytes = sizeof(FileHeader);
  h.n_ticks = n;
  h.seed = seed;
  if (!symbol.empty()) std::memcpy(h.symbol, symbol.data(), symbol.size());
  std::uint64_t off = align_up(sizeof(FileHeader), kCacheLine);
  for (std::size_t c = 0; c < kNumColumns; ++c) {
    h.off[c] = off;
    off = align_up(off + n * kElemBytes[c], kCacheLine);
  }
  h.file_bytes = std::max<std::uint64_t>(off, align_up(sizeof(FileHeader), kCacheLine));

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open " + path.string() + " for writing");
  out.write(reinterpret_cast<const char*>(&h), sizeof h);
  write_column(out, cols.ts, h.off[0]);
  write_column(out, cols.price, h.off[1]);
  write_column(out, cols.size, h.off[2]);
  write_column(out, cols.side, h.off[3]);
  write_column(out, cols.bid, h.off[4]);
  write_column(out, cols.ask, h.off[5]);
  write_column(out, cols.bid_size, h.off[6]);
  write_column(out, cols.ask_size, h.off[7]);
  // Pad to file_bytes so the last column is followed by a full cache line.
  out.seekp(static_cast<std::streamoff>(h.file_bytes - 1));
  out.put('\0');
  if (!out) throw std::runtime_error("short write to " + path.string());
}

MappedTickFile::MappedTickFile(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw_errno("open " + path.string());
  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw_errno("fstat " + path.string());
  }
  bytes_ = static_cast<std::size_t>(st.st_size);
  if (bytes_ < sizeof(FileHeader)) {
    ::close(fd);
    throw std::runtime_error(path.string() + ": too small to be a tick file");
  }
  base_ = ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);  // the mapping keeps its own reference to the file
  if (base_ == MAP_FAILED) {
    base_ = nullptr;
    throw_errno("mmap " + path.string());
  }
  // Backtests scan front to back; tell the kernel to read ahead aggressively.
  ::madvise(base_, bytes_, MADV_SEQUENTIAL);

  header_ = static_cast<const FileHeader*>(base_);
  const FileHeader& h = *header_;
  const auto fail = [&](const std::string& why) {
    unmap();
    throw std::runtime_error(path.string() + ": " + why);
  };
  if (std::memcmp(h.magic, kMagic, sizeof kMagic) != 0) fail("bad magic");
  if (h.version != kFormatVersion)
    fail("unsupported format version " + std::to_string(h.version) + " (expected " +
         std::to_string(kFormatVersion) + "; regenerate the file)");
  if (h.file_bytes > bytes_) fail("truncated");
  const std::uint64_t n = h.n_ticks;
  for (std::size_t c = 0; c < kNumColumns; ++c) {
    if (h.off[c] % kCacheLine != 0) fail("misaligned column");
    if (n != 0 && (h.off[c] > bytes_ || n > (bytes_ - h.off[c]) / kElemBytes[c])) fail("column out of range");
  }

  cols_.ts = column<std::int64_t>(base_, h.off[0], n);
  cols_.price = column<double>(base_, h.off[1], n);
  cols_.size = column<std::uint32_t>(base_, h.off[2], n);
  cols_.side = column<std::int8_t>(base_, h.off[3], n);
  cols_.bid = column<double>(base_, h.off[4], n);
  cols_.ask = column<double>(base_, h.off[5], n);
  cols_.bid_size = column<std::uint32_t>(base_, h.off[6], n);
  cols_.ask_size = column<std::uint32_t>(base_, h.off[7], n);
}

MappedTickFile::~MappedTickFile() { unmap(); }

MappedTickFile::MappedTickFile(MappedTickFile&& o) noexcept
    : base_(std::exchange(o.base_, nullptr)),
      bytes_(std::exchange(o.bytes_, 0)),
      header_(std::exchange(o.header_, nullptr)),
      cols_(std::exchange(o.cols_, {})) {}

MappedTickFile& MappedTickFile::operator=(MappedTickFile&& o) noexcept {
  if (this != &o) {
    unmap();
    base_ = std::exchange(o.base_, nullptr);
    bytes_ = std::exchange(o.bytes_, 0);
    header_ = std::exchange(o.header_, nullptr);
    cols_ = std::exchange(o.cols_, {});
  }
  return *this;
}

std::string MappedTickFile::symbol() const {
  const char* s = header_->symbol;
  return {s, strnlen(s, kSymbolBytes)};
}

void MappedTickFile::unmap() noexcept {
  if (base_ != nullptr) ::munmap(base_, bytes_);
  base_ = nullptr;
  bytes_ = 0;
  header_ = nullptr;
  cols_ = {};
}

// ---- CSV import -------------------------------------------------------------

namespace {

std::vector<std::string_view> split(std::string_view line) {
  std::vector<std::string_view> out;
  std::size_t start = 0;
  while (true) {
    const std::size_t comma = line.find(',', start);
    std::string_view f = line.substr(start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
    while (!f.empty() && (f.front() == ' ' || f.front() == '\t')) f.remove_prefix(1);
    while (!f.empty() && (f.back() == ' ' || f.back() == '\t' || f.back() == '\r')) f.remove_suffix(1);
    out.push_back(f);
    if (comma == std::string_view::npos) break;
    start = comma + 1;
  }
  return out;
}

template <class T>
T parse(std::string_view f, std::size_t line_no, std::string_view name) {
  T v{};
  const auto [ptr, ec] = std::from_chars(f.data(), f.data() + f.size(), v);
  if (ec != std::errc{} || ptr != f.data() + f.size())
    throw std::runtime_error("CSV line " + std::to_string(line_no) + ": bad " + std::string(name) + " '" +
                             std::string(f) + "'");
  return v;
}

}  // namespace

TickVectors read_tick_csv(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error(path.string() + ": empty file");

  constexpr std::array<std::string_view, kNumColumns> kNames = {"ts",  "price", "size",     "side",
                                                                "bid", "ask",   "bid_size", "ask_size"};
  std::array<int, kNumColumns> idx{};
  idx.fill(-1);
  const auto header = split(line);
  for (std::size_t i = 0; i < header.size(); ++i)
    for (std::size_t c = 0; c < kNumColumns; ++c)
      if (header[i] == kNames[c]) idx[c] = static_cast<int>(i);
  for (std::size_t c = 0; c < 4; ++c)
    if (idx[c] < 0) throw std::runtime_error(path.string() + ": missing column '" + std::string(kNames[c]) + "'");
  const bool quotes = std::all_of(idx.begin() + 4, idx.end(), [](int i) { return i >= 0; });

  TickVectors out;
  std::size_t line_no = 1;
  std::int64_t last_ts = INT64_MIN;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty() || line == "\r") continue;
    const auto f = split(line);
    if (f.size() < header.size())
      throw std::runtime_error("CSV line " + std::to_string(line_no) + ": expected " + std::to_string(header.size()) +
                               " fields");
    const auto get = [&](std::size_t c) { return f[static_cast<std::size_t>(idx[c])]; };
    Tick t{};
    t.ts = parse<std::int64_t>(get(0), line_no, "ts");
    t.price = parse<double>(get(1), line_no, "price");
    t.size = parse<std::uint32_t>(get(2), line_no, "size");
    const int side = parse<int>(get(3), line_no, "side");
    if (side != 1 && side != -1) throw std::runtime_error("CSV line " + std::to_string(line_no) + ": side must be 1 or -1");
    t.side = static_cast<std::int8_t>(side);
    if (t.ts < last_ts) throw std::runtime_error("CSV line " + std::to_string(line_no) + ": timestamps go backwards");
    if (!(t.price > 0.0)) throw std::runtime_error("CSV line " + std::to_string(line_no) + ": price must be positive");
    last_ts = t.ts;
    if (quotes) {
      t.bid = parse<double>(get(4), line_no, "bid");
      t.ask = parse<double>(get(5), line_no, "ask");
      t.bid_size = parse<std::uint32_t>(get(6), line_no, "bid_size");
      t.ask_size = parse<std::uint32_t>(get(7), line_no, "ask_size");
      if (!(t.bid > 0.0) || t.ask < t.bid)
        throw std::runtime_error("CSV line " + std::to_string(line_no) + ": need 0 < bid <= ask");
    } else {
      t.bid = t.ask = t.price;
      t.bid_size = t.ask_size = t.size;
    }
    out.push_back(t);
  }
  return out;
}

}  // namespace tickforge
