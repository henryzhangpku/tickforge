#include "tickforge/tick_store.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <bit>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#include "tickforge/memory.hpp"

namespace tickforge {

static_assert(std::endian::native == std::endian::little, "tick files are little-endian");

namespace {

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

void write_tick_file(const std::filesystem::path& path, const TickColumns& cols, std::uint64_t seed) {
  const std::size_t n = cols.size_ticks();
  if (cols.ts.size() != n || cols.size.size() != n || cols.side.size() != n)
    throw std::invalid_argument("write_tick_file: column lengths differ");

  FileHeader h{};
  std::memcpy(h.magic, kMagic, sizeof kMagic);
  h.version = kFormatVersion;
  h.header_bytes = sizeof(FileHeader);
  h.n_ticks = n;
  h.seed = seed;
  std::uint64_t off = align_up(sizeof(FileHeader), kCacheLine);
  h.off_ts = off;
  off = align_up(off + n * sizeof(std::int64_t), kCacheLine);
  h.off_price = off;
  off = align_up(off + n * sizeof(double), kCacheLine);
  h.off_size = off;
  off = align_up(off + n * sizeof(std::uint32_t), kCacheLine);
  h.off_side = off;
  h.file_bytes = align_up(off + n * sizeof(std::int8_t), kCacheLine);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open " + path.string() + " for writing");
  out.write(reinterpret_cast<const char*>(&h), sizeof h);
  write_column(out, cols.ts, h.off_ts);
  write_column(out, cols.price, h.off_price);
  write_column(out, cols.size, h.off_size);
  write_column(out, cols.side, h.off_side);
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
  const auto fail = [&](const char* why) {
    unmap();
    throw std::runtime_error(path.string() + ": " + why);
  };
  if (std::memcmp(h.magic, kMagic, sizeof kMagic) != 0) fail("bad magic");
  if (h.version != kFormatVersion) fail("unsupported version");
  if (h.file_bytes > bytes_) fail("truncated");
  const std::uint64_t n = h.n_ticks;
  if (h.off_side + n > bytes_ || h.off_size + 4 * n > h.off_side || h.off_price + 8 * n > h.off_size ||
      h.off_ts + 8 * n > h.off_price)
    fail("column offsets out of range");
  if ((h.off_ts | h.off_price | h.off_size | h.off_side) % kCacheLine != 0) fail("misaligned column");

  cols_.ts = column<std::int64_t>(base_, h.off_ts, n);
  cols_.price = column<double>(base_, h.off_price, n);
  cols_.size = column<std::uint32_t>(base_, h.off_size, n);
  cols_.side = column<std::int8_t>(base_, h.off_side, n);
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

void MappedTickFile::unmap() noexcept {
  if (base_ != nullptr) ::munmap(base_, bytes_);
  base_ = nullptr;
  bytes_ = 0;
  header_ = nullptr;
  cols_ = {};
}

}  // namespace tickforge
