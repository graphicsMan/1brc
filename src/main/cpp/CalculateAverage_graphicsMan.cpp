/**
 *  Copyright 2023 The original authors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <stdint.h>

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <print>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// ============================================================================
// Compiler-specific macros
// ============================================================================

#if defined(_MSC_VER)
  #define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
  #define FORCE_INLINE __attribute__((always_inline)) inline
#else
  #define FORCE_INLINE inline
#endif

// ============================================================================
// SIMD Configuration - Uncomment ONE of these to force a specific implementation
// ============================================================================
// #define USE_GENERIC      // Force generic (no SIMD) implementation
// #define USE_AUTO         // Use best available (default if none defined)

// ============================================================================
// Uncomment to get statistics and other information
// ============================================================================
//#define DEBUG_PRINT

// Determine which SIMD to use based on configuration and compiler support
#if defined(USE_GENERIC)
  // No SIMD macros defined
#else  // USE_AUTO or nothing defined - use best available
  #if defined(__AVX2__)
    #define USE_AVX2
  #elif defined(__ARM_NEON)
    #define USE_NEON
  #endif
#endif

#if defined(USE_NEON)
#include <arm_neon.h>
#elif defined(USE_AVX2)
#include <immintrin.h>
#endif

constexpr char kFile[] = "./measurements.txt";

class MMap {
public:
  MMap() = default;
  ~MMap() { close(); }

  // Disable copy and move
  MMap(const MMap&) = delete;
  MMap& operator=(const MMap&) = delete;
  MMap(MMap&&) = delete;
  MMap& operator=(MMap&&) = delete;

  bool open(const char* path) {
#if defined(_WIN32)
    hFile_ = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                         OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (hFile_ == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile_, &fileSize)) {
      close();
      return false;
    }
    size_ = fileSize.QuadPart;
    hMap_ = CreateFileMappingA(hFile_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap_) {
      close();
      return false;
    }
    data_ = static_cast<const char*>(MapViewOfFile(hMap_, FILE_MAP_READ, 0, 0, 0));
    if (!data_) {
      close();
      return false;
    }
#else
    fd_ = ::open(path, O_RDONLY);
    if (fd_ == -1) return false;
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
      close();
      return false;
    }
    size_ = sb.st_size;
    data_ = static_cast<const char*>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      close();
      return false;
    }
    madvise(const_cast<char*>(data_), size_, MADV_SEQUENTIAL);
#endif
    return true;
  }

  void close() {
#if defined(_WIN32)
    if (data_) {
      UnmapViewOfFile(data_);
      data_ = nullptr;
    }
    if (hMap_) {
      CloseHandle(hMap_);
      hMap_ = nullptr;
    }
    if (hFile_ != INVALID_HANDLE_VALUE) {
      CloseHandle(hFile_);
      hFile_ = INVALID_HANDLE_VALUE;
    }
#else
    if (data_) {
      munmap(const_cast<char*>(data_), size_);
      data_ = nullptr;
    }
    if (fd_ != -1) {
      ::close(fd_);
      fd_ = -1;
    }
#endif
  }

  const char* data() const { return data_; }
  size_t size() const { return size_; }

private:
  const char* data_ = nullptr;
  size_t size_ = 0;
#if defined(_WIN32)
  HANDLE hFile_ = INVALID_HANDLE_VALUE;
  HANDLE hMap_ = nullptr;
#else
  int fd_ = -1;
#endif
};

struct MeasurementAggregator {
  int64_t sum = 0;
  uint32_t count = 0;
  int16_t min = std::numeric_limits<int16_t>::max();
  int16_t max = std::numeric_limits<int16_t>::min();

  void add(int value) {
    min = std::min<int16_t>(min, value);
    max = std::max<int16_t>(max, value);
    sum += value;
    ++count;
  }

  void merge(const MeasurementAggregator& other) {
    min = std::min<int16_t>(min, other.min);
    max = std::max<int16_t>(max, other.max);
    sum += other.sum;
    count += other.count;
  }
};

struct ResultRow {
  double min;
  double mean;
  double max;

  ResultRow(double min, double mean, double max)
      : min(min), mean(mean), max(max) {}

  static double round(double value) {
    return std::round(value * 10.0) / 10.0;
  }

  std::string toString() const {
    return std::format("{:.1f}/{:.1f}/{:.1f}", round(min), round(mean), round(max));
  }
};

// SWAR constants for finding specific bytes (used in NEON/fallback paths)
[[maybe_unused]] constexpr uint64_t kSemicolonPattern = 0x3B3B3B3B3B3B3B3BULL;
[[maybe_unused]] constexpr uint64_t kNewlinePattern = 0x0A0A0A0A0A0A0A0AULL;
[[maybe_unused]] constexpr uint64_t kSwarLow = 0x0101010101010101ULL;
[[maybe_unused]] constexpr uint64_t kSwarHigh = 0x8080808080808080ULL;

// ============================================================================
// Key32: Platform-specific 32-byte key type and operations
// ============================================================================

#if defined(USE_AVX2)
// AVX2: Use __m256i for 32-byte keys
using Key32 = __m256i;
static constexpr size_t kKey32Align = 32;

static const __m256i kIndices32 = _mm256_setr_epi8(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);

struct Masks32 {
  __m256i masks[32];
  Masks32() {
    for (int i = 0; i < 32; ++i) {
      alignas(32) uint8_t m[32] = {};
      for (int j = 0; j < i; ++j) m[j] = 0xFF;
      masks[i] = _mm256_load_si256(reinterpret_cast<const __m256i*>(m));
    }
  }
};
static const Masks32 kMasks32;

FORCE_INLINE Key32 loadMasked32(const char* data, size_t len) {
  __m256i loaded = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
  return _mm256_and_si256(loaded, kMasks32.masks[len]);
}

FORCE_INLINE bool eq32(Key32 a, Key32 b) {
  __m256i cmp = _mm256_cmpeq_epi8(a, b);
  return _mm256_movemask_epi8(cmp) == static_cast<int32_t>(0xFFFFFFFF);
}

FORCE_INLINE bool isZero32(Key32 v) {
  return _mm256_testz_si256(v, v);
}

FORCE_INLINE Key32 zeroKey32() {
  return _mm256_setzero_si256();
}

FORCE_INLINE uint64_t extractFirst8(Key32 key) {
  return static_cast<uint64_t>(_mm256_extract_epi64(key, 0));
}

FORCE_INLINE const char* keyData32(const Key32* key) {
  return reinterpret_cast<const char*>(key);
}

#elif defined(USE_NEON)
// NEON: Use uint8x16x2_t for 32-byte keys (two 16-byte vectors)
using Key32 = uint8x16x2_t;
static constexpr size_t kKey32Align = 16;

FORCE_INLINE Key32 loadMasked32(const char* data, size_t len) {
  uint8x16_t lo = vld1q_u8(reinterpret_cast<const uint8_t*>(data));
  uint8x16_t hi = vld1q_u8(reinterpret_cast<const uint8_t*>(data + 16));

  // Create masks for low and high halves
  static const uint8x16_t kIndicesLo = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  static const uint8x16_t kIndicesHi = {16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};

  uint8x16_t lenVec = vdupq_n_u8(static_cast<uint8_t>(len));
  uint8x16_t maskLo = vcgtq_u8(lenVec, kIndicesLo);
  uint8x16_t maskHi = vcgtq_u8(lenVec, kIndicesHi);

  Key32 result;
  result.val[0] = vandq_u8(lo, maskLo);
  result.val[1] = vandq_u8(hi, maskHi);
  return result;
}

FORCE_INLINE bool eq32(Key32 a, Key32 b) {
  uint8x16_t cmpLo = vceqq_u8(a.val[0], b.val[0]);
  uint8x16_t cmpHi = vceqq_u8(a.val[1], b.val[1]);
  uint8x16_t cmpAll = vandq_u8(cmpLo, cmpHi);
  return vminvq_u8(cmpAll) == 0xFF;
}

FORCE_INLINE bool isZero32(Key32 v) {
  uint8x16_t orBoth = vorrq_u8(v.val[0], v.val[1]);
  return vmaxvq_u8(orBoth) == 0;
}

FORCE_INLINE Key32 zeroKey32() {
  Key32 result;
  result.val[0] = vdupq_n_u8(0);
  result.val[1] = vdupq_n_u8(0);
  return result;
}

FORCE_INLINE uint64_t extractFirst8(Key32 key) {
  return vgetq_lane_u64(vreinterpretq_u64_u8(key.val[0]), 0);
}

FORCE_INLINE const char* keyData32(const Key32* key) {
  return reinterpret_cast<const char*>(&key->val[0]);
}

#else
// Generic: Use struct with uint64_t[4] for efficient operations
struct Key32 {
  union {
    uint64_t u64[4];
    uint8_t u8[32];
  };
};
static constexpr size_t kKey32Align = 8;

FORCE_INLINE Key32 loadMasked32(const char* data, size_t len) {
  Key32 result{};
  std::memcpy(result.u8, data, std::min(len, size_t{32}));
  return result;
}

FORCE_INLINE bool eq32(const Key32& a, const Key32& b) {
  return a.u64[0] == b.u64[0] && a.u64[1] == b.u64[1] &&
         a.u64[2] == b.u64[2] && a.u64[3] == b.u64[3];
}

FORCE_INLINE bool isZero32(const Key32& v) {
  return (v.u64[0] | v.u64[1] | v.u64[2] | v.u64[3]) == 0;
}

FORCE_INLINE Key32 zeroKey32() {
  return Key32{};
}

FORCE_INLINE uint64_t extractFirst8(const Key32& key) {
  return key.u64[0];
}

FORCE_INLINE const char* keyData32(const Key32* key) {
  return reinterpret_cast<const char*>(key->u8);
}
#endif

// ============================================================================
// Hash function (shared across platforms)
// ============================================================================

// Hash function selection (auto-detected, or override by defining HASH_FUNCTION):
//   1 = wyhash-style (requires __uint128_t for fast 128-bit multiply)
//   2 = aHash AES-based 2 rounds (requires AES-NI)
//   3 = Murmur3 finalizer (XOR-shift + multiply) - fallback
//
// Auto-selection priority:
//   1. __uint128_t available -> wyhash (1) - best quality + speed
//   2. AES-NI available -> AES 2 rounds (2) - fast but slightly more collisions
//   3. Otherwise -> Murmur3 (3) - reliable fallback
#ifndef HASH_FUNCTION
  #if defined(__SIZEOF_INT128__)
    #define HASH_FUNCTION 1
  #elif defined(__AES__) && defined(USE_AVX2)
    #define HASH_FUNCTION 2
  #else
    #define HASH_FUNCTION 3
  #endif
#endif

#if HASH_FUNCTION == 1
// wyhash-style - extremely fast multiply-xor-multiply (requires __uint128_t)
FORCE_INLINE uint64_t wymul128(uint64_t a, uint64_t b) {
  __uint128_t r = static_cast<__uint128_t>(a) * b;
  return static_cast<uint64_t>(r) ^ static_cast<uint64_t>(r >> 64);
}

FORCE_INLINE size_t hashKey32(const Key32& key) {
  uint64_t h = extractFirst8(key);
  constexpr uint64_t WYP0 = 0xa0761d6478bd642fULL;
  constexpr uint64_t WYP1 = 0xe7037ed1a0b428dbULL;
  return wymul128(h ^ WYP0, h ^ WYP1);
}

#elif HASH_FUNCTION == 2
// aHash AES-based - uses hardware AES instructions
#if defined(__AES__) && defined(USE_AVX2)
FORCE_INLINE size_t hashKey32(const Key32& key) {
  __m128i enc = _mm_set_epi64x(0, static_cast<int64_t>(extractFirst8(key)));
  __m128i k1 = _mm_set_epi64x(0x452821E638D01377LL, 0x243F6A8885A308D3LL);
  __m128i k2 = _mm_set_epi64x(0xBE5466CF34E90C6CLL, 0xA4093822299F31D0LL);
  enc = _mm_aesenc_si128(enc, k1);
  enc = _mm_aesenc_si128(enc, k2);
  return static_cast<size_t>(_mm_cvtsi128_si64(enc));
}
#else
#error "HASH_FUNCTION 2 (aHash AES) requires AES-NI and AVX2"
#endif

#elif HASH_FUNCTION == 3
// Murmur3 finalizer - XOR-shift and multiply (fallback)
FORCE_INLINE size_t hashKey32(const Key32& key) {
  uint64_t h = extractFirst8(key);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return h;
}

#else
#error "Unknown HASH_FUNCTION value (valid: 1=wyhash, 2=AES, 3=Murmur3)"
#endif

// ============================================================================
// ParseResult: Common structure for parsing results
// ============================================================================

struct ParseResult {
  Key32 maskedKey;
  const char* stationPtr;  // For large stations (>31 bytes)
  int value;
  const char* next;
  uint8_t stationLen;
};

// ============================================================================
// parseNext: Platform-specific line parsing
// ============================================================================

FORCE_INLINE std::pair<int, int> parseTemperature(uint64_t tempData) {
  int bytesConsumed = 0;
  int sign = 1;
  if ((tempData & 0xFF) == '-') {
    sign = -1;
    tempData >>= 8;
    bytesConsumed = 1;
  }

  int b0 = (tempData & 0xFF);
  int b1 = (tempData >> 8) & 0xFF;
  int b2 = (tempData >> 16) & 0xFF;
  int b3 = (tempData >> 24) & 0xFF;

  int value;
  if (b1 == '.') {
    // d.d format
    value = (b0 - '0') * 10 + (b2 - '0');
    bytesConsumed += 3;
  } else {
    // dd.d format
    value = (b0 - '0') * 100 + (b1 - '0') * 10 + (b3 - '0');
    bytesConsumed += 4;
  }

  return {sign * value, bytesConsumed};
}

#if defined(USE_AVX2)
inline ParseResult parseNext(const char* p) {
  // Load 32 bytes to capture station + temperature in most cases
  __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));

  // Find semicolon
  __m256i vsemi = _mm256_set1_epi8(';');
  __m256i cmpSemi = _mm256_cmpeq_epi8(chunk, vsemi);
  int maskSemi = _mm256_movemask_epi8(cmpSemi);

  int semiPos;
  __m256i maskedKey;

  if (maskSemi) {
    // Fast path: semicolon in first 32 bytes (~99% of cases)
    semiPos = __builtin_ctz(maskSemi);
    // Mask the key directly from already-loaded chunk
    __m256i keyMask = _mm256_cmpgt_epi8(
        _mm256_set1_epi8(static_cast<int8_t>(semiPos)), kIndices32);
    maskedKey = _mm256_and_si256(chunk, keyMask);
  } else {
    // Fallback: search for semicolon sequentially
    const char* semi = p + 1;
    while (*semi != ';') ++semi;
    semiPos = static_cast<int>(semi - p);

    // For fallback, load and mask the key
    if (semiPos <= 31) {
      maskedKey = loadMasked32(p, semiPos);
    } else {
      maskedKey = zeroKey32();  // Will use stationPtr instead
    }
  }

  const char* temp = p + semiPos + 1;
  uint64_t tempData;
  std::memcpy(&tempData, temp, 8);

  auto [value, bytesConsumed] = parseTemperature(tempData);

  return {maskedKey, p, value, temp + bytesConsumed, static_cast<uint8_t>(semiPos)};
}
#elif defined(USE_NEON)
inline ParseResult parseNext(const char* p) {
  // Lambda to find character position using SWAR on 16-byte NEON register
  auto findChar = [](uint8x16_t chunk, uint64_t targetPattern) -> int {
    uint64_t chunkLo = vgetq_lane_u64(vreinterpretq_u64_u8(chunk), 0);
    uint64_t chunkHi = vgetq_lane_u64(vreinterpretq_u64_u8(chunk), 1);

    uint64_t xorLo = chunkLo ^ targetPattern;
    uint64_t hasLo = (xorLo - kSwarLow) & ~xorLo & kSwarHigh;

    if (hasLo) {
      return __builtin_ctzll(hasLo) >> 3;
    }

    uint64_t xorHi = chunkHi ^ targetPattern;
    uint64_t hasHi = (xorHi - kSwarLow) & ~xorHi & kSwarHigh;
    return 8 + (__builtin_ctzll(hasHi) >> 3);
  };

  // Load 16 bytes first (covers 60% of cases)
  uint8x16_t chunk = vld1q_u8(reinterpret_cast<const uint8_t*>(p));

  // Find semicolon and newline
  uint8x16_t vsemi = vdupq_n_u8(';');
  uint8x16_t vnewline = vdupq_n_u8('\n');
  uint8x16_t cmpSemi = vceqq_u8(chunk, vsemi);
  uint8x16_t cmpNewline = vceqq_u8(chunk, vnewline);

  int semiPos;
  bool bothInFirst16 = vmaxvq_u8(cmpSemi) && vmaxvq_u8(cmpNewline);

  if (vmaxvq_u8(cmpSemi)) {
    // Semicolon in first 16 bytes
    semiPos = findChar(chunk, kSemicolonPattern);
  } else {
    // Fallback: sequential search for semicolon
    const char* semi = p + 1;
    while (*semi != ';') ++semi;
    semiPos = static_cast<int>(semi - p);
  }

  // Create masked key
  Key32 maskedKey;
  if (semiPos <= 31) {
    maskedKey = loadMasked32(p, semiPos);
  } else {
    maskedKey = zeroKey32();  // Will use stationPtr instead
  }

  // Extract temperature data
  const char* temp = p + semiPos + 1;
  uint64_t tempData;

  if (bothInFirst16) {
    // Both delimiters in first 16 bytes - extract from loaded chunk
    int tempOffset = semiPos + 1;
    uint64_t chunkLo = vgetq_lane_u64(vreinterpretq_u64_u8(chunk), 0);
    uint64_t chunkHi = vgetq_lane_u64(vreinterpretq_u64_u8(chunk), 1);

    if (tempOffset < 8) {
      tempData = chunkLo >> (tempOffset * 8);
      if (tempOffset > 0) {
        tempData |= (chunkHi << ((8 - tempOffset) * 8));
      }
    } else {
      tempData = chunkHi >> ((tempOffset - 8) * 8);
    }
  } else {
    std::memcpy(&tempData, temp, 8);
  }

  auto [value, bytesConsumed] = parseTemperature(tempData);

  return {maskedKey, p, value, temp + bytesConsumed, static_cast<uint8_t>(semiPos)};
}
#else
inline ParseResult parseNext(const char* p) {
  // Find semicolon - optimized to skip first character (guaranteed min 1 char)
  const char* semi = p + 1;

  // SWAR version for ARM and other platforms - search 8 bytes at a time
  while (true) {
    uint64_t chunk;
    std::memcpy(&chunk, semi, 8);
    uint64_t xorResult = chunk ^ kSemicolonPattern;
    uint64_t hasSemi = (xorResult - kSwarLow) & ~xorResult & kSwarHigh;

    if (hasSemi) {
      semi += __builtin_ctzll(hasSemi) >> 3;
      break;
    }
    semi += 8;
  }

  size_t semiPos = static_cast<size_t>(semi - p);

  // Create masked key
  Key32 maskedKey;
  if (semiPos <= 31) {
    maskedKey = loadMasked32(p, semiPos);
  } else {
    maskedKey = zeroKey32();  // Will use stationPtr instead
  }

  // Parse temperature
  const char* temp = semi + 1;
  uint64_t tempData;
  std::memcpy(&tempData, temp, 8);

  auto [value, bytesConsumed] = parseTemperature(tempData);

  return {maskedKey, p, value, temp + bytesConsumed, static_cast<uint8_t>(semiPos)};
}
#endif

struct LgHash {
  size_t operator()(std::string_view v) const {
    uint64_t h = 0;
    std::memcpy(&h, v.data(), std::min(size_t{8}, v.size()));
    return h * uint64_t{0x9e3779b97f4a7c15};
  }
};

class Maps {
public:
  Maps() {
    measurementsLg_.reserve(128);
    allocateArrays(kStartLen);
  }

  ~Maps() {
    std::free(keys_);
    std::free(values_);
  }

  // Disable copy and move
  Maps(const Maps&) = delete;
  Maps& operator=(const Maps&) = delete;
  Maps(Maps&&) = delete;
  Maps& operator=(Maps&&) = delete;

  void add(const ParseResult& result) {
    if (result.stationLen > 31) {
      std::string_view station(result.stationPtr, result.stationLen);
      measurementsLg_[station].add(result.value);
    } else {
      addSmall(result.maskedKey, result.value);
    }
  }

  // Legacy add for tail processing
  void add(std::string_view station, int value) {
    if (station.size() > 31) {
      measurementsLg_[station].add(value);
    } else {
      Key32 maskedKey = loadMasked32(station.data(), station.size());
      addSmall(maskedKey, value);
    }
  }

  void merge(const Maps& other) {
    // Merge large maps
    for (const auto& [station, agg] : other.measurementsLg_) {
      measurementsLg_[station].merge(agg);
    }

    // Merge small maps (keys are already masked)
    for (size_t i = 0; i < other.len_; ++i) {
      if (!isZero32(other.keys_[i])) {
        Key32 maskedKey = other.keys_[i];
        size_t hash = hashKey32(maskedKey);
        size_t idx = hash & (len_ - 1);

        for (size_t probe = 0; probe < len_; ++probe) {
          if (tryAddSmall(idx, maskedKey)) {
            values_[idx].merge(other.values_[i]);
            break;
          }
          idx = (idx + 1) & (len_ - 1);
        }
      }
    }

#if defined(DEBUG_PRINT)
    totalLookups_ += other.totalLookups_;
    probeCollisions_ += other.probeCollisions_;
    rehashCount_ += other.rehashCount_;
    if (other.maxProbeDepth_ > maxProbeDepth_) {
      maxProbeDepth_ = other.maxProbeDepth_;
    }
#endif
  }

  void consolidate() {
    // Merge small map into large map
    for (size_t i = 0; i < len_; ++i) {
      if (!isZero32(keys_[i])) {
        const char* kd = keyData32(&keys_[i]);
        size_t keyLen = 0;
        while (keyLen < 32 && kd[keyLen] != '\0') ++keyLen;
        std::string_view station(kd, keyLen);
        measurementsLg_[station].merge(values_[i]);
      }
    }
    // Note: Don't clear keys - string_views in measurementsLg_ point to them
  }

#if defined(DEBUG_PRINT)
  void printStats() const {
    std::print("Hash table stats:\n");
    std::print("  Total lookups: {}\n", totalLookups_);
    std::print("  Probe collisions: {}\n", probeCollisions_);
    std::print("  Collision rate: {:.2f}%\n",
               totalLookups_ > 0 ? (100.0 * probeCollisions_ / totalLookups_) : 0.0);
    std::print("  Rehash count: {}\n", rehashCount_);
    std::print("  Max probe depth: {}\n", maxProbeDepth_);
    std::print("  Final table size: {}\n", len_);
  }
#endif

  const auto& measurements() const { return measurementsLg_; }

private:
  static constexpr size_t kStartLen = 16384;
  static constexpr size_t kMaxProbes = 8;

  std::unordered_map<std::string_view, MeasurementAggregator, LgHash> measurementsLg_;
  Key32* keys_ = nullptr;
  MeasurementAggregator* values_ = nullptr;
  size_t len_ = kStartLen;

#if defined(DEBUG_PRINT)
  size_t totalLookups_ = 0;
  size_t probeCollisions_ = 0;
  size_t rehashCount_ = 0;
  size_t maxProbeDepth_ = 0;
#endif

  void allocateArrays(size_t newLen) {
    constexpr size_t alignment = 32;
    keys_ = static_cast<Key32*>(std::aligned_alloc(alignment, newLen * sizeof(Key32)));
    values_ = static_cast<MeasurementAggregator*>(
        std::aligned_alloc(alignof(MeasurementAggregator),
                           newLen * sizeof(MeasurementAggregator)));

    std::memset(keys_, 0, newLen * sizeof(Key32));
    for (size_t i = 0; i < newLen; ++i) {
      new (&values_[i]) MeasurementAggregator();
    }
  }

  FORCE_INLINE bool tryAddSmall(size_t idx, Key32 maskedKey) {
    Key32 existing = keys_[idx];

    // Empty slot - insert here
    if (isZero32(existing)) {
      keys_[idx] = maskedKey;
      return true;
    }

    // Key matches - already exists
    if (eq32(existing, maskedKey)) {
      return true;
    }

    // Collision - different key
    return false;
  }

  void rehash() {
    size_t oldLen = len_;
    size_t newLen = len_ * 2;

    Key32* oldKeys = keys_;
    MeasurementAggregator* oldValues = values_;

    len_ = newLen;
    allocateArrays(newLen);

    // Re-insert all existing entries (keys are already masked)
    for (size_t i = 0; i < oldLen; ++i) {
      if (!isZero32(oldKeys[i])) {
        size_t hash = hashKey32(oldKeys[i]);
        size_t idx = hash & (len_ - 1);

        while (!tryAddSmall(idx, oldKeys[i])) {
          idx = (idx + 1) & (len_ - 1);
        }
        values_[idx] = oldValues[i];
      }
    }

    std::free(oldKeys);
    std::free(oldValues);
  }

  void addSmall(Key32 maskedKey, int value) {
    size_t hash = hashKey32(maskedKey);
    size_t idx = hash & (len_ - 1);

#if defined(DEBUG_PRINT)
    ++totalLookups_;
#endif

    for (size_t probe = 0; probe < kMaxProbes; ++probe) {
      if (tryAddSmall(idx, maskedKey)) {
        values_[idx].add(value);
#if defined(DEBUG_PRINT)
        if (probe > maxProbeDepth_) {
          maxProbeDepth_ = probe;
        }
#endif
        return;
      }
#if defined(DEBUG_PRINT)
      ++probeCollisions_;
#endif
      idx = (idx + 1) & (len_ - 1);
    }

#if defined(DEBUG_PRINT)
    ++rehashCount_;
#endif
    rehash();
    addSmall(maskedKey, value);
  }
};

void processChunk(const char* roughStart, const char* roughEnd,
                  const char* fileStart, const char* fileEnd, Maps& maps) {
  const char* start = roughStart;
  const char* end = roughEnd;

  // If not the first chunk, skip to the first complete line
  if (start > fileStart) {
    while (start < fileEnd && *(start - 1) != '\n') {
      ++start;
    }
  }

  // If not the last chunk, find the end of the last complete line in our range
  if (end < fileEnd) {
    while (end < fileEnd && *end != '\n') {
      ++end;
    }
    if (end < fileEnd) {
      ++end;  // Move past the newline
    }
  }

  const char* p = start;

  // Process all complete lines except the last few to ensure parseNext has buffer space
  const char* safeEnd = end - 32;  // Conservative: ensure enough space for parseNext reads
  if (safeEnd < p) safeEnd = p;

  while (p < safeEnd) {
    ParseResult result = parseNext(p);
    maps.add(result);
    p = result.next + 1;
  }

  // Process remaining tail with boundary checks
  while (p < end) {
    const char* semi = p;
    while (semi < end && *semi != ';') ++semi;
    if (semi >= end) break;

    std::string_view station(p, static_cast<size_t>(semi - p));

    // Parse temperature inline
    const char* temp = semi + 1;
    if (temp >= end) break;
    int sign = 1;
    if (*temp == '-') {
      sign = -1;
      temp++;
    }
    if (temp >= end) break;

    int value = *temp++ - '0';
    if (temp < end && *temp != '.') {
      value = value * 10 + (*temp++ - '0');
      if (temp < end && *temp != '.') {
        value = value * 10 + (*temp++ - '0');
      }
    }
    if (temp >= end) break;
    temp++;  // skip '.'
    if (temp >= end) break;
    value = value * 10 + (*temp++ - '0');
    value *= sign;

    const char* lineEnd = temp;
    while (lineEnd < end && *lineEnd != '\n') ++lineEnd;

    maps.add(station, value);
    p = lineEnd + 1;
  }

  // Consolidate small/medium maps into large map
  maps.consolidate();
}

int main() {
  using Clock = std::chrono::high_resolution_clock;
  auto totalStart = Clock::now();

  // Get number of threads
  unsigned int numThreads = std::thread::hardware_concurrency() - 1;
  if (numThreads == 0) numThreads = 1;

#if defined(DEBUG_PRINT)
  std::print("Using {} threads\n", numThreads);
  #if defined(USE_AVX2)
  std::print("SIMD: AVX2\n");
  #elif defined(USE_NEON)
  std::print("SIMD: NEON\n");
  #else
  std::print("SIMD: Generic (no SIMD)\n");
  #endif
#endif

  // === MMAP PHASE ===
  auto mmapStart = Clock::now();
  MMap mmap;
  if (!mmap.open(kFile)) {
    std::print(stderr, "Failed to open file: {}\n", kFile);
    return 1;
  }
  auto mmapEnd = Clock::now();

  // Compute chunk boundaries
  size_t chunkSize = mmap.size() / numThreads;
  const char* fileStart = mmap.data();
  const char* fileEnd = mmap.data() + mmap.size();

  // Calculate number of merge rounds: ceil(log2(numThreads))
  unsigned int mergeRounds = 0;
  for (unsigned int n = numThreads; n > 1; n = (n + 1) / 2) {
    ++mergeRounds;
  }

  // Create barriers for synchronization using deque (stable references)
  // - parseComplete: signals all threads have finished parsing
  // - roundComplete[r]: signals all threads have finished round r
  std::deque<std::barrier<>> barriers;
  barriers.emplace_back(numThreads);  // parseComplete
  for (unsigned int r = 0; r < mergeRounds; ++r) {
    barriers.emplace_back(numThreads);  // roundComplete[r]
  }

  // === PARSE PHASE ===
  auto parseStart = Clock::now();

  // Create Maps storage - Maps objects will be created in parallel by each thread
  std::vector<std::unique_ptr<Maps>> threadMaps(numThreads);

  // Spawn worker threads - each thread creates its own Maps and participates in merge
  std::vector<std::thread> threads;
  threads.reserve(numThreads - 1);
  for (unsigned int i = 1; i < numThreads; ++i) {
    const char* chunkStart = mmap.data() + (i * chunkSize);
    const char* chunkEnd = (i == numThreads - 1) ? fileEnd : (mmap.data() + ((i + 1) * chunkSize));

    threads.emplace_back([&, i, chunkStart, chunkEnd]() {
      // Create Maps and process chunk
      threadMaps[i] = std::make_unique<Maps>();
      processChunk(chunkStart, chunkEnd, fileStart, fileEnd, *threadMaps[i]);

      // Signal parsing complete and wait for all threads
      barriers[0].arrive_and_wait();

      // Participate in log(n) merge rounds
      for (unsigned int r = 0; r < mergeRounds; ++r) {
        unsigned int step = 1u << r;  // 1, 2, 4, 8, ...
        unsigned int stride = step * 2;  // 2, 4, 8, 16, ...

        // Thread i merges from thread (i + step) if:
        // - i is divisible by stride
        // - (i + step) < numThreads
        if ((i % stride) == 0 && (i + step) < numThreads) {
          threadMaps[i]->merge(*threadMaps[i + step]);
        }

        // Wait for all threads to complete this round
        barriers[1 + r].arrive_and_wait();
      }
    });
  }

  // Main thread (index 0) creates its Maps and processes its chunk
  threadMaps[0] = std::make_unique<Maps>();
  processChunk(fileStart, mmap.data() + chunkSize, fileStart, fileEnd, *threadMaps[0]);

  // Signal parsing complete and wait for all threads
  barriers[0].arrive_and_wait();
  auto parseEnd = Clock::now();

#if defined(DEBUG_PRINT)
  std::print("All threads completed parsing, starting parallel merge...\n");
#endif

  // === MERGE PHASE (parallel log(n) merge) ===
  auto mergeStart = Clock::now();

  // Main thread participates in merge rounds
  for (unsigned int r = 0; r < mergeRounds; ++r) {
    unsigned int step = 1u << r;  // 1, 2, 4, 8, ...

    // Thread 0 merges from thread step if step < numThreads
    if (step < numThreads) {
      threadMaps[0]->merge(*threadMaps[step]);
    }

    // Wait for all threads to complete this round
    barriers[1 + r].arrive_and_wait();
  }

  auto mergeEnd = Clock::now();

  // Detach threads - they've completed all work
  for (auto& thread : threads) {
    thread.detach();
  }

#if defined(DEBUG_PRINT)
  threadMaps[0]->printStats();
  std::print("Total stations: {}\n", threadMaps[0]->measurements().size());
#endif

  // === OUTPUT PHASE ===
  auto outputStart = Clock::now();

  // Copy to vector and sort by station name
  std::vector<std::pair<std::string_view, MeasurementAggregator>> sorted;
  sorted.reserve(threadMaps[0]->measurements().size());
  for (const auto& entry : threadMaps[0]->measurements()) {
    sorted.push_back(entry);
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  // Output results (everything is in the large map now)
  std::print("{{");
  bool first = true;
  for (const auto& [station, agg] : sorted) {
    if (!first) std::print(", ");
    first = false;
    double mean = (agg.sum / 10.0) / agg.count;
    ResultRow result(agg.min / 10.0, mean, agg.max / 10.0);
    std::print("{}={}", station, result.toString());
  }
  std::print("}}\n");
  auto outputEnd = Clock::now();

  auto totalEnd = Clock::now();

  // Print timing breakdown
  auto toMs = [](auto duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
  };

#if defined(DEBUG_PRINT)
  std::print("\n");
#endif
  std::print(stderr, "=== Timing Breakdown ===\n");
  std::print(stderr, "  mmap:   {:8.3f} ms\n", toMs(mmapEnd - mmapStart));
  std::print(stderr, "  parse:  {:8.3f} ms\n", toMs(parseEnd - parseStart));
  std::print(stderr, "  merge:  {:8.3f} ms\n", toMs(mergeEnd - mergeStart));
  std::print(stderr, "  output: {:8.3f} ms\n", toMs(outputEnd - outputStart));
  std::print(stderr, "  total:  {:8.3f} ms\n", toMs(totalEnd - totalStart));

  return 0;
}
