/*
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

#include <iostream>
#include <string_view>
#include <unordered_map>
#include <map>
#include <flat_map>
#include <sstream>
#include <iomanip>
#include <limits>
#include <deque>
#include <cmath>
#include <stdint.h>
#include <tuple>

#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

constexpr char kFile[] = "./measurements.txt";

struct MMap {
    char* data = nullptr;
    size_t size = 0;
#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap = nullptr;
#else
    int fd = -1;
#endif

    bool open(const char* path) {
#ifdef _WIN32
        hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(hFile, &fileSize)) { close(); return false; }
        size = fileSize.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) { close(); return false; }
        data = (char*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!data) { close(); return false; }
#else
        fd = ::open(path, O_RDONLY);
        if (fd == -1) return false;
        struct stat sb;
        if (fstat(fd, &sb) == -1) { close(); return false; }
        size = sb.st_size;
        data = (char*)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { data = nullptr; close(); return false; }
        madvise(data, size, MADV_SEQUENTIAL);
#endif
        return true;
    }

    void close() {
#ifdef _WIN32
        if (data) { UnmapViewOfFile(data); data = nullptr; }
        if (hMap) { CloseHandle(hMap); hMap = nullptr; }
        if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
#else
        if (data) { munmap(data, size); data = nullptr; }
        if (fd != -1) { ::close(fd); fd = -1; }
#endif
    }

    ~MMap() { close(); }
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
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1)
            << round(min) << "/" << round(mean) << "/" << round(max);
        return oss.str();
    }
};


#ifdef __ARM_NEON
std::array<uint8x16_t, 16> makeMasks() {
  std::array<uint8x16_t, 16> res;
  uint8_t arr[16];
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      arr[j] = (j < i) ? 0xff : 0x0;
    }
    res[i] = vld1q_u8(arr);
  }
  return res;
};
#endif

#ifdef __AVX2__
std::array<__m128i, 16> makeMasks() {
  std::array<__m128i, 16> res;
  uint8_t arr[16];
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      arr[j] = (j < i) ? 0xff : 0x0;
    }
    res[i] = _mm_loadu_si128((__m128i*)arr);
  }
  return res;
};
#endif

inline bool compareEqStrs(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) {return false;}
  else if (a.size() > 16) {
    return a == b;
  }

#ifdef __AVX2__
  static const std::array<__m128i, 16> kMask = makeMasks();

  auto mask = kMask[a.size()];

  __m128i av = _mm_and_si128(_mm_loadu_si128((__m128i*)a.data()), mask);
  __m128i bv = _mm_and_si128(_mm_loadu_si128((__m128i*)b.data()), mask);
  __m128i cmp = _mm_cmpeq_epi8(av, bv);

  return _mm_movemask_epi8(cmp) == 0xFFFF;
#elif defined(__ARM_NEON)
  static const std::array<uint8x16_t, 16> kMask = makeMasks();

  auto mask = kMask[a.size()];

  auto av = vandq_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(a.data())), mask);
  auto bv = vandq_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(b.data())), mask);

  return vmaxvq_u8(vsubq_u8(av, bv)) == 0;
#else
  return a == b;
#endif
}

// TODO: bug right now if index goes past name.size() and we don't have space in our direct_ map
class Trie {
public:
  void add(std::vector<std::pair<std::string_view, MeasurementAggregator>> &agg, std::string_view name, int index, int val) {
    for (size_t i = 0; i < count_; ++i) {
      if (compareEqStrs(name, std::get<0>(direct_[i]))) {
        agg[std::get<1>(direct_[i])].second.add(val);
        return;
      }
    }
    
    if (count_ >= 6) {
      indirect_[index].add(agg, name, index+1, val);
      return;
    }

   
    std::get<0>(direct_[count_]) = name;
    std::get<1>(direct_[count_]) = agg.size();
    agg.emplace_back(name, MeasurementAggregator{});
    ++count_;

    if (count_ == 6) {
      indirect_ = std::make_unique<Trie[]>(256);
    }
  }

private:
  std::unique_ptr<Trie[]> indirect_;
  std::array<std::tuple<std::string_view, size_t>, 6> direct_;
  size_t count_ = {0};
};


    struct suint128 {
      uint64_t a, b;

      void set(std::string_view s) {
        b = 0;
        std::memcpy(this, s.data(), s.size());
      }

      bool operator == (suint128 oth) const {
        return (a == oth.a) & (b == oth.b);
      }

      
    };

bool operator <(suint128 l, suint128 r)  {
        return (l.a < r.a) || ((l.a == r.a) & (l.b < r.b));
      }

// SWAR constants for finding specific bytes
constexpr uint64_t kSemicolonPattern = 0x3B3B3B3B3B3B3B3BULL;  // ';' repeated 8 times
constexpr uint64_t kNewlinePattern = 0x0A0A0A0A0A0A0A0AULL;    // '\n' repeated 8 times
constexpr uint64_t kSwarLow = 0x0101010101010101ULL;           // For zero-byte detection
constexpr uint64_t kSwarHigh = 0x8080808080808080ULL;          // For zero-byte detection

#ifdef __AVX2__
__attribute__((noinline)) std::tuple<std::string_view, int, char*> parseNext(char *p) {
    // Load 32 bytes to capture station + temperature in most cases
    __m256i chunk = _mm256_loadu_si256((__m256i*)p);

    // Find semicolon
    __m256i vsemi = _mm256_set1_epi8(';');
    __m256i cmp_semi = _mm256_cmpeq_epi8(chunk, vsemi);
    int mask_semi = _mm256_movemask_epi8(cmp_semi);

    // Find newline
    __m256i vnewline = _mm256_set1_epi8('\n');
    __m256i cmp_newline = _mm256_cmpeq_epi8(chunk, vnewline);
    int mask_newline = _mm256_movemask_epi8(cmp_newline);

    int semi_pos, newline_pos;

    if (mask_semi && mask_newline) {
        // Fast path: both in first 32 bytes (~99% of cases)
        semi_pos = __builtin_ctz(mask_semi);
        newline_pos = __builtin_ctz(mask_newline);
    } else {
        // Fallback: search for semicolon/newline sequentially
        char* semi = p + 1;
        while (*semi != ';') ++semi;
        semi_pos = semi - p;

        char* lineEnd = semi + 1;
        while (*lineEnd != '\n') ++lineEnd;
        newline_pos = lineEnd - p;
    }

    std::string_view station(p, semi_pos);

    // Parse temperature from the loaded chunk
    char* temp = p + semi_pos + 1;
    int temp_len = newline_pos - semi_pos - 1;

    // Extract temperature bytes from chunk
    uint64_t temp_data;
    memcpy(&temp_data, temp, 8);

    int sign = 1;
    if ((temp_data & 0xFF) == '-') {
        sign = -1;
        temp_data >>= 8;
        temp_len--;
    }

    int value;
    int b0 = (temp_data & 0xFF) - '0';
    int b1 = (temp_data >> 8) & 0xFF;
    int b2 = (temp_data >> 16) & 0xFF;
    int b3 = (temp_data >> 24) & 0xFF;

    // Use temp_len to determine format (branchless where possible)
    if (temp_len == 3) {
        // d.d
        value = b0 * 10 + ((b2 - '0'));
    } else if (temp_len == 4) {
        // dd.d
        value = b0 * 100 + ((b1 - '0') * 10) + ((b3 - '0'));
    } else {
        // ddd.d
        int b4 = (temp_data >> 32) & 0xFF;
        value = b0 * 1000 + ((b1 - '0') * 100) + ((b2 - '0') * 10) + ((b4 - '0'));
    }

    return {station, sign * value, p + newline_pos};
}
#elif defined(__ARM_NEON)
inline std::tuple<std::string_view, int, char*> parseNext(char *p) {
    // Lambda to find character position using SWAR on 16-byte NEON register
    auto findChar = [](uint8x16_t chunk, uint64_t target_pattern) -> int {
        uint64_t chunk_lo = vgetq_lane_u64(vreinterpretq_u64_u8(chunk), 0);
        uint64_t chunk_hi = vgetq_lane_u64(vreinterpretq_u64_u8(chunk), 1);

        uint64_t xor_lo = chunk_lo ^ target_pattern;
        uint64_t has_lo = (xor_lo - kSwarLow) & ~xor_lo & kSwarHigh;

        if (has_lo) {
            return __builtin_ctzll(has_lo) >> 3;
        }

        uint64_t xor_hi = chunk_hi ^ target_pattern;
        uint64_t has_hi = (xor_hi - kSwarLow) & ~xor_hi & kSwarHigh;
        return 8 + (__builtin_ctzll(has_hi) >> 3);
    };

    // Load 16 bytes first (covers 60% of cases)
    uint8x16_t chunk = vld1q_u8((uint8_t*)p);

    // Find semicolon and newline
    uint8x16_t vsemi = vdupq_n_u8(';');
    uint8x16_t vnewline = vdupq_n_u8('\n');
    uint8x16_t cmp_semi = vceqq_u8(chunk, vsemi);
    uint8x16_t cmp_newline = vceqq_u8(chunk, vnewline);

    int semi_pos = -1;
    int newline_pos = -1;

    // Check if both semicolon and newline are in first 16 bytes
    if (vmaxvq_u8(cmp_semi) && vmaxvq_u8(cmp_newline)) {
        // Fast path: both found in 16 bytes
        semi_pos = findChar(chunk, kSemicolonPattern);

        // Find newline, masking out positions before semicolon
        uint64_t chunk_lo = vgetq_lane_u64(vreinterpretq_u64_u8(chunk), 0);
        uint64_t chunk_hi = vgetq_lane_u64(vreinterpretq_u64_u8(chunk), 1);

        uint64_t xor_newline_lo = chunk_lo ^ kNewlinePattern;
        uint64_t has_newline_lo = (xor_newline_lo - kSwarLow) & ~xor_newline_lo & kSwarHigh;

        if (semi_pos < 8) {
            uint64_t mask = (1ULL << (semi_pos * 8 + 8)) - 1;
            has_newline_lo &= ~mask;
        }

        if (has_newline_lo) {
            newline_pos = __builtin_ctzll(has_newline_lo) >> 3;
        } else {
            uint64_t xor_newline_hi = chunk_hi ^ kNewlinePattern;
            uint64_t has_newline_hi = (xor_newline_hi - kSwarLow) & ~xor_newline_hi & kSwarHigh;
            newline_pos = 8 + (__builtin_ctzll(has_newline_hi) >> 3);
        }
    } else if (vmaxvq_u8(cmp_semi)) {
        // Semicolon in first 16, newline beyond
        semi_pos = findChar(chunk, kSemicolonPattern);

        // Load next 16 bytes for newline
        uint8x16_t chunk2 = vld1q_u8((uint8_t*)(p + 16));
        uint8x16_t cmp_newline2 = vceqq_u8(chunk2, vnewline);

        if (vmaxvq_u8(cmp_newline2)) {
            newline_pos = 16 + findChar(chunk2, kNewlinePattern);
        } else {
            // Fallback: sequential search for newline beyond 32 bytes
            char* lineEnd = p + 32;
            while (*lineEnd != '\n') ++lineEnd;
            newline_pos = lineEnd - p;
        }
    } else {
        // Fallback: sequential search for both
        char* semi = p + 1;
        while (*semi != ';') ++semi;
        semi_pos = semi - p;

        char* lineEnd = semi + 1;
        while (*lineEnd != '\n') ++lineEnd;
        newline_pos = lineEnd - p;
    }

    std::string_view station(p, semi_pos);

    // Parse temperature using known positions
    char* temp = p + semi_pos + 1;
    int temp_len = newline_pos - semi_pos - 1;

    // Extract temperature - try to use already-loaded data when possible
    uint64_t temp_data;
    int temp_offset = semi_pos + 1;

    if (vmaxvq_u8(cmp_semi) && vmaxvq_u8(cmp_newline)) {
        // Both delimiters in first 16 bytes - all data is loaded
        uint64_t chunk_lo = vgetq_lane_u64(vreinterpretq_u64_u8(chunk), 0);
        uint64_t chunk_hi = vgetq_lane_u64(vreinterpretq_u64_u8(chunk), 1);

        if (temp_offset < 8) {
            // Temperature starts in low 8 bytes
            temp_data = chunk_lo >> (temp_offset * 8);
            // If we need data from high bytes, combine them
            if (temp_offset > 0) {
                temp_data |= (chunk_hi << ((8 - temp_offset) * 8));
            }
        } else {
            // Temperature starts in high 8 bytes
            temp_data = chunk_hi >> ((temp_offset - 8) * 8);
        }
    } else {
        // Fallback: use memcpy for temperatures beyond first chunk
        memcpy(&temp_data, temp, 8);
    }

    int sign = 1;
    if ((temp_data & 0xFF) == '-') {
        sign = -1;
        temp_data >>= 8;
        temp_len--;
    }

    int value;
    int b0 = (temp_data & 0xFF) - '0';
    int b1 = (temp_data >> 8) & 0xFF;
    int b2 = (temp_data >> 16) & 0xFF;
    int b3 = (temp_data >> 24) & 0xFF;

    // Determine format based on temp_len
    if (temp_len == 3) {
        // d.d
        value = b0 * 10 + ((b2 - '0'));
    } else if (temp_len == 4) {
        // dd.d
        value = b0 * 100 + ((b1 - '0') * 10) + ((b3 - '0'));
    } else {
        // ddd.d (temp_len == 5)
        int b4 = (temp_data >> 32) & 0xFF;
        value = b0 * 1000 + ((b1 - '0') * 100) + ((b2 - '0') * 10) + ((b4 - '0'));
    }

    return {station, sign * value, p + newline_pos};
}
#else
__attribute__((noinline)) std::tuple<std::string_view, int, char*> parseNext(char *p) {
    // Find semicolon - optimized to skip first character (guaranteed min 1 char)
    char* semi = p + 1;

    // SWAR version for ARM and other platforms - search 8 bytes at a time
    while (true) {
        uint64_t chunk;
        memcpy(&chunk, semi, 8);
        uint64_t xor_result = chunk ^ kSemicolonPattern;
        uint64_t has_semi = (xor_result - kSwarLow) & ~xor_result & kSwarHigh;

        if (has_semi) {
            semi += __builtin_ctzll(has_semi) >> 3;
            break;
        }
        semi += 8;
    }

    std::string_view station(p, semi - p);

    // Parse temperature with fast paths for common formats
    char* temp = semi + 1;

    // Load 8 bytes containing temperature
    uint64_t temp_data;
    memcpy(&temp_data, temp, 8);

    int sign = 1;
    if ((temp_data & 0xFF) == '-') {
        sign = -1;
        temp_data >>= 8;
    }

    // Extract bytes
    int b0 = (temp_data & 0xFF);
    int b1 = (temp_data >> 8) & 0xFF;
    int b2 = (temp_data >> 16) & 0xFF;
    int b3 = (temp_data >> 24) & 0xFF;

    int value;
    char* lineEnd;

    // Fast path: d.d format (most common)
    if (b1 == '.') {
        value = ((b0 - '0') * 10 + (b2 - '0'));
        lineEnd = temp + 3;
    }
    // Fast path: dd.d format (second most common)
    else if (b2 == '.') {
        value = ((b0 - '0') * 100 + (b1 - '0') * 10 + (b3 - '0'));
        lineEnd = temp + 4;
    }
    // Rare: ddd.d format
    else {
        int b4 = (temp_data >> 32) & 0xFF;
        value = ((b0 - '0') * 1000 + (b1 - '0') * 100 + (b2 - '0') * 10 + (b4 - '0'));
        lineEnd = temp + 5;
    }

    return {station, sign * value, lineEnd};
}
#endif

struct DumbMap {
  std::deque<std::pair<uint64_t, MeasurementAggregator>> vec[4096];

  MeasurementAggregator &operator[] (uint64_t dat) {
    auto idx = (dat* uint64_t{0x9e3779b97f4a7c15}) & 4095;

    for (auto &ma : vec[idx]) {
      if (ma.first == dat) {
        return ma.second;
      }
    }

    vec[idx].emplace_back(dat, MeasurementAggregator{});
    return vec[idx].back().second;
  }
};

int main() {
    MMap mmap;
    if (!mmap.open(kFile)) {
        std::cerr << "Failed to open file: " << kFile << std::endl;
        return 1;
    }

    struct sLghash {
      size_t operator() (std::string_view v) const {
        uint64_t h = 0;
        memcpy(&h, v.data(), std::min<uint64_t>(8, v.size()));
               
        return h * uint64_t{0x9e3779b97f4a7c15};
      }
    };
    
    std::unordered_map<std::string_view, MeasurementAggregator, sLghash> measurementsLg;
    measurementsLg.reserve(256);
    //std::flat_map<std::string_view, MeasurementAggregator> measurementsLg;

    struct s16hash {
      size_t operator() (suint128 v) const {
        return (v.a ^ v.b) * uint64_t{0x9e3779b97f4a7c15};
      }
    };

    std::unordered_map<suint128, MeasurementAggregator, s16hash> measurementsMd;
    measurementsMd.reserve(4096);
    //std::flat_map<suint128, MeasurementAggregator> measurementsMd;
   
    struct s8hash {
      size_t operator() (uint64_t v) const {
        return v;// * uint64_t{0x9e3779b97f4a7c15};
      }
    };
    
    std::unordered_map<uint64_t, MeasurementAggregator, s8hash> measurementsSm;
    measurementsSm.reserve(4096);

    DumbMap measurementsDumb;

    char* p = mmap.data;
    char* end = mmap.data + mmap.size / 2;

    // Process all complete lines except the last few to ensure parseNext has buffer space
    // We'll process the tail separately
    char* safeEnd = end - 32; // Conservative: ensure enough space for parseNext reads
    if (safeEnd < p) safeEnd = p;

    int cnt = 0;


    Trie measurementsTable[256];

    std::vector<std::pair<std::string_view, MeasurementAggregator>> measurements;

    while (p < safeEnd) {
        if ((cnt & 0xffffff) == 0) {
            std::cout << "cnt=" << cnt << "\n";
        }
        ++cnt;

        auto [station, value, lineEnd] = parseNext(p);

        if (station.size() < 8) {
            uint64_t st8 = 0;
            std::memcpy(&st8, station.data(), station.size());
            measurementsSm[st8].add(value);
        }
        else if (station.size() < 16) {
            suint128 st16;
            st16.set(station);
            measurementsMd[st16].add(value);
        }
        else {
            measurementsLg[station].add(value);
        }

        p = lineEnd + 1;
    }

    // Process remaining tail with boundary checks
    while (p < end) {
        char* semi = p;
        while (semi < end && *semi != ';') ++semi;
        if (semi >= end) break;

        std::string_view station(p, semi - p);

        // Parse temperature inline
        char* temp = semi + 1;
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
        temp++; // skip '.'
        if (temp >= end) break;
        value = value * 10 + (*temp++ - '0');
        value *= sign;

        char* lineEnd = temp;
        while (lineEnd < end && *lineEnd != '\n') ++lineEnd;

        if (station.size() < 8) {
            uint64_t st8 = 0;
            std::memcpy(&st8, station.data(), station.size());
            measurementsSm[st8].add(value);
        } else if (station.size() < 16) {
            suint128 st16;
            st16.set(station);
            measurementsMd[st16].add(value);
        } else {
            measurementsLg[station].add(value);
        }

        p = lineEnd + 1;
    }

    std::cout << "small: " << measurementsSm.size() << std::endl;
    std::cout << "medium: " << measurementsMd.size() << std::endl;
    std::cout << "large: " << measurementsLg.size() << std::endl;


    // for (auto &outer : measurementsDumb.vec) {
    //   for (auto &inner : outer) {
    //     measurementsSm.insert(inner);
    //   }
    // }
    
    // for (int i = 0; i < 256; ++i) {
    //   for (int j = 0; j < 256; ++j) {

    //     if (!measurementsTable[i][j].empty()) {
    //       std::print("======================================\n");
    //       for (const auto& [station, index, count] : measurementsTable[i][j]) {
    //         std::print("{} : {}, {}\n", station, index, count);

    //     }
    //   }
    // }
    // }

    std::cout << "{";
    bool first = true;
    for (const auto& [station, agg] : measurementsLg) {
        if (!first) std::cout << ", ";
        first = false;
        double mean = (agg.sum / 10.0) / agg.count;
        ResultRow result(agg.min / 10.0, mean, agg.max / 10.0);
        std::cout << station << "=" << result.toString();
    }
    for (const auto& [st8, agg] : measurementsSm) {
      std::string_view station(reinterpret_cast<const char *>(&st8));
        if (!first) std::cout << ", ";
        first = false;
        double mean = (agg.sum / 10.0) / agg.count;
        ResultRow result(agg.min / 10.0, mean, agg.max / 10.0);
        std::cout << station << "=" << result.toString();
    }

      for (const auto& [station, agg] : measurements) {
        if (!first) std::cout << ", ";
        first = false;
        double mean = (agg.sum / 10.0) / agg.count;
        ResultRow result(agg.min / 10.0, mean, agg.max / 10.0);
        std::cout << station << "=" << result.toString();
    }
    
    // for (const auto& [st16, agg] : measurementsMd) {
    //   std::string_view station(reinterpret_cast<const char *>(&st16));
    //     if (!first) std::cout << ", ";
    //     first = false;
    //     double mean = ResultRow::round(agg.sum * 10.0) / 10.0 / agg.count;
    //     ResultRow result(agg.min, mean, agg.max);
    //     std::cout << station << "=" << result.toString();
    // }
    
    std::cout << "}" << std::endl;

    return 0;
}
