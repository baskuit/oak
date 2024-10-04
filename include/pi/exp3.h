#pragma once

#include <cstdint>
#include <array>

namespace Exp3 {


// struct Exp3EntryTest {
//   static_assert(sizeof(Exp3Entry) == 8);
// };

#pragma pack(push, 1)
struct uint24_t {
    uint8_t bytes[3]; // Array of 3 bytes

    uint24_t() = default;
    
    // Optional: Constructor to initialize from a uint32_t
    uint24_t(uint32_t value) {
        bytes[0] = value & 0xFF;        // Least significant byte
        bytes[1] = (value >> 8) & 0xFF; // Middle byte
        bytes[2] = (value >> 16) & 0xFF; // Most significant byte
    }

    // Optional: Conversion to uint32_t
    operator uint32_t() const {
        return (static_cast<uint32_t>(bytes[0]) |
                (static_cast<uint32_t>(bytes[1]) << 8) |
                (static_cast<uint32_t>(bytes[2]) << 16));
    }

    uint24_t &operator++() noexcept {
      return *this;
    }
};
#pragma pack(pop)

static_assert(sizeof(uint24_t) == 3);

// prolly faster to have the float data properly aligned
// #pragma pack(push, 1)
// struct Exp3Entry {
//   float gain;
//   uint24_t visits;
// };
// #pragma pop()

// struct Exp3EntryTest {
//   static_assert(sizeof(Exp3Entry == 7));
// };

#pragma pack(push, 1)
class JointBanditData {
private:
  std::array<float, 9> row_gains;
  std::array<float, 9> col_gains;
  std::array<uint24_t, 9> row_visits;
  std::array<uint24_t, 9> col_visits;
  uint8_t _rows;
  uint8_t _cols;

public:

  void init(auto rows, auto cols) {
    _rows = rows;
    _cols = cols;
    // row_gains.resize(rows);
    // row_visits.resize(rows);
    // col_gains.resize(cols);
    // col_visits.resize(cols);
  }

  bool is_init() const noexcept {
    return _rows != 0;
  }

  template <typename Outcome> 
  void update(const Outcome &outcome) noexcept {
    ++row_visits[outcome.row_idx];
    ++col_visits[outcome.col_idx];
  }

  template <typename Outcome>
  void select(Outcome &outcome) const noexcept {
    outcome.row_idx = 0;
    outcome.col_idx = 0;
  }
};
#pragma pack(pop)

struct JointBanditDataTest {
  // hopefully that shit about double cache lines is true
  static_assert(sizeof(JointBanditData) == 128);
};

};