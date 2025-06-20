

#pragma once

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

namespace UCB {

template <typename T, size_t shift> T scaled_int_div(T x, T y) {
  return (x << shift) / y;
}

#pragma pack(push, 1)
struct uint24_t {
  std::array<uint8_t, 3> _data;

  constexpr uint24_t() = default;

  constexpr uint24_t(uint32_t value) noexcept {
    _data[0] = value & 0xFF;
    _data[1] = (value >> 8) & 0xFF;
    _data[2] = (value >> 16) & 0xFF;
  }

  constexpr operator uint32_t() const noexcept {
    return (static_cast<uint32_t>(_data[0]) |
            (static_cast<uint32_t>(_data[1]) << 8) |
            (static_cast<uint32_t>(_data[2]) << 16));
  }

  constexpr auto value() const noexcept { return static_cast<uint32_t>(*this); }

  constexpr uint24_t &operator++() noexcept {
    uint32_t value = static_cast<uint32_t>(*this) + 1;
    *this = uint24_t(value);
    return *this;
  }
};
#pragma pack(pop)

struct uint24_t_test {
  static_assert(sizeof(uint24_t) == 3);

  static consteval uint24_t overflow() {
    uint24_t x{};
    for (size_t i = 0; i < (1 << 24); ++i) {
      ++x;
    }
    return x;
  }
};

#pragma pack(push, 1)
class JointBanditData {
public:
  std::array<uint32_t, 9> p1_score;
  std::array<uint32_t, 9> p2_score;
  std::array<uint32_t, 9> p1_visits;
  std::array<uint32_t, 9> p2_visits;
  uint8_t _rows;
  uint8_t _cols;

  struct Outcome {
    float value;
    uint8_t p1_index;
    uint8_t p2_index;
  };

  void init(auto rows, auto cols) noexcept {
    _rows = rows;
    _cols = cols;
    std::fill(p1_score.begin(), p1_score.begin() + rows, 0);
    std::fill(p2_score.begin(), p2_score.begin() + cols, 0);
    std::fill(this->p1_visits.begin(), this->p1_visits.begin() + rows,
              1);
    std::fill(this->p2_visits.begin(), this->p2_visits.begin() + cols,
              1);
  }

  bool is_init() const noexcept { return this->_rows != 0; }

  template <typename Outcome> void update(const Outcome &outcome) noexcept {
    ++this->p1_visits[outcome.p1_index];
    ++this->p2_visits[outcome.p2_index];
    p1_score[outcome.p1_index] += 2 * outcome.value;
    p2_score[outcome.p2_index] += 2 - 2 * outcome.value;
  }

  template <typename PRNG, typename Outcome>
  void select(PRNG &device, Outcome &outcome) const noexcept {
    if (_rows < 2) {
      outcome.p1_index = 0;
    } else {
      std::array<uint64_t, 9> q{};
      uint64_t N{};
      for (auto i = 0; i < _rows; ++i) {
        q[i] = scaled_int_div<uint32_t, 9>(p1_score[i], p1_visits[i]);
        N += p1_visits[i];
      }
      double sqrt_log_N = sqrt(log(N));
      double max = 0;
      for (auto i = 0; i < _rows; ++i) {
        double e = sqrt_log_N / p1_visits[i];
        double a = e + q[i] / 1024.0;
        // std::cout << a << ' ';
        if (a > max) {
            max = a;
            outcome.p1_index = i;
        }
      }
      // std::cout << std::endl;
    }

    if (_cols < 2) {
      outcome.p2_index = 0;
    } else {
      std::array<uint64_t, 9> q{};
      uint64_t N{};
      for (auto i = 0; i < _cols; ++i) {
        q[i] = scaled_int_div<uint32_t, 9>(p2_score[i], p2_visits[i]);
        N += p2_visits[i];
      }
      double sqrt_log_N = sqrt(log(N));
      double max = 0;
      for (auto i = 0; i < _cols; ++i) {
        double e = sqrt_log_N / p2_visits[i];
        double a = e + q[i] / 1024.0;
        if (a > max) {
            max = a;
            outcome.p2_index = i;
        }
      }
    }
  }

  std::string visit_string() const {
    std::stringstream sstream{};
    sstream << "V1: ";
    for (auto i = 0; i < _rows; ++i) {
    sstream << std::to_string(this->p1_visits[i]) << " ";
    }
    sstream << "V2: ";
    for (auto i = 0; i < _cols; ++i) {
    sstream << std::to_string(this->p2_visits[i]) << " ";
    }
    sstream.flush();
    return sstream.str();
  }

  std::pair<std::vector<float>, std::vector<float>>
  policies(float iterations) const {

    std::vector<float> p1{};
    std::vector<float> p2{};

    p1.resize(_rows);
    p2.resize(_cols);

    for (auto i = 0; i < _rows; ++i) {
    p1[i] = this->p1_visits[i] / (iterations - 1);
    }
    for (auto i = 0; i < _cols; ++i) {
    p2[i] = this->p2_visits[i] / (iterations - 1);
    }
    
    return {p1, p2};
  }
};
#pragma pack(pop)

template <typename Container>
std::vector<float> empirical_strategies(const Container &container) {
  std::vector<float> result{};
  result.resize(container.size());
  float sum = 0;
  for (auto i = 0; i < result.size(); ++i) {
    result[i] = static_cast<float>(container[i]);
    sum += result[i];
  }
  assert(sum != 0);
  for (auto i = 0; i < result.size(); ++i) {
    result[i] /= sum;
  }
  return result;
}

}; // namespace UCB
