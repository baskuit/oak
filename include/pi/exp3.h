#pragma once

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <vector>

namespace Exp3 {

#pragma pack(push, 1)
struct uint24_t {
  std::array<uint8_t, 3> _data;

  uint24_t() = default;

  uint24_t(uint32_t value) {
    _data[0] = value & 0xFF;
    _data[1] = (value >> 8) & 0xFF;
    _data[2] = (value >> 16) & 0xFF;
  }

  operator uint32_t() const noexcept {
    return (static_cast<uint32_t>(_data[0]) |
            (static_cast<uint32_t>(_data[1]) << 8) |
            (static_cast<uint32_t>(_data[2]) << 16));
  }

  uint24_t &operator++() noexcept {
    uint32_t value = static_cast<uint32_t>(*this) + 1;
    *this = uint24_t(value); // Wrap around if needed
    return *this;
  }
};
#pragma pack(pop)

struct uint24_t_test {
  static_assert(sizeof(uint24_t) == 3);

  uint24_t_test() {
    uint24_t a{};
    assert(static_cast<uint32_t>(a) == 0);
    ++a;
    assert(static_cast<uint32_t>(a) == 1);

    uint24_t b{};
    for (int i = 0; i < 1 << 24; ++i) {
      ++b;
    }
    assert(static_cast<uint32_t>(b) == 0);
  }
};

// template <bool keep_selections>
// struct BaseData;

// template <>
// struct BaseData<false>
// {
//   std::array<float, 9> row_gains;
//   std::array<float, 9> col_gains;
//   uint8_t _rows;
//   uint8_t _cols;
// };

// template <>
// struct BaseData<true>
// {
//   std::array<float, 9> row_gains;
//   std::array<float, 9> col_gains;
//   std::array<uint24_t, 9> row_visits;
//   std::array<uint24_t, 9> col_visits;
//   uint8_t _rows;
//   uint8_t _cols;
// };

#pragma pack(push, 1)
class JointBanditData {
  // private:
public:
  std::array<float, 9> row_gains;
  std::array<float, 9> col_gains;
  std::array<uint24_t, 9> row_visits;
  std::array<uint24_t, 9> col_visits;
  uint8_t _rows;
  uint8_t _cols;

public:
  struct Outcome {
    float value;
    float row_mu;
    float col_mu;
    uint8_t row_idx;
    uint8_t col_idx;
  };

  void init(auto rows, auto cols) {
    _rows = rows;
    _cols = cols;
    std::fill(row_gains.begin(), row_gains.begin() + rows, 0);
    std::fill(row_visits.begin(), row_visits.begin() + rows, uint24_t{});
    std::fill(col_gains.begin(), col_gains.begin() + cols, 0);
    std::fill(col_visits.begin(), col_visits.begin() + cols, uint24_t{});
  }

  bool is_init() const noexcept { return _rows != 0; }

  template <typename Outcome> void update(const Outcome &outcome) noexcept {
    ++row_visits[outcome.row_idx];
    ++col_visits[outcome.col_idx];
    if ((row_gains[outcome.row_idx] += outcome.value / outcome.row_mu) >= 0) {
      const auto max = row_gains[outcome.row_idx];
      for (auto &v : row_gains) {
        v -= max;
      }
    }
    if ((col_gains[outcome.col_idx] += (1 - outcome.value) / outcome.col_mu) >=
        0) {
      const auto max = col_gains[outcome.col_idx];
      for (auto &v : col_gains) {
        v -= max;
      }
    }
  }

  template <typename PRNG, typename Outcome>
  void select(PRNG &device, Outcome &outcome) const noexcept {
    static const float gamma = .03f;
    static const float one_minus_gamma = .97f;
    std::array<float, 9> row_forecast{};
    std::array<float, 9> col_forecast{};

    if (_rows == 1) {
      outcome.row_idx = 0;
      outcome.row_mu = 1;
    } else {
      const float eta{gamma / _rows};
      softmax(row_forecast, row_gains, _rows, eta);
      std::transform(row_forecast.begin(), row_forecast.begin() + _rows,
                     row_forecast.begin(), [eta](const float value) {
                       return one_minus_gamma * value + eta;
                     });
      outcome.row_idx = device.sample_pdf(row_forecast);
      outcome.row_mu = row_forecast[outcome.row_idx];
    }
    if (_cols == 1) {
      outcome.col_idx = 0;
      outcome.col_mu = 1;
    } else {
      const float eta{gamma / _cols};
      softmax(col_forecast, col_gains, _cols, eta);
      std::transform(col_forecast.begin(), col_forecast.begin() + _cols,
                     col_forecast.begin(), [eta](const float value) {
                       return one_minus_gamma * value + eta;
                     });
      outcome.col_idx = device.sample_pdf(col_forecast);
      outcome.col_mu = col_forecast[outcome.col_idx];
    }
    assert(outcome.row_idx < _rows);
    assert(outcome.col_idx < _cols);
  }

private:
  void softmax(auto &forecast, const auto &gains, const auto k,
               float eta) const {
    float sum = 0;
    for (auto i = 0; i < k; ++i) {
      const float y = std::exp(gains[i] * eta);
      forecast[i] = y;
      sum += y;
    }
    for (auto i = 0; i < k; ++i) {
      forecast[i] /= sum;
    }
  };
};
#pragma pack(pop)

struct JointBanditDataTest {
  // hopefully that shit about double cache lines is true
  static_assert(sizeof(JointBanditData) == 128);
};

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

}; // namespace Exp3