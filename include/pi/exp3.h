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
  // static_assert(static_cast<uint32_t>(overflow()) == 0);
};

#pragma pack(push, 1)
class JointBanditData {
  // private:
public:
  std::array<float, 9> p1_gains;
  std::array<float, 9> p2_gains;
  std::array<uint24_t, 9> p1_visits;
  std::array<uint24_t, 9> p2_visits;
  uint8_t _rows;
  uint8_t _cols;

public:
  struct Outcome {
    float value;
    float p1_mu;
    float p2_mu;
    uint8_t p1_index;
    uint8_t p2_index;
  };

  void init(auto rows, auto cols) {
    _rows = rows;
    _cols = cols;
    std::fill(p1_gains.begin(), p1_gains.begin() + rows, 0);
    std::fill(p1_visits.begin(), p1_visits.begin() + rows, uint24_t{});
    std::fill(p2_gains.begin(), p2_gains.begin() + cols, 0);
    std::fill(p2_visits.begin(), p2_visits.begin() + cols, uint24_t{});
  }

  bool is_init() const noexcept { return _rows != 0; }

  template <typename Outcome> void update(const Outcome &outcome) noexcept {
    ++p1_visits[outcome.p1_index];
    ++p2_visits[outcome.p2_index];
    if ((p1_gains[outcome.p1_index] += outcome.value / outcome.p1_mu) >= 0) {
      const auto max = p1_gains[outcome.p1_index];
      for (auto &v : p1_gains) {
        v -= max;
      }
    }
    if ((p2_gains[outcome.p2_index] += (1 - outcome.value) / outcome.p2_mu) >=
        0) {
      const auto max = p2_gains[outcome.p2_index];
      for (auto &v : p2_gains) {
        v -= max;
      }
    }
  }

  template <typename PRNG, typename Outcome>
  void select(PRNG &device, Outcome &outcome) const noexcept {
    static const float gamma = .03f;
    static const float one_minus_gamma = .97f;
    std::array<float, 9> forecast{};
    if (_rows == 1) {
      outcome.p1_index = 0;
      outcome.p1_mu = 1;
    } else {
      const float eta{gamma / _rows};
      softmax(forecast, p1_gains, _rows, eta);
      std::transform(
          forecast.begin(), forecast.begin() + _rows, forecast.begin(),
          [eta](const float value) { return one_minus_gamma * value + eta; });
      outcome.p1_index = device.sample_pdf(forecast);
      outcome.p1_mu = forecast[outcome.p1_index];
    }
    if (_cols == 1) {
      outcome.p2_index = 0;
      outcome.p2_mu = 1;
    } else {
      const float eta{gamma / _cols};
      softmax(forecast, p2_gains, _cols, eta);
      std::transform(
          forecast.begin(), forecast.begin() + _cols, forecast.begin(),
          [eta](const float value) { return one_minus_gamma * value + eta; });
      outcome.p2_index = device.sample_pdf(forecast);
      outcome.p2_mu = forecast[outcome.p2_index];
    }
    assert(outcome.p1_index < _rows);
    assert(outcome.p2_index < _cols);
  }

  std::string visit_string() const {
    std::stringstream sstream{};
    sstream << "V1: ";
    for (auto i = 0; i < _rows; ++i) {
      sstream << std::to_string(p1_visits[i]) << " ";
    }
    sstream << "V2: ";
    for (auto i = 0; i < _cols; ++i) {
      sstream << std::to_string(p2_visits[i]) << " ";
    }
    sstream.flush();
    return sstream.str();
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
