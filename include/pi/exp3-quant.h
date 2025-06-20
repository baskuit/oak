

#pragma once

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace Exp3 {

// represents float scaled by 1 << 14;
using u8 = uint8_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u64 = uint64_t;

using Weight = u32;
constexpr int bits = 32;
using Work = u64;

constexpr int weight_shift = 16;
constexpr Weight ONE = Weight{1} << weight_shift;
constexpr int halve_shift = 8;

constexpr float to_float(auto a) { return (float)a / ONE; }

void print_q(const auto &v) {
  for (const auto x : v) {
    std::cout << (float)x / ONE << ' ';
  }
  std::cout << std::endl;
}

void print(const auto &v) {
  for (const auto x : v) {
    std::cout << x << ' ';
  }
  std::cout << std::endl;
}

void print_q(const auto &v, auto sum) {
  for (const auto x : v) {
    std::cout << ((float)x) / (float)sum << ' ';
  }
  std::cout << std::endl;
}

Work exp32(Work x) {
  Work x2 = (x * x) >> weight_shift;
  Work x3 = (x2 * x) >> weight_shift;
  Work x4 = (x3 * x) >> weight_shift;
  Work x5 = (x4 * x) >> weight_shift;
  Work x6 = (x5 * x) >> weight_shift;
  return ONE + x + (x2 / 2) + (x3 / 6) + (x4 / 24) + (x5 / 120) + (x6 / 720);
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

template <bool enabled> struct JointBanditDataBase;

template <> struct JointBanditDataBase<true> {
  std::array<Weight, 9> p1_weights;
  std::array<Weight, 9> p2_weights;
  std::array<uint24_t, 9> p1_visits;
  std::array<uint24_t, 9> p2_visits;
  uint8_t _rows;
  uint8_t _cols;
};

template <> struct JointBanditDataBase<false> {
  std::array<Weight, 9> p1_weights;
  std::array<Weight, 9> p2_weights;
  uint8_t _rows;
  uint8_t _cols;
};

#pragma pack(push, 1)
template <bool enable_visits = false>
class JointBanditData : public JointBanditDataBase<enable_visits> {
public:
  using JointBanditDataBase<enable_visits>::p1_weights;
  using JointBanditDataBase<enable_visits>::p2_weights;
  using JointBanditDataBase<enable_visits>::_rows;
  using JointBanditDataBase<enable_visits>::_cols;

  struct Outcome {
    float value;
    uint8_t p1_index;
    uint8_t p2_index;
    float p1_eta_over_mu;
    float p2_eta_over_mu;
    float gamma{.001};
    float eta{.01};
  };

  void init(auto rows, auto cols) noexcept {
    _rows = rows;
    _cols = cols;
    std::fill(p1_weights.begin(), p1_weights.begin() + rows, ONE);
    std::fill(p2_weights.begin(), p2_weights.begin() + cols, ONE);
    if constexpr (enable_visits) {
      std::fill(this->p1_visits.begin(), this->p1_visits.begin() + rows,
                uint24_t{});
      std::fill(this->p2_visits.begin(), this->p2_visits.begin() + cols,
                uint24_t{});
    }
  }

  bool is_init() const noexcept { return this->_rows != 0; }

  template <typename Outcome> void update(const Outcome &outcome) noexcept {
    if constexpr (enable_visits) {
      ++this->p1_visits[outcome.p1_index];
      ++this->p2_visits[outcome.p2_index];
    }
    const auto value = 2 * outcome.value - 1;
    const float p1_exp = exp(value * outcome.p1_eta_over_mu);
    const float p2_exp = exp((0 - value) * outcome.p2_eta_over_mu);
    p1_weights[outcome.p1_index] *= p1_exp;
    p2_weights[outcome.p2_index] *= p2_exp;
    // std::cout << p1_exp << ' ' << p2_exp << std::endl;
  }

  template <typename PRNG, typename Outcome>
  void select(PRNG &device, Outcome &outcome) noexcept {
    std::array<float, 9> forecast{};
    if (_rows == 1) {
      outcome.p1_index = 0;
      outcome.p1_eta_over_mu = outcome.eta;
    } else {
      bool should_halve = false;
      bool should_double = false;
      Work sum = 0;
      Work w_sum = 0;
      std::array<Work, 9> probs{};

      for (auto i = 0; i < _rows; ++i) {
        if (p1_weights[i] >= (Work{1} << (bits - halve_shift))) {
          should_halve = true;
        }
        // Work update = (priors[i] * gamma_over_one_minus_gamma / 256);
        probs[i] = p1_weights[i] + outcome.gamma * ONE;
        sum += probs[i];
        w_sum += p1_weights[i];
      }

      if (sum < ONE) {
        should_double = true;
      }

      const Work r = device.random_int(sum);
      outcome.p1_index = 0xFF;
      Work acc = 0;
      for (auto i = 0; i < _rows; ++i) {
        acc += probs[i];
        outcome.p1_index += 1;
        if (r < acc) {
          break;
        }
      }

      assert((outcome.p1_index >= 0) && outcome.p1_index < _rows);

      double mu_float = (double)probs[outcome.p1_index] / sum;
      outcome.p1_eta_over_mu = (outcome.eta / (mu_float + outcome.gamma));

      if (should_halve) {
        // std::cout << "halve p1\n";
        // print_q(p1_weights, sum);
        for (auto &w : p1_weights) {
          w += 1;
          w /= 2;
        }
        // print_q(p1_weights, sum);
        outcome.p1_eta_over_mu /= 2;
      }
      if (should_double) {
        for (auto &w : p1_weights) {
          w *= 2;
        }
      }
    }
    if (_cols == 1) {
      outcome.p2_index = 0;
      outcome.p2_eta_over_mu = outcome.eta;
    } else {

      bool should_halve = false;
      bool should_double = false;
      Work sum = 0;
      Work w_sum = 0;
      std::array<Work, 9> probs{};

      for (auto i = 0; i < _cols; ++i) {
        if (p2_weights[i] >= (Work{1} << (bits - halve_shift))) {
          should_halve = true;
        }
        Work update = 0;
        probs[i] = p2_weights[i] + outcome.gamma * ONE;
        sum += probs[i];
        w_sum += p2_weights[i];
      }

      if (sum < ONE) {
        should_double = true;
      }

      const Work r = device.random_int(sum);
      outcome.p2_index = 0xFF;
      Work acc = 0;
      for (auto i = 0; i < _cols; ++i) {
        acc += probs[i];
        outcome.p2_index += 1;
        if (r < acc) {
          break;
        }
      }

      double mu_float = (double)probs[outcome.p2_index] / sum;
      outcome.p2_eta_over_mu = (outcome.eta / (mu_float + outcome.gamma));

      if (should_halve) {
        // std::cout << "halve p2\n";
        for (auto &w : p2_weights) {
          w += 1;
          w /= 2;
        }
        outcome.p2_eta_over_mu /= 2;
      }
      if (should_double) {
        for (auto &w : p2_weights) {
          w *= 2;
        }
      }
    }
    assert(outcome.p1_index < _rows);
    assert(outcome.p2_index < _cols);
  }

  std::string visit_string() const {
    std::stringstream sstream{};
    if constexpr (enable_visits) {
      sstream << "V1: ";
      for (auto i = 0; i < _rows; ++i) {
        sstream << std::to_string(this->p1_visits[i]) << " ";
      }
      sstream << "V2: ";
      for (auto i = 0; i < _cols; ++i) {
        sstream << std::to_string(this->p2_visits[i]) << " ";
      }
      sstream.flush();
    }
    return sstream.str();
  }

  std::pair<std::vector<float>, std::vector<float>>
  policies(float iterations) const {

    std::vector<float> p1{};
    std::vector<float> p2{};

    p1.resize(_rows);
    p2.resize(_cols);

    if constexpr (enable_visits) {
      for (auto i = 0; i < _rows; ++i) {
        p1[i] = this->p1_visits[i].value() / (iterations - 1);
      }
      for (auto i = 0; i < _cols; ++i) {
        p2[i] = this->p2_visits[i].value() / (iterations - 1);
      }
    }
    return {p1, p2};
  }
};
#pragma pack(pop)

// struct JointBanditDataTest {
//   // hopefully that shit about double cache lines is true
//   static_assert(sizeof(JointBanditData<.1f, true>) == 128);
//   static_assert(sizeof(JointBanditData<.1f, false>) == 76);
// };

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
