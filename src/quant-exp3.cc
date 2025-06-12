#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <util/random.h>

// represents float scaled by 1 << 14;
using u8 = uint8_t;
using u32 = uint32_t;
using u16 = uint16_t;
using i32 = int32_t;
using i16 = int16_t;

constexpr int weight_shift = 4;
constexpr u16 ONE = 1 << weight_shift;

void print_q(const auto& v) {
  for (const auto x : v) {
    std::cout << (float)x / ONE << ' ';
  }
  std::cout << std::endl;
}

int32_t approx_exp_q16(int32_t x) {

  // for (auto d = 1; d <= degree; ++d) {

  // }
  int32_t x2 = (x * x) >> weight_shift;
  int32_t x3 = (x2 * x) >> weight_shift;
  int32_t x4 = (x3 * x) >> weight_shift;
  return ONE + x + (x2 / 2) + (x3 / 6) + (x4 / 24);
}

template <typename T> auto random_matrix(prng &device, auto m, auto n) {
  std::array<std::array<T, 9>, 9> matrix{};
  for (auto i = 0; i < m; ++i) {
    for (auto j = 0; j < n; ++j) {
      matrix[i][j] = static_cast<T>(device.uniform());
    }
  }
  return matrix;
}

auto expl(const auto &matrix, const auto &p1, const auto &p2) {
  const auto m = p1.size();
  const auto n = p2.size();
  std::vector<float> q1{}, q2{};
  q1.resize(m);
  q2.resize(n);
  for (auto i = 0; i < m; ++i) {
    const auto x = p1[i];
    for (auto j = 0; j < n; ++j) {
      const auto y = p2[j];
      q1[i] += y * matrix[i][j];
      q2[j] += x * (matrix[i][j]);
    }
  }
  std::pair<float, float> result;
  result.first = *std::max_element(q1.begin(), q1.end());
  result.second = *std::min_element(q2.begin(), q2.end());
  return result;
}

struct QuantBandit {
  std::vector<u16> weights; // exp(gain)
  std::vector<u8> priors;
  std::vector<uint32_t> probs; // Probabilities
  std::vector<uint32_t> chosen;
  std::mt19937 rng;
  u32 gamma, eta, k, gamma_over_one_minus_gamma;
  u32 eta_over_mu;

  QuantBandit(auto num_arms, float g) {
    k = static_cast<u32>(num_arms);
    probs.resize(k, ONE / k);
    weights.resize(k, ONE);
    chosen.resize(k, 0);
    priors.resize(k, 256 / k);
    std::mt19937 rng{std::random_device{}()};
    gamma = (uint32_t)(g * ONE);
    eta = gamma / k;
    std::cout << "gamma: " << gamma << " eta: " << eta << std::endl;
    gamma_over_one_minus_gamma = (u32)(g / (1 - g) * ONE);
  }

  int select() {

    bool should_halve = false;
    u32 sum = 0;
    std::cout << "weights: " << std::endl;
    print_q(weights);
    for (auto i = 0; i < k; ++i) {
      if (weights[i] >= 1 << 15) {
        should_halve = true;
      }
      probs[i] = weights[i] + priors[i] * gamma_over_one_minus_gamma / 256;
      sum += probs[i];
    }

    std::cout << "probs: " << std::endl;
    print_q(probs);

    std::uniform_int_distribution<uint32_t> dist(0, sum);
    uint32_t r = dist(rng);

    int c = 0;
    u32 acc = 0;
    for (int i = 0; i < weights.size(); ++i) {
      acc += probs[i];
      if (r < acc) {
        break;
      }
      ++c;
    }
    ++chosen[c];

    std::cout << " sum: " << sum;
    std::cout << " probs[c]: " << probs[c];
    std::cout << " eta: " << eta;
    std::cout << std::endl;


    eta_over_mu = sum * ONE / probs[c] * eta / ONE;

    if (should_halve) {
      for (auto &w : weights) {
        w /= 2;
      }
      eta_over_mu /= 2;
    }

    return c;
  }

  void update(auto c, float r) { weights[c] += r * eta_over_mu; }
};

auto try_poo(QuantBandit &b1, QuantBandit &b2, const auto &matrix, auto iter) {
  for (auto n = 0; n < iter; ++n) {
    auto i = b1.select();
    auto j = b2.select();
    const float r = matrix[i][j];
    const float s = 1 - r;
    b1.update(i, r);
    b2.update(j, s);
  }

  const auto m = b1.weights.size();
  const auto n = b2.weights.size();
  std::vector<float> p1;
  p1.resize(m);
  for (auto i = 0; i < m; ++i) {
    p1[i] = static_cast<float>(b1.chosen[i]) / iter;
  }
  std::vector<float> p2;
  p2.resize(n);
  for (auto i = 0; i < n; ++i) {
    p2[i] = static_cast<float>(b2.chosen[i]) / iter;
  }

  const auto e = expl(matrix, p1, p2);
  const auto x = e.first - e.second;
  return x;
}

int main(int argc, char **argv) {

  if (argc < 6) {
    std::cerr << "m n iter trials gamma; get average expl. for quantized exp3."
              << std::endl;
    return 1;
  }


  prng device{5564564563};
  const auto m = std::atoi(argv[1]);
  const auto n = std::atoi(argv[2]);
  const auto iter = std::atoi(argv[3]);
  const auto trials = std::atoi(argv[4]);
  const float gamma = std::atof(argv[5]);

  std::cout << "ONE: " << ONE << std::endl;

  std::cout << " m: " << m;
  std::cout << " n: " << n;
  std::cout << " iter: " << iter;
  std::cout << " trails: " << trials;
  std::cout << " gamma: " << gamma << std::endl;

  double total_x = 0;
  double total_x_uniform = 0;
  auto start = std::chrono::high_resolution_clock::now();

  for (auto t = 0; t < trials; ++t) {
    auto matrix = random_matrix<double>(device, m, n);

    QuantBandit b1{m, gamma};
    QuantBandit b2{n, gamma};
    const auto x = try_poo(b1, b2, matrix, iter);
    total_x += x;

    std::vector<double> p1_uniform{};
    p1_uniform.resize(m, 1.0 / m);
    std::vector<double> p2_uniform{};
    p2_uniform.resize(n, 1.0 / n);

    const auto x_uniform = expl(matrix, p1_uniform, p2_uniform);
    total_x_uniform += (x_uniform.first - x_uniform.second);
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto d =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << d.count() << std::endl;

  std::cout << "expl: " << total_x / trials << std::endl;
  std::cout << "expl uniform: " << total_x_uniform / trials << std::endl;

  return 0;
}