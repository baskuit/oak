#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <util/random.h>

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

constexpr float to_float(auto a) {
  return (float) a / ONE;
}

void print_q(const auto& v) {
  for (const auto x : v) {
    std::cout << (float)x / ONE << ' ';
  }
  std::cout << std::endl;
}

void print(const auto& v) {
  for (const auto x : v) {
    std::cout << x << ' ';
  }
  std::cout << std::endl;
}

void print_q(const auto& v, auto sum) {
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
  std::vector<Weight> weights; // exp(gain)
  std::vector<u8> priors;
  std::vector<Work> probs; // Probabilities
  std::vector<Work> chosen;
  std::mt19937 rng;
  Work gamma, eta, k, gamma_over_one_minus_gamma;
  Work eta_over_mu;
  float eta_over_mu_f;
  float g;
  Work sum = 0;


  QuantBandit(auto num_arms, float g) {
    k = static_cast<Work>(num_arms);
    probs.resize(k, 0);
    weights.resize(k, ONE);
    chosen.resize(k, 0);
    priors.resize(k, 256 / k);
    std::mt19937 rng{std::random_device{}()};
    gamma = (Work)(g * ONE);
    eta = gamma / k;
    gamma_over_one_minus_gamma = (Work)(g / (1 - g) * ONE);
    // std::cout << "gamma: " << to_float(gamma) << " eta: " << to_float(eta) << " z: " << to_float(gamma_over_one_minus_gamma) << std::endl;
    // std::cout << "gamma: " << g << " eta: " << g/k << " z: " << g/(1-g) << std::endl;
    this->g = g;
  }

  int select() {

    bool should_halve = false;
    sum = 0;
    Work w_sum = 0;
    // std::cout << "weights: " << std::endl;
    // print_q(weights);
    for (auto i = 0; i < k; ++i) {
      if (weights[i] >= (Work{1} << (bits - 8))) {
        should_halve = true;
      }
      Work update = (priors[i] * gamma_over_one_minus_gamma / 256);
      // std::cout << "u: " << update << " ~ " << to_float(update) << std::endl;
      probs[i] = weights[i] + update;
      sum += probs[i];
      w_sum += weights[i];
    }

    std::uniform_int_distribution<Work> dist(0, sum);
    Work r = dist(rng);

    int c = -1;
    Work acc = 0;
    for (int i = 0; i < weights.size(); ++i) {
      acc += probs[i];
      c += 1;
      if (r < acc) {
        break;
      }
    }
    ++chosen[c];

    double mu_float = (double)probs[c] / sum;
    double eta_float = g / k;

    eta_over_mu = (eta_float / mu_float) * ONE;
    eta_over_mu_f = (eta_float / mu_float);

    if (should_halve) {
      std::cout << "halve" << std::endl;
      for (auto &w : weights) {
        w += 1;
        w /= 2;
      }
      eta_over_mu += 1;
      eta_over_mu /= 2;
    }

    return c;
  }

  void update(auto c, float r) { 
    // double factor = exp(r * to_float(eta_over_mu));
    double factor = exp(r * eta_over_mu_f);
    Work fac = exp32(eta_over_mu * r);
    // weights[c] *= fac;
    // weights[c] /= ONE;
    weights[c] *= factor;
  }
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

  print(p1);
  print(p2);

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

    // if ((m == 2) && (n == 2)) {
    //   matrix[0][0] = 1.0;
    //   matrix[0][1] = 0.0;
    //   matrix[1][0] = 0.0;
    //   matrix[1][1] = 1.0;
    // }

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