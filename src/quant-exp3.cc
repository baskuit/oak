#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

constexpr int FP_SHIFT = 16;
constexpr uint32_t FP_ONE = 1 << FP_SHIFT;

double de(uint32_t n) {
  return n / (double) FP_ONE;
}

// Approximate exp(x) in Q16.16 using 1 + x + x²/2 + x³/6
uint32_t approx_exp_q16(uint32_t x_q16) {
  uint32_t x2 = (x_q16 * x_q16) >> FP_SHIFT;
  uint32_t x3 = (x2 * x_q16) >> FP_SHIFT;
  return FP_ONE + x_q16 + (x2 >> 1) + (x3 / 6);
}

// Sample an arm using fixed-point probabilities
int sample_arm(const std::vector<uint32_t> &probs_q16, std::mt19937 &rng) {
  std::uniform_int_distribution<uint32_t> dist(0, FP_ONE - 1);
  uint32_t r = dist(rng);
  uint32_t acc = 0;
  for (int i = 0; i < probs_q16.size(); ++i) {
    acc += probs_q16[i];
    if (r < acc)
      return i;
  }
  return probs_q16.size() - 1;
}

#include <nnue/accumulator.h>
#include <pi/exp3.h>
#include <util/random.h>

#include <algorithm>
#include <chrono>
#include <iostream>

constexpr float g_ = .03;

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
  assert(result.first > result.second);
  return result;
}

struct QuantBandit {
  std::vector<uint16_t> gains;   // Quantized gain
  std::vector<uint32_t> weights; // exp(gain)
  std::vector<uint32_t> probs;   // Probabilities
  std::vector<uint32_t> chosen;
  std::mt19937 rng;
  uint32_t GAMMA, GAMMA_OVER_K, NUM_ARMS;

  QuantBandit(size_t NUM_ARMS, float gamma) {
    this->NUM_ARMS = NUM_ARMS;
    gains.resize(NUM_ARMS, 0);
    probs.resize(NUM_ARMS, FP_ONE / NUM_ARMS);
    weights.resize(NUM_ARMS, 0);
    chosen.resize(NUM_ARMS, 0);
    std::mt19937 rng{std::random_device{}()};
    GAMMA = (uint32_t)(gamma * FP_ONE);
    GAMMA_OVER_K = GAMMA / NUM_ARMS;
  }

  int select() {

    // Compute weights as exp(gain * gamma / k)
    uint64_t weight_sum = 0;
    for (int i = 0; i < NUM_ARMS; ++i) {
      uint32_t gain_scaled = (gains[i] * GAMMA_OVER_K); // Q16.16
      weights[i] = approx_exp_q16(gain_scaled);
      std::cout << de(gain_scaled) << " : " << de(weights[i]) << ' ';

      weight_sum += weights[i];
      // std::cout << weights[i] / (float) FP_ONE << ' ';
    }
    std::cout << '\n';
    // Compute probabilities in Q16.16
    for (int i = 0; i < NUM_ARMS; ++i) {
      uint32_t prob_exploit =
          (uint32_t)(((uint64_t)weights[i] << FP_SHIFT) / weight_sum); // Q16.16
      uint32_t prob_explore = FP_ONE / NUM_ARMS; // Uniform
      // (1 - gamma) * exploit + gamma * explore
      probs[i] =
          ((FP_ONE - GAMMA) * prob_exploit + GAMMA * prob_explore) >> FP_SHIFT;
      std::cout << probs[i] / (float) FP_ONE << ' ';
    }
    std::cout << '\n';
    // Choose an arm
    int c = sample_arm(probs, rng);
    chosen[c]++;
    return c;
  }

  void update(uint32_t reward, auto c) {
    uint32_t estimated =
        ((uint64_t)reward << FP_SHIFT) / probs[c]; // Q16.16
    // Convert estimated reward to scaled integer gain and clip
    uint32_t gain_update =
        estimated >> (FP_SHIFT - 4); // scale down to uint16_t range
    gains[c] = std::min<uint32_t>(FP_ONE, gains[c] + gain_update);
  }
};

auto try_poo(QuantBandit& b1, QuantBandit& b2, const auto &matrix, auto iter) {
  for (auto n = 0; n < iter; ++n) {
    auto i = b1.select();
    auto j = b2.select();
    const uint32_t r = matrix[i][j] * FP_ONE;
    const uint32_t s = FP_ONE - r;
    b1.update(r, i);
    b2.update(s, j);
  }

  const auto m = b1.gains.size();
  const auto n = b2.gains.size();
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
    std::cerr << "m n iter trials gamma; get average expl for float16/32 exp3"
              << std::endl;
    return 1;
  }
  prng device{5564564563};
  const auto m = std::atoi(argv[1]);
  const auto n = std::atoi(argv[2]);
  const auto iter = std::atoi(argv[3]);
  const auto trials = std::atoi(argv[4]);
  const float gamma = std::atof(argv[5]);

  float total_x = 0;
  auto start = std::chrono::high_resolution_clock::now();

  for (auto t = 0; t < trials; ++t) {
    auto matrix = random_matrix<float>(device, m, n);
    matrix[0][1] = matrix[0][0] = 1.0;
    matrix[1][1] = matrix[1][0] = 0.0;

    QuantBandit b1{m, gamma};
    QuantBandit b2{n, gamma};
    const auto x = try_poo(b1, b2, matrix, iter);
    total_x += x;
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto d = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << d.count() << std::endl;

  std::cout << "expl: " << total_x / trials << std::endl;

  return 0;
}