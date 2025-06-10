#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

constexpr int NUM_ARMS = 5;
constexpr int FP_SHIFT = 16;
constexpr uint32_t FP_ONE = 1 << FP_SHIFT;
constexpr uint32_t GAMMA = (uint32_t)(0.07 * FP_ONE); // 0.07 in Q16.16
constexpr uint32_t GAMMA_OVER_K = GAMMA / NUM_ARMS;

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
  // assert(result.first > result.second);
  if (result.first <= result.second) {
    std::cout << result.first << ' ' << result.second << std::endl;
    for (auto i = 0; i < m; ++i) {
      std::cout << q1[i] << ' ';
    }
    std::cout << std::endl;
    for (auto i = 0; i < n; ++i) {
      std::cout << q2[i] << ' ';
    }
    std::cout << std::endl;
  }
  return result;
}

struct QuantBandit {
  std::vector<uint16_t> gains;   // Quantized gain
  std::vector<uint32_t> weights; // exp(gain)
  std::vector<uint32_t> probs;   // Probabilities
  std::vector<uint32_t> chosen;
  std::mt19937 rng;

  QuantBandit(size_t NUM_ARMS) {
    gains.resize(NUM_ARMS, 0);
    probs.resize(NUM_ARMS, FP_ONE / NUM_ARMS);
    weights.resize(NUM_ARMS, 0);
    chosen.resize(NUM_ARMS, 0);
    std::mt19937 rng{std::random_device{}()};
    // std::cout << NUM_ARMS << ' ' << gains.size() << std::endl;
  }

  int select() {

    // Compute weights as exp(gain * gamma / k)
    uint64_t weight_sum = 0;
    for (int i = 0; i < NUM_ARMS; ++i) {
      uint32_t gain_scaled = (gains[i] * GAMMA_OVER_K); // Q16.16
      weights[i] = approx_exp_q16(gain_scaled);
      weight_sum += weights[i];
    }
    // Compute probabilities in Q16.16
    for (int i = 0; i < NUM_ARMS; ++i) {
      uint32_t prob_exploit =
          (uint32_t)(((uint64_t)weights[i] << FP_SHIFT) / weight_sum); // Q16.16
      uint32_t prob_explore = FP_ONE / NUM_ARMS; // Uniform
      // (1 - gamma) * exploit + gamma * explore
      probs[i] =
          ((FP_ONE - GAMMA) * prob_exploit + GAMMA * prob_explore) >> FP_SHIFT;
    }
    // Choose an arm
    int c = sample_arm(probs, rng);
    chosen[c]++;
    return c;
  }

  void update(uint32_t reward, auto chosen) {
    uint32_t estimated =
        ((uint64_t)reward << FP_SHIFT) / probs[chosen]; // Q16.16
    // Convert estimated reward to scaled integer gain and clip
    uint32_t gain_update =
        estimated >> (FP_SHIFT - 4); // scale down to uint16_t range
    gains[chosen] = std::min<uint32_t>(65535, gains[chosen] + gain_update);
  }
};

auto try_poo(QuantBandit& b1, QuantBandit& b2, const auto &matrix, auto iter) {
  for (auto n = 0; n < iter; ++n) {
    auto i = b1.select();
    auto j = b2.select();
    const uint32_t r = matrix[i][j] * FP_ONE;
    const uint32_t s = (1 - matrix[i][j]) * FP_ONE;
    b1.update(r, i);
    b2.update(s, j);
  }

  const auto m = b1.gains.size();
  const auto n = b2.gains.size();
  std::vector<float> p1;
  p1.resize(m);
  for (auto i = 0; i < m; ++i) {
    p1[i] = static_cast<float>(b1.chosen[i]) / iter;
    std::cout << p1[i] << ' ';
  }
  std::cout << std::endl;
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
    const auto matrix = random_matrix<float>(device, m, n);
    QuantBandit b1{m};
    QuantBandit b2{n};
    const auto x = try_poo(b1, b2, matrix, iter);
    total_x += x;

//     const auto float_x =
//         try_poo<float>(float_matrix_device, matrix, m, n, iter, gamma);
//     total_float_x += float_x;
//   }
//   auto end = std::chrono::high_resolution_clock::now();
//   auto d = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//   std::cout << d.count() << std::endl;
//   start = std::chrono::high_resolution_clock::now();
//   for (auto t = 0; t < trials; ++t) {
//     const auto matrix = random_matrix<_Float16>(half_matrix_device, m, n);
//     const auto half_x = try_poo<_Float16>(half_device, matrix, m, n, iter,
//                                           static_cast<_Float16>(gamma));
//     total_half_x += half_x;
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto d = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << d.count() << std::endl;

  std::cout << "expl: " << total_x / trials << std::endl;

  return 0;
}