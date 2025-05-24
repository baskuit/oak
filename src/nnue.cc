#include "nnue/nnue_architecture.h"

#include <random>

int main(int argc, char **argv) {

  const auto size = 1 << 20;
  const auto trials{1 << 22};

  Stockfish::Eval::NNUE::NetworkArchitecture nn{};

  uint8_t ones[32];
  int32_t output[1];

  Stockfish::Eval::NNUE::Layers::AffineTransform<32, 1> layer{};
  for (auto i = 0; i < 32; ++i) {
    ones[i] = 127;
    layer.weights[i] = 64; // aka 64 = 1.0
  }
  layer.propagate(ones, output);
  std::cout << output[0] << std::endl;
  return 0;

  // layer.write_parameters()

  uint64_t seed;

  seed = std::atoi(argv[1]);
  std::cout << "seed: " << seed << std::endl;

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<uint32_t> randint{0, size - 512};

  uint8_t randomNoise[size];
  uint32_t offset[trials];
  for (auto i = 0; i < size; ++i) {
    randomNoise[i] = gen() % 128;
  }
  for (auto i = 0; i < trials; ++i) {
    offset[i] = randint(gen);
  }

  for (int i = 0; i < 10; ++i) {
    std::cout << offset[i] << ' ';
  }
  std::cout << std::endl;

  for (auto i = 0; i < trials; ++i) {
    const auto result = nn.propagate(randomNoise + offset[i]);
  }

  const auto result = nn.propagate(randomNoise);

  std::cout << "result <i32, 16>  : " << result << std::endl;

  return 0;
}