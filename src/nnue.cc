#include "nnue/nnue_architecture.h"

#include <fstream>
#include <random>

int main(int argc, char **argv) {

  const auto size = 1 << 20;
  const auto trials{1 << 22};

  Stockfish::Eval::NNUE::NetworkArchitecture nn{};

  std::ifstream accs, w1s, b1s, w2s, b2s, dummys;

  uint8_t acc[512];
  uint8_t dummy[32];

  int32_t fc_0_out[32];
  uint8_t ac_0_out[32];

  accs.open("./weights/accd", std::ios::binary);
  Stockfish::Eval::NNUE::read_little_endian(accs, acc, 512);
  dummys.open("./weights/accd", std::ios::binary);
  Stockfish::Eval::NNUE::read_little_endian(dummys, dummy, 32);

  std::cout << "Trunaced acc:\n";
  for (auto i = 0; i < 32; ++i) {
    std::cout << (int)dummy[i] << ' ';
  }
  std::cout << std::endl;

  w1s.open("./weights/w1", std::ios::binary);
  b1s.open("./weights/b1", std::ios::binary);
  nn.fc_1.read_parameters(w1s, b1s);

  nn.fc_1.propagate(dummy, fc_0_out);
  nn.ac_0.propagate(fc_0_out, ac_0_out);

  std::cout << "Output of 32x32 layer on truncated acc:\n";
  for (auto i = 0; i < 32; ++i) {
    std::cout << (int)ac_0_out[i] << ' ';
  }
  std::cout << std::endl;

  return 0;
}