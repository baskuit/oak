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
  Stockfish::Eval::NNUE::read_litt  le_endian(dummys, dummy, 32);


  w1s.open("./weights/w2d", std::ios::binary);
  b1s.open("./weights/b2d", std::ios::binary);
  nn.fc_1.read_parameters(w1s, b1s);


  return 0;
}