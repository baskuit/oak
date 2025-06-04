#include "nnue/nnue_architecture.h"

#include <fstream>
#include <random>

int main(int argc, char **argv) {

  const auto size = 1 << 20;
  const auto trials{1 << 22};

  Stockfish::Eval::NNUE::NetworkArchitecture nn{};

  std::ifstream accs, w0s, b0s, w1s, b1s, w2s, b2s;

  uint8_t acc[512];

  accs.open("./weights/accd", std::ios::binary);
  Stockfish::Eval::NNUE::read_little_endian(accs, acc, 512);

  w0s.open("./weights/w0", std::ios::binary);
  b0s.open("./weights/b0", std::ios::binary);
  w1s.open("./weights/w1", std::ios::binary);
  b1s.open("./weights/b1", std::ios::binary);
  w2s.open("./weights/w2", std::ios::binary);
  b2s.open("./weights/b2", std::ios::binary);

  const bool success = nn.fc_0.read_parameters(w0s, b0s) &&
                       nn.fc_1.read_parameters(w1s, b1s) &&
                       nn.fc_2.read_parameters(w2s, b2s);

  if (!success) {
    std::cerr << "read_params failed." << std::endl;
    return 1;
  }

  auto out = nn.propagate(acc);

  std::cout << "last: " << out << std::endl;
  std::cout << "last (float): " << (float)out / (127 * 64) << std::endl;
  return 0;
}