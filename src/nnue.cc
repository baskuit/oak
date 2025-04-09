#include "nnue/nnue_architecture.h"

#include <random>

int main (int argc, char **argv) {

    const auto size = 1 << 20;
    const auto trials{1 << 22};

    Stockfish::Eval::NNUE::NetworkArchitecture nn{};

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