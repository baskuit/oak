#include <ii/random-battles/randbat.h>

int generate_team(int argc, char **argv) {
  if (argc != 5) {
    return 1;
  }

  uint16_t seed_raw[4];
  seed_raw[3] = std::atoi(argv[1]);
  seed_raw[2] = std::atoi(argv[2]);
  seed_raw[1] = std::atoi(argv[3]);
  seed_raw[0] = std::atoi(argv[4]);
  int64_t seed = *reinterpret_cast<int64_t *>(seed_raw);

  //   std::cout << "seed: ";
  //   for (int i = 0; i < 4; ++i) {
  //     std::cout << seed_raw[3 - i] << ' ';
  //   }
  //   std::cout << " = " << seed << std::endl;

  RandomBattles::Teams generator{RandomBattles::PRNG{seed}};
  const auto team = generator.randomTeam();
  team.print();
  return 0;
}

int main(int argc, char **argv) { return generate_team(argc, argv); }