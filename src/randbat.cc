#include <randbat.h>

int benchmark(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "enter number of teams to generate" << std::endl;
    return 1;
  }

  RandomBattles::PRNG prng{32049283049280};
  for (int i = 0; i < std::atoi(argv[1]); ++i) {
    prng.next();
    RandomBattles::Teams generator{prng};
    generator.randomTeam();
  }
  return 0;
}

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

  std::cout << "seed: ";
  for (int i = 0; i < 4; ++i) {
    std::cout << seed_raw[3 - i] << ' ';
  }
  std::cout << " = " << seed << std::endl;

  RandomBattles::Teams generator{RandomBattles::PRNG{seed}};
  const auto team = generator.randomTeam();
  team.print();
  return 0;
}

int matches(int argc, char **argv) {
  RandomBattles::Teams generator{RandomBattles::PRNG{0}};
  const auto team = generator.randomTeam();

  RandomBattles::PartialTeam partial{};
  partial.species_slots[0] = {Data::Species::Jolteon, 0};
  partial.move_sets[0] = {Data::Moves::None, Data::Moves::None, Data::Moves::None, Data::Moves::None};

  std::cout << "team matches partial: " << team.matches(partial) << std::endl;
  std::cout << "partial matches team: " << partial.matches(team) << std::endl;

  team.print();
  partial.print();

  return 0;
}

int main(int argc, char **argv) { return benchmark(argc, argv); }