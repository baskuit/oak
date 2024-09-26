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
  if (argc != 2) {
    std::cout << "enter number of tries to match partial." << std::endl;
    return 1;
  }

  RandomBattles::PRNG prng{0};

  RandomBattles::PartialTeam partial{};
  partial.species_slots[0] = {Data::Species::Jolteon, 0};
  partial.species_slots[1] = {Data::Species::Shellder, 1};
  partial.species_slots[2] = {Data::Species::Grimer, 2};
  partial.move_sets[0] = {Data::Moves::Agility, Data::Moves::BodySlam,
                          Data::Moves::None, Data::Moves::None};
  partial.move_sets[1] = {Data::Moves::Blizzard, Data::Moves::DoubleEdge,
                          Data::Moves::None, Data::Moves::None};
  partial.move_sets[2] = {Data::Moves::FireBlast, Data::Moves::BodySlam,
                          Data::Moves::None, Data::Moves::None};

  size_t tries = std::atoi(argv[1]);
  std::cout << "using " << tries << " tries; find teams matching:" << std::endl;
  partial.print();
  std::cout << std::endl;

  std::unordered_map<int64_t, bool> seed_set{};

  for (int i = 0; i < tries; ++i) {
    RandomBattles::Teams generator{prng};
    const auto seed = prng.seed;
    const auto team = generator.randomTeam();

    if (partial.matches(team)) {
      std::cout << "match found! seed: " << seed << std::endl;
      team.print();
      std::cout << std::endl;
      seed_set[seed] = true;
    };
    prng.next();
  }

  std::cout << "\nTOTAL DISTINCT MATCHING SEEDS: " << seed_set.size()
            << std::endl;

  return 0;
}

int main(int argc, char **argv) { return matches(argc, argv); }