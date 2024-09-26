#include <randbat.h>

#include <mutex>
#include <thread>
#include <random>

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

struct Sync {
  std::mutex mutex;
  struct Data {
    size_t n{};
    RandomBattles::PartialTeam team;
  };
  std::unordered_map<int64_t, Data> map;
};

template <typename Sync>
void sample_teams(Sync *sync, const RandomBattles::PartialTeam partial) {

  while (true) {
    std::mt19937_64 rng{};
    rng.seed(std::random_device{}());
    int64_t seed = rng();
    RandomBattles::Teams generator{RandomBattles::PRNG{seed}};
    const auto team = generator.randomTeam();
    if (partial.matches(team)) {
      sync->mutex.lock();
      auto &data = sync->map[seed];
      data.team = team;
      ++data.n;
      if (data.n == 1) {
        std::cout << "\nseed: " << seed << std::endl;
        team.print();
      }
      sync->mutex.unlock();
    }
  }
}

consteval auto get_partial() noexcept {
  RandomBattles::PartialTeam partial{};
  partial.species_slots[0] = {Data::Species::Jolteon, 0};
  partial.move_sets[0] = {Data::Moves::Agility, Data::Moves::ThunderWave,
                          Data::Moves::Thunderbolt, Data::Moves::None};

  partial.species_slots[1] = {Data::Species::Shellder, 1};
  partial.move_sets[1] = {Data::Moves::Explosion, Data::Moves::Blizzard,
                          Data::Moves::Surf, Data::Moves::None};

  partial.species_slots[2] = {Data::Species::Grimer, 2};
  partial.move_sets[2] = {Data::Moves::Explosion, Data::Moves::FireBlast,
                          Data::Moves::MegaDrain, Data::Moves::None};

  partial.species_slots[3] = {Data::Species::Farfetchd, 3};
  partial.move_sets[3] = {Data::Moves::Slash, Data::Moves::Agility,
                          Data::Moves::BodySlam, Data::Moves::None};

  partial.species_slots[4] = {Data::Species::Bellsprout, 4};
  partial.move_sets[4] = {Data::Moves::SleepPowder, Data::Moves::StunSpore,
                          Data::Moves::RazorLeaf, Data::Moves::None};

  partial.species_slots[5] = {Data::Species::None, 5};
  partial.move_sets[5] = {Data::Moves::None, Data::Moves::None,
                          Data::Moves::None, Data::Moves::None};
  return partial;
}

int deep_sample(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "Usage: enter thread count to sample/match against 'partial; "
                 "The program will display output continuously.'"
              << std::endl;
    return 1;
  }

  const auto partial = get_partial();
  const int n_threads = std::atoi(argv[1]);

  Sync sync{};

  auto *const thread_arr = new std::thread[n_threads];
  for (auto i = 0; i < n_threads; ++i) {
    thread_arr[i] = std::thread(&sample_teams<Sync>, &sync, partial);
  }

  for (auto i = 0; i < n_threads; ++i) {
    thread_arr[i].join();
  }

  delete[] thread_arr;
  return 0;
}

int main(int argc, char **argv) { return deep_sample(argc, argv); }