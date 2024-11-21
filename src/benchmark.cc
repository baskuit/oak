#include <battle/init.h>
#include <data/sample-teams.h>
#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/tree.h>
#include <util/random.h>

#include <chrono>

namespace Types {
  constexpr bool enable_visits{false};
  using Obs = std::array<uint8_t, 16>;
  using Node = Tree::Node<Exp3::JointBanditData<enable_visits>, Obs>;
};

int benchmark() {
  const auto p1 = SampleTeams::teams[0];
  const auto p2 = SampleTeams::teams[1];
  const uint64_t seed = 123456789;
  prng device{seed};
  auto battle = Init::battle(p1, p2, seed);

  MCTS<false> search{};
  size_t iterations = 1 << 20;
  auto result = pkmn_gen1_battle_update(&battle, 0, 0, &search.options);
  Types::Node node{};
  pkmn_gen1_chance_durations durations{};

  const auto start = std::chrono::high_resolution_clock::now();
  const auto output =
      search.run(iterations, device, node, &battle, result, &durations);
  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << duration.count() << " ms." << std::endl;
  return 0;
}

int main() { return benchmark(); }