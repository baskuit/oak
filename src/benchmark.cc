#include <battle/sample-teams.h>
#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/tree.h>
#include <util/random.h>

#include <chrono>

struct BattleData {
  pkmn_gen1_battle battle;
  pkmn_gen1_chance_durations durations;
  pkmn_result result;
};

struct Model {
  prng device;
};

namespace Types {}; // namespace Types

int benchmark(int argc, char **argv) {
  constexpr bool mcts_root_visits{false};

  using Obs = std::array<uint8_t, 16>;
  using Node = Tree::Node<Exp3::JointBanditData<.03f, mcts_root_visits>, Obs>;

  const auto p1 = SampleTeams::teams[0];
  const auto p2 = SampleTeams::teams[1];
  const uint64_t seed = 1111111;
  Model model;
  model.device = prng{seed};
  BattleData input{};
  input.battle = Init::battle(p1, p2, seed);

  MCTS search{};
  int exp = 20;
  if (argc == 2) {
    exp = std::atoi(argv[1]);
  }
  exp = std::max(0, std::min(20, exp));
  size_t iterations = 1 << exp;
  input.result = Init::update(input.battle, 0, 0, search.options);
  Node node{};

  const auto output = search.run(iterations, node, input, model);
  std::cout << output.duration.count() << " ms." << std::endl;

  return 0;
}

int main(int argc, char **argv) { return benchmark(argc, argv); }