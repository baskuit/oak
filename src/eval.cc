#include <battle/debug-log.h>
#include <battle/init.h>
#include <data/options.h>
#include <data/sample-teams.h>
#include <data/strings.h>
#include <iostream>
#include <pi/abstract.h>
#include <pi/eval.h>
#include <util/print.h>
#include <util/random.h>

#include <thread>

using Obs = std::array<uint8_t, 16>;
using Node = Tree::Node<Exp3::JointBanditData<.15f, false>, Obs>;

void monte_carlo(const Eval::Input &battle_data, size_t ms) {}

// struct MonteCarlo {
//   MCTS search{};

//   std::pair<uint8_t, uint8_t> go(const BattleData &battle_data, size_t ms,
//                                  uint64_t seed) const {
//     Node node;
//     Model model;
//     model.device = {seed};
//     const auto output = search.run(ms, node, battle_data, model);
//     return {};
//   }
// };

struct FastModel {

  Eval::GlobalMEM global;
};

void simul_test() {}

static_assert(Options::calc && Options::chance && !Options::log);

int abstract_test(int argc, char **argv) {

  const uint64_t seed = std::atoi(argv[1]);
  prng device{seed};

  Eval::Input input;

  const auto p1 = SampleTeams::teams[0];
  const auto p2 = SampleTeams::teams[1];
  input.battle = Init::battle(p1, p2);
  MCTS search{};
  input.result = Init::update(input.battle, 0, 0, search.options);
  input.abstract = {input.battle};

  Eval::GlobalMEM global{};
  const bool cache_loaded = global.load("./global");
  std::cout << "Global load : " << cache_loaded << std::endl;
  Node node;

  Eval::Model model{23402342, Eval::CachedEval{p1, p2, global}};

  const auto output = search.run(1 << 20, node, input, model);

  for (auto i = 0; i < output.p1.size(); ++i) {
    std::cout << side_choice_string(input.battle.bytes, output.choices1[i])
              << " : " << output.p1[i] << ", ";
  }
  std::cout << std::endl;
  for (auto i = 0; i < output.p2.size(); ++i) {
    std::cout << side_choice_string(input.battle.bytes + Offsets::side,
                                    output.choices2[i])
              << " : " << output.p2[i] << ", ";
  }
  std::cout << std::endl;

  return 0;
}

int main(int argc, char **argv) { return abstract_test(argc, argv); }
