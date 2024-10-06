#include <battle/battle.h>
#include <battle/util.h>

#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/pgame.h>
#include <pi/tree.h>

#include <model/monte-carlo-model.h>

#include <iostream>
#include <numeric>

template <typename Container> void print(const Container &container) {
  const auto n = container.size();
  for (int i = 0; i < n - 1; ++i) {
    std::cout << container[i] << ' ';
  }
  std::cout << container[n - 1] << std::endl;
}

namespace Teams {
using Data::Moves;
using Data::Species;
using enum Moves;
struct Set {
  Species species;
  std::vector<Moves> moves;
};

Set jynx{Species::Jynx, {Blizzard, LovelyKiss, Psychic, Rest}};
Set chansey{Species::Chansey, {IceBeam, Sing, SoftBoiled, Thunderbolt}};
Set cloyster{Species::Cloyster, {Blizzard, Clamp, Explosion, HyperBeam}};
Set rhydon{Species::Rhydon, {BodySlam, Earthquake, RockSlide, Substitute}};
Set starmie{Species::Starmie, {Blizzard, Recover, Thunderbolt, ThunderWave}};
Set tauros{Species::Tauros, {Blizzard, BodySlam, Earthquake, HyperBeam}};
};

struct Types {
  using State = Battle<0, true, true>;
  using Model = MonteCarloModel<prng, State, 16>;
  using Obs = std::array<uint8_t, 16>;
  using Node = Tree::Node<Exp3::JointBanditData, Obs>;
};

int mcts_1v1(int argc, char **argv) {
  prng device{1111111};
  Types::State battle{std::vector<Teams::Set>{Teams::tauros}, std::vector<Teams::Set>{Teams::rhydon}};
  battle.apply_actions(0, 0);
  battle.get_actions();
  Types::Model model{device.uniform_64()};
  Types::Node node{};

  const auto iterations = 1 << 20;
  double total_value = 0;
  const size_t window_size = 1 << 8;
  std::array<float, window_size> window{};

  for (auto i = 0; i < iterations; ++i) {
    auto battle_copy{battle};
    battle_copy.randomize_transition(device);
    const auto value = MCTS::run_iteration(device, &node, battle_copy, model);
    total_value += value;
    window[i % window_size] = value;
  }

  const double rolling_average =
      std::accumulate(window.begin(), window.end(), 0) / (double)window_size;

  const auto row_strategy = Exp3::empirical_strategies(node.data().row_visits);
  const auto col_strategy = Exp3::empirical_strategies(node.data().col_visits);
  std::cout << "row strategy" << std::endl;
  for (int i = 0; i < battle.rows(); ++i) {
    std::cout << side_choice_string(battle.battle().bytes, battle.row_actions[i]) <<  " : " << row_strategy[i] << std::endl;
  }
  std::cout << std::endl;

  std::cout << "col strategy" << std::endl;
  for (int i = 0; i < battle.cols(); ++i) {
    std::cout << side_choice_string(battle.battle().bytes + 184, battle.col_actions[i]) <<  " : " << col_strategy[i] << std::endl;
  }
  std::cout << std::endl;

  std::cout << "rolling average: " << rolling_average << std::endl;
  std::cout << "average value: " << total_value / iterations << std::endl;
  std::cout << "total nodes: " << MCTS::total_nodes << std::endl;
  std::cout << "average depth: "
            << MCTS::total_depth / (double)MCTS::total_nodes << std::endl;

  return 0;
}

int main(int argc, char **argv) { return mcts_1v1(argc, argv); }
