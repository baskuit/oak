#include <battle/battle.h>
#include <battle/sides.h>

#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/pgame.h>
#include <pi/tree.h>

#include <model/monte-carlo-model.h>

#include <iostream>

template <typename Container>
void print(const Container &container) {
  const auto n = container.size();
  for (int i = 0; i < n - 1; ++i) {
    std::cout << container[i] << ' ';
  }
  std::cout << container[n - 1] << std::endl;
}

int main() {

//   Exp3::uint24_t_test int_test{};
  prng device{1111111};
  Battle<64, true> battle{sides[0], sides[1]};
  battle.apply_actions(0, 0);
  battle.get_actions();

  MonteCarloModel<prng, Battle<64, true>> model{device.uniform_64()};
  using Obs = std::array<uint8_t, 16>;
  using Exp3Node = Tree::Node<Exp3::JointBanditData, Obs>;
  Exp3Node node{};

  const auto iterations = 1 << 20;
  for (auto i = 0; i < iterations; ++i) {
    auto battle_copy{battle};
      MCTS::run_iteration(device, &node, battle_copy, model);
  }

  const auto row_strategy = Exp3::empirical_strategies(node.data().row_visits);
  print(node.data().row_visits);
  print(row_strategy);

  std::cout << "total nodes: " << MCTS::total_nodes << std::endl;

  return 0;
}
