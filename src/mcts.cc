#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/pgame.h>
#include <pi/tree.h>

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
  const PGame game{3};
  PGameModel model{};
  using Exp3Node = Tree::Node<Exp3::JointBanditData, int>;
  Exp3Node node{};

  const auto iterations = 1 << 20;
  for (auto i = 0; i < iterations; ++i) {
    auto game_copy{game};
      MCTS::run_iteration(device, &node, game_copy, model);
  }

  const auto row_strategy = Exp3::empirical_strategies(node.data().row_visits);
  print(node.data().row_visits);
  print(row_strategy);

  std::cout << "total nodes: " << MCTS::total_nodes << std::endl;

  return 0;
}