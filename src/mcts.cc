#include <pi/mcts.h>
#include <pi/pgame.h>
#include <pi/tree.h>
#include <pi/exp3.h>

int main () {

    PGame game{5};
    PGameModel model{};
    using Exp3Node = Tree::Node<Exp3::JointBanditData, int>;
    Exp3::JointBanditData data{};
    Exp3Node node{};

    const auto iterations = 1 << 20;
    for (auto i = 0; i < iterations; ++i) {
        MCTS::run_iteration(&node, game, model);
    }

    return 0;

}