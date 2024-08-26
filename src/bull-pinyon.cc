#include <pinyon.h>

#include "../include/battle.h"
#include "../include/clamp.h"
#include "../include/mc-average.h"
#include "../include/sides.h"

void q_value() {
  // depth-2 alpha beta solve with MC-AVG at leafs. damage rolls are clamped at
  // the start
  using B = Battle<64, 3, ChanceObs, float, float>;
  using LeafModel =
      Clamped<SearchModel<AlphaBetaForce<MonteCarloModelAverage<B>>, false,
                          false, false>>;

  // the bull 1v1 exhaustively explored up to depth 1. at 'terminal' nodes, we
  // apply the above value estimation
  using Types = FullTraversal<NullModel<MappedState<LeafModel, true>>>;

  prng device{22524};

  LeafModel::State battle{sides[0], sides[0]};
  battle.apply_actions(0, 0);
  battle.get_actions();

  const size_t leaf_depth = 2;
  LeafModel::Model leaf_model{
      leaf_depth, prng{0}, {prng{0}, 1 << 1}, {0, 1 << 7, 0.0f}};

  const size_t mapping_tries = 1 << 18;
  const size_t mapping_depth = 1;
  Types::State mapped_state{mapping_depth, mapping_tries, prng{3}, leaf_model,
                            battle};
  std::cout << "mapped state initialized!" << std::endl;
  std::cout << "tries: " << mapping_tries << "; "
            << mapped_state.node->count_matrix_nodes() << " nodes."
            << std::endl;

  Types::Model model{};
  Types::Search search{};
  Types::MatrixNode node{};
  const size_t threads = 8;

  search.run(1, device, mapped_state, model, node, threads);

  // the point of all this compute. this allows us to approximate the q-value
  // matrix that bull.cc provides
  node.stats.nash_payoff_matrix.print();

  // Matrix<float> float_matrix{4, 4};
  // for (int i = 0; i < 4; ++i) {
  // for (int j = 0; j < 4; ++j) {
  //     float_matrix.get(i, j) =
  //     math::to_float(node.stats.nash_payoff_matrix.get(i,
  //     j).get_row_value());
  // }}
  // float_matrix.print();
}

int main(int argc, char **argv) {
  q_value();
  return 0;
}