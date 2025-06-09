#include <battle/init.h>
#include <battle/sample-teams.h>
#include <pi/exp3.h>
#include <pi/mcts.h>
#include <util/random.h>

#include <nnue/accumulator.h>
#include <nnue/nnue_architecture.h>

#include <cmath>
#include <fstream>
#include <random>

using Node =
    Tree::Node<Exp3::JointBanditData<.03f, false>, std::array<uint8_t, 16>>;

struct Model {

  prng device{34545645};
  NNUE::WordCaches nnue_caches{};
  Stockfish::Eval::NNUE::NetworkArchitecture nn{};
  std::array<uint8_t, 512> acc{};

  float inference(const pkmn_gen1_battle &battle,
                  const pkmn_gen1_chance_durations &durations) {
    NNUE::Abstract abstract{battle, durations};
    nnue_caches.write_acc(abstract, acc.data());
    const float out = nn.propagate(acc.data()) / (127 * 64);
    // std::cout << out << std::endl;
    return 1 / (1 + std::exp(-out));
  }
};

int main(int argc, char **argv) {

  auto battle =
      Init::battle(SampleTeams::teams[0], SampleTeams::teams[1]);
  pkmn_gen1_battle_options options{};
  auto result = Init::update(battle, 0, 0, options); 
  prng device{312312};
  std::bit_cast<uint64_t *>(battle.bytes + Offsets::seed)[0] =
      device.uniform_64();
  pkmn_gen1_chance_durations durations{};
  MCTS search{};
  Node node{};
  Model model{};
  auto &nn = model.nn;

  std::ifstream accs, w0s, b0s, w1s, b1s, w2s, b2s;
  w0s.open("./weights/w0", std::ios::binary);
  b0s.open("./weights/b0", std::ios::binary);
  w1s.open("./weights/w1", std::ios::binary);
  b1s.open("./weights/b1", std::ios::binary);
  w2s.open("./weights/w2", std::ios::binary);
  b2s.open("./weights/b2", std::ios::binary);

  const bool success = nn.fc_0.read_parameters(w0s, b0s) &&
                       nn.fc_1.read_parameters(w1s, b1s) &&
                       nn.fc_2.read_parameters(w2s, b2s);

  if (!success) {
    std::cerr << "read_params failed." << std::endl;
    return 1;
  }

  MonteCarlo::Input input{battle, durations, result};
  auto output = search.run(1 << 20, node, input, model);


  model.nnue_caches.p1.print_sizes();
  model.nnue_caches.p2.print_sizes();

  return 0;
}