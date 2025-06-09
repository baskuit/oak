#include <battle/init.h>
#include <battle/sample-teams.h>
#include <pi/exp3.h>
#include <pi/mcts.h>
#include <util/random.h>

#include <nnue/accumulator.h>
#include <nnue/nnue_architecture.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

using Node =
    Tree::Node<Exp3::JointBanditData<.03f, false>, std::array<uint8_t, 16>>;

struct Model {

  prng device{34545645};
  NNUE::WordCaches nnue_caches;
  Stockfish::Eval::NNUE::WordNet<198, 64, 39> pokemon_net;
  Stockfish::Eval::NNUE::WordNet<198 + 14, 64, 55> active_net;
  Stockfish::Eval::NNUE::NetworkArchitecture nn;
  std::array<uint8_t, 512> acc;

  float inference(const pkmn_gen1_battle &battle,
                  const pkmn_gen1_chance_durations &durations) {
    NNUE::Abstract abstract{battle, durations};
    nnue_caches.write_acc(abstract, acc.data());
    const float out = nn.propagate(acc.data()) / (127 * 64);
    // std::cout << out << std::endl;
    return 1 / (1 + std::exp(-out));
  }
};

bool read_params_from_dir(std::filesystem::path path, auto &pokemon_net,
                          auto &active_net, auto &main_net) {

  const auto read = [](auto &nn, const std::filesystem::path path,
                       const std::string tag) {
    std::ifstream accs, w0s, b0s, w1s, b1s, w2s, b2s;
    std::filesystem::path p = path / (tag + "w0");
    std::cout << p.string() << std::endl;
    w0s.open(path / (tag + "w0"), std::ios::binary);
    b0s.open(path / (tag + "b0"), std::ios::binary);
    w1s.open(path / (tag + "w1"), std::ios::binary);
    b1s.open(path / (tag + "b1"), std::ios::binary);
    w2s.open(path / (tag + "w2"), std::ios::binary);
    b2s.open(path / (tag + "b2"), std::ios::binary);
    // std::cout << std::filesystem::exists(path / (tag + "w0")) << std::endl;
    // std::cout << std::filesystem::exists(path / (tag + "b0")) << std::endl;
    const bool success = nn.fc_0.read_parameters(w0s, b0s) &&
           nn.fc_1.read_parameters(w1s, b1s) &&
           nn.fc_2.read_parameters(w2s, b2s);
    std::cout << success << std::endl;
    return success;
  };
  return read(active_net, path, "a") && read(pokemon_net, path, "p") &&
         read(main_net, path, "nn");
}

int main(int argc, char **argv) {

  auto battle = Init::battle(SampleTeams::teams[0], SampleTeams::teams[1]);
  pkmn_gen1_battle_options options{};
  auto result = Init::update(battle, 0, 0, options);
  prng device{312312};
  std::bit_cast<uint64_t *>(battle.bytes + Offsets::seed)[0] =
      device.uniform_64();
  pkmn_gen1_chance_durations durations{};
  MCTS search{};
  Node node{};
  Model model{};

  bool success = read_params_from_dir("./weights/9500", model.pokemon_net, model.active_net, model.nn);
  std::cout << success << std::endl;
  // MonteCarlo::Input input{battle, durations, result};
  // MonteCarlo::Model mcm{device};
  // std::chrono::milliseconds dur{100};
  // size_t count = 1 << 20;
  // auto output = search.run(count, node, input, mcm);
  // std::cout << output.iterations << std::endl;

  // model.nnue_caches.p1.print_sizes();
  // model.nnue_caches.p2.print_sizes();

  return 0;
}