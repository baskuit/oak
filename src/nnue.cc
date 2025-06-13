#include <battle/init.h>
#include <battle/sample-teams.h>
#include <battle/strings.h>
#include <pi/exp3.h>
#include <pi/mcts.h>
#include <util/random.h>

#include <nnue/accumulator.h>
#include <nnue/nnue_architecture.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

using Node =
    Tree::Node<Exp3::JointBanditData<.03f, false>, std::array<uint8_t, 16>>;

struct Model {

  prng device{34545645};
  NNUE::WordCaches nnue_caches;
  NNUE::WordNet<198, 32, 39> pokemon_net;
  NNUE::WordNet<198 + 14, 32, 55> active_net;
  NNUE::NetworkArchitecture nn;
  std::array<uint8_t, 512> acc;

  float inference(const pkmn_gen1_battle &battle,
                  const pkmn_gen1_chance_durations &durations) {
    NNUE::Abstract abstract{battle, durations};
    nnue_caches.write_acc(abstract, acc.data());
    for (const auto byte : acc) {
      std::cout << (int)byte << ' ';
    }
    std::cout << std::endl;

    const float out = nn.propagate(acc.data()) / (127 * 64);
    const float val = 1 / (1 + std::exp(-out));
    std::cout << "out: " << val << " = sigmoid " << out << std::endl;
    return val;
  }
};

bool read_net(std::filesystem::path path, auto &net, std::string tag) {
  std::ifstream accs, w0s, b0s, w1s, b1s, w2s, b2s;
  w0s.open(path / (tag + "w0"), std::ios::binary);
  b0s.open(path / (tag + "b0"), std::ios::binary);
  w1s.open(path / (tag + "w1"), std::ios::binary);
  b1s.open(path / (tag + "b1"), std::ios::binary);
  w2s.open(path / (tag + "w2"), std::ios::binary);
  b2s.open(path / (tag + "b2"), std::ios::binary);
  const bool success = net.fc_0.read_parameters(w0s, b0s) &&
                       net.fc_1.read_parameters(w1s, b1s) &&
                       net.fc_2.read_parameters(w2s, b2s);
  return success;
}

bool read_params_from_dir(std::filesystem::path path, auto &pokemon_net,
                          auto &active_net, auto &main_net) {

  const auto read = [](auto &nn, const std::filesystem::path path,
                       const std::string tag) {
    std::ifstream accs, w0s, b0s, w1s, b1s, w2s, b2s;
    std::filesystem::path p = path / (tag + "w0");
    // std::cout << p.string() << std::endl;
    w0s.open(path / (tag + "w0"), std::ios::binary);
    b0s.open(path / (tag + "b0"), std::ios::binary);
    w1s.open(path / (tag + "w1"), std::ios::binary);
    b1s.open(path / (tag + "b1"), std::ios::binary);
    w2s.open(path / (tag + "w2"), std::ios::binary);
    b2s.open(path / (tag + "b2"), std::ios::binary);
    const bool success = nn.fc_0.read_parameters(w0s, b0s) &&
                         nn.fc_1.read_parameters(w1s, b1s) &&
                         nn.fc_2.read_parameters(w2s, b2s);
    // std::cout << "affine read success :" << std::endl;
    return success;
  };
  return read(active_net, path, "a") && read(pokemon_net, path, "p") &&
         read(main_net, path, "nn");
}

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cerr << "Input: nn path." << std::endl;
    return 1;
  }

  std::filesystem::path path{std::string{argv[1]}};

  if (!std::filesystem::exists(path)) {
    std::cerr << "Path does not exist." << std::endl;
    return 1;
  }

  NNUE::WordNet<NNUE::pokemon_in_dim, 32, NNUE::pokemon_out_dim> p;
  const bool success = read_net(path, p, "p");
  std::cout << success << std::endl;


  for (const auto f : p.fc_0.biases) {
    std::cout << f << ' ';
  }
  std::cout << std::endl;

  return 0;
}

// int poo(int argc, char **argv) {

//   if (argc < 3) {
//     std::cerr << "Input: nn path, iterations." << std::endl;
//     return 1;
//   }

//   std::string nn_path{argv[1]};
//   size_t iter = std::atoll(argv[2]);

//   std::cout << nn_path << ' ' << iter << std::endl;

//   auto battle = Init::battle(SampleTeams::teams[0], SampleTeams::teams[1]);
//   prng device{312312};
//   std::bit_cast<uint64_t *>(battle.bytes + Offsets::seed)[0] =
//       device.uniform_64();
//   pkmn_gen1_battle_options options{};
//   auto result = Init::update(battle, 0, 0, options);

//   pkmn_gen1_chance_durations durations{};
//   MCTS search{};
//   Node node{};
//   Model model{};

//   bool success = read_params_from_dir(nn_path, model.pokemon_net,
//                                       model.active_net, model.nn);
//   std::cout << "nn read success: " << success << std::endl;
//   MonteCarlo::Input input{battle, durations, result};
//   // MonteCarlo::Model mcm{device};

//   auto output = search.run(iter, node, input, model);
//   std::cout << "time: " << output.duration.count() / 1000.0 << std::endl;

//   std::cout << "p1 word cache sizes:" << std::endl;
//   model.nnue_caches.p1.print_sizes();
//   std::cout << "p2 word cache sizes:" << std::endl;
//   model.nnue_caches.p2.print_sizes();

//   std::cout << "search policies:" << std::endl;
//   for (auto i = 0; i < output.m; ++i) {
//     std::cout << output.p1[i] << ' ';
//   }
//   std::cout << std::endl;
//   for (auto i = 0; i < output.n; ++i) {
//     std::cout << output.p2[i] << ' ';
//   }
//   std::cout << std::endl;

//   return 0;
// }