#include <battle/init.h>
#include <battle/sample-teams.h>
#include <battle/strings.h>
#include <pi/exp3.h>
#include <pi/frame.h>
#include <pi/mcts.h>
#include <util/random.h>

#include <nnue/accumulator.h>
#include <nnue/nnue_architecture.h>

#include <cmath>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <atomic>
#include <thread>

using Obs = std::array<uint8_t, 16>;
using Node = Tree::Node<Exp3::JointBanditData<.03f, false>, Obs>;

void print_p(const auto& v) {
  for (const auto x : v) {
    std::cout << x <<  ' ';
  }
  std::cout << std::endl;
}

namespace NNUE {
struct Model {
  prng device{304958340593840}; // for mcts..
  NNUE::PokemonNet pokemon_net;
  NNUE::ActiveNet active_net;
  NNUE::NNUECache nnue_cache;

  NNUE::NetworkArchitecture main_net;
  std::array<uint8_t, 512> acc;

  Model(const pkmn_gen1_battle &battle)
      : pokemon_net{}, active_net{},
        nnue_cache{battle, pokemon_net, active_net}, main_net{}, acc{} {}

  float inference(const pkmn_gen1_battle &battle,
                  const pkmn_gen1_chance_durations &durations) {
    NNUE::BattleKeys battle_keys{battle, durations};
    nnue_cache.accumulate(battle_keys, acc.data());
    // NNUE::print_acc(acc);
    const float out = main_net.propagate(acc.data()) / (8128.0);
    // std::cout << "model scaled out: " << out << std::endl;
    const float val = 1 / (1 + std::exp(-out));
    return val;
  }
};
}

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
    w0s.open(path / (tag + "w0"), std::ios::binary);
    b0s.open(path / (tag + "b0"), std::ios::binary);
    w1s.open(path / (tag + "w1"), std::ios::binary);
    b1s.open(path / (tag + "b1"), std::ios::binary);
    w2s.open(path / (tag + "w2"), std::ios::binary);
    b2s.open(path / (tag + "b2"), std::ios::binary);
    const bool success = nn.fc_0.read_parameters(w0s, b0s) &&
                         nn.fc_1.read_parameters(w1s, b1s) &&
                         nn.fc_2.read_parameters(w2s, b2s);
    return success;
  };
  const bool m_success = read(main_net, path, "nn");
  const bool p_success = read(pokemon_net, path, "p");
  const bool a_success = read(active_net, path, "a");
  std::cout << "net reads: " << m_success << p_success << a_success
            << std::endl;
  return m_success && p_success && a_success;
}

void thread_fn(std::atomic<size_t> *score, std::atomic<size_t> *counter,
               size_t trials, size_t iterations, std::string net_path) {

  for (int trial = 0; trial < trials; ++trial) {

    prng device{std::random_device{}()};
    const auto team1 =
        SampleTeams::teams[device.random_int(SampleTeams::teams.size())];
    const auto team2 =
        SampleTeams::teams[device.random_int(SampleTeams::teams.size())];

    auto battle = Init::battle(team1, team2);
    pkmn_gen1_battle_options options{};
    auto result = Init::update(battle, 0, 0, options);
    pkmn_gen1_chance_durations durations{};

    NNUE::Model nnue{battle};


    const bool read_success = read_params_from_dir(
        net_path, nnue.pokemon_net, nnue.active_net, nnue.main_net);
    if (!read_success) {
      std::cerr << "Failed to read net params from " << net_path << std::endl;
      return;
    }

    while (!pkmn_result_type(result)) {

      pkmn_choice c1, c2;

      const auto [choices1, choices2] = Init::choices(battle, result);

      {
        MCTS search{};
        Node node{};
        MonteCarlo::Input input{battle, durations, result};
        MonteCarlo::Model model{device.uniform_64()};
        const auto output = search.run(iterations, node, input, model);
        c1 = choices1[device.sample_pdf(output.p1)];
        // print_p(output.p1);
        // print_p(output.p2);
      }
      {
        MCTS search{};
        Node node{};
        MonteCarlo::Input input{battle, durations, result};
        const auto output = search.run(iterations, node, input, nnue);
        c2 = choices2[device.sample_pdf(output.p2)];
        // print_p(output.p1);
        // print_p(output.p2);
      }

      // std::cout << '!' << std::endl;

      result = Init::update(battle, c1, c2, options);
    }

    const size_t s = 2 * Init::score(result);
    score->fetch_add(s);
    counter->fetch_add(1);

  }
}

void print_thread_fn(std::atomic<size_t> *score, std::atomic<size_t> *counter, bool* run) {
  while(*run) {
    sleep(10);
    std::cout << "score: " << score->load() << "; counter: " << counter->load() << " = " <<
    (score->load() / 2.0) / counter->load() << std::endl;
  }
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Input: net-path, threads, trials, iterations." << std::endl;
    return 1;
  }

  std::string net_path{argv[1]};
  const size_t threads = std::atoll(argv[2]);
  const size_t trials = std::atoll(argv[3]);
  const size_t iterations = std::atoll(argv[4]);

  std::atomic<size_t> score{};
  std::atomic<size_t> counter{};
  bool run_print = true;

  std::thread thread_pool[threads];
  for (auto t = 0; t < threads; ++t) {
    thread_pool[t] = std::thread{&thread_fn, &score, &counter, trials, iterations, net_path};
  }
  std::thread print_thread{&print_thread_fn, &score, &counter, &run_print};

  for (auto t = 0; t < threads; ++t) {
    thread_pool[t].join();
  }
  
  run_print = false;

  print_thread.join();

  return 0;
}
