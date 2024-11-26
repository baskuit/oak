#include <battle/debug-log.h>
#include <battle/init.h>
#include <data/durations.h>
#include <data/options.h>
#include <data/sample-teams.h>
#include <data/strings.h>
#include <iostream>
#include <pi/abstract.h>
#include <pi/eval.h>
#include <util/print.h>
#include <util/random.h>

#include <thread>

static_assert(Options::calc && Options::chance && !Options::log);

struct PokeModel {
  prng device;
  struct Eval {
    float value(const pkmn_gen1_battle &battle) const {
      return PokeEngine::evaluate_battle(battle);
    }
  };
  Eval eval{};
};

template <typename Team, typename Dur>
void versus(const Team p1, const Team p2, size_t trials, Dur dur, uint64_t seed,
            size_t *n, size_t *score) {

  using Obs = std::array<uint8_t, 16>;

  const auto half = [dur](auto x, auto y, auto seed) -> float {
    auto battle = Init::battle(x, y);
    pkmn_gen1_chance_durations durations{};
    pkmn_gen1_battle_options options{};
    auto result = Init::update(battle, 0, 0, options);

    Eval::GlobalMEM global{};
    global.load("./global");

    MCTS search{};
    MonteCarlo::Model mcm{seed};
    Eval::Model eval{mcm.device.uniform_64(), Eval::CachedEval{x, y, global}};
    PokeModel poke_eval{mcm.device.uniform_64()};

    while (!pkmn_result_type(result)) {

      const auto [choices1, choices2] = Init::choices(battle, result);

      auto i = 0;
      if (choices1.size() > 1) {
        using Node = Tree::Node<Exp3::JointBanditData<.15f, false>, Obs>;
        Node node{};
        MonteCarlo::Input mc_input{battle, durations, result};
        auto mc_output = search.run(dur, node, mc_input, mcm);
        // print(mc_output.p1);
        i = mcm.device.sample_pdf(mc_output.p1);
      }

      auto j = 0;
      if (choices2.size() > 1) {
        using Node = Tree::Node<Exp3::JointBanditData<.03f, false>, Obs>;
        Node node{};
        Eval::Input eval_input{battle, durations, battle, result};
        auto eval_output = search.run(dur, node, eval_input, poke_eval);
        // print(eval_output.p2);
        j = eval.device.sample_pdf(eval_output.p2);
      }

      result = Init::update(battle, choices1[i], choices2[j], options);
      durations = *pkmn_gen1_battle_options_chance_durations(&options);

      // std::cout << side_choice_string(battle.bytes, choices1[i]) << ", "
      //           << side_choice_string(battle.bytes + Offsets::side,
      //           choices2[i])
      //           << std::endl;
    }
    return Init::score(result);
  };

  prng device{seed};

  float mc_total = 0;
  for (auto i = 0; i < trials; ++i) {
    *score += 2 * half(p1, p2, device.uniform_64());
    *n += 1;
    *score += 2 * half(p2, p1, device.uniform_64());
    *n += 1;
  }
}

void print_score(bool *flag, size_t *n, size_t *score) {
  while (!*flag) {
    float n_ = static_cast<float>(*n) + 1.0f * (*n == 0);
    float avg = static_cast<float>(*score) / 2 / n_;
    std::cout << "n: " << *n << " avg: " << avg << std::endl;
    sleep(10);
  }
}

int abstract_test(int argc, char **argv) {

  using Team = std::array<SampleTeams::Set, 6>;

  size_t ms = 100;
  size_t threads = 1;
  size_t trials = 5;
  uint64_t seed = 9028342938;

  if (argc != 5) {
    std::cerr << "Usage: ms, threads, trials, seed" << std::endl;
    return 1;
  }

  ms = std::atoi(argv[1]);
  threads = std::atoi(argv[2]);
  trials = std::atoi(argv[3]);
  seed = std::atoi(argv[4]);

  std::cout << "Input: " << ms << ' ' << threads << ' ' << trials << ' ' << seed
            << std::endl;

  prng device{seed};

  const auto p1 = SampleTeams::teams[0];
  const auto p2 = SampleTeams::teams[1];

  size_t n = 0;
  size_t score = 0;

  std::thread thread_pool[threads];

  for (auto t = 0; t < threads; ++t) {
    thread_pool[t] = std::thread{&versus<Team, std::chrono::milliseconds>,
                                 p1,
                                 p2,
                                 trials,
                                 std::chrono::milliseconds{ms},
                                 device.uniform_64(),
                                 &n,
                                 &score};
  }

  bool flag = false;

  std::thread print_thread{&print_score, &flag, &n, &score};

  for (auto t = 0; t < threads; ++t) {
    thread_pool[t].join();
  }

  flag = true;
  print_thread.join();

  return 0;
}

int main(int argc, char **argv) { return abstract_test(argc, argv); }
