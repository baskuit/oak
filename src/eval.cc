#include <data/durations.h>
#include <data/options.h>
#include <data/sample-teams.h>

#include <battle/init.h>
#include <battle/strings.h>
#include <battle/view.h>

#include <pi/eval.h>

#include <util/print.h>
#include <util/random.h>

#include <iostream>
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
void versus(std::atomic<int> *index, size_t max, Dur dur, uint64_t seed,
            size_t *n, size_t *score) {

  while (index->fetch_add(1) < max) {

    const auto half = [dur](auto x, auto y, auto seed) -> float {
      auto battle = Init::battle(x, y);
      pkmn_gen1_chance_durations durations{};
      pkmn_gen1_battle_options options{};
      auto result = Init::update(battle, 0, 0, options);

      Eval::OVODict global{};
      global.load("./cache");

      MCTS search{};
      MonteCarlo::Model mcm{seed};
      Eval::Model eval{mcm.device.uniform_64(), Eval::CachedEval{x, y, global}};
      PokeModel poke_eval{mcm.device.uniform_64()};

      std::vector<float> eval_values{};

      while (!pkmn_result_type(result)) {

        const auto [choices1, choices2] = Init::choices(battle, result);

        auto i = 0;
        if (choices1.size() > 1) {
          using Node = Tree::Node<Exp3::JointBanditData<.03f, false>,
                                  std::array<uint8_t, 16>>;
          Node node{};
          MonteCarlo::Input input1{battle, durations, result};
          auto output1 =
              search.run<true, false, false, true>(dur, node, input1, mcm);
          i = mcm.device.sample_pdf(output1.p1);
        }

        auto j = 0;
        if (choices2.size() > 1) {
          using Node = Tree::Node<Exp3::JointBanditData<.03f, false>,
                                  std::array<uint8_t, 16>>;
          Node node{};
          Eval::Input input2{battle, durations,
                             Eval::Abstract{battle, eval.eval.ovo_matrix},
                             result};
          // MonteCarlo::Input input2{battle, durations, result};
          auto output2 =
              search.run<true, false, true, false>(dur, node, input2, eval);
          j = eval.device.sample_pdf(output2.p2);
          // input2.abstract.print();
          const auto v = eval.eval.value(input2.abstract);
          std::cout << v << std::endl;
          // std::cout << "iter: " << output2.iterations << std::endl;
          eval_values.push_back(v);
        } else {
          std::cout << "-" << std::endl;
        }

        std::cout << Strings::battle_to_string(battle) << std::endl;

        result = Init::update(battle, choices1[i], choices2[j], options);
        durations = *pkmn_gen1_battle_options_chance_durations(&options);
      }

      for (auto it = eval_values.end() - 5; it < eval_values.end(); ++it) {
        std::cout << *it << ' ';
      }
      const auto score = Init::score(result);
      std::cout << "~ " << score << std::endl;

      return score;
    };

    prng device{seed};

    const auto p1 = SampleTeams::teams[device.random_int(100)];
    const auto p2 = SampleTeams::teams[device.random_int(100)];

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
  size_t threads = 3;
  size_t max = 1000;
  uint64_t seed = 2934828342938;

  if (argc != 5) {
    std::cerr << "Input: ms, threads, max games, seed" << std::endl;
    std::cerr
        << "Compares Monte Carlo to Eval::Cached Eval with sample team games"
        << std::endl;
    return 1;
  }

  ms = std::atoi(argv[1]);
  threads = std::atoi(argv[2]);
  max = std::atoi(argv[3]);
  seed = std::atoi(argv[4]);

  std::cout << "Input: " << ms << ' ' << threads << ' ' << max << ' ' << seed
            << std::endl;

  std::atomic<int> index{};
  prng device{seed};

  size_t n = 0;
  size_t score = 0;

  std::thread thread_pool[threads];

  for (auto t = 0; t < threads; ++t) {
    thread_pool[t] = std::thread{&versus<Team, std::chrono::milliseconds>,
                                 &index,
                                 max,
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
