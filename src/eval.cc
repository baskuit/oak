#include <data/options.h>

#include <battle/sample-teams.h>
#include <battle/strings.h>
#include <battle/view.h>

#include <pi/ovo-eval.h>

#include <util/print.h>
#include <util/random.h>

#include <iostream>
#include <thread>

static_assert(Options::calc && Options::chance && !Options::log);

template <typename Team, typename Dur>
void versus(std::atomic<int> *index, Eval::OVODict *eval_dict,  size_t max, Dur dur, uint64_t seed,
            size_t *n, size_t *score) {

  while (index->fetch_add(1) < max) {

    const auto half = [dur, eval_dict](auto side1, auto side2, auto seed) -> float {
      auto battle = Init::battle(side1, side2);
      pkmn_gen1_battle_options options{};
      pkmn_gen1_chance_durations durations{};
      auto result = Init::update(battle, 0, 0, options);

      MCTS search{};
      Eval::Model eval{seed, Eval::CachedEval{side1, side2, *eval_dict}};

      while (!pkmn_result_type(result)) {

        const auto [choices1, choices2] = Init::choices(battle, result);

        auto i = 0;
        auto j = 0;
        if ((choices1.size() > 1) || (choices2.size() > 1)) {
          using Node = Tree::Node<Exp3::JointBanditData<.03f, false>,
                                  std::array<uint8_t, 16>>;
          Node node{};
          Eval::Input input{battle, durations, 
            Eval::Abstract{battle, eval.eval.ovo_matrix}, result};
          auto output =
              search.run<MCTS::Options<true, 3, 3, 1>>(dur, node, input, eval);
          i = eval.device.sample_pdf(output.p1);
          j = eval.device.sample_pdf(output.p2);
        }
        result = Init::update(battle, choices1[i], choices2[j], options);
        durations = *pkmn_gen1_battle_options_chance_durations(&options);
      }
      const auto score = Init::score(result);
      return score;
    };

    prng device{seed};
    const auto n_teams = SampleTeams::teams.size();
    const auto p1 = SampleTeams::teams[device.random_int(n_teams)];
    const auto p2 = SampleTeams::teams[device.random_int(n_teams)];
    *score += 2 * half(p1, p2, device.uniform_64());
    *n += 1;
  }
}

int generate_ovo_games(int argc, char **argv) {

  using Team = std::array<Init::Set, 6>;

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

  std::cout << "Input - " << "ms: " <<  ms << " threads: " << threads << " max: " << max << " seed: " << seed
            << std::endl;

  std::atomic<int> index{};
  prng device{seed};


  size_t n = 0;
  size_t score = 0;

  std::thread thread_pool[threads];

  Eval::OVODict eval_dict{};
  if(!eval_dict.load("./cache")){
    std::cerr << "could not load cache" << std::endl;
    return 1;
  }

  for (auto t = 0; t < threads; ++t) {
    thread_pool[t] = std::thread{&versus<Team, std::chrono::milliseconds>,
                                 &index,
                                 &eval_dict,
                                 max,
                                 std::chrono::milliseconds{ms},
                                 device.uniform_64(),
                                 &n,
                                 &score};
  }

  // TODO period save to disc thread
  // TODO ctrl+c cleanup

  for (auto t = 0; t < threads; ++t) {
    thread_pool[t].join();
  }

  return 0;
}

int main(int argc, char **argv) { return generate_ovo_games(argc, argv); }
