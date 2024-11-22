#include <battle/debug-log.h>
#include <battle/init.h>
#include <data/options.h>
#include <data/sample-teams.h>
#include <data/strings.h>
#include <iostream>
#include <pi/abstract.h>
#include <pi/eval.h>
#include <util/random.h>

static_assert(Options::calc && Options::chance && !Options::log);

int abstract_test(int argc, char **argv) {

  prng device{std::atoi(argv[1])};

  const auto p1 = SampleTeams::teams[0];
  const auto p2 = SampleTeams::teams[1];
  auto battle = Init::battle(p1, p2);
  MCTS<false, true> search{};
  auto result = Init::update(battle, 0, 0, search.options);
  Abstract::Battle abstract{battle};

  Eval::GlobalMEM global{};
  const bool cache_loaded = global.load("./global");
  std::cout << "Global load : " << cache_loaded << std::endl;
  Eval::CachedEval model{p1, p2, global};

  while (!pkmn_choice_type(result)) {

    float value = model.value(abstract);
    std::cout << "value: " << value << std::endl;

    const auto [choices1, choices2] = Init::choices(battle, result);

    auto i = device.random_int(choices1.size());
    auto j = device.random_int(choices2.size());

    result = Init::update(battle, choices1[i], choices2[j], search.options);
    abstract.update(battle);
  }

  float payoff;
  switch (pkmn_result_type(result)) {
  case PKMN_RESULT_WIN: {
    payoff = 1.0;
    break;
  }
  case PKMN_RESULT_LOSE: {
    payoff = 0.0;
    break;
  }
  case PKMN_RESULT_TIE: {
    payoff = 0.5;
    break;
  }
  default: {
    assert(false);
    payoff = 0.5;
  }
  };

  std::cout << "payoff: " << payoff << std::endl;

  return 0;
}

int main(int argc, char **argv) { return abstract_test(argc, argv); }
