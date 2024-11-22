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
  const auto p1 = SampleTeams::teams[0];
  const auto p2 = SampleTeams::teams[0];
  auto battle = Init::battle(p1, p2);
  MCTS<false, true> search{};
  auto result = Init::update(battle, 0, 0, search.options);
  Abstract::Battle abstract{battle};

  Eval::GlobalMEM global{};
  const bool cache_loaded = global.load("./global");
  std::cout << "Global load : " << cache_loaded << std::endl;
  Eval::CachedEval model{p1, p2, global};

  return 0;
}

int main(int argc, char **argv) { return abstract_test(argc, argv); }
