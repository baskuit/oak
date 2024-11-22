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

int init_1v1() {}

int abstract_test(int argc, char **argv) {
  // if (argc != 3) {
  //   std::cout
  //       << "Usage: provide seed and tries for Abstract::Battle update test"
  //       << std::endl;
  //   return 1;
  // }
  // const uint64_t seed = std::atoi(argv[1]);
  // const size_t tries = std::atoi(argv[2]);
  // prng device{seed};
  // for (size_t t = 0; t < tries; ++t) {

  //   const int i = device.random_int(SampleTeams::teams.size());
  //   const int j = device.random_int(SampleTeams::teams.size());

  //   auto battle = Init::battle(SampleTeams::teams[i], SampleTeams::teams[j]);
  //   pkmn_gen1_battle_options options;
  //   auto result = Init::update(battle, 0, 0, options);
  //   Abstract::Battle abstract{battle};

  //   int turns = 0;
  //   while (!pkmn_result_type(result)) {
  //     const auto [choices1, choices2] = Init::choices(battle, result);
  //     auto c1 = choices1[device.random_int(choices1.size())];
  //     auto c2 = choices2[device.random_int(choices2.size())];
  //     result = Init::update(battle, c1, c2, options);

  //     abstract.update(battle);
  //     Abstract::Battle new_abstract{battle};
  //     assert(abstract == new_abstract);
  //     ++turns;
  //   }
  //   // std::cout << turns << std::endl;
  // }

  Eval::Pokemon lax{
      Species::Snorlax,
      {Moves::BodySlam, Moves::Rest, Moves::Earthquake, Moves::Reflect},
      .1};
  Eval::Pokemon chansey{
      Species::Chansey,
      {Moves::SeismicToss, Moves::Rest, Moves::Earthquake, Moves::Reflect},
      1};

  // auto mem = Eval::compute_table(SampleTeams::teams[0][0],
  //                                SampleTeams::teams[0][1], 123123, 1 << 16);
  // Eval::print_mem(mem);

  Eval::GlobalMEM global{};

  // auto mem = global(lax, chansey);

  // global.save("./global");
  global.load("./global");

  auto mem = global(lax, chansey);
  Eval::print_mem(mem);

  return 0;
}

int main(int argc, char **argv) { return abstract_test(argc, argv); }
