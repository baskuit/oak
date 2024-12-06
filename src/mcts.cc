#include <util/random.h>

#include <data/options.h>
#include <data/strings.h>

#include <battle/sample-teams.h>

#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/tree.h>

#include <iostream>
#include <numeric>
#include <sstream>

static_assert(Options::calc && Options::chance && !Options::log);

namespace Sets {
struct SetCompare {
  constexpr bool operator()(Init::Set a, Init::Set b) const {
    if (a.species == b.species) {
      return a.moves < b.moves;
    } else {
      return a.species < b.species;
    }
  }
};

auto get_sorted_set_array() {
  std::map<Init::Set, size_t, SetCompare> map{};
  for (const auto &team : SampleTeams::teams) {
    for (const auto &set : team) {
      auto set_sorted = set;
      std::sort(set_sorted.moves.begin(), set_sorted.moves.end());
      ++map[set_sorted];
    }
  }
  std::vector<std::pair<Init::Set, size_t>> sets_sorted_by_use{};
  for (const auto &[set, n] : map) {
    sets_sorted_by_use.emplace_back(set, n);
  }
  std::sort(sets_sorted_by_use.begin(), sets_sorted_by_use.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });
  return sets_sorted_by_use;
}

std::string set_string(auto set) {
  std::stringstream stream{};
  stream << Names::species_string(set.species) << " { ";
  for (const auto move : set.moves) {
    stream << Names::move_string(move) << ", ";
  }
  stream << "}";
  stream.flush();
  return stream.str();
}
} // namespace Sets

void print_output(const auto &battle_data, const auto &output) {

  const auto [p1_choices, p2_choices] =
      Init::choices(battle_data.battle, battle_data.result);
  std::cout << "P1 Policy:" << std::endl;
  const auto m = output.p1.size();
  for (auto i = 0; i < m; ++i) {
    std::cout << side_choice_string(battle_data.battle.bytes, p1_choices[i])
              << " : " << output.p1[i] << std::endl;
  }
  std::cout << "P2 Policy:" << std::endl;
  const auto n = output.p2.size();
  for (auto i = 0; i < n; ++i) {
    std::cout << side_choice_string(battle_data.battle.bytes + Offsets::side,
                                    p2_choices[i])
              << " : " << output.p2[i] << std::endl;
  }
  std::cout << "Value: " << output.average_value << std::endl;

  std::cout << "Visits Matrix:" << std::endl;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << output.visit_matrix[i][j] << "\t";
    }
    std::cout << std::endl;
  }
}

int print1v1(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "Usage: provide seed, two set indices, and mcts iterations"
              << std::endl;
    return 1;
  }

  using Obs = std::array<uint8_t, 16>;
  using Node = Tree::Node<Exp3::JointBanditData<.03f, false>, Obs>;

  const uint64_t seed = std::atoi(argv[1]);
  const int i = std::atoi(argv[2]);
  const int j = std::atoi(argv[3]);
  const size_t iterations = std::atoi(argv[4]);

  prng device{seed};

  // sorting, printing the sets take from sample teams
  const auto sorted_set_array = Sets::get_sorted_set_array();
  std::vector<Init::Set> sets{};
  for (const auto &pair : sorted_set_array) {
    const auto &set = pair.first;
    sets.emplace_back(set);
  }

  const auto set_a = sets[i];
  const auto set_b = sets[j];
  const auto set_a_str = Sets::set_string(set_a);
  const auto set_b_str = Sets::set_string(set_b);

  std::cout << set_a_str << " vs " << set_b_str << std::endl;

  MonteCarlo::Input battle_data;
  battle_data.battle = Init::battle(std::vector<Init::Set>{set_a},
                                    std::vector<Init::Set>{set_b});
  MonteCarlo::Model model;
  model.device = prng{device.uniform_64()};
  MCTS search{};
  battle_data.result = Init::update(battle_data.battle, 0, 0, search.options);
  Node node{};
  pkmn_gen1_chance_durations durations{};

  const auto output = search.run(iterations, node, battle_data, model);

  print_output(battle_data, output);

  return 0;
}

int main(int argc, char **argv) { return print1v1(argc, argv); }
