#include <data/sample-teams.h>
#include <data/strings.h>

#include <battle/init.h>

#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/tree.h>

#include <util/random.h>

#include <iostream>
#include <numeric>
#include <sstream>

#include <data/options.h>

static_assert(Options::calc && Options::chance && !Options::log);

namespace Sets {
struct SetCompare {
  constexpr bool operator()(SampleTeams::Set a, SampleTeams::Set b) const {
    if (a.species == b.species) {
      return a.moves < b.moves;
    } else {
      return a.species < b.species;
    }
  }
};

auto get_sorted_set_array() {
  std::map<SampleTeams::Set, size_t, SetCompare> map{};
  for (const auto &team : SampleTeams::teams) {
    for (const auto &set : team) {
      auto set_sorted = set;
      std::sort(set_sorted.moves.begin(), set_sorted.moves.end());
      ++map[set_sorted];
    }
  }
  std::vector<std::pair<SampleTeams::Set, size_t>> sets_sorted_by_use{};
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

struct Types {
  using Obs = std::array<uint8_t, 16>;
  using Node = Tree::Node<Exp3::JointBanditData<false>, Obs>;
};

int all_1v1(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "Usage: provide seed, two set indices, and mcts iterations"
              << std::endl;
    return 1;
  }
  const uint64_t seed = std::atoi(argv[1]);
  const int i = std::atoi(argv[2]);
  const int j = std::atoi(argv[3]);
  const size_t iterations = std::atoi(argv[4]);

  prng device{seed};

  // sorting, printing the sets take from sample teams
  const auto sorted_set_array = Sets::get_sorted_set_array();
  std::vector<SampleTeams::Set> sets{};
  for (const auto &pair : sorted_set_array) {
    const auto &set = pair.first;
    sets.emplace_back(set);
  }

  const auto set_a = sets[i];
  const auto set_b = sets[j];
  const auto set_a_str = Sets::set_string(set_a);
  const auto set_b_str = Sets::set_string(set_b);

  std::cout << set_a_str << " vs " << set_b_str << std::endl;

  auto battle = Init::battle(std::vector<SampleTeams::Set>{set_a},
                             std::vector<SampleTeams::Set>{set_b});
  constexpr bool debug_print = false;
  MCTS<debug_print> search{};
  auto result = pkmn_gen1_battle_update(&battle, 0, 0, &search.options);
  Types::Node node{};
  pkmn_gen1_chance_durations durations{};

  const auto output =
      search.run(iterations, device, node, &battle, result, &durations);
  const auto [p1_choices, p2_choices] = Init::choices(&battle, result);
  std::cout << "P1 Policy:" << std::endl;
  const auto m = output.p1.size();
  for (auto i = 0; i < m; ++i) {
    std::cout << side_choice_string(battle.bytes, p1_choices[i]) << " : "
              << output.p1[i] << std::endl;
  }
  std::cout << "P2 Policy:" << std::endl;
  const auto n = output.p2.size();
  for (auto i = 0; i < n; ++i) {
    std::cout << side_choice_string(battle.bytes + Offsets::side, p2_choices[i])
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

  return 0;
}

int main(int argc, char **argv) { return all_1v1(argc, argv); }
