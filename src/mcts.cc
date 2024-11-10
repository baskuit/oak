#include <data/sample-teams.h>
#include <data/strings.h>

#include <battle/init.h>

#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/tree.h>

#include <types/random.h>

#include <iostream>
#include <numeric>
#include <sstream>

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
  using Node = Tree::Node<Exp3::JointBanditData, Obs>;
};

int all_1v1(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: provide seed, mcts iterations" << std::endl;
    return 1;
  }
  const uint64_t seed = std::atoi(argv[1]);
  const size_t iterations = std::atoi(argv[2]);
  prng device{seed};

  std::cout << "Calculating 1v1 for all sets found in SampleTeams::teams, "
               "ordered by frequency"
            << std::endl;
  std::cout << "MCTS ITERATIONS: " << iterations << '\n' << std::endl;

  // sorting, printing the sets take from sample teams
  const auto sorted_set_array = Sets::get_sorted_set_array();
  std::vector<SampleTeams::Set> sets{};
  for (const auto &pair : sorted_set_array) {
    const auto &set = pair.first;
    sets.emplace_back(set);
  }

  std::cout << "total sets: " << sets.size() << '\n' << std::endl;

  const auto top = sets.begin() + 10;

  // iterate through all pairs and search the 1v1
  const auto n = sets.size();
  for (auto i = sets.begin(); i != top; ++i) {
    const auto set_a = *i;
    const auto set_a_str = Sets::set_string(set_a);
    for (auto j = i + 1; j != top; ++j) {
      const auto set_b = *j;
      const auto set_b_str = Sets::set_string(set_b);

      auto battle = Init::battle(std::vector<SampleTeams::Set>{set_a},
                                 std::vector<SampleTeams::Set>{set_b});
      MCTS<true> search{};
      auto result = pkmn_gen1_battle_update(&battle, 0, 0, &search.options);
      Types::Node node{};
      pkmn_gen1_chance_durations durations{};
    
      search.run(iterations, device, node, &battle, result, &durations);
    
      return 0;
    }
  }
  return 0;
}

int main(int argc, char **argv) { return all_1v1(argc, argv); }
