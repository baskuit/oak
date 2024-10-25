#include <battle/battle.h>
#include <battle/chance.h>
#include <battle/util.h>

#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/pgame.h>
#include <pi/tree.h>
#include <pi/abstract.h>

#include <model/monte-carlo-model.h>

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
  using State = Battle<0, true, true>;
  using Model = MonteCarloModel<prng, State, 16>;
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

  // iterate through all pairs and search the 1v1
  const auto n = sets.size();
  for (auto i = 0; i < n; ++i) {
    const auto set_a = sets[i];
    const auto set_a_str = Sets::set_string(set_a);
    for (auto j = i + 1; j < n; ++j) {
      const auto set_b = sets[j];
      const auto set_b_str = Sets::set_string(set_b);
      Types::State battle{std::vector<SampleTeams::Set>{set_a},
                          std::vector<SampleTeams::Set>{set_b}};
      battle.apply_actions(0, 0);
      battle.get_actions();
      Types::Model model{device.uniform_64()};
      Types::Node node{};
      MCTS mcts{};
      std::cout << set_a_str << " vs " << set_b_str << std::endl;
      pkmn_result result{};
      pkmn_gen1_chance_durations durations{};
      mcts.run(iterations, device, node, &battle.battle(), result, &durations);

      // for (int i = 0; i < battle.rows(); ++i) {
      //   std::cout << side_choice_string(battle.battle().bytes,
      //                                   battle.row_actions[i])
      //             << " : " << output.row_strategy[i] << ", ";
      // }
      // std::cout << std::endl;

      // for (int i = 0; i < battle.cols(); ++i) {
      //   std::cout << side_choice_string(battle.battle().bytes + 184,
      //                                   battle.col_actions[i])
      //             << " : " << output.col_strategy[i] << ", ";
      // }
      // std::cout << std::endl;
      // std::cout << "average value: " << output.average_value
      //           << " rolling average: " << output.rolling_average_value
      //           << std::endl;
      // std::cout << "average depth: " << output.average_depth << std::endl;
      return 0;
    }
  }
  return 0;
}

int main(int argc, char **argv) { return all_1v1(argc, argv); }
