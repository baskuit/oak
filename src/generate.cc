#include <util/random.h>

#include <data/options.h>
#include <data/sample-teams.h>
#include <data/strings.h>

#include <battle/init.h>

#include <pi/eval.h>
#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/tree.h>

#include <atomic>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

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

void thread_fn(std::atomic<int> *const atomic,
               const std::vector<SampleTeams::Set> *sets, Eval::OVODict *dict) {
  const auto n = sets->size();
  const auto max = n * (n - 1) / 2;
  auto index = 0;
  while ((index = atomic->fetch_add(1)) < max) {
    // seek manually lol
    auto k = 0;
    for (auto i = 0; i < n; ++i) {
      for (auto j = i + 1; j < n; ++i) {
        if (k == index) {
          auto s1 = (*sets)[i];
          auto s2 = (*sets)[j];
          dict->add(s1, s2);
          dict->save("./cache");
          std::cout << Sets::set_string(s1) << " vs " << Sets::set_string(s2)
                    << std::endl;
          break;
        }
        ++k;
      }
    }
  }
}

int generate(int argc, char **argv) {
  
  size_t threads = 2;
  size_t exp = 16;
  if (argc >= 2) {
    threads = std::atoi(argv[1]);
  }

  if (argc >= 3) {
    exp = std::atoi(argv[2]);
  }

  std::cout << "Usage: ./generate threasd exp (2^exp search iterations)" << std::endl;

  const auto sorted_set_array = Sets::get_sorted_set_array();
  std::vector<SampleTeams::Set> sets{};
  for (const auto &pair : sorted_set_array) {
    const auto &set = pair.first;
    sets.emplace_back(set);
  }

  std::cout << "Number of sets:" << sets.size() << std::endl;

  auto *thread_pool = new std::thread[threads];

  std::atomic<int> index{};

  Eval::OVODict global{};
  global.iterations = 1 << exp;

  for (auto i = 0; i < threads; ++i) {
    thread_pool[i] = std::thread{&thread_fn, &index, &sets, &global};
  }
  for (auto i = 0; i < threads; ++i) {
    thread_pool[i].join();
  }

  return 0;
}

int main(int argc, char **argv) { return generate(argc, argv); }
