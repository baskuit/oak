#include <util/random.h>

#include <data/options.h>
#include <data/strings.h>

#include <battle/sample-teams.h>

#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/ovo-eval.h>
#include <pi/tree.h>

#include <atomic>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

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
} // namespace Sets

void thread_fn(std::atomic<int> *const atomic,
               const std::vector<Init::Set> *sets, Eval::OVODict *dict,
               std::mutex *mutex) {
  const auto n = sets->size();
  const auto seek = [sets, dict, n, mutex](const auto index) {
    auto k = 0;
    for (auto i = 0; i < n; ++i) {
      for (auto j = i; j < n; ++j) {
        if (k == index) {
          auto s1 = (*sets)[i];
          auto s2 = (*sets)[j];
          dict->get(s1, s2);
          dict->save("./cache");
          {
            std::unique_lock lock{*mutex};
            std::cout << index << " " << i << " " << j << std::endl;
            std::cout << set_string(s1) << " vs " << set_string(s2)
                      << std::endl;
            std::cout << "value: " << dict->get(s1, s2)[2][0][2][0]
                      << std::endl;
          }
          return true;
        }
        ++k;
      }
    }
    return false;
  };

  while (seek(atomic->fetch_add(1))) {
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

  std::cout << "Usage: './generate n exp'\n Generate OVODict using n threads "
               "and 2^exp search iterations"
            << std::endl;

  const auto sorted_set_array = Sets::get_sorted_set_array();
  std::vector<Init::Set> sets{};
  for (const auto &pair : sorted_set_array) {
    const auto &set = pair.first;
    sets.emplace_back(set);
  }

  std::cout << "Number of sets:" << sets.size() << std::endl;

  std::atomic<int> index{};
  std::mutex write{};
  prng device{12231256};

  Eval::OVODict global{};
  global.iterations = 1 << exp;

  // global.load("/home/user/oak/cache");

  auto *thread_pool = new std::thread[threads];

  for (auto i = 0; i < threads; ++i) {
    thread_pool[i] = std::thread{&thread_fn, &index, &sets, &global, &write};
  }
  for (auto i = 0; i < threads; ++i) {
    thread_pool[i].join();
  }

  delete[] thread_pool;

  global.save("./cache");
  global.save("./test");
  Eval::OVODict test{};
  test.load("./test");

  for (const auto &pair : global.OVOData) {
    assert(test.OVOData[pair.first] == pair.second);
    if (test.OVOData[pair.first] != pair.second) {
      std::cout << '!' << std::endl;
      return 1;
    }
  }

  return 0;
}

int main(int argc, char **argv) { return generate(argc, argv); }
