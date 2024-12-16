#pragma once

#include <data/moves.h>
#include <data/species.h>

#include <battle/init.h>

#include <pi/eval.h>
#include <pi/exp3.h>
#include <pi/mcts.h>
#include <pi/tree.h>

#include <process.h>
#include <sides.h>

#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>

#include <log.h>

namespace Lab {

namespace Games {

struct SearchOutputs {
  MCTS::Output head;
  std::vector<MCTS::Output> tail;
};

using Node =
    Tree::Node<Exp3::JointBanditData<.03f, false>, std::array<uint8_t, 16>>;

struct State {
  pkmn_gen1_battle battle;
  pkmn_gen1_battle_options options;
  pkmn_result result;

  // interim info
  uint64_t seed;
  pkmn_choice c1;
  pkmn_choice c2;

  size_t m;
  size_t n;
  std::array<pkmn_choice, 9> choices1;
  std::array<pkmn_choice, 9> choices2;

  std::vector<SearchOutputs> outputs;
};

struct StateSearchData {
  std::vector<std::unique_ptr<Node>> nodes;
};

struct History {
  std::vector<State> states;
  Eval::CachedEval eval;
  DebugLog debug_log{};
};

using HistorySearchData = std::vector<StateSearchData>;

struct ManagedData {
  std::map<std::string, History> history_map;
  std::map<std::string, std::vector<StateSearchData>> search_data_map;
  Eval::OVODict ovo_dict;
};

struct ManagerData {

  struct Loc {
    size_t depth;
    std::string key;
    std::array<size_t, 3> current;

    bool operator<(const Loc &other) const {
      if (depth > other.depth) {
        return false;
      }
      if (key != other.key) {
        return false;
      }
      for (auto i = 0; i < depth - 1; ++i) {
        if (current[i] != other.current[i]) {
          return false;
        }
      }
      return true;
    }
  };

  Loc loc;
  std::array<size_t, 3> bounds;

  // thread access
  std::map<Loc, bool> locked_map;
  std::mutex mutex{};

  prng device{static_cast<uint64_t>(rand())};
  std::filesystem::path pkmn_debug_path{"./extern/engine/bin/pkmn-debug"};
};

class Program : public ProgramBase<true, true> {
public:
  using Base = ProgramBase<true, true>;
  using Base::Base;

  ManagedData data;
  ManagerData mgmt{};

  std::string prompt() const override;

  bool handle_command(const std::span<const std::string> words) override;

  bool save(std::filesystem::path) override;
  bool load(std::filesystem::path) override;

  bool create(const std::string key, const Init::Config p1,
              const Init::Config p2);

private:
  bool search(const std::span<const std::string> words);
  void loc() const;
  bool rollout();
  bool update(pkmn_choice c1, pkmn_choice c2);
  bool update(std::string c1, std::string c2);
  bool rm(const std::span<const std::string>);
  void print() const;
  bool cd(const std::span<const std::string>);
  bool up();
  bool next();
  bool prev();
  bool first();
  bool last();
  bool trunc();
  bool cp(const std::span<const std::string>);
  bool log() const;
  bool pkmn_debug(std::string key);
  bool battle_bytes() const;
  bool side_bytes() const;
  bool pokemon_bytes() const;

  bool search();

  History &history();
  State &state();
  StateSearchData &search_data();
  Node *node();
  SearchOutputs &search_outputs();
  MCTS::Output &output();

  const History &history() const;
  const State &state() const;
  const StateSearchData &search_data() const;
  const Node *node() const;
  const SearchOutputs &search_outputs() const;
  const MCTS::Output &output() const;

  bool lock_location(const ManagerData::Loc &loc) {
    std::unique_lock lock{mgmt.mutex};
    for (const auto [key, value] : mgmt.locked_map) {
      if ((key < loc) || (loc < key)) {
        return false;
      }
    }
    mgmt.locked_map[loc] = true;
    return true;
  }

  bool unlock_location(const ManagerData::Loc &loc) {
    std::unique_lock lock{mgmt.mutex};
    if (mgmt.locked_map.contains(loc)) {
      mgmt.locked_map.erase(loc);
      return true;
    }
    return false;
  }
};

} // namespace Games
} // namespace Lab