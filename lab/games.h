#pragma once

#include <data/moves.h>
#include <data/species.h>

#include <battle/init.h>

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

using History = std::vector<State>;
using HistorySearchData = std::vector<StateSearchData>;

struct ManagedData {
  std::map<std::string, std::vector<State>> history_map;
  std::map<std::string, std::vector<StateSearchData>> search_data_map;
};

struct ManagerData {

  struct Loc {
    std::string key;
    size_t depth;
    std::array<size_t, 3> current;
  };

  // home = nullopt, 0, {0, 0, 0}
  // bottom = "key", 3, {1, 1, 1}

  Loc loc;
  std::array<size_t, 3> bounds;

  // thread access
  std::map<Loc, bool> locked_map;
  std::mutex mutex{};
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
  bool rm(std::string key);
  void print() const;
  bool cd(const std::span<const std::string> words);
  bool up();
  bool next();
  bool prev();
  bool first();
  bool last();

  bool battle_bytes() const;
  bool side_bytes() const;
  bool pokemon_bytes() const;

  bool search();

  History &history();
  State &state();
  StateSearchData &search_data();
  Node &node();
  SearchOutputs &search_outputs();
  MCTS::Output &output();

  const History &history() const;
  const State &state() const;
  const StateSearchData &search_data() const;
  const Node &node() const;
  const SearchOutputs &search_outputs() const;
  const MCTS::Output &output() const;
};

} // namespace Games
} // namespace Lab