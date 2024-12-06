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

struct BattleData {
  pkmn_gen1_battle battle;
  pkmn_gen1_battle_options options;
  pkmn_result result;

  uint64_t seed;
  size_t m;
  size_t n;
  std::array<pkmn_choice, 9> choices1;
  std::array<pkmn_choice, 9> choices2;
};

struct SearchOutputs {
  MCTS::Output head;
  std::vector<MCTS::Output> tail;
};

using Node =
    Tree::Node<Exp3::JointBanditData<.03f, false>, std::array<uint8_t, 16>>;

struct State {

  BattleData battle_data;
  std::vector<SearchOutputs> outputs;

  std::vector<std::unique_ptr<Node>> nodes;
  std::mutex mutex;
};

struct History {
  std::vector<std::unique_ptr<State>> states;
  std::mutex mutex;
};

struct ManagedData {
  std::map<std::string, std::unique_ptr<History>> histories;
};

struct ManagerData {
  std::optional<std::string> cli_key;
  std::optional<size_t> cli_state;
  std::optional<size_t> cli_node;
  std::optional<size_t> cli_search;
  std::mutex mutex{};
};

class Program : public ProgramBase<true, true> {
public:
  using Base = ProgramBase<true, true>;
  using Base::Base;

  ManagedData data;
  ManagerData mgmt;

  std::string prompt() const noexcept override;

  bool
  handle_command(const std::span<const std::string> words) noexcept override;

  bool save(std::filesystem::path) noexcept override;
  bool load(std::filesystem::path) noexcept override;

  bool create(const std::string key, const Init::Config p1,
              const Init::Config p2);

private:
  bool rollout();
  bool update(pkmn_choice c1, pkmn_choice c2) noexcept;
  bool update(std::string c1, std::string c2) noexcept;
  bool rm(std::string key) noexcept;
  size_t size() const noexcept;
  void print() const noexcept;
  bool cd(const std::span<const std::string> words) noexcept;
  size_t depth() const noexcept;
  bool up() noexcept;
  bool next() noexcept;
  bool prev() noexcept;
  bool first() noexcept;
  bool last() noexcept;
  std::optional<size_t> &cli_current();

  History &history();
  State &state();
  Node &node();
  MCTS::Output &outputs();

  const History &history() const;
  const State &state() const;
  const Node &node() const;
  const MCTS::Output &outputs() const;
};

} // namespace Games
} // namespace Lab