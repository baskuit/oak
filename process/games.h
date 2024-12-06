#pragma once

#include <data/moves.h>
#include <data/species.h>

#include <pi/mcts.h>

#include <process.h>

#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>

namespace Process {

namespace Games {

struct BattleData {
  // battle
  // result
  // options
  // seed/obs?
};

struct SearchOutputs {
  MCTS::Output head;
  std::vector<MCTS::Output> outputs;
};

using Node = std::map<int, std::unique_ptr<int>>;

struct NodeData {

  Node node;
  SearchOutputs outputs;
};

struct State {

  BattleData battle_data;
  std::vector<SearchOutputs> output_data;

  std::vector<std::unique_ptr<Node>> node_data;
  std::mutex mutex;

  // State(State &&other) : battle_data{other.battle_data}, node_data{} {}
};

struct History {
  std::vector<std::unique_ptr<State>> histories;
  std::mutex mutex;
};

struct ManagedData {
  std::map<std::string, History> histories;
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

  bool create(const std::string key, const auto p1, const auto p2) {
    std::unique_lock lock{mgmt.mutex};
    if (data.histories.contains(key)) {
      err("create: '", key, "' already present.");
      return false;
    }

    // SideInitializer p1;
    // SideInitializer p2;

    // for (const auto set : )

    data.histories[key];
    return true;
  }

private:
  size_t size() const noexcept;
  void print() const noexcept;
  bool cd(const std::span<const std::string> words) noexcept;
  size_t depth() const noexcept;
  bool up() noexcept;
};

} // namespace Games
} // namespace Process