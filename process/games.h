#pragma once

#include <data/moves.h>
#include <data/species.h>

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

struct Output {};

struct SearchOutputs {
  Output head;
  std::vector<Output> outputs;
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

  State(State &&other) : battle_data{other.battle_data}, node_data{} {}
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
  std::optional<size_t> cli_index;
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
      err("add: '", key, "' already present.");
      return false;
    }
    data.histories.emplace(key);
    return true;
  }

private:
  void print() const noexcept;

  size_t depth() const noexcept {
    if (mgmt.cli_search.has_value()) {
      return 4;
    } else {
      if (mgmt.cli_node.has_value()) {
        return 3;
      } else {
        if (mgmt.cli_index.has_value()) {
          return 2;
        } else {
          if (mgmt.cli_key.has_value()) {
            return 1;
          } else {
            return 0;
          }
        }
      }
    }
  }
};

} // namespace Games
} // namespace Process