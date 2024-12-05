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

struct BattleInfo {
  // battle
  // result
  // options?
};

struct History {

  std::mutex mutex;
  std::vector<size_t> vec;
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
    data.histories[key] = {};
    return true;
  }
};

} // namespace Games
} // namespace Process