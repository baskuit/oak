#pragma once

#include <data/moves.h>
#include <data/species.h>

#include <process.h>

#include <optional>
#include <map>
#include <span>
#include <mutex>
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

};

struct ManagedData {
  std::map<std::string, History> sides;
};

struct ManagerData {
  std::optional<std::string> cli_key;
  std::optional<size_t> cli_slot;
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
};
}
}