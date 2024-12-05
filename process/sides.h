#pragma once

#include <data/moves.h>
#include <data/species.h>

#include <process.h>

#include <map>
#include <optional>
#include <span>
#include <sstream>

namespace Process {

namespace Sides {

struct Set {
  Data::Species species;
  Data::OrderedMoveSet moves;
  uint8_t dur_sleep;
};

struct SideConfig {
  std::array<Set, 6> party;
  // View::ActivePokemon active;
};

struct ManagedData {
  std::map<std::string, SideConfig> sides;
};

struct ManagerData {
  std::optional<std::string> cli_key;
  std::optional<size_t> cli_slot;
};

class Program : public ProgramBase<false, true> {
public:
  using Base = ProgramBase<false, true>;
  using Base::Base;

  ManagedData data;
  ManagerData mgmt;

  std::string prompt() const noexcept override;

  bool
  handle_command(const std::span<const std::string> words) noexcept override;

  bool save(std::filesystem::path) noexcept override;
  bool load(std::filesystem::path) noexcept override;

  bool add(const std::string key) noexcept;
  bool rm(const std::string key) noexcept;

  bool in(const std::string l) noexcept {}
  bool out() noexcept {
    if (mgmt.cli_slot.has_value()) {
      mgmt.cli_slot = std::nullopt;
      return true;
    } else {
      if (mgmt.cli_key.has_value()) {
        mgmt.cli_key = std::nullopt;
        return true;
      } else {
        return false;
      }
    }
  }

  bool hp(const std::string);
  bool status(const std::string);
  bool set(const std::span<const std::string> words) noexcept;

private:
  void print() const noexcept;

  size_t depth() const noexcept {
    if (mgmt.cli_key.has_value()) {
      if (mgmt.cli_slot.has_value()) {
        return 2;
      } else {
        return 1;
      }
    } else {
      return 0;
    }
  }
};

} // namespace Sides

} // namespace Process