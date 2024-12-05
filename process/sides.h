#pragma once

#include <data/moves.h>

#include <battle/init.h>
#include <battle/strings.h>

#include <pi/eval.h>
#include <pi/mcts.h>
#include <pi/tree.h>

#include <process.h>

#include <pkmn.h>

#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Process {

namespace Sides {

struct Set {
  Data::Species species;
  Data::OrderedMoveSet moves;
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

  bool save(std::filesystem::path) noexcept override { return false; }
  bool load(std::filesystem::path) noexcept override { return false; }
};

} // namespace Sides

} // namespace Process