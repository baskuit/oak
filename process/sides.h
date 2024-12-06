#pragma once

#include <data/moves.h>
#include <data/species.h>

#include <battle/init.h>

#include <process.h>

#include <map>
#include <optional>
#include <span>
#include <sstream>

namespace Process {

namespace Sides {

struct ManagedData {
  std::map<std::string, Init::Config> sides;
};

struct ManagerData {
  std::optional<std::string> cli_key;
  std::optional<size_t> cli_slot;
};

class Program : public ProgramBase<false, true> {
public:
  Program(std::ostream *out, std::ostream *err);

  ManagedData data;
  ManagerData mgmt;

  std::string prompt() const noexcept override;

  bool
  handle_command(const std::span<const std::string> words) noexcept override;

  bool save(std::filesystem::path) noexcept override;
  bool load(std::filesystem::path) noexcept override;

private:
  bool add(const std::string key) noexcept;
  bool rm(const std::string key) noexcept;

  bool cd(const std::span<const std::string> words) noexcept;
  bool cp(const std::span<const std::string> words) noexcept;

  bool set(const std::span<const std::string> words) noexcept;

  bool hp(const std::string);
  bool status(const std::string);

  void print() const noexcept;

  size_t depth() const noexcept;

  bool up() noexcept;
};

} // namespace Sides

} // namespace Process