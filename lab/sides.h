#pragma once

#include <data/moves.h>
#include <data/species.h>

#include <battle/init.h>

#include <process.h>

#include <map>
#include <optional>
#include <span>
#include <sstream>

namespace Lab {

namespace Sides {

struct ManagedData {
  std::map<std::string, Init::Config> sides;
};

struct ManagerData {
  std::optional<std::string> key;
  std::optional<size_t> slot;
};

class Program : public ProgramBase<false, true> {
public:
  Program(std::ostream *out, std::ostream *err);

  ManagedData data;
  ManagerData mgmt;

  std::string prompt() const override;

  bool handle_command(const std::span<const std::string> words) override;

  bool save(std::filesystem::path) override;
  bool load(std::filesystem::path) override;

private:
  bool add(const std::string key);
  bool rm(const std::string key);

  bool cd(const std::span<const std::string> words);
  bool cp(const std::span<const std::string> words);

  bool set(const std::span<const std::string> words);

  bool hp(const std::string);
  bool status(const std::string);

  void print() const;

  size_t depth() const;

  bool up();
};

} // namespace Sides

} // namespace Lab