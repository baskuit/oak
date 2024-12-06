#include <sides.h>

#include <util/fs.h>

#include <data/strings.h>

namespace Process {
namespace Sides {

std::string Program::prompt() const noexcept {
  std::string p{"sides"};
  if (mgmt.cli_key.has_value()) {
    p += "/" + mgmt.cli_key.value();
  }
  if (mgmt.cli_slot.has_value()) {
    const auto slot = mgmt.cli_slot.value();
    p += "/" + (slot == 0 ? "Active" : std::to_string(slot));
  }
  p += "$ ";
  return p;
}

bool Program::handle_command(
    const std::span<const std::string> words) noexcept {
  if (words.empty()) {
    return false;
  }
  const auto &command = words.front();
  if (command == "print" || command == "p") {
    print();
    return true;
  }

  if (command == "save" || command == "load") {

    if (words.size() < 2) {
      err(command, ": missing arg(s).");
      return false;
    }
    bool success;
    std::filesystem::path path{words[1]};
    if (command == "save") {
      success = save(path);
    } else {
      success = load(path);
    }
    if (!success) {
      err(command, ": Failed.");
    } else {
      log(command, ": Operation at path: '", path.string(), "' succeeded.");
    }
    return success;

  } else if (command == "cd") {

    return cd({words.begin() + 1, words.size() - 1});

  } else if (command == "add") {

    return add(words[1]);

  } else if (command == "set") {

    return set({words.begin() + 1, words.size() - 1});

  } else if (command == "rm") {

    // return rm(words[1]);

  }

  err("error: command '", command, "' not recognized");
  return false;
}

bool Program::save(std::filesystem::path path) noexcept {
  return FS::save(path, data.sides);
}
bool Program::load(std::filesystem::path path) noexcept {
  return FS::load(path, data.sides);
}

bool Program::add(std::string key) noexcept {
  if (data.sides.contains(key)) {
    err("add: ", key, " already present.");
    return false;
  } else {
    data.sides.emplace(key, SideConfig{});
    return true;
  }
}

bool Program::set(const std::span<const std::string> words) noexcept {
  if (depth() != 2) {
    err("set: a slot must be in focus.");
    return false;
  }
  if (words.size()) {
    err("set: Missing args.");
    return false;
  }

  Data::Species species{};
  Data::Moves move{};
  const auto parse = [&species, &move](std::string word) {
    try {

    } catch (...) {
    }
  };

  for (const auto &word : words) {
    parse(word);
  }

  if (species == Data::Species::None) {
    return false;
  }
  return true;
}

bool Program::cd(const std::span<const std::string> words) noexcept {
  if (words.size()) {
    err("cd: Missing args.");
    return false;
  }

  const auto handle_word = [this](std::string s) {
    if (s == "..") {
      return up();
    }

    std::optional<size_t> slot;
    try {
      slot = std::stoi(s);
      if (slot > 6) {
        slot = std::nullopt;
      }
    } catch (...) {
      slot = std::nullopt;
    }

    switch (depth()) {
    case 2:
      return false;
    case 1:
      if (slot.has_value()) {
        mgmt.cli_slot = slot;
        return true;
      }
      if (s == "active") {
        mgmt.cli_slot = 0;
        return true;
      }
    case 0:
      if (data.sides.contains(s)) {
        mgmt.cli_key = s;
        return true;
      }
    default:
      return false;
    }
  };

  for (const auto &p : words) {
    if (!handle_word(p)) {
      return false;
    }
  }
  return true;
}

void Program::print() const noexcept {
  switch (depth()) {
  case 0: {
    log(data.sides.size(), " sides:");
    for (const auto &[key, value] : data.sides) {
      log(key);
    }
    return;
  }
  case 1: {
    const auto &party = data.sides.at(mgmt.cli_key.value()).party;
    for (const auto &pokemon : party) {
      log_(Names::species_string(pokemon.species), " : ");
      for (const auto move : pokemon.moves._data) {
        log_(Names::move_string(move), ' ');
      }
      log("");
    }
    return;
  }
  case 2: {
    const auto slot = mgmt.cli_slot.value();
    if (slot == 0) {
      log("print: TODO Active.");
    } else {
      const auto &pokemon = data.sides.at(mgmt.cli_key.value()).party.at(slot);
      log_(Names::species_string(pokemon.species), " : ");
      for (const auto move : pokemon.moves._data) {
        log_(Names::move_string(move), ' ');
      }
    }
    return;
  }
  default: {
    return;
  }
  }
}

} // namespace Sides
} // namespace Process