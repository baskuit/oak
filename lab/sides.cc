#include <sides.h>

#include <util/fs.h>

#include <data/strings.h>

#include <battle/sample-teams.h>
#include <battle/strings.h>

namespace Lab {
namespace Sides {

constexpr size_t max_sample_teams{10};

Program::Program(std::ostream *out, std::ostream *err)
    : ProgramBase<false, true>{out, err} {
  for (auto i = 0; i < max_sample_teams; ++i) {
    Init::Config config{};
    config.pokemon = SampleTeams::teams[i];
    data.sides[std::to_string(i)] = config;
  }
}

std::string Program::prompt() const noexcept {
  std::string p{"sides"};
  if (mgmt.key.has_value()) {
    p += "/" + mgmt.key.value();
  }
  if (mgmt.slot.has_value()) {
    const auto slot = mgmt.slot.value();
    p += "/" + (slot == 0 ? "active" : std::to_string(slot));
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
  if (command == "print" || command == "ls") {
    print();
    return true;
  }

  const std::span<const std::string> tail{words.begin() + 1, words.size() - 1};

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

    return cd(tail);

  } else if (command == "set") {

    return set(tail);

  } else if (command == "cp") {

    return cp(tail);

  } else if (command == "add") {

    return add(words[1]);

  } else if (command == "rm") {

    return rm(words[1]);
  }

  err("sides: command '", command, "' not recognized");
  return false;
}

bool Program::save(std::filesystem::path path) noexcept {
  constexpr bool overwrite = true;
  const auto mode =
      overwrite ? std::ios::binary : std::ios::binary | std::ios::trunc;
  std::ofstream file(path, mode);
  if (!file.is_open()) {
    return false;
  }

  size_t s;
  for (const auto &[key, value] : data.sides) {
    s = key.size();
    file.write(std::bit_cast<const char *>(&s), sizeof(size_t));
    file.write(std::bit_cast<const char *>(key.data()), s);
    file.write(std::bit_cast<const char *>(&value), sizeof(Init::Config));
  }

  file.close();
  return true;
}

bool Program::load(std::filesystem::path path) noexcept {
  return FS::load(path, data.sides);
}

bool Program::add(std::string key) noexcept {
  if (data.sides.contains(key)) {
    err("add: ", key, " already present.");
    return false;
  } else {
    data.sides.emplace(key, Init::Config{});
    return true;
  }
}

bool Program::rm(std::string key) noexcept {
  if (depth() != 0) {
    err("rm: A side cannot be in focus");
    return false;
  }
  if (!data.sides.contains(key)) {
    err("rm: ", key, " not present.");
    return false;
  } else {
    data.sides.erase(key);
    return true;
  }
}

bool Program::set(const std::span<const std::string> words) noexcept {
  if (depth() != 2) {
    err("set: A slot must be in focus.");
    return false;
  }
  if (words.empty()) {
    err("set: Missing args.");
    return false;
  }

  Data::OrderedMoveSet move_set{};

  Data::Species species{};
  Data::Moves move{};
  const auto parse = [&species, &move, &move_set](std::string word) {
    try {
      species = Strings::string_to_species(word);
      if (species != Data::Species::None) {
        return true;
      }
    } catch (...) {
    }
    try {
      move = Strings::string_to_move(word);
      return move_set.insert(move);
    } catch (...) {
    }
    return false;
  };

  for (const auto &word : words) {
    if (!parse(word)) {
      err("set: '", word, "' could not be matched to species/move.");
      return false;
    }
  }

  if (species == Data::Species::None) {
    err("set: Could not parse species");
    return false;
  }

  auto &pokemon =
      data.sides.at(mgmt.key.value()).pokemon.at(mgmt.slot.value() - 1);
  pokemon.moves = move_set._data;
  pokemon.species = species;
  return true;
}

bool Program::cp(const std::span<const std::string> words) noexcept {
  if (words.empty()) {
    err("cp: Missing source.");
    return false;
  }
  const auto source = words[0];
  if (!data.sides.contains(source)) {
    err("cp: Source '", source, "' not found.");
    return false;
  }

  std::string dest;
  if (words.size() >= 2) {
    dest = words[1];
    if (data.sides.contains(dest)) {
      err("cp: Destination '", dest, "' already present.");
      return false;
    }
  } else {
    size_t i = 1;
    do {
      dest = source + "(" + std::to_string(i) + ")";
      ++i;
    } while (data.sides.contains(dest));
  }

  data.sides[dest] = data.sides[source];
  return true;
}

bool Program::cd(const std::span<const std::string> words) noexcept {
  if (words.empty()) {
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
        mgmt.slot = slot;
        return true;
      }
      if (s == "active") {
        mgmt.slot = 0;
        return true;
      }
    case 0:
      if (data.sides.contains(s)) {
        mgmt.key = s;
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
      log_(key, '\t');
    }
    log("");
    return;
  }
  case 1: {
    const auto &party = data.sides.at(mgmt.key.value()).pokemon;
    size_t i = 0;
    for (const auto &pokemon : party) {
      log_(++i, " : ", Names::species_string(pokemon.species), " : ");
      for (const auto move : pokemon.moves) {
        log_(Names::move_string(move), ' ');
      }
      log("");
    }
    return;
  }
  case 2: {
    const auto slot = mgmt.slot.value();
    if (slot == 0) {
      log("print: TODO Active.");
    } else {
      const auto &pokemon =
          data.sides.at(mgmt.key.value()).pokemon.at(slot - 1);
      log_(Names::species_string(pokemon.species), " : ");
      for (const auto move : pokemon.moves) {
        log_(Names::move_string(move), ' ');
      }
      log("");
    }
    return;
  }
  default: {
    return;
  }
  }
}

size_t Program::depth() const noexcept {
  if (mgmt.key.has_value()) {
    if (mgmt.slot.has_value()) {
      return 2;
    } else {
      return 1;
    }
  } else {
    return 0;
  }
}

bool Program::up() noexcept {
  if (mgmt.slot.has_value()) {
    mgmt.slot = std::nullopt;
    return true;
  }
  if (mgmt.key.has_value()) {
    mgmt.key = std::nullopt;
    return true;
  }
  return true; // better than failing surely!
}

} // namespace Sides
} // namespace Lab