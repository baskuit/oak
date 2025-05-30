#include <sides.h>

#include <util/fs.h>

#include <data/strings.h>

#include <battle/sample-teams.h>
#include <battle/strings.h>

namespace Lab {
namespace Sides {

constexpr size_t max_sample_teams{10};

Program::Program(std::ostream *out, std::ostream *err)
    : ProgramBase<false, true>{out, err}, data{}, mgmt{} {
  for (auto i = 0; i < max_sample_teams; ++i) {
    Init::Config config{};
    config.pokemon = SampleTeams::teams[i];
    data.sides[std::to_string(i)] = config;
  }
}

std::string Program::prompt() const {
  const std::string esc{"\033[0m"};
  std::string p{"\033[32m(sides)"};
  p += esc;
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

bool Program::handle_command(const std::span<const std::string> words) {
  if (words.empty()) {
    return false;
  }
  const auto &command = words.front();
  if (command == "print" || command == "ls") {
    print();
    return true;
  }

  if (words.size() < 2) {
    return false;
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
  } else if (command == "hp") {
    return hp(tail);
  } else if (command == "status") {
    return status(tail);
  }

  err("sides: command '", command, "' not recognized");
  return false;
}

bool Program::p1_select(const std::span<const std::string> words) {}

bool Program::save(std::filesystem::path path) {
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

bool Program::load(std::filesystem::path path) {
  return FS::load(path, data.sides);
}

bool Program::add(std::string key) {
  if (data.sides.contains(key)) {
    err("add: ", key, " already present.");
    return false;
  } else {
    data.sides.emplace(key, Init::Config{});
    return true;
  }
}

bool Program::rm(std::string key) {
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
bool Program::hp(const std::span<const std::string> words) {
  if (depth() != 2) {
    err("hp: A non-active slot must be in focus.");
    return false;
  }
  if (mgmt.slot.value() == 0) {
    err("hp: A non-active slot must be in focus.");
    return false;
  }
  if (words.empty()) {
    err("hp: Missing args.");
    return false;
  }

  size_t hp;
  try {
    hp = std::min(100, std::stoi(words[0]));
  } catch (...) {
    err("hp: Could not parse arg (percent).");
    return false;
  }
  auto &pokemon =
      data.sides.at(mgmt.key.value()).pokemon.at(mgmt.slot.value() - 1);
  pokemon.hp = hp / 100.0f;
  return true;
}

bool Program::status(const std::span<const std::string> words) {
  if (depth() != 2) {
    err("hp: A non-active slot must be in focus.");
    return false;
  }
  if (mgmt.slot.value() == 0) {
    err("hp: A non-active slot must be in focus.");
    return false;
  }
  if (words.empty()) {
    err("hp: Missing args.");
    return false;
  }

  const auto &first = words[0];
  auto &pokemon =
      data.sides.at(mgmt.key.value()).pokemon.at(mgmt.slot.value() - 1);
  if (first == "slp") {
    if (words.size() < 2) {
      err("status: Missing sleep turns [0, 7].");
      return false;
    }
    size_t dur;
    try {
      dur = std::stoi(words[1]);
    } catch (...) {
      err("status: Could not parse sleep turns.");
      return false;
    }
    if (dur > 7) {
      err("status: Sleep turns too high.");
      return false;
    }
    pokemon.status = static_cast<uint8_t>(Data::Status::Sleep1);
    pokemon.sleep = dur;
  } else if (first == "rest") {
    if (words.size() < 2) {
      err("status: Missing rest duration [1, 2].");
      return false;
    }
    uint8_t dur;
    try {
      dur = std::stoi(words[1]);
    } catch (...) {
      err("status: Could not rest duration.");
      return false;
    }
    if ((dur > 2) || (dur < 1)) {
      err("status: Invalid rest duration.");
      return false;
    }
    pokemon.status = dur | 0b10000000;
  } else if (first == "none" || first == "clear" || first == "clr") {
    pokemon.status = 0;
  } else if (first == "par") {
    pokemon.status = static_cast<uint8_t>(Data::Status::Paralysis);
  } else if (first == "psn") {
    pokemon.status = static_cast<uint8_t>(Data::Status::Poison);
  } else if (first == "frz") {
    pokemon.status = static_cast<uint8_t>(Data::Status::Freeze);
  } else if (first == "brn") {
    pokemon.status = static_cast<uint8_t>(Data::Status::Burn);
  } else {
    err("status: Invalid input");
    return false;
  }
  return true;
}

bool Program::set(const std::span<const std::string> words) {
  if (depth() != 2 || mgmt.slot == 0) {
    err("set: A non-active slot must be in focus.");
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
  std::sort(pokemon.moves.begin(), pokemon.moves.end(),
            std::greater<Data::Moves>());
  pokemon.species = species;
  return true;
}

bool Program::cp(const std::span<const std::string> words) {
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

bool Program::cd(const std::span<const std::string> words) {
  if (words.empty()) {
    err("cd: Missing args.");
    return false;
  }

  const auto handle_word = [this](std::string s) {
    if (s == "..") {
      return up();
    } else if (s == "/") {
      mgmt.key = std::nullopt;
      mgmt.slot = std::nullopt;
      return true;
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

void Program::print() const {

  const auto print_poke = [this](const auto &pokemon) {
    log_(Names::species_string(pokemon.species));
    if (pokemon.status) {
      log_(" ", Strings::status(pokemon.status), " (", pokemon.sleep, ")");
    }
    log_(" : ");
    for (const auto move : pokemon.moves) {
      log_(Names::move_string(move), ' ');
    }
    log("");
  };

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
      print_poke(pokemon);
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
      print_poke(pokemon);
    }
    return;
  }
  default: {
    return;
  }
  }
}

size_t Program::depth() const {
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

bool Program::up() {
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