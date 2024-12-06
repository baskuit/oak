#include <games.h>

#include <battle/sample-teams.h>

#include <sides.h>

namespace Process {
namespace Games {

// This is probably bad, but I still think of the return type as the actual
// initialization type
auto convert_config(const Sides::SideConfig config) {
  std::array<Init::Set, 6> side;
  for (auto i = 0; i < 6; ++i) {
    side[i].species = config.party[i].species;
    for (auto j = 0; j < 4; ++j) {
      side[i].moves[j] = config.party[i].moves._data[j];
    }
  }
  return side;
}

std::string Program::prompt() const noexcept {
  std::string r{};
  switch (depth()) {
  case 4:
    r = "/" + std::to_string(mgmt.cli_search.value()) + r;
  case 3:
    r = "/" + std::to_string(mgmt.cli_node.value()) + r;
  case 2:
    r = "/" + std::to_string(mgmt.cli_state.value()) + r;
  case 1:
    r = "/" + mgmt.cli_key.value() + r;
  default:
    r = "games" + r;
    r += "$ ";
  }
  return r;
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
  err("games: command '", command, "' not recognized");
  return false;
}

bool Program::save(std::filesystem::path path) noexcept { return false; }
bool Program::load(std::filesystem::path path) noexcept { return false; }

void Program::print() const noexcept {
  switch (depth()) {
  case 0:
    log(data.histories.size(), " games:");
    for (const auto &[key, value] : data.histories) {
      log_(key, '\t');
    }
    log("");
    return;
  case 1:
    log(data.histories.at(mgmt.cli_key.value()).histories.size(), " states.");
    return;
  case 2:
    log("TODO print state");
    return;
  case 3:
    log("TODO print node");
    return;
  case 4:
    log("TODO print output");
    return;
  }
}

bool Program::create(const std::string key, const Sides::SideConfig p1,
                     const Sides::SideConfig p2) {
  std::unique_lock lock{mgmt.mutex};
  if (data.histories.contains(key)) {
    err("create: '", key, "' already present.");
    return false;
  }

  const auto battle = Init::battle(convert_config(p1), convert_config(p2));
  const auto &history = data.histories[key];

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

    const auto d = depth();

    if (d == 0) {
      if (data.histories.contains(s)) {
        mgmt.cli_key = s;
        return true;
      } else {
        return false;
      }
    }

    size_t slot;
    try {
      slot = std::stoi(s);
    } catch (...) {
      return false;
    }

    std::optional<size_t> *datum;
    switch (d) {
    case 3: {
      datum = &mgmt.cli_search;
    }
    case 2: {
      datum = &mgmt.cli_node;
      break;
    }
    case 1: {
      datum = &mgmt.cli_state;
      break;
    }
    default: { // case 4
      return false;
    }
    }
    if (slot < size()) {
      *datum = slot;
      return true;
    } else {
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

size_t Program::depth() const noexcept {
  if (mgmt.cli_search.has_value()) {
    return 4;
  } else {
    if (mgmt.cli_node.has_value()) {
      return 3;
    } else {
      if (mgmt.cli_state.has_value()) {
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

bool Program::up() noexcept {
  if (mgmt.cli_search.has_value()) {
    mgmt.cli_search = std::nullopt;
    return true;
  } else if (mgmt.cli_node.has_value()) {
    mgmt.cli_node = std::nullopt;
    return true;
  } else if (mgmt.cli_state.has_value()) {
    mgmt.cli_state = std::nullopt;
    return true;
  } else if (mgmt.cli_key.has_value()) {
    mgmt.cli_key = std::nullopt;
    return true;
  }
  return true;
}

size_t Program::size() const noexcept {
  if (mgmt.cli_search.has_value()) {
    return data.histories.at(mgmt.cli_key.value())
        .histories.at(mgmt.cli_state.value())
        ->output_data.at(mgmt.cli_node.value())
        .outputs.size();
  } else if (mgmt.cli_node.has_value()) {
    return data.histories.at(mgmt.cli_key.value())
        .histories.at(mgmt.cli_state.value())
        ->node_data.size();
  } else if (mgmt.cli_state.has_value()) {
    return data.histories.at(mgmt.cli_key.value()).histories.size();
  } else if (mgmt.cli_key.has_value()) {
    return data.histories.size();
  } else {
    return 0;
  }
}

} // namespace Games
} // namespace Process