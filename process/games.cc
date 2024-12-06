#include <games.h>

#include <battle/sample-teams.h>

#include <sides.h>

namespace Process {
namespace Games {

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
  if (command == "print" || command == "ls") {
    print();
    return true;
  }

  const std::span<const std::string> tail{words.begin() + 1, words.size() - 1};

  if (command == "cd") {

    return cd(tail);
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
    log(data.histories.at(mgmt.cli_key.value())->states.size(), " states.");
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

bool Program::create(const std::string key, const Init::Config p1,
                     const Init::Config p2) {
  std::unique_lock lock{mgmt.mutex};
  if (data.histories.contains(key)) {
    err("create: '", key, "' already present.");
    return false;
  }

  const auto battle = Init::battle(p1, p2);
  // data.histories.insert(key);
  data.histories[key] = std::make_unique<History>();
  auto &history = *data.histories[key];
  // history = std
  history.states.emplace_back(std::make_unique<State>());
  auto &state = *history.states.front();
  state.battle_data.battle = battle;
  state.battle_data.options = {};
  state.nodes.emplace_back(new Node{});
  state.nodes.emplace_back(new Node{});
  state.outputs.resize(2);

  return true;
}

bool Program::cd(const std::span<const std::string> words) noexcept {
  if (words.empty()) {
    err("cd: Missing args.");
    return false;
  }

  // worst code ever
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
  if (!mgmt.cli_key.has_value()) {
    return data.histories.size();
  }
  const auto &history = *data.histories.at(mgmt.cli_key.value());
  if (!mgmt.cli_state.has_value()) {
    return history.states.size();
  }
  const auto &state = *history.states.at(mgmt.cli_state.value());
  if (!mgmt.cli_node.has_value()) {
    return state.nodes.size();
  }
  const auto &output = state.outputs[mgmt.cli_node.value()];
  if (!mgmt.cli_search.has_value()) {
    return output.tail.size();
  }
  return 0;
}

} // namespace Games
} // namespace Process