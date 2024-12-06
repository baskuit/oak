#include <games.h>

#include <battle/sample-teams.h>
#include <battle/strings.h>

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
  } else if (command == "rollout") {
    return rollout();
  }

  const std::span<const std::string> tail{words.begin() + 1, words.size() - 1};

  if (command == "cd") {

    return cd(tail);
  }

  if (command == "update") {
    if (words.size() < 3) {
      return false;
    }
    return update(words[1], words[2]);
  }

  err("games: command '", command, "' not recognized");
  return false;
}

bool Program::save(std::filesystem::path path) noexcept { return false; }
bool Program::load(std::filesystem::path path) noexcept { return false; }

void Program::print() const noexcept {
  switch (depth()) {
  case 0: {
    log(data.histories.size(), " games:");
    for (const auto &[key, value] : data.histories) {
      log_(key, '\t');
    }
    log("");
    return;
  }
  case 1: {
    log(data.histories.at(mgmt.cli_key.value())->states.size(), " states.");
    return;
  }
  case 2: {
    const auto &bd = data.histories.at(mgmt.cli_key.value())
                         ->states.at(mgmt.cli_state.value())
                         ->battle_data;
    log(Strings::battle_to_string(bd.battle));
    log(bd.m, " ", bd.n);
    for (auto i = 0; i < bd.m; ++i) {
      log_((int)bd.choices1[i], ' ');
    }
    log("");
    for (auto i = 0; i < bd.n; ++i) {
      log_((int)bd.choices2[i], ' ');
    }
    log("");
    return;
  }
  case 3: {
    log("TODO print node");
    return;
  }
  case 4: {
    log("TODO print output");
    return;
  }
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
  state.battle_data.m = 1;
  state.battle_data.n = 1;

  return true;
}

bool Program::update(std::string str1, std::string str2) noexcept {
  uint8_t x, y;
  try {
    x = std::stoi(str1);
    y = std::stoi(str2);
  } catch (...) {
    err("update: bad w/e.");
    return false;
  }
  return update(x, y);
}

bool Program::update(pkmn_choice c1, pkmn_choice c2) noexcept {
  if (!mgmt.cli_key.has_value()) {
    err("update: A game must be in focus");
    return false;
  }
  auto &history = *data.histories[mgmt.cli_key.value()];
  const auto &state = *history.states.back();

  auto next = std::make_unique<State>();
  auto &bd = next->battle_data;
  bd.battle = state.battle_data.battle;
  bd.seed = *std::bit_cast<const uint64_t *>(bd.battle.bytes + Offsets::seed);
  bd.options = state.battle_data.options;
  bd.result = Init::update(bd.battle, c1, c2, bd.options);
  const auto [choices1, choices2] = Init::choices(bd.battle, bd.result);

  if (pkmn_result_type(bd.result)) {
    bd.m = 0;
    bd.n = 0;
  } else {
    bd.m = choices1.size();
    bd.n = choices2.size();
  }
  std::copy(choices1.begin(), choices1.end(), bd.choices1.begin());
  std::copy(choices2.begin(), choices2.end(), bd.choices2.begin());

  next->nodes.emplace_back(std::make_unique<Node>());
  next->nodes.emplace_back(std::make_unique<Node>());
  next->outputs.resize(2);

  history.states.emplace_back(std::move(next));

  return true;
}

bool Program::rollout() {
  if (depth() == 0) {
    err("rollout: A Game must be in focus.");
    return false;
  }
  auto &history = *data.histories[mgmt.cli_key.value()];
  const auto *state = history.states.back().get();
  prng device{123123};
  while ((state->battle_data.m * state->battle_data.n) > 0) {
    const auto i = device.random_int(state->battle_data.m);
    const auto j = device.random_int(state->battle_data.n);
    const bool success =
        update(state->battle_data.choices1[i], state->battle_data.choices2[j]);
    if (!success) {
      err("rollout: Bad update.");
      return false;
    }
    state = history.states.back().get();
  }
  log("rollout: ", history.states.size(), " states.");
  return true;
};

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

bool Program::rm(std::string key) noexcept {
  if (depth() != 0) {
    err("rm: A game cannot be in focus");
    return false;
  }
  if (!data.histories.contains(key)) {
    err("rm: ", key, " not present.");
    return false;
  } else {
    data.histories.erase(key);
    return true;
  }
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

History &Program::history() { return *data.histories.at(mgmt.cli_key.value()); }
State &Program::state() { return *history().states.at(mgmt.cli_state.value()); }
Node &Program::node() { return *state().nodes.at(mgmt.cli_node.value()); }
MCTS::Output &Program::outputs() {
  return state()
      .outputs.at(mgmt.cli_node.value())
      .tail.at(mgmt.cli_search.value());
}

} // namespace Games
} // namespace Process