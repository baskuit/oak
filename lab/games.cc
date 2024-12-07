#include <games.h>

#include <battle/sample-teams.h>
#include <battle/strings.h>

#include <sides.h>

namespace Lab {
namespace Games {

std::string Program::prompt() const  {
  std::stringstream p{};
  p << "games";
  if (mgmt.loc.depth >= 1) {
    p << "/" << mgmt.loc.key;
  }
  for (int i = 0; i < static_cast<int>(mgmt.loc.depth) - 1; ++i) {
    p << '/' << mgmt.loc.current[i];
  }
  p << "$ ";

  return p.str();
}

void Program::loc() const  {
  std::stringstream p{};
  p << "depth: " << mgmt.loc.depth << " index: " << mgmt.loc.current[0] << ' '
    << mgmt.loc.current[1] << ' ' << mgmt.loc.current[2] << std::endl;
  p << "bound: " << mgmt.bounds[0] << ' ' << mgmt.bounds[1] << ' '
    << mgmt.bounds[2] << std::endl;
  log(p.str());
}

bool Program::handle_command(
    const std::span<const std::string> words)  {
  if (words.empty()) {
    return false;
  }
  const auto &command = words.front();
  if (command == "ls") {
    print();
    return true;
  } else if (command == "loc") {
    loc();
    return true;
  } else if (command == "rollout") {
    return rollout();
  } else if (command == "prev") {
    return prev();
  } else if (command == "next") {
    return next();
  } else if (command == "first") {
    return first();
  } else if (command == "last") {
    return last();
  }
  // if (command == "update") {
  //   if (words.size() < 3) {
  //     return false;
  //   }
  //   return update(words[1], words[2]);
  // }

  const std::span<const std::string> tail{words.begin() + 1, words.size() - 1};

  if (command == "cd") {
    return cd(tail);
  }
  err("games: command '", command, "' not recognized");
  return false;
}

bool Program::save(std::filesystem::path path)  { return false; }
bool Program::load(std::filesystem::path path)  { return false; }

void Program::print() const  {
  switch (mgmt.loc.depth) {
  case 0: {
    log(data.history_map.size(), " games:");
    for (const auto &[key, value] : data.history_map) {
      log_(key, '\t');
    }
    log("");
    return;
  }
  case 1: {
    const auto &states = data.history_map.at(mgmt.loc.key);
    log(states.size(), " states.");
    return;
  }
  case 2: {
    const auto &s = state();
    log(Strings::battle_to_string(s.battle));
    for (auto i = 0; i < s.m; ++i) {
      const auto c = s.choices1[i];
      log_(side_choice_string(s.battle.bytes, c), ' ');
    }
    log("");
    for (auto i = 0; i < s.n; ++i) {
      const auto c = s.choices2[i];
      log_(side_choice_string(s.battle.bytes + Offsets::side, c), ' ');
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

bool Program::up()  {
  if (mgmt.loc.depth == 0) {
    return true;
  } else if (mgmt.loc.depth == 1) {
    mgmt.bounds[0] = 0;
    mgmt.loc.key = "";
    --mgmt.loc.depth;
  } else {
    if (mgmt.loc.depth < 3) {
      mgmt.bounds[mgmt.loc.depth] = 0;
    }
    mgmt.loc.current[mgmt.loc.depth - 1] = 0;
    --mgmt.loc.depth;
  }
  return true;
}

bool Program::cd(const std::span<const std::string> words) {
  if (words.empty()) {
    err("cd: Missing args.");
    return false;
  }

  const auto handle_word = [this](std::string word) {
    if (word == "..") {
      return up();
    }

    if (mgmt.loc.depth == 4) {
      err("cd: Only '..' allowed here.");
      return false;
    } else if (mgmt.loc.depth == 0) {
      if (data.history_map.contains(word)) {
        mgmt.loc.key = word;
        ++mgmt.loc.depth;
        return true;
      } else {
        err("cd: Game key '", word, "' not found.");
        return false;
      }
    }

    size_t index;
    try {
      index = std::stoi(word);
    } catch (...) {
      err("cd: Bad argument; expecting index.");
      return false;
    }

    if (mgmt.bounds[mgmt.loc.depth - 1] <= index) {
      err("cd: {depth: ", mgmt.loc.depth, " index: ", index,
          " bound: ", mgmt.bounds[mgmt.loc.depth - 1]);
      return false;
    }

    mgmt.loc.current[mgmt.loc.depth - 1] = index;
    ++mgmt.loc.depth;

    return true;
  };

  const auto update_bounds = [this]() {
    switch (mgmt.loc.depth) {
    case 0: {
      return;
    }
    case 1: {
      mgmt.bounds[0] = history().size();
      return;
    }
    case 2: {
      mgmt.bounds[1] = search_data().nodes.size();
      return;
    }
    case 3: {
      mgmt.bounds[2] = search_outputs().tail.size();
      return;
    }
    default:
    }
  };

  for (const auto &p : words) {
    if (!handle_word(p)) {
      return false;
    } else {
      update_bounds();
    }
  }
  return true;
}

bool Program::prev()  {
  if (mgmt.loc.depth < 2) {
    err("prev: A list must be in focus.");
    return false;
  }
  auto &cur = mgmt.loc.current[mgmt.loc.depth - 2];
  if (cur != 0) {
    --cur;
  }
  return true;
}
bool Program::next()  {
  if (mgmt.loc.depth < 2) {
    err("next: A list must be in focus.");
    return false;
  }
  auto &cur = mgmt.loc.current[mgmt.loc.depth - 2];
  if (cur < mgmt.bounds[mgmt.loc.depth - 2] - 1) {
    ++cur;
  }
  return true;
}
bool Program::first()  {
  if (mgmt.loc.depth < 2) {
    err("first: A list must be in focus.");
    return false;
  }
  mgmt.loc.current[mgmt.loc.depth - 2] = 0;
  return true;
}
bool Program::last()  {
  if (mgmt.loc.depth < 2) {
    err("last: A list must be in focus.");
    return false;
  }
  mgmt.loc.current[mgmt.loc.depth - 2] = mgmt.bounds[mgmt.loc.depth - 2] - 1;
  return true;
}

History &Program::history() { return data.history_map.at(mgmt.loc.key); }
State &Program::state() { return history().at(mgmt.loc.current[0]); }
SearchOutputs &Program::search_outputs() {
  return data.history_map.at(mgmt.loc.key)
      .at(mgmt.loc.current[0])
      .outputs.at(mgmt.loc.current[2]);
}
StateSearchData &Program::search_data() {
  return data.search_data_map.at(mgmt.loc.key).at(mgmt.loc.current[0]);
}
Node &Program::node() {
  return *data.search_data_map.at(mgmt.loc.key)
              .at(mgmt.loc.current[0])
              .nodes.at(mgmt.loc.current[1]);
}
MCTS::Output &Program::output() {
  return search_outputs().tail.at(mgmt.loc.current[2]);
}

const History &Program::history() const {
  return data.history_map.at(mgmt.loc.key);
}
const State &Program::state() const {
  return history().at(mgmt.loc.current[0]);
}
const SearchOutputs &Program::search_outputs() const {
  return data.history_map.at(mgmt.loc.key)
      .at(mgmt.loc.current[0])
      .outputs.at(mgmt.loc.current[2]);
}
const StateSearchData &Program::search_data() const {
  return data.search_data_map.at(mgmt.loc.key).at(mgmt.loc.current[0]);
}

const Node &Program::node() const {
  return *data.search_data_map.at(mgmt.loc.key)
              .at(mgmt.loc.current[0])
              .nodes.at(mgmt.loc.current[1]);
}
const MCTS::Output &Program::output() const {
  return search_outputs().tail.at(mgmt.loc.current[2]);
}

bool Program::create(const std::string key, const Init::Config p1,
                     const Init::Config p2) {
  std::unique_lock lock{mgmt.mutex};
  if (data.history_map.contains(key)) {
    err("create: '", key, "' already present.");
    return false;
  }

  const auto battle = Init::battle(p1, p2);
  data.history_map[key] = {};
  auto &history = data.history_map[key];
  history.emplace_back();
  auto &state = history.front();
  state.battle = battle;
  state.options = {};
  state.m = 1;
  state.n = 1;
  state.outputs.resize(2);

  auto &search_data = data.search_data_map[key];
  search_data.emplace_back();
  search_data.front().nodes.emplace_back(std::unique_ptr<Node>());
  search_data.front().nodes.emplace_back(std::unique_ptr<Node>());

  return true;
}

// bool Program::update(std::string str1, std::string str2)  {
//   uint8_t x, y;
//   try {
//     x = std::stoi(str1);
//     y = std::stoi(str2);
//   } catch (...) {
//     err("update: bad w/e.");
//     return false;
//   }
//   return update(x, y);
// }

bool Program::update(pkmn_choice c1, pkmn_choice c2)  {
  // if (mgmt.loc.depth == 0) {
  //   err("update: A game must be in focus");
  //   return false;
  // }
  auto &h = history();
  auto &state = h.back();

  state.c1 = c1;
  state.c2 = c2;  

  State next{};
  next.battle = state.battle;
  next.seed =
      *std::bit_cast<const uint64_t *>(next.battle.bytes + Offsets::seed);
  next.options = state.options;
  next.result = Init::update(next.battle, c1, c2, next.options);
  const auto [choices1, choices2] = Init::choices(next.battle, next.result);
  next.outputs.resize(2);

  if (pkmn_result_type(next.result)) {
    next.m = 0;
    next.n = 0;
  } else {
    next.m = choices1.size();
    next.n = choices2.size();
  }
  std::copy(choices1.begin(), choices1.end(), next.choices1.begin());
  std::copy(choices2.begin(), choices2.end(), next.choices2.begin());

  h.emplace_back(next);

  auto &search_data_history = data.search_data_map.at(mgmt.loc.key);
  search_data_history.emplace_back();
  auto &s = search_data_history.at(h.size() - 1);
  s.nodes.emplace_back(std::make_unique<Node>());
  s.nodes.emplace_back(std::make_unique<Node>());

  ++mgmt.bounds[0];

  return true;
}

bool Program::rollout() {
  if (mgmt.loc.depth == 0) {
    err("rollout: A Game must be in focus.");
    return false;
  }
  auto &h = history();
  const auto *state = &h.back();
  prng device{123123};
  while ((state->m * state->n) > 0) {
    const auto i = device.random_int(state->m);
    const auto j = device.random_int(state->n);
    const bool success = update(state->choices1[i], state->choices2[j]);
    if (!success) {
      err("rollout: Bad update.");
      return false;
    }
    state = &h.back();
  }
  log("rollout: ", h.size(), " states.");
  return true;
};

// bool Program::rm(std::string key)  {
//   if (depth() != 0) {
//     err("rm: A game cannot be in focus");
//     return false;
//   }
//   if (!data.histories.contains(key)) {
//     err("rm: ", key, " not present.");
//     return false;
//   } else {
//     data.histories.erase(key);
//     return true;
//   }
// }

} // namespace Games
} // namespace Lab