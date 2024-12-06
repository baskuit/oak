#include <games.h>

#include <battle/sample-teams.h>
#include <battle/strings.h>

#include <sides.h>

namespace Lab {
namespace Games {

std::string Program::prompt() const noexcept {
  std::stringstream p{};
  p << "games";
  if (mgmt.loc.depth > 0) {
    p << "/" << mgmt.loc.key;
    for (auto i = 0; i < mgmt.loc.depth - 1; ++i) {
      p << '/' << mgmt.loc.current[i];
    }
  }
  p << "$" << mgmt.loc.depth << " ";
  return p.str();
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
    // } else if (command == "rollout") {
    //   return rollout();
    // } else if (command == "prev") {
    //   return prev();
    // } else if (command == "next") {
    //   return next();
    // } else if (command == "first") {
    //   return first();
    // } else if (command == "last") {
    //   return last();
    // } else if (command == "size") {
    //   log("size: ", size());
    //   return true;
  }

  const std::span<const std::string> tail{words.begin() + 1, words.size() - 1};

  if (command == "cd") {
    return cd(tail);
  }

  // if (command == "update") {
  //   if (words.size() < 3) {
  //     return false;
  //   }
  //   return update(words[1], words[2]);
  // }

  err("games: command '", command, "' not recognized");
  return false;
}

bool Program::save(std::filesystem::path path) noexcept { return false; }
bool Program::load(std::filesystem::path path) noexcept { return false; }

void Program::print() const noexcept {
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

bool Program::up() noexcept {
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

bool Program::cd(const std::span<const std::string> words) noexcept {
  if (words.empty()) {
    err("cd: Missing args.");
    return false;
  }

  const auto handle_word = [this](std::string word) {
    if (word == "..") {
      return up();
    }

    if (mgmt.loc.depth == 4) {
      return false;
    } else if (mgmt.loc.depth == 0) {
      if (data.history_map.contains(word)) {
        mgmt.loc.key = word;
        ++mgmt.loc.depth;
        return true;
      } else {
        return false;
      }
    }

    size_t index;

    if (mgmt.bounds[mgmt.loc.depth - 1] <= index) {
      return false;
    }
    mgmt.loc.current[mgmt.loc.depth - 1] = index;
    return true;
  };

  const auto update_bounds = [this]() {
    switch (mgmt.loc.depth) {
      case 1: {
        mgmt.bounds[0] = history().size();
        return;
      }
      case 2: {
        mgmt.bounds[1] = data.search_data_map.at(mgmt.loc.key)
              .at(mgmt.loc.current[0]).nodes.size();
        return;
      }
      case 3: {
        mgmt.bounds[2] = search_outputs().tail.size();
        return;
      }
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

History &Program::history() { return data.history_map.at(mgmt.loc.key); }
State &Program::state() { return history().at(mgmt.loc.current[0]); }
SearchOutputs &Program::search_outputs() {
  return data.history_map.at(mgmt.loc.key)
      .at(mgmt.loc.current[0])
      .outputs.at(mgmt.loc.current[1]);
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
      .outputs.at(mgmt.loc.current[1]);
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


// bool Program::update(std::string str1, std::string str2) noexcept {
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

// bool Program::update(pkmn_choice c1, pkmn_choice c2) noexcept {
//   if (!mgmt.cli_key.has_value()) {
//     err("update: A game must be in focus");
//     return false;
//   }
//   auto &history = *data.histories[mgmt.cli_key.value()];
//   const auto &state = *history.states.back();

//   auto next = std::make_unique<State>();
//   auto &bd = next->battle_data;
//   bd.battle = state.battle_data.battle;
//   bd.seed = *std::bit_cast<const uint64_t *>(bd.battle.bytes +
//   Offsets::seed); bd.options = state.battle_data.options; bd.result =
//   Init::update(bd.battle, c1, c2, bd.options); const auto [choices1,
//   choices2] = Init::choices(bd.battle, bd.result);

//   if (pkmn_result_type(bd.result)) {
//     bd.m = 0;
//     bd.n = 0;
//   } else {
//     bd.m = choices1.size();
//     bd.n = choices2.size();
//   }
//   std::copy(choices1.begin(), choices1.end(), bd.choices1.begin());
//   std::copy(choices2.begin(), choices2.end(), bd.choices2.begin());

//   next->nodes.emplace_back(std::make_unique<Node>());
//   next->nodes.emplace_back(std::make_unique<Node>());
//   next->outputs.resize(2);

//   history.states.emplace_back(std::move(next));

//   return true;
// }

// bool Program::rollout() {
//   if (depth() == 0) {
//     err("rollout: A Game must be in focus.");
//     return false;
//   }
//   auto &history = *data.histories[mgmt.cli_key.value()];
//   const auto *state = history.states.back().get();
//   prng device{123123};
//   while ((state->battle_data.m * state->battle_data.n) > 0) {
//     const auto i = device.random_int(state->battle_data.m);
//     const auto j = device.random_int(state->battle_data.n);
//     const bool success =
//         update(state->battle_data.choices1[i],
//         state->battle_data.choices2[j]);
//     if (!success) {
//       err("rollout: Bad update.");
//       return false;
//     }
//     state = history.states.back().get();
//   }
//   log("rollout: ", history.states.size(), " states.");
//   return true;
// };

// bool Program::rm(std::string key) noexcept {
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