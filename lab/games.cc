#include <games.h>

#include <data/strings.h>

#include <battle/sample-teams.h>
#include <battle/strings.h>

#include <pi/eval.h>

#include <sides.h>

namespace Lab {
namespace Games {

std::string Program::prompt() const {
  std::stringstream p{};
  const std::string esc{"\033[0m"};
  p << "\033[31m" << "(games)";

  std::array<std::string, 4> color_codes{"\033[33m Game ", "\033[34m State ",
                                         "\033[35m Node ", "\033[36m Output "};

  if (mgmt.loc.depth >= 1) {
    p << color_codes[0] << mgmt.loc.key;
  }
  for (int i = 0; i < static_cast<int>(mgmt.loc.depth) - 1; ++i) {
    p << color_codes[i + 1] << mgmt.loc.current[i];
  }
  p << "$ " << esc;
  return p.str();
}

void Program::loc() const {
  std::stringstream p{};
  p << "depth: " << mgmt.loc.depth << " index: " << mgmt.loc.current[0] << ' '
    << mgmt.loc.current[1] << ' ' << mgmt.loc.current[2] << std::endl;
  p << "bound: " << mgmt.bounds[0] << ' ' << mgmt.bounds[1] << ' '
    << mgmt.bounds[2] << std::endl;
  log(p.str());
}

bool Program::handle_command(const std::span<const std::string> words) {
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
  } else if (command == "battle") {
    return battle_bytes();
  } else if (command == "pokemon") {
    return pokemon_bytes();
  } else if (command == "trunc") {
    return trunc();
  }

  const std::span<const std::string> tail{words.begin() + 1, words.size() - 1};

  if (command == "update") {
    if (words.size() < 3) {
      err("update: Please enter u8 value of c1, c2 as a decimal (e.g. 5, 17)");
      return false;
    }
    return update(words[1], words[2]);
  } else if (command == "search") {
    return search(tail);
  } else if (command == "cp") {
    return cp(tail);
  } else if (command == "cd") {
    return cd(tail);
  }
  err("games: command '", command, "' not recognized");
  return false;
}

bool Program::trunc() {

  if (mgmt.loc.depth < 2) {
    err("trunc: A State must be in focus.");
    return false;
  }
  std::unique_lock lock{mgmt.mutex};

  auto &h = history();
  const auto n = mgmt.loc.current[0] + 1;
  h.states.resize(n);
  data.search_data_map.at(mgmt.loc.key).resize(n);
  return true;
}

bool Program::cp(const std::span<const std::string> words) {
  if (words.empty()) {
    err("cp: Missing source.");
    return false;
  }
  const auto source = words[0];
  if (!data.history_map.contains(source)) {
    err("cp: Source '", source, "' not found.");
    return false;
  }

  std::string dest;
  if (words.size() >= 2) {
    dest = words[1];
    if (data.history_map.contains(dest)) {
      err("cp: Destination '", dest, "' already present.");
      return false;
    }
  } else {
    size_t i = 1;
    do {
      dest = source + "(" + std::to_string(i) + ")";
      ++i;
    } while (data.history_map.contains(dest));
  }

  data.history_map[dest] = data.history_map[source];
  data.search_data_map[dest].clear();
  data.search_data_map[dest].resize(data.history_map[source].states.size());

  for (auto &vec : data.search_data_map[dest]) {
    vec.nodes.emplace_back(std::make_unique<Node>());
    vec.nodes.emplace_back(std::make_unique<Node>());
  }

  return true;
}

bool Program::save(std::filesystem::path path) { return false; }
bool Program::load(std::filesystem::path path) { return false; }

void Program::print() const {
  const auto print_output = [this](const MCTS::Output &o) {
    log("average value: ", o.average_value);
    log("iterations: ", o.iterations);
    for (auto i = 0; i < o.m; ++i) {
      for (auto j = 0; j < o.n; ++j) {
        auto value = std::format(
            "{:5.3f}",
            o.value_matrix[i][j] / std::max(uint32_t{1}, o.visit_matrix[i][j]));
        value = std::string{"", 5 - value.size()} + value;
        log_(value, " ");
      }
      log("");
    }
    const auto &battle = state().battle;
    for (auto i = 0; i < o.m; ++i) {
      log_(side_choice_string(battle.bytes, o.choices1[i]),
           std::format("{:>5.3f}", o.p1[i]), ' ');
    }
    log("");
    for (auto i = 0; i < o.n; ++i) {
      log_(side_choice_string(battle.bytes + Offsets::side, o.choices2[i]),
           std::format("{:>5.3f}", o.p2[i]), ' ');
    }
    log("");
  };

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
    log(history().states.size(), " states.");
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
    const auto &o = search_outputs().head;
    print_output(o);
    return;
  }
  case 4: {
    print_output(output());
    return;
  }
  }
}

bool Program::up() {
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
      mgmt.bounds[0] = history().states.size();
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
    default: {
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

bool Program::prev() {
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
bool Program::next() {
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
bool Program::first() {
  if (mgmt.loc.depth < 2) {
    err("first: A list must be in focus.");
    return false;
  }
  mgmt.loc.current[mgmt.loc.depth - 2] = 0;
  return true;
}
bool Program::last() {
  if (mgmt.loc.depth < 2) {
    err("last: A list must be in focus.");
    return false;
  }
  mgmt.loc.current[mgmt.loc.depth - 2] = mgmt.bounds[mgmt.loc.depth - 2] - 1;
  return true;
}

History &Program::history() { return data.history_map.at(mgmt.loc.key); }
State &Program::state() { return history().states.at(mgmt.loc.current[0]); }
SearchOutputs &Program::search_outputs() {
  return data.history_map.at(mgmt.loc.key)
      .states.at(mgmt.loc.current[0])
      .outputs.at(mgmt.loc.current[1]);
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
  return history().states.at(mgmt.loc.current[0]);
}
const SearchOutputs &Program::search_outputs() const {
  return data.history_map.at(mgmt.loc.key)
      .states.at(mgmt.loc.current[0])
      .outputs.at(mgmt.loc.current[1]);
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
  history.states.emplace_back();
  history.eval = Eval::CachedEval{p1.pokemon, p2.pokemon, data.ovo_dict};
  auto &state = history.states.front();
  state.battle = battle;
  state.options = {};
  state.m = 1;
  state.n = 1;
  state.outputs.resize(2);

  auto *durations = pkmn_gen1_battle_options_chance_durations(&state.options);
  auto &d = View::ref(*durations);
  for (auto s = 0; s < 2; ++s) {
    for (auto i = 0; i < 6; ++i) {
      const auto &set = s == 0 ? p1.pokemon[i] : p2.pokemon[i];
      if (Data::is_sleep(set.status)) {
        d.duration(s).set_sleep(i, set.sleep);
      }
    }
  }

  auto &search_data = data.search_data_map[key];
  search_data.emplace_back();
  search_data.front().nodes.emplace_back(std::make_unique<Node>());
  search_data.front().nodes.emplace_back(std::make_unique<Node>());

  return true;
}

bool Program::update(std::string str1, std::string str2) {
  if (mgmt.loc.depth == 0) {
    err("update: A game must be in focus");
    return false;
  }

  const auto &s = history().states.back();
  uint8_t x, y;
  try {
    x = std::stoi(str1);
  } catch (...) {
    std::array<std::string, 9> str_choices{};
    std::transform(
        s.choices1.begin(), s.choices1.begin() + s.m, str_choices.begin(),
        [&s](const auto c) { return side_choice_string(s.battle.bytes, c); });
    int i = Strings::unique_index(str_choices, str1);
    log("update: str1: ", str1, " i: ", i);
    if (i == -1) {
      err("update: Could not parse c1: ", str1);
      return false;
    }
    x = static_cast<size_t>(s.choices1[i]);
  }
  try {
    y = std::stoi(str2);
  } catch (...) {
    std::array<std::string, 9> str_choices{};
    std::transform(s.choices2.begin(), s.choices2.begin() + s.n,
                   str_choices.begin(), [&s](const auto c) {
                     return side_choice_string(s.battle.bytes + Offsets::side,
                                               c);
                   });
    int i = Strings::unique_index(str_choices, str2);
    log("update: str2: ", str2, " i: ", i);
    if (i == -1) {
      err("update: Could not parse c2: ", str2);
      return false;
    }
    y = static_cast<size_t>(s.choices2[i]);
  }

  return update(x, y);
}

bool Program::update(pkmn_choice c1, pkmn_choice c2) {
  if (mgmt.loc.depth == 0) {
    err("update: A game must be in focus");
    return false;
  }
  auto &h = history();
  auto &state = h.states.back();

  state.c1 = c1;
  state.c2 = c2;

  State next{};
  next.battle = state.battle;
  next.seed =
      *std::bit_cast<const uint64_t *>(next.battle.bytes + Offsets::seed);
  next.options = state.options;
  next.result = Init::update(next.battle, c1, c2, next.options);
  const auto [choices1, choices2] = Init::choices(next.battle, next.result);
  if (pkmn_result_type(next.result)) {
    next.m = 0;
    next.n = 0;
  } else {
    next.m = choices1.size();
    next.n = choices2.size();
  }
  std::copy(choices1.begin(), choices1.end(), next.choices1.begin());
  std::copy(choices2.begin(), choices2.end(), next.choices2.begin());

  // set up head of search outputs
  next.outputs.emplace_back();
  auto &o = next.outputs.front().head;
  o.m = next.m;
  o.n = next.n;
  o.choices1 = next.choices1;
  o.choices2 = next.choices2;
  next.outputs.emplace_back();
  next.outputs.back().head = o;

  h.states.emplace_back(next);

  auto &search_data_history = data.search_data_map.at(mgmt.loc.key);
  search_data_history.emplace_back();
  auto &s = search_data_history.at(h.states.size() - 1);
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
  const auto *state = &h.states.back();
  prng device{123123};
  while ((state->m * state->n) > 0) {
    const auto i = device.random_int(state->m);
    const auto j = device.random_int(state->n);
    const bool success = update(state->choices1[i], state->choices2[j]);
    if (!success) {
      err("rollout: Bad update.");
      return false;
    }
    state = &h.states.back();
  }
  log("rollout: ", h.states.size(), " states.");
  return true;
};

void accumulate(MCTS::Output &head, MCTS::Output &foot) {
  head.iterations += foot.iterations;
  head.duration += foot.duration;
  head.total_value += foot.total_value;
  head.average_value = head.total_value / head.iterations;
  head.p1 = {};
  head.p2 = {};
  for (auto i = 0; i < 9; ++i) {
    for (auto j = 0; j < 9; ++j) {
      head.visit_matrix[i][j] += foot.visit_matrix[i][j];
      head.value_matrix[i][j] += foot.value_matrix[i][j];
      head.p1[i] += head.visit_matrix[i][j];
      head.p2[j] += head.visit_matrix[i][j];
    }
  }
  for (auto i = 0; i < 9; ++i) {
    head.p1[i] /= head.iterations;
  }
  for (auto j = 0; j < 9; ++j) {
    head.p2[j] /= head.iterations;
  }
}

bool Program::battle_bytes() const {
  if (mgmt.loc.depth < 2) {
    err("battle: State must be in focus.");
    return false;
  }
  const auto &battle = state().battle;
  log(buffer_to_string(battle.bytes, 384));
  return true;
}
bool Program::pokemon_bytes() const {
  if (mgmt.loc.depth < 2) {
    err("battle: State must be in focus.");
    return false;
  }
  const auto &battle = state().battle;
  for (auto s = 0; s < 2; ++s) {
    for (auto p = 0; p < 6; ++p) {
      log(buffer_to_string(battle.bytes + (s * Offsets::side) +
                               (p * Offsets::pokemon),
                           Offsets::pokemon));
    }
  }
  return true;
}

bool Program::search(const std::span<const std::string> words) {
  if (mgmt.loc.depth < 3) {
    err("words: Node must be in focus.");
    return false;
  }
  if (words.size() != 3) {
    err("words: Invalid Args. Expecting 'mc'/'eval', 'time'/'count', <n> ");
    return false;
  }
  bool mc;
  if (words[0] == "mc") {
    mc = true;
  } else if (words[0] == "eval") {
    mc = false;
  } else {
    err("search: Could not parse value estimatation mode e.g. mc/eval");
    return false;
  }
  bool iter;
  if (words[1] == "time" || words[1] == "ms") {
    iter = false;
  } else if (words[1] == "count" || words[1] == "n") {
    iter = true;
  } else {
    err("search: Could not parse mode e.g. time/count");
    return false;
  }
  size_t n;
  try {
    n = std::stoi(words[2]);
  } catch (...) {
    err("search: Could not parse Arg #3");
    return false;
  }

  const auto &s = state();
  MCTS search{};
  MonteCarlo::Input input;
  input.battle = s.battle;
  input.durations = *pkmn_gen1_battle_options_chance_durations(&s.options);
  input.result = s.result;
  const auto &d = View::ref(input.durations);
  MCTS::Output output;
  MonteCarlo::Model model{9823457230948};
  Eval::Input eval_input{};
  eval_input.battle = input.battle;
  eval_input.durations = input.durations;
  eval_input.result = input.result;
  Eval::Model eval{240923840923, history().eval};
  try {
    if (iter) {
      if (mc) {
        output = search.run(n, node(), input, model);
      } else {
        output = search.run(n, node(), eval_input, eval);
        log("searching using eval");
      }
    } else {
      if (mc) {
        output = search.run(std::chrono::milliseconds{n}, node(), input, model);
      } else {
        log("searching using eval");
        output =
            search.run(std::chrono::milliseconds{n}, node(), eval_input, eval);
      }
    }
  } catch (const std::exception &e) {
    err("search: Exception thrown during run: ", e.what());
  }
  search_outputs().tail.push_back(output);
  accumulate(search_outputs().head, output);
  ++mgmt.bounds[2];
  log(output.iterations, " MCTS iterations completed in ",
      search_outputs().tail.back().duration.count(), " ms.");
  return true;
}

// try {
//   auto &x =
// }

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