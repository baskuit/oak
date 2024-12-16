#include <games.h>

#include <util/fs.h>

#include <data/strings.h>

#include <battle/sample-teams.h>
#include <battle/strings.h>

#include <pi/eval.h>

#include <sides.h>
// #include <log.h>

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
  Base::log(p.str());
}

bool Program::handle_command(const std::span<const std::string> words) {
  if (words.empty()) {
    return false;
  }
  const auto &command = words.front();

  if (command == "save" || command == "load") {

    if (words.size() < 2) {
      Base::err(command, ": missing arg(s).");
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
      Base::err(command, ": Failed.");
    } else {
      Base::log(command, ": Operation at path: '", path.string(), "' succeeded.");
    }
    return success;
  } else if (command == "ls") {
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
  } else if (command == "log") {
    return log();
  }
  
  const std::span<const std::string> tail{words.begin() + 1, words.size() - 1};

  if (command == "update") {
    if (words.size() < 3) {
      Base::err("update: Please enter u8 value of c1, c2 as a decimal (e.g. 5, 17)");
      return false;
    }
    return update(words[1], words[2]);
  } else if (command == "search") {
    return search(tail);
  } else if (command == "cp") {
    return cp(tail);
  } else if (command == "cd") {
    return cd(tail);
  } else if (command == "rm") {
    return rm(tail);
  }
  Base::err("games: command '", command, "' not recognized");
  return false;
}

bool Program::pkmn_debug(std::string word) {
  mgmt.pkmn_debug_path = word;
  return true;
}

bool Program::log() const {
  if (mgmt.loc.depth < 1) {
    return false;
  }
  // std::system(mgmt.pkmn_debug_path)
  const auto &h = history();
  h.debug_log.save_data_to_path("aa");

  return true;
}

bool Program::trunc() {

  if (mgmt.loc.depth < 2) {
    Base::err("trunc: A State must be in focus.");
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
    Base::err("cp: Missing source.");
    return false;
  }
  const auto source = words[0];
  if (!data.history_map.contains(source)) {
    Base::err("cp: Source '", source, "' not found.");
    return false;
  }

  std::string dest;
  if (words.size() >= 2) {
    dest = words[1];
    if (data.history_map.contains(dest)) {
      Base::err("cp: Destination '", dest, "' already present.");
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

bool Program::save(std::filesystem::path path) {

  constexpr bool overwrite = true;
  const auto mode =
      overwrite ? std::ios::binary : std::ios::binary | std::ios::trunc;
  std::ofstream file(path, mode);
  if (!file.is_open()) {
    return false;
  }

  for (const auto &[key, history] : data.history_map) {
    size_t s = key.size();
    file.write(std::bit_cast<const char *>(&s), sizeof(size_t));
    file.write(std::bit_cast<const char *>(key.data()), s);

    size_t n_states = history.states.size();
    file.write(std::bit_cast<const char *>(&n_states), sizeof(size_t));

    for (const auto &state : history.states) {

      file.write(std::bit_cast<const char *>(&state.battle),
                 PKMN_GEN1_BATTLE_SIZE);
      file.write(std::bit_cast<const char *>(&state.options),
                 PKMN_GEN1_BATTLE_OPTIONS_SIZE);
      file.write(std::bit_cast<const char *>(&state.result), 1);
      file.write(std::bit_cast<const char *>(&state.seed), 8);
      file.write(std::bit_cast<const char *>(&state.c1), 1);
      file.write(std::bit_cast<const char *>(&state.c2), 1);
      file.write(std::bit_cast<const char *>(&state.m), sizeof(size_t));
      file.write(std::bit_cast<const char *>(&state.n), sizeof(size_t));
      file.write(std::bit_cast<const char *>(&state.choices1), 9);
      file.write(std::bit_cast<const char *>(&state.choices2), 9);

      size_t n_nodes = state.outputs.size();
      file.write(std::bit_cast<const char *>(&n_nodes), sizeof(size_t));
      for (int i = 0; i < n_nodes; ++i) {
        const auto &o = state.outputs[i];
        file.write(std::bit_cast<const char *>(&o.head), sizeof(MCTS::Output));

        size_t n_searches = state.outputs[i].tail.size();
        file.write(std::bit_cast<const char *>(&n_searches), sizeof(size_t));

        for (int j = 0; j < n_searches; ++j) {
          file.write(std::bit_cast<const char *>(&o.tail[j]),
                     sizeof(MCTS::Output));
        }
      }
    }
  }

  file.close();
  return true;
}

bool Program::load(std::filesystem::path path) {
  std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  std::map<std::string, History> history_map{};

  file.seekg(0, std::ios::beg);
  while (file.peek() != EOF) {

    size_t n_key;
    if (!FS::try_read<size_t>(file, n_key)) {
      return false;
    }
    std::vector<char> buffer{};
    buffer.resize(n_key);
    file.read(buffer.data(), n_key);
    if (const auto g = file.gcount(); g != n_key) {
      return false;
    }
    std::string key{buffer.data(), n_key};
    auto &history = history_map[key];

    size_t n_states;
    if (!FS::try_read<size_t>(file, n_states)) {
      return false;
    }

    for (auto i = 0; i < n_states; ++i) {
      history.states.emplace_back();
      State &s = history.states.back();
      if (!FS::try_read<pkmn_gen1_battle>(file, s.battle)) {
        return false;
      }
      if (!FS::try_read<pkmn_gen1_battle_options>(file, s.options)) {
        return false;
      }
      if (!FS::try_read<pkmn_result>(file, s.result)) {
        return false;
      }
      if (!FS::try_read<uint64_t>(file, s.seed)) {
        return false;
      }
      if (!FS::try_read<pkmn_choice>(file, s.c1)) {
        return false;
      }
      if (!FS::try_read<pkmn_choice>(file, s.c2)) {
        return false;
      }
      if (!FS::try_read<size_t>(file, s.m)) {
        return false;
      }
      if (!FS::try_read<size_t>(file, s.n)) {
        return false;
      }
      if (!FS::try_read<std::array<pkmn_choice, 9>>(file, s.choices1)) {
        return false;
      }
      if (!FS::try_read<std::array<pkmn_choice, 9>>(file, s.choices2)) {
        return false;
      }

      size_t n_nodes;
      if (!FS::try_read<size_t>(file, n_nodes)) {
        return false;
      }
      for (int j = 0; j < n_nodes; ++j) {
        s.outputs.emplace_back();

        auto &so = s.outputs.back();
        if (!FS::try_read<MCTS::Output>(file, so.head)) {
          return false;
        }
        size_t n_outputs;
        if (!FS::try_read<size_t>(file, n_outputs)) {
          return false;
        }
        for (int k = 0; k < n_outputs; ++k) {
          so.tail.emplace_back();
          if (!FS::try_read<MCTS::Output>(file, so.tail.back())) {
            return false;
          }
        }
      }
    }
  }

  // set
  data.history_map = history_map;
  data.search_data_map.clear();
  for (const auto &[key, value] : history_map) {
    auto &search_data = data.search_data_map[key];

    for (const auto &state : value.states) {
      search_data.emplace_back();
      search_data.back().nodes.resize(state.outputs.size());
    }
  }

  file.close();
  return true;
}

// struct StateSearchData {
//   std::vector<std::unique_ptr<Node>> nodes;
// };

// struct History {
//   std::vector<State> states;
//   Eval::CachedEval eval;
// };

// // using History = std::vector<State>;
// using HistorySearchData = std::vector<StateSearchData>;

// struct ManagedData {
//   std::map<std::string, History> history_map;
//   std::map<std::string, std::vector<StateSearchData>> search_data_map;
//   Eval::OVODict ovo_dict;
// };

void Program::print() const {
  const auto print_output = [this](const MCTS::Output &o) {
    Base::log("average value: ", o.average_value);
    Base::log("iterations: ", o.iterations);
    for (auto i = 0; i < o.m; ++i) {
      for (auto j = 0; j < o.n; ++j) {
        auto value = std::format(
            "{:5.3f}",
            o.value_matrix[i][j] / std::max(uint32_t{1}, o.visit_matrix[i][j]));
        value = std::string{"", 5 - value.size()} + value;
        Base::log_(value, " ");
      }
      Base::log("");
    }
    const auto &battle = state().battle;
    for (auto i = 0; i < o.m; ++i) {
      Base::log_(side_choice_string(battle.bytes, o.choices1[i]),
           std::format("{:>5.3f}", o.p1[i]), ' ');
    }
    Base::log("");
    for (auto i = 0; i < o.n; ++i) {
      Base::log_(side_choice_string(battle.bytes + Offsets::side, o.choices2[i]),
           std::format("{:>5.3f}", o.p2[i]), ' ');
    }
    Base::log("");
  };

  switch (mgmt.loc.depth) {
  case 0: {
    Base::log(data.history_map.size(), " games:");
    for (const auto &[key, value] : data.history_map) {
      Base::log_(key, '\t');
    }
    Base::log("");
    return;
  }
  case 1: {
    Base::log(history().states.size(), " states.");
    return;
  }
  case 2: {
    const auto &s = state();
    Base::log(Strings::battle_to_string(s.battle));
    for (auto i = 0; i < s.m; ++i) {
      const auto c = s.choices1[i];
      Base::log_(side_choice_string(s.battle.bytes, c), ' ');
    }
    Base::log("");
    for (auto i = 0; i < s.n; ++i) {
      const auto c = s.choices2[i];
      Base::log_(side_choice_string(s.battle.bytes + Offsets::side, c), ' ');
    }
    Base::log("");
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
    Base::err("cd: Missing args.");
    return false;
  }

  const auto handle_word = [this](std::string word) {
    if (word == "..") {
      return up();
    } else if (word == "/") {
      mgmt.loc = {};
      mgmt.bounds = {};
      return true;
    }

    if (mgmt.loc.depth == 4) {
      Base::err("cd: Only '..' allowed here.");
      return false;
    } else if (mgmt.loc.depth == 0) {
      if (data.history_map.contains(word)) {
        mgmt.loc.key = word;
        ++mgmt.loc.depth;
        return true;
      } else {
        Base::err("cd: Game key '", word, "' not found.");
        return false;
      }
    }

    size_t index;
    try {
      index = std::stoi(word);
    } catch (...) {
      Base::err("cd: Bad argument; expecting index.");
      return false;
    }

    if (mgmt.bounds[mgmt.loc.depth - 1] <= index) {
      Base::err("cd: {depth: ", mgmt.loc.depth, " index: ", index,
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
    Base::err("prev: A list must be in focus.");
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
    Base::err("next: A list must be in focus.");
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
    Base::err("first: A list must be in focus.");
    return false;
  }
  mgmt.loc.current[mgmt.loc.depth - 2] = 0;
  return true;
}
bool Program::last() {
  if (mgmt.loc.depth < 2) {
    Base::err("last: A list must be in focus.");
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
Node *Program::node() {
  return data.search_data_map.at(mgmt.loc.key)
      .at(mgmt.loc.current[0])
      .nodes.at(mgmt.loc.current[1])
      .get();
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

const Node *Program::node() const {
  return data.search_data_map.at(mgmt.loc.key)
      .at(mgmt.loc.current[0])
      .nodes.at(mgmt.loc.current[1])
      .get();
}
const MCTS::Output &Program::output() const {
  return search_outputs().tail.at(mgmt.loc.current[2]);
}

bool Program::create(const std::string key, const Init::Config p1,
                     const Init::Config p2) {
  std::unique_lock lock{mgmt.mutex};
  if (data.history_map.contains(key)) {
    Base::err("create: '", key, "' already present.");
    return false;
  }

  const auto battle = Init::battle(p1, p2);
  data.history_map[key] = {};
  auto &history = data.history_map[key];
  history.states.emplace_back();
  // history.eval = Eval::CachedEval{p1.pokemon, p2.pokemon, data.ovo_dict};
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
        d.duration(s).set_sleep(i, set.sleep); // TODO null mons will break this
      }
    }
  }

  auto &search_data = data.search_data_map[key];
  search_data.emplace_back();
  search_data.front().nodes.emplace_back(std::make_unique<Node>());
  search_data.front().nodes.emplace_back(std::make_unique<Node>());

  history.debug_log.set_header(battle);

  return true;
}

bool Program::rm(const std::span<const std::string> words) {
  if (mgmt.loc.depth != 0) {
    Base::err("rm: A game cannot be in focus");
    return false;
  }
  if (words.size() == 0) {
    Base::err("rm: Missing args.");
    return false;
  }
  for (const auto key : words) {
    if (!data.history_map.contains(key)) {
      Base::err("rm: ", key, " not present.");
      return false;
    } else {
      data.history_map.erase(key);
    }
  }
  return true;
}

bool Program::update(std::string str1, std::string str2) {
  if (mgmt.loc.depth == 0) {
    Base::err("update: A game must be in focus");
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
    Base::log("update: str1: ", str1, " i: ", i);
    if (i == -1) {
      Base::err("update: Could not parse c1: ", str1);
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
    Base::log("update: str2: ", str2, " i: ", i);
    if (i == -1) {
      Base::err("update: Could not parse c2: ", str2);
      return false;
    }
    y = static_cast<size_t>(s.choices2[i]);
  }

  return update(x, y);
}

bool Program::update(pkmn_choice c1, pkmn_choice c2) {
  if (mgmt.loc.depth == 0) {
    Base::err("update: A game must be in focus");
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
  next.result = h.debug_log.update(next.battle, c1, c2, next.options);
  // next.result = Init::update(next.battle, c1, c2, next.options);
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
  next.outputs.emplace_back();
  for (auto &so : next.outputs) {
    auto &o = so.head;
    o.m = next.m;
    o.n = next.n;
    o.choices1 = next.choices1;
    o.choices2 = next.choices2;
  }

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
    Base::err("rollout: A Game must be in focus.");
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
      Base::err("rollout: Bad update.");
      return false;
    }
    state = &h.states.back();
  }
  Base::log("rollout: ", h.states.size(), " states.");
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
    Base::err("battle: State must be in focus.");
    return false;
  }
  const auto &battle = state().battle;
  Base::log(buffer_to_string(battle.bytes, 384));
  return true;
}
bool Program::pokemon_bytes() const {
  if (mgmt.loc.depth < 2) {
    Base::err("battle: State must be in focus.");
    return false;
  }
  const auto &battle = state().battle;
  for (auto s = 0; s < 2; ++s) {
    for (auto p = 0; p < 6; ++p) {
      Base::log(buffer_to_string(battle.bytes + (s * Offsets::side) +
                               (p * Offsets::pokemon),
                           Offsets::pokemon));
    }
  }
  return true;
}

bool Program::search(const std::span<const std::string> words) {
  if (mgmt.loc.depth < 3) {
    Base::err("search: Node must be in focus.");
    return false;
  }
  if (words.size() != 3) {
    Base::err("search: Invalid Args. Expecting 'mc'/'eval', 'time'/'count', <n> ");
    return false;
  }
  if (node() == nullptr) {
    Base::err("search: Cannot search on loaded (archived) games. Copy this game then "
        "search on that.");
    return false;
  }
  bool mc;
  if (words[0] == "mc") {
    mc = true;
  } else if (words[0] == "eval") {
    mc = false;
  } else {
    Base::err("search: Could not parse value estimatation mode e.g. mc/eval");
    return false;
  }
  bool iter;
  if (words[1] == "time" || words[1] == "ms") {
    iter = false;
  } else if (words[1] == "count" || words[1] == "n") {
    iter = true;
  } else {
    Base::err("search: Could not parse mode e.g. time/count");
    return false;
  }
  size_t n;
  try {
    n = std::stoi(words[2]);
  } catch (...) {
    Base::err("search: Could not parse Arg #3");
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
        output = search.run(n, *node(), input, model);
      } else {
        output = search.run(n, *node(), eval_input, eval);
        Base::log("searching using eval");
      }
    } else {
      if (mc) {
        output =
            search.run(std::chrono::milliseconds{n}, *node(), input, model);
      } else {
        Base::log("searching using eval");
        output =
            search.run(std::chrono::milliseconds{n}, *node(), eval_input, eval);
      }
    }
  } catch (const std::exception &e) {
    Base::err("search: Exception thrown during run: ", e.what());
  }
  search_outputs().tail.push_back(output);
  accumulate(search_outputs().head, output);
  ++mgmt.bounds[2];
  Base::log(output.iterations, " MCTS iterations completed in ",
      search_outputs().tail.back().duration.count(), " ms.");
  return true;
}

} // namespace Games
} // namespace Lab