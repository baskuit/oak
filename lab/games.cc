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

bool Program::playout() {
  if (mgmt.loc.depth == 0) {
    Base::err("playout: A game must be in focus.");
    return false;
  }
  auto &h = history();
  const auto *state = &h.states.back();
  prng device{mgmt.device.uniform_64()};
  while ((state->choices.m * state->choices.n) > 0) {

    MCTS search{};
    MonteCarlo::Input input;
    input.battle = state->frame.battle;
    input.durations =
        *pkmn_gen1_battle_options_chance_durations(&state->frame.options);
    input.result = state->frame.result;
    MonteCarlo::Model model{mgmt.device.uniform_64()};
    Node node{};
    const auto output = search.run(1000000, node, input, model);

    const auto i = device.sample_pdf(output.p1);
    const auto j = device.sample_pdf(output.p2);

    // const auto i = device.random_int(state->choices.m);
    // const auto j = device.random_int(state->choices.n);
    const bool success = update(state->choices.p1[i], state->choices.p2[j]);
    if (!success) {
      Base::err("rollout: Bad update.");
      return false;
    }
    state = &h.states.back();
  }
  return true;
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
      Base::log(command, ": Operation at path: '", path.string(),
                "' succeeded.");
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
  } else if (command == "playout") {
    return playout();
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
    return update(tail);
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
    Base::err("log: A game is not in focus.");
    return false;
  }
  // std::system(mgmt.pkmn_debug_path)
  const auto key = mgmt.loc.key;
  const auto &h = history();
  std::filesystem::path path{key};
  std::fstream file;
  file.open(path, std::ios::binary | std::ios::app);
  if (!file.is_open()) {
    Base::err("log: Can't write to disk.");
    return false;
  }
  const char bytes[8] = {1, 1, 0, 1, 0, 0, 0, 0};
  file.write(bytes, 8);
  file.write(std::bit_cast<const char *>(&h.header.battle),
             PKMN_GEN1_BATTLE_SIZE);
  for (const auto &state : h.states) {
    file.write(std::bit_cast<const char *>(state.frame.log.data()), 256);
    file.write(std::bit_cast<const char *>(&state.frame.battle),
               PKMN_GEN1_BATTLE_SIZE);
    file.write(std::bit_cast<const char *>(&state.frame.result), 1);
    file.write(std::bit_cast<const char *>(&state.frame.c1), 1);
    file.write(std::bit_cast<const char *>(&state.frame.c2), 1);
  }
  file.close();
  const std::string cmd =
      mgmt.pkmn_debug_path.string() + " ./" + key + " > index.html";
  // const std::string cmd = mgmt.pkmn_debug_path.string() + " ./" + key + " > "
  // + key + ".html";
  const int r = std::system(cmd.data());

  return (r == 0);
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
      const auto &frame = state.frame;
      const auto &choices = state.choices;
      file.write(std::bit_cast<const char *>(&frame.battle),
                 PKMN_GEN1_BATTLE_SIZE);
      file.write(std::bit_cast<const char *>(&frame.options),
                 PKMN_GEN1_BATTLE_OPTIONS_SIZE);
      file.write(std::bit_cast<const char *>(&frame.result), 1);
      file.write(std::bit_cast<const char *>(&frame.seed), 8);
      file.write(std::bit_cast<const char *>(&frame.c1), 1);
      file.write(std::bit_cast<const char *>(&frame.c2), 1);
      file.write(std::bit_cast<const char *>(&choices.m),
                 sizeof(decltype(choices.m)));
      file.write(std::bit_cast<const char *>(&choices.n),
                 sizeof(decltype(choices.n)));
      file.write(std::bit_cast<const char *>(&choices.p1), 9);
      file.write(std::bit_cast<const char *>(&choices.p2), 9);

      const size_t n_nodes = state.outputs.size();
      file.write(std::bit_cast<const char *>(&n_nodes), sizeof(size_t));
      for (int i = 0; i < n_nodes; ++i) {
        const auto &o = state.outputs[i];
        file.write(std::bit_cast<const char *>(&o.head), sizeof(MCTS::Output));

        const size_t n_searches = state.outputs[i].tail.size();
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
      auto &frame = s.frame;
      auto &choices = s.choices;
      if (!FS::try_read<pkmn_gen1_battle>(file, frame.battle)) {
        return false;
      }
      if (!FS::try_read<pkmn_gen1_battle_options>(file, frame.options)) {
        return false;
      }
      if (!FS::try_read<pkmn_result>(file, frame.result)) {
        return false;
      }
      if (!FS::try_read<uint64_t>(file, frame.seed)) {
        return false;
      }
      if (!FS::try_read<pkmn_choice>(file, frame.c1)) {
        return false;
      }
      if (!FS::try_read<pkmn_choice>(file, frame.c2)) {
        return false;
      }
      if (!FS::try_read<decltype(choices.m)>(file, choices.m)) {
        return false;
      }
      if (!FS::try_read<decltype(choices.n)>(file, choices.n)) {
        return false;
      }
      if (!FS::try_read<std::array<pkmn_choice, 9>>(file, choices.p1)) {
        return false;
      }
      if (!FS::try_read<std::array<pkmn_choice, 9>>(file, choices.p2)) {
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
  Base::log("load: Successful.");
  return true;
}

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
    const auto &battle = state().frame.battle;
    for (auto i = 0; i < o.m; ++i) {
      Base::log_(side_choice_string(battle.bytes, o.choices1[i]),
                 std::format("{:>5.3f}", o.p1[i]), ' ');
    }
    Base::log("");
    for (auto i = 0; i < o.n; ++i) {
      Base::log_(
          side_choice_string(battle.bytes + Offsets::side, o.choices2[i]),
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
    Base::log("Eval enabled: ", history().header.eval_enabled);
    return;
  }
  case 2: {
    const auto &s = state();
    Base::log(Strings::battle_to_string(s.frame.battle));
    for (auto i = 0; i < s.choices.m; ++i) {
      const auto c = s.choices.p1[i];
      Base::log_(side_choice_string(s.frame.battle.bytes, c), ' ');
    }
    Base::log("");
    for (auto i = 0; i < s.choices.n; ++i) {
      const auto c = s.choices.p2[i];
      Base::log_(side_choice_string(s.frame.battle.bytes + Offsets::side, c),
                 ' ');
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
NodeData &Program::search_outputs() {
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
const NodeData &Program::search_outputs() const {
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

  Header header;
  header.p1 = p1;
  header.p2 = p2;
  header.battle = Init::battle(p1, p2);

  Frame frame{};
  frame.battle = header.battle;
  auto *durations = pkmn_gen1_battle_options_chance_durations(&frame.options);
  auto &d = View::ref(*durations);
  for (auto s = 0; s < 2; ++s) {
    for (auto i = 0; i < 6; ++i) {
      const auto &set = s == 0 ? p1.pokemon[i] : p2.pokemon[i];
      if (Data::is_sleep(set.status)) {
        d.duration(s).set_sleep(i,
                                set.sleep); // TODO null mons will break this?
      }
    }
  }

  State first{};
  const bool success = update(header.battle, 0, 0, first);
  if (!success) {
    Base::err(
        "create: Initial 0, 0 update returned non null result. Bad sides.");
    return false;
  }

  auto &history = data.history_map[key];
  history.header = header;
  auto &dict = data.ovo_dict;
  if ([&history, p1, p2, &dict]() {
        for (const auto &set1 : p1.pokemon) {
          for (const auto &set2 : p2.pokemon) {
            if (!dict.contains(set1, set2)) {
              return false;
            }
          }
        }
        return true;
      }()) {
    history.header.eval_enabled = true;
    history.eval = Eval::CachedEval{p1.pokemon, p2.pokemon, data.ovo_dict};
  }

  history.states.emplace_back(first);
  auto &search_data_history = data.search_data_map[key];
  search_data_history.emplace_back();
  auto &search_data = search_data_history.back();
  search_data.nodes.emplace_back(std::make_unique<Node>());
  search_data.nodes.emplace_back(std::make_unique<Node>());
  ++mgmt.bounds[0];
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

bool Program::update(const std::span<const std::string> words) {
  if (mgmt.loc.depth == 0) {
    Base::err("update: A game must be in focus");
    return false;
  }
  if (words.size() < 2) {
    Base::err(
        "update: Please enter u8 value of c1, c2 as a decimal (e.g. 5, 17)");
    return false;
  }
  const auto str1 = words[0];
  const auto str2 = words[1];
  const auto &s = history().states.back();
  uint8_t x, y;
  try {
    x = std::stoi(str1);
  } catch (...) {
    std::array<std::string, 9> str_choices{};
    std::transform(s.choices.p1.begin(), s.choices.p1.begin() + s.choices.m,
                   str_choices.begin(), [&s](const auto c) {
                     return side_choice_string(s.frame.battle.bytes, c);
                   });
    int i = Strings::unique_index(str_choices, str1);
    Base::log("update: str1: ", str1, " i: ", i);
    if (i == -1) {
      Base::err("update: Could not parse c1: ", str1);
      return false;
    }
    x = static_cast<size_t>(s.choices.p1[i]);
  }
  try {
    y = std::stoi(str2);
  } catch (...) {
    std::array<std::string, 9> str_choices{};
    std::transform(s.choices.p2.begin(), s.choices.p2.begin() + s.choices.n,
                   str_choices.begin(), [&s](const auto c) {
                     return side_choice_string(
                         s.frame.battle.bytes + Offsets::side, c);
                   });
    int i = Strings::unique_index(str_choices, str2);
    Base::log("update: str2: ", str2, " i: ", i);
    if (i == -1) {
      Base::err("update: Could not parse c2: ", str2);
      return false;
    }
    y = static_cast<size_t>(s.choices.p2[i]);
  }

  return update(x, y);
}

bool Program::update(pkmn_choice c1, pkmn_choice c2) {
  const auto &s = history().states.back();
  State next{};
  const bool success = update(s.frame.battle, c1, c2, next);
  if (success) {
    history().states.emplace_back(next);
    auto &search_data_history = data.search_data_map.at(mgmt.loc.key);
    search_data_history.emplace_back();
    auto &s = search_data_history.back();
    s.nodes.emplace_back(std::make_unique<Node>());
    s.nodes.emplace_back(std::make_unique<Node>());
    ++mgmt.bounds[0];
  }
  return success;
}

bool Program::update(const pkmn_gen1_battle &battle, pkmn_choice c1,
                     pkmn_choice c2, State &state) {
  auto &frame = state.frame;
  frame.c1 = c1;
  frame.c2 = c2;
  frame.battle = battle;
  frame.seed =
      *std::bit_cast<const uint64_t *>(frame.battle.bytes + Offsets::seed);
  frame.options = state.frame.options;
  pkmn_gen1_log_options log_options{state.frame.log.data(), 256};
  Log::set(frame.options, &log_options, nullptr, nullptr);
  frame.result = Log::update(frame.battle, c1, c2, frame.options);
  const auto [choices1, choices2] = Init::choices(frame.battle, frame.result);

  if (pkmn_result_type(frame.result) == PKMN_RESULT_ERROR) {
    Base::err("update: Error during update call");
    return false;
  }

  auto &choices = state.choices;
  if (pkmn_result_type(state.frame.result)) {
    choices.m = 0;
    choices.n = 0;
  } else {
    const auto [choices1, choices2] =
        Init::choices(state.frame.battle, state.frame.result);
    choices.m = choices1.size();
    choices.n = choices2.size();
    std::copy(choices1.begin(), choices1.end(), choices.p1.begin());
    std::copy(choices2.begin(), choices2.end(), choices.p2.begin());
  }

  auto &outputs = state.outputs;
  outputs.resize(2);
  for (auto &so : outputs) {
    auto &o = so.head;
    o.m = choices.m;
    o.n = choices.n;
    o.choices1 = choices.p1;
    o.choices2 = choices.p2;
  }
  return true;
}

bool Program::rollout() {
  if (mgmt.loc.depth == 0) {
    Base::err("rollout: A game must be in focus.");
    return false;
  }
  auto &h = history();
  const auto *state = &h.states.back();
  prng device{mgmt.device.uniform_64()};
  while ((state->choices.m * state->choices.n) > 0) {
    const auto i = device.random_int(state->choices.m);
    const auto j = device.random_int(state->choices.n);
    const bool success = update(state->choices.p1[i], state->choices.p2[j]);
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
  const auto &battle = state().frame.battle;
  Base::log(buffer_to_string(battle.bytes, 384));
  return true;
}
bool Program::pokemon_bytes() const {
  if (mgmt.loc.depth < 2) {
    Base::err("battle: State must be in focus.");
    return false;
  }
  const auto &battle = state().frame.battle;
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
    Base::err(
        "search: Invalid Args. Expecting 'mc'/'eval', 'time'/'count', <n> ");
    return false;
  }
  if (node() == nullptr) {
    Base::err(
        "search: Cannot search on loaded (archived) games. Copy this game then "
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
  input.battle = s.frame.battle;
  input.durations =
      *pkmn_gen1_battle_options_chance_durations(&s.frame.options);
  input.result = s.frame.result;
  const auto &d = View::ref(input.durations);
  MCTS::Output output;
  MonteCarlo::Model model{mgmt.device.uniform_64()};
  Eval::Input eval_input{};
  eval_input.battle = input.battle;
  eval_input.durations = input.durations;
  eval_input.result = input.result;
  Eval::Model eval{mgmt.device.uniform_64(), history().eval};
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