#include <data/sample-teams.h>

#include <battle/init.h>
#include <battle/strings.h>

#include <pi/eval.h>
#include <pi/mcts.h>
#include <pi/tree.h>

#include <pkmn.h>

#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

using Node =
    Tree::Node<Exp3::JointBanditData<.01f, false>, std::array<uint8_t, 16>>;

// return bool tells you if the command succeeded. in particular, false means
// Program was not mutated
struct Program {
private:
  Eval::OVODict one_versus_one_tables;

  using Sets = std::map<std::string, SampleTeams::Set>;
  using Teams = std::map<std::string, std::array<SampleTeams::Set, 6>>;
  // any decision point in the battle, along with search data
  struct State {
    MonteCarlo::Input search_input;
    struct SearchData {
      std::unique_ptr<Node> node;
      using Output = MCTS::Output;
      Output output;
      std::vector<Output> component_outputs;
    };
    std::array<SearchData, 2> search_data;

    pkmn_choice c1;
    pkmn_choice c2;
  };
  struct Game {
    std::vector<State> states;
    Eval::Model eval;

    Game(const auto p1, const auto p2, const Eval::OVODict &ovo_dict,
         auto seed = 79283457290384)
        : states{}, eval{seed, Eval::CachedEval{p1, p2, ovo_dict}} {}
  };
  using Games = std::map<std::string, Game>;

  Sets sets;
  Teams teams;
  Games games;

  prng device;
  size_t set_counter;
  size_t team_counter;
  size_t game_counter;

  struct Worker {
    std::thread thread;
  };
  std::map<size_t, Worker> workers;
  Worker *get_worker() {
    for (auto &[key, worker] : workers) {
      if (!worker.thread.joinable()) {
        return &worker;
      }
    }
    return nullptr;
  }

  // Determines what commands are offered and what is printed
  struct InterfaceFocus {
    std::optional<std::string> game_key;
    std::optional<size_t> state_index;
    std::optional<size_t> tree_index;
    std::optional<size_t> output_index;
  };
  InterfaceFocus loc;

public:
  Program() = default;

  bool load_tables(std::filesystem::path path) {
    return one_versus_one_tables.load(path);
  }
  bool load_teams(std::filesystem::path path) {}
  bool save_tables(std::filesystem::path path) const {}
  bool save_teams(std::filesystem::path path) const {}

  bool add_set(std::string string) {}
  bool add_team(std::string string) {}
  bool delete_set(std::string string) {}
  bool delete_team(std::string string) {}

  bool create_game(std::string team1, std::string team2) {}
  bool delete_game(std::string index) {}

  bool search(auto iterations) {
    if (depth() < 3) {
      return false;
    }
    try {
      auto &search_data = data();
      MCTS search{};
      const auto output = search.run(iterations, search_data.node,
                                     state().search_input, game().eval);
    } catch (std::exception &e) {
      std::cerr << e.what() << std::endl;
      return false;
    }
    return true;
  }
  bool c1() {
    if (depth() < 2) {
      return false;
    }
    return true;
  }
  bool c2() {
    if (depth() < 2) {
      return false;
    }
    return true;
  }
  bool update() {
    if (depth() < 2) {
      return false;
    }
    return true;
  }
  bool update(auto c1, auto c2) {
    if (depth() < 2) {
      return false;
    }
    return true;
  }
  bool undo() {}
  bool clone() {}
  bool trunc() {}
  bool name() {}

  // User Interface

  bool out() {}
  bool in() {}

  void print_loc() const {}

  void print_help() const {}

private:
  size_t depth() const {
    if (loc.output_index.has_value()) {
      return 4;
    }
    if (loc.tree_index.has_value()) {
      return 3;
    }
    if (loc.state_index.has_value()) {
      return 2;
    }
    if (loc.game_key.has_value()) {
      return 1;
    }
    return 0;
  }

  Game &game() { return games.at(loc.game_key.value()); }

  State &state() { return game().states.at(loc.state_index.value()); }

  State::SearchData &data() {
    return State().search_data.at(loc.output_index.value());
  }
};

int main_loop(int argc, char **argv) {

  // ProgramData data{};

  while (true) {

    std::string message;

    std::cin >> message;
    Data::Moves x;
    try {
      x = Strings::string_to_move(message);
    } catch (const std::exception &e) {
      std::cout << e.what() << std::endl;
      continue;
    }

    std::cout << (int)x << std::endl;
  }
}

int main(int argc, char **argv) { return main_loop(argc, argv); }