#pragma once

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

using SampleTeams::Set;

struct ActivePokemon {
  // Data::Species species;
  struct Boosts {
    int atk;
    int def;
    int spe;
    int spc;
  };
  uint8_t slot;
  bool leech_seed;
};

struct Side {
  ActivePokemon active;
  std::array<Set, 6> party;
};

struct Sides {

  Sides (std::ostream *out, std::ostream *err)  : out{out}, err{err} {}


  std::ostream *out;
  std::ostream *err;

  struct Location {
    enum class Slot {
      Active,
      One,
      Two,
      Three,
      Four,
      Five,
      Six,
    };
    std::optional<std::string> key;
    std::optional<Slot> slot;
  };

  std::map<std::string, Side> sides;
  Location loc;

  void show () const noexcept {

    if (loc.key.has_value()) {
      const auto &side = sides.at(loc.key.value());
      if (loc.slot.has_value()) {
        *out << "TODO printing party poke" << std::endl;
      } else {
        *out << loc.key.value() << ":" << std::endl;
        for (const auto &set : side.party) {
          *out << Names::species_string(set.species) << ' ';
        }
        *out << std::endl;
      }
    } else {
      const auto n = sides.size();
      *out << n << " sides." << std::endl;
    }

  }

  bool add(std::string key) {
    if (sides.contains(key)) {
      *err << key << " already present." << std::endl;
      return false;
    } else {
      sides[key] = {};
      *out << key << " added to sides." << std::endl;
      return true;
    }
  }

  
};

struct Program {
  std::ostream *out;
  std::ostream *err;

  Sides sides;

  enum class Location {
    Sides,
    Games,
  };

  Location loc;

  Program (std::ostream *out, std::ostream *err)  : sides{out, err} {}

  const char* prompt () const noexcept {
    static const char s[]{"Sides$ "};
    static const char g[]{"Games$ "};
    switch (loc) {
      case Location::Sides:
      return s;
      case Location::Games:
      return g;
    }
  }

  void goto_sides() noexcept {
    loc = Location::Sides;
  }

  void goto_games() noexcept {
    loc = Location::Games;
  }

};

// using Node =
//     Tree::Node<Exp3::JointBanditData<.01f, false>, std::array<uint8_t, 16>>;

// // return bool tells you if the command succeeded. in particular, false means
// // Program was not mutated
// struct Program {
// private:
//   // std::ostream out{std::cout};
//   // std::ostream err{std::cerr};

//   Eval::OVODict one_versus_one_tables;

//   using Sets = std::map<std::string, SampleTeams::Set>;
//   using Teams = std::map<std::string, std::array<SampleTeams::Set, 6>>;

//   // any decision point in the battle, along with search data
//   struct State {
//     MonteCarlo::Input search_input;
//     struct SearchData {
//       std::unique_ptr<Node> node{std::make_unique<Node>()};
//       using Output = MCTS::Output;
//       Output output;
//       std::vector<Output> component_outputs;
//     };
//     std::array<SearchData, 2> search_data;

//     pkmn_choice c1;
//     pkmn_choice c2;

//     State() = default;
//     State(State &&) = default;
//   };

//   struct Game {
//     std::vector<State> states;
//     Eval::Model eval;

//     Game(const auto p1, const auto p2, const Eval::OVODict &ovo_dict,
//          auto seed = 79283457290384)
//         : states{}, eval{seed, Eval::CachedEval{p1, p2, ovo_dict}} {}

//     Game() = default;
//     Game(Game &&) = default;
//   };

//   using Games = std::map<std::string, Game>;

//   Sets sets;
//   Teams teams;
//   Games games;

//   prng device;
//   size_t set_counter;
//   size_t team_counter;
//   size_t game_counter;

//   struct Worker {
//     // std::thread thread;
//   };
//   std::map<size_t, Worker> workers;
//   Worker *get_worker() {
//     // for (auto &[key, worker] : workers) {
//     //   if (!worker.thread.joinable()) {
//     //     return &worker;
//     //   }
//     // }
//     return nullptr;
//   }

//   // Determines what commands are offered and what is printed
//   struct InterfaceFocus {
//     std::optional<std::string> game_key;
//     std::optional<size_t> state_index;
//     std::optional<size_t> tree_index;
//     std::optional<size_t> output_index;
//   };
//   InterfaceFocus loc;

// public:
//   Program() = default;

//   bool load_tables(std::filesystem::path path) {
//     return one_versus_one_tables.load(path);
//   }
//   bool load_teams(std::filesystem::path path) { return false; }
//   bool save_tables(std::filesystem::path path) {
//     return one_versus_one_tables.save(path);
//   }
//   bool save_teams(std::filesystem::path path) const { return false; }

//   bool add_set(std::string string) { return false; }
//   bool add_team(std::string string) { return false; }
//   bool delete_set(std::string string) { return false; }
//   bool delete_team(std::string string) { return false; }

//   bool create_game(std::string team1, std::string team2) { return false; }
//   bool delete_game(std::string index) { return false; }

//   bool search(auto iterations) {
//     if (depth() < 3) {
//       return false;
//     }
//     try {
//       auto &search_data = data();
//       MCTS search{};
//       const auto output = search.run(iterations, search_data.node,
//                                      state().search_input, game().eval);
//     } catch (std::exception &e) {
//       std::cerr << e.what() << std::endl;
//       return false;
//     }
//     return true;
//   }
//   bool c1() {
//     if (depth() < 2) {
//       return false;
//     }
//     return true;
//   }
//   bool c2() {
//     if (depth() < 2) {
//       return false;
//     }
//     return true;
//   }
//   bool update(auto c1, auto c2) {
//     if (depth() < 2) {
//       return false;
//     }
//     return true;
//   }

//   bool clone() {
//     if (depth() < 2) {
//       return false;
//     }

//     return true;
//   }
//   bool trunc() {
//     if (depth() < 2) {
//       return false;
//     }
//     auto &g = game();
//     g.states.resize(loc.state_index.value() + 1);
//     return true;
//   }
//   // User Interface

//   bool out() { return false; }
//   bool in() { return false; }

//   void print_loc() const {
//     std::stringstream sstream{};
//     switch (depth()) {
//     case 4:
//       sstream << "Output: " << loc.tree_index.value() << ' ';
//     case 3:
//       sstream << "Node: " << loc.tree_index.value() << ' ';
//     case 2:
//       sstream << "State: " << loc.tree_index.value() << ' ';
//     case 1:
//       sstream << "Game: " << loc.tree_index.value() << ' ';
//       break;
//     default:
//       sstream << "No game in focus. " << games.size() << " games in memory.";
//     }
//     std::cout << sstream.str() << std::endl;
//   }

//   void print_help() const {}

// private:
//   void clone_game() {
//     Game g{};
//     const auto &h = game();
//     const auto n = h.states.size();
//     g.states.resize(n);
//     for (auto i = 0; i < n; ++i) {
//       auto &s = g.states[i];
//       const auto &t = h.states[i];

//       s.search_input = t.search_input;
//       s.c1 = t.c1;
//       s.c2 = t.c2;
//     }
//     auto c = ++game_counter;
//     while (games.contains(std::to_string(c))) {
//       c = ++game_counter;
//     }
//   }

//   size_t depth() const {
//     if (loc.output_index.has_value()) {
//       return 4;
//     }
//     if (loc.tree_index.has_value()) {
//       return 3;
//     }
//     if (loc.state_index.has_value()) {
//       return 2;
//     }
//     if (loc.game_key.has_value()) {
//       return 1;
//     }
//     return 0;
//   }

//   Game &game() { return games.at(loc.game_key.value()); }

//   State &state() { return game().states.at(loc.state_index.value()); }

//   State::SearchData &data() {
//     return State().search_data.at(loc.output_index.value());
//   }
// };