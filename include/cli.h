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

template <typename Out, typename... Args>
void log(Out *out, const Args &...messages) {
  ((*out << messages << " "), ...) << std::endl;
}

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

  std::ostream *out;
  std::ostream *err;

  std::map<std::string, Side> sides;
  Location loc;

  Sides(std::ostream *out, std::ostream *err) : out{out}, err{err} {}

  void show() const noexcept {
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
    if (loc.key.has_value()) {
      log(err, "add: Need to run'out' first");
      return false;
    }
    if (sides.contains(key)) {
      *err << key << " already present." << std::endl;
      return false;
    } else {
      sides[key] = {};
      *out << key << " added to sides." << std::endl;
      return true;
    }
  }

  bool rm(std::string key) {
    if (loc.key.has_value()) {
      log(err, "rm: Need to run'out' first");
      return false;
    }
    if (sides.contains(key)) {
      sides.erase(key);
      *out << key << " removed from sides." << std::endl;
      return true;
    } else {
      *err << key << " not in sides." << std::endl;
      return false;
    }
  }

  bool set_mon(std::span<const std::string> words) {

    if (!loc.slot.has_value()) {
      log(err, "set: Not viewing a side or slot.");
      return false;
    }

    if (loc.slot.value() == Location::Slot::Active) {
      log(err, "set: In active slot; use TODO instead.");
      return false;
    }

    auto &slot =
        sides[loc.key.value()].party[static_cast<size_t>(loc.slot.value())];

    Data::Species species{};
    std::vector<Data::Moves> moves{};

    Data::Species in_species{};
    Data::Moves in_move{};

    const auto update = [this, &moves, &species](const auto data) {
      const auto err = this->err;
      using T = std::remove_reference_t<decltype(data)>;
      if constexpr (std::is_convertible_v<T, Data::Moves>) {
        if (data == Data::Moves::None) {
          log(err, "set: 'None' read.");
          return false;
        }
        const auto it = std::find(moves.begin(), moves.end(), data);
        if (it == moves.end()) {
          if (moves.size() >= 4) {
            log(err, "set: Too many moves.");
            return false;
          } else {
            moves.push_back(data);
            return true;
          }
        } else {
          log(err, "set: Read duplicate move.");
          return false;
        }
      } else if constexpr (std::is_convertible_v<T, Data::Species>) {
        if (data == Data::Species::None) {
          log(err, "set: 'None' read.");
          return false;
        }
        if (species == Data::Species::None) {
          species = data;
          return true;
        } else {
          log(err, "set: Species already read.");
          return false;
        }
      } else {
        log(err, "set: Invalid data type in update lambda???");
        std::cout << typeid(data).name() << std::endl;
        return false;
      }
    };

    for (const auto &word : words) {
      try {
        in_species = Strings::string_to_species(word);
      } catch (...) {
        try {
          in_move = Strings::string_to_move(word);
        } catch (...) {
          log(err, "set: Invalid species/move.");
          return false;
        }
        if (!update(in_move)) {
          log(err, "set: Update move failed.");

          return false;
        }
        continue;
      }
      if (!update(in_species)) {
        log(err, "set: Update species failed");

        return false;
      }
      continue;
    }
    if (species == Data::Species::None) {
      log(err, "set: Failed to read species.");
      return false;
    }

    slot.species = species;
    std::copy(moves.begin(), moves.end(), slot.moves.begin());
    return true;
  }

  bool in(std::string s) {
    if (loc.key.has_value()) {
      if (loc.slot.has_value()) {
        return false;
      } else {
        int slot;
        try {
          slot = std::stoi(s);
        } catch (...) {
          if (s == "active") {
            loc.slot = Location::Slot::Active;
            return true;
          }
          *err << "in: could not read index" << std::endl;
          return false;
        }
        if ((slot < 0) || (slot > 6)) {
          *err << "in: index out of range [0 (active), 1 (party), ..., 6] "
               << std::endl;
          return false;
        } else {
          loc.slot = static_cast<Location::Slot>(slot);
          return true;
        }
      }
    } else {
      if (sides.contains(s)) {
        loc.key = s;
        return true;
      } else {
        log(err, "in: Side ", s, " could not be found.");
        return false;
      }
    }
  }

  bool _out() {
    if (loc.slot.has_value()) {
      loc.slot = {};
      return true;
    } else {
      if (loc.key.has_value()) {
        loc.key = {};
        return true;
      } else {
        return false;
      }
    }
  }

  bool handle_command(const auto &words) {
    const auto &command = words[0];

    if (command == "show") {
      show();
      return true;
    } else if (command == "out") {
      _out();
      return true;
    }

    if (words.size() < 2) {
      return false;
    }
    const auto &first = words[1];

    if (command == "add") {
      return add(first);
    } else if (command == "rm") {
      return rm(first);
    } else if (command == "in") {
      return in(words[1]);
    } else if (command == "set") {
      return set_mon({words.begin() + 1, words.size() - 1});
    }

    log(err, "sides: Invalid command.");
    return false;
  }

  std::string prompt_data{};

  const char *prompt() noexcept {

    prompt_data = "Sides";

    if (loc.key.has_value()) {
      prompt_data += "/";
      prompt_data += loc.key.value();
      if (loc.slot.has_value()) {
        prompt_data += "/";
        prompt_data += std::to_string(static_cast<size_t>(loc.slot.value()));
      }
    }
    prompt_data += " $ ";
    return prompt_data.data();
  }
};

struct Games {

  Games(std::ostream *out, std::ostream *err) : out{out}, err{err} {}

  std::ostream *out;
  std::ostream *err;

  struct Game {
    // std::mutex mutex;
    struct State {
      // std::mutex mutex;
    };

    std::vector<State> states{};
  };

  std::map<std::string, Game> games;

  bool handle_command(const auto &words) { return false; }
};

struct Threads {
  struct Thread {
    std::thread thread;
    std::optional<std::string> locked_key;
    std::optional<size_t> locked_index;
    std::string name;

    Thread(Thread &&) = default;
    Thread(const Thread &) = delete;
  };
  std::map<size_t, Thread> threads;

  bool kill(size_t key) {}
};

struct Program {

  std::ostream *out;
  std::ostream *err;

  Sides sides;
  Games games;
  Threads threads;

  enum class Location {
    Sides,
    Games,
    // unused
    Threads,
    Models, // ovo_dicts
  };
  Location loc;

  Program(std::ostream *out, std::ostream *err)
      : sides{out, err}, games{out, err}, loc{} {}

  void set_loc(Location l) { loc = l; }

  const char *prompt() noexcept {
    static const char s[]{"Sides$ "};
    static const char g[]{"Games$ "};
    static const char t[]{"Threads$ "};
    static const char m[]{"Models$ "};

    switch (loc) {
    case Location::Sides:
      return sides.prompt();
    case Location::Games:
      return g;
    case Location::Threads:
      return t;
    case Location::Models:
      return m;
    default:
      return nullptr;
    }
  }

  bool handle_command(const auto &words) {
    const auto &command = words[0];

    if (command == "sides") {
      loc = Location::Sides;
      return true;
    } else if (command == "games") {
      loc = Location::Games;
      return true;
    } else if (command == "clear") {
      std::system("clear");
      return true;
    }
    switch (loc) {
    case Location::Sides:
      return sides.handle_command(words);
    case Location::Games:
      return games.handle_command(words);
    default:
      return false;
    }
  }
};