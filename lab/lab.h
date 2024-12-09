#pragma once

#include <games.h>
#include <process.h>
#include <sides.h>
#include <threads.h>

namespace Lab {

struct ManagementData {
  enum class Focus {
    S,
    G,
  };
};

class Program : public ProgramBase<false, false> {
public:
  using Base = ProgramBase<false, false>;

  Lab::Sides::Program sides_process;
  Lab::Games::Program games_process;

  ManagementData::Focus focus{};

  ThreadManager thread_manager;

public:
  Program(std::ostream *out, std::ostream *err)
      : Base{out, err}, sides_process{out, err}, games_process{out, err},
        focus{}, thread_manager{} {}

  std::string prompt() const {
    switch (focus) {
    case ManagementData::Focus::S:
      return sides_process.prompt();
    case ManagementData::Focus::G:
      return games_process.prompt();
    default:
      return " $ ";
    }
  }

  bool handle_command(const std::span<const std::string> words) {
    if (words.empty()) {
      return false;
    }

    const auto &command = words.front();
    if (command == "sides") {
      focus = ManagementData::Focus::S;
      return true;
    } else if (command == "games") {
      focus = ManagementData::Focus::G;
      return true;
    } else if (command == "clear") {
      std::system("clear");
      return true;
    } else if (command == "help" || command == "h") {
      if (words.size() > 1) {
        log("Commands");
        log("sides: Switch to Sides context; create teams and battle states.");
        log("games: Switch to Games context; play out and analyze battles.");
        log("clear: Clear terminal.");
        return true;
      }
    }

    if (command == "create") {
      if (words.size() < 4) {
        err("create: Invalid args.");
        return false;
      }
      uint64_t seed = 0x123456;
      if (words.size() == 5) {
      }
      return create_history(words[1], words[2], words[3]);
    }

    // if (command == "detatch") {}

    switch (focus) {
    case ManagementData::Focus::S:
      return sides_process.handle_command(words);
    case ManagementData::Focus::G:
      return games_process.handle_command(words);
    default:
      err("Invalid focus.");
      return false;
    }
  }

  bool save(std::filesystem::path) { return false; }

  bool load(std::filesystem::path) { return false; }

  bool create_history(const std::string key, const std::string p1_key,
                      const std::string p2_key, const uint64_t seed = 0) {
    if (games_process.data.history_map.contains(key)) {
      err("create: key '", key, "' already present.");
      return false;
    }
    const auto &sides = sides_process.data.sides;
    if (!sides.contains(p1_key)) {
      err("create: p1 key '", p1_key, "' not found in sides/.");
      return false;
    }
    if (!sides.contains(p2_key)) {
      err("create: p2 key '", p2_key, "' not found sides/.");
      return false;
    }
    auto p1 = sides.at(p1_key);
    auto p2 = sides.at(p2_key);
    return games_process.create(key, p1, p2);
  }
};

} // namespace Lab
