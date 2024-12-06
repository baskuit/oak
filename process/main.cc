#include <games.h>
#include <process.h>
#include <sides.h>

#include <battle/sample-teams.h>

#include <cstdlib>
#include <getopt.h>
#include <iostream>

#include <readline/history.h>
#include <readline/readline.h>

namespace Process {

struct ManagementData {
  enum class Focus {
    S,
    G,
  };
};

class Program : public ProgramBase<false, false> {
  using Base = ProgramBase<false, false>;

  Process::Sides::Program sides_process;
  Process::Games::Program games_process;

  ManagementData::Focus focus{};

public:
  Program(std::ostream *out, std::ostream *err)
      : Base{out, err}, sides_process{out, err}, games_process{out, err} {}

  std::string prompt() const noexcept {
    switch (focus) {
    case ManagementData::Focus::S:
      return sides_process.prompt();
    case ManagementData::Focus::G:
      return games_process.prompt();
    default:
      return " $ ";
    }
  }

  bool handle_command(const std::span<const std::string> words) noexcept {
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

  bool save(std::filesystem::path) noexcept { return false; }

  bool load(std::filesystem::path) noexcept { return false; }

  bool create_history(const std::string key, const std::string p1_key,
                      const std::string p2_key, const uint64_t seed = 0) {
    if (games_process.data.histories.contains(key)) {
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

} // namespace Process

std::string trim(const std::string &str) {
  auto start = str.find_first_not_of(" ");
  if (start == std::string::npos)
    return ""; // All spaces
  auto end = str.find_last_not_of(" ");
  return str.substr(start, end - start + 1);
}

std::vector<std::vector<std::string>> parse_line(const char *data) {
  std::stringstream ss{data};

  std::vector<std::vector<std::string>> commands{};
  for (std::string line; std::getline(ss, line, ';');) {
    line = trim(line);
    if (!line.empty()) {
      std::vector<std::string> command{};
      std::istringstream stream(line);
      for (std::string word; stream >> word;) {
        command.push_back(word);
      }
      if (!command.empty()) {
        commands.push_back(command);
      }
    }
  }
  return commands;
}

int debug(int argc, char *argv[]) {
  Process::Program program{&std::cout, &std::cerr};
  std::vector<std::string> com0{"create", "123", "0", "1"};
  program.handle_command(com0);
  // program.handle_command(com1);
  return 0;
}

int loop(int argc, char *argv[]) {

  Process::Program program{&std::cout, &std::cerr};

  while (const auto input =
             std::unique_ptr<const char>(readline(program.prompt().data()))) {

    const auto commands = parse_line(input.get());
    bool commands_succeeded = true;

    for (const auto &command : commands) {
      bool command_succeeded;
      try {
        command_succeeded =
            program.handle_command({command.data(), command.size()});
        commands_succeeded &= command_succeeded;
      } catch (const std::exception &e) {
        std::cerr << "Uncaught exception in handle_commands: " << e.what()
                  << std::endl;
        commands_succeeded = false;
      }
    }

    if (input.get()) {
      add_history(input.get());
    }
  }

  return 0;
}
void print_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name << " [options]\n"
            << "Options:\n"
            << "  -h, --help       Show this help message\n"
            << "  -v, --version    Print version information\n";
}

int main(int argc, char *argv[]) {
  const char *const short_opts = "hv";
  const option long_opts[] = {{"help", no_argument, nullptr, 'h'},
                              {"version", no_argument, nullptr, 'v'},
                              {nullptr, 0, nullptr, 0}};

  while (true) {
    const int opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
    if (opt == -1)
      break;
    switch (opt) {
    case 'h':
      print_usage(argv[0]);
      return 0;
    case 'v':
      std::cout << "Version 1.0.0\n";
      return 0;
    case '?': // Unrecognized option
    default:
      print_usage(argv[0]);
      return 1;
    }
  }

  loop(argc, argv);
  // return debug(argc, argv);

  return 0;
}