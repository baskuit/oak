#include <process.h>
#include <sides.h>
#include <games.h>

#include <cstdlib>
#include <iostream>
#include <getopt.h>

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
};

} // namespace Process

std::string trim(const std::string &str) {
  auto start = str.find_first_not_of(" ");
  if (start == std::string::npos)
    return ""; // All spaces
  auto end = str.find_last_not_of(" ");
  return str.substr(start, end - start + 1);
}

int loop(int argc, char *argv[]) {

  Process::Program program{&std::cout, &std::cerr};

  while (const auto input =
             std::unique_ptr<const char>(readline(program.prompt().data()))) {

    std::vector<std::string> commands{};
    std::stringstream ss{input.get()};

    bool commands_succeeded = true;

    for (std::string line; std::getline(ss, line, ';');) {
      line = trim(line);
      if (line.empty())
        continue;
      commands.push_back(line);

      std::vector<std::string> words{};
      std::istringstream stream(line);
      for (std::string word; stream >> word;) {
        words.push_back(word);
      }

      bool command_succeeded;

      if (!words.empty()) {
        try {
          command_succeeded =
              program.handle_command({words.data(), words.size()});
          commands_succeeded &= command_succeeded;
        } catch (const std::exception &e) {
          std::cerr << "Uncaught exception in handle_commands: " << e.what()
                    << std::endl;
          commands_succeeded = false;
        }
      } else {
        // no error message, just re prompt
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

  return 0;
}