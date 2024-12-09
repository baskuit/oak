#include <lab.h>

#include <getopt.h>
#include <iostream>

#include <readline/history.h>
#include <readline/readline.h>

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
  Lab::Program program{&std::cout, &std::cerr};

  std::vector<std::string> lines{};

  // lines.push_back("add tauros");
  // lines.push_back("cd tauros 1");
  // lines.push_back("set tauros bodyslam scratch earthq strength");
  // lines.push_back("hp 9");
  // lines.push_back("status slp 6");

  // lines.push_back("cd .. ..");
  // lines.push_back("add snorlax");
  // lines.push_back("cd snorlax 1");
  // lines.push_back("set snorlax sonicb");
  // lines.push_back("hp 1");

  lines.push_back("add abr-53; cd abr-53");
  lines.push_back("cd 1; set tauros bodys hyperb earthq blizz; hp 48; cd ..");
  lines.push_back(
      "cd 2; set jolt thunderb thunderw doublek pinm; status slp 1; cd ..");
  lines.push_back("cd 3; set chans sing softb seismic iceb; hp 82; cd ..");
  lines.push_back("cd 4; set snorlax bodys reflec rest iceb; cd ..");
  lines.push_back("cd ..; add uoc-54; cd uoc-54");
  lines.push_back("cd 1; set snor bodys reflec rest iceb; status rest 2;cd ..");
  lines.push_back("cd 2; set jynx love blizz psychic rest; hp 22; cd ..");
  lines.push_back("cd 3; set rhydon earthq subs bodysl; hp 89; cd ..");

  lines.push_back("games");
  lines.push_back("create a abr-53 uoc-54");
  // lines.push_back("create a 0 1");

  lines.push_back("cd a");
  lines.push_back("update 0 0; cd 1 0");
  // lines.push_back("search mc n 1000");
  // lines.push_back("ls");

  for (const auto line : lines) {

    const auto commands = parse_line(line.data());
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
  }

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

int loop(int argc, char *argv[]) {

  Lab::Program program{&std::cout, &std::cerr};

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

int lab(int argc, char *argv[]) {
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

int main(int argc, char *argv[]) {
  // return lab(argc, argv);
  return debug(argc, argv);
}