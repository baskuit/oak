#include <cstdlib>
#include <getopt.h>
#include <iostream>

#include <readline/history.h>
#include <readline/readline.h>

#include <cli.h>


std::string trim(const std::string& str) {
    auto start = str.find_first_not_of(" ");
    if (start == std::string::npos) return ""; // All spaces
    auto end = str.find_last_not_of(" ");
    return str.substr(start, end - start + 1);
}

int loop(int argc, char* argv[]) {
    Program program{&std::cout, &std::cerr};

    while (const auto input =
               std::unique_ptr<const char>(readline(program.prompt()))) {

        std::vector<std::string> commands{};
        std::stringstream ss{input.get()};
        for (std::string line; std::getline(ss, line, ';');) {
            line = trim(line); // Trim the command
            if (line.empty()) continue; // Skip empty commands
            commands.push_back(line);

            std::vector<std::string> words{};
            std::istringstream stream(line);
            for (std::string word; stream >> word;) { // Use stream >> word
                words.push_back(word);
                std::cout << word << "-";
            }
            std::cout << std::endl;
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