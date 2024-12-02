#include <cstdlib>
#include <getopt.h>
#include <iostream>

#include <readline/history.h>
#include <readline/readline.h>

#include <cli.h>

void print_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name << " [options]\n"
            << "Options:\n"
            << "  -h, --help       Show this help message\n"
            << "  -v, --version    Print version information\n";
}

int loop(int argc, char *argv[]) {
  const char *prompt = "";

  while (true) {
    char *input = readline(prompt);
    //  ctrl + d
    if (!input)
      break;

    if (*input)
      add_history(input);

    std::cout << input << std::endl;

    free(input);
  }

  return 0;
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

// int main

// int main_loop(int argc, char **argv) {

//   Program data{};

//   while (true) {

//     std::string message;

//     std::getline(std::cin, message);
//     std::vector<std::string> split{};
//     std::stringstream ss{message};
//     for (std::string item; std::getline(ss, item, ';');
//     split.push_back(item)) {
//     }

//     for (const auto &w : split) {
//       std::cout << "w: " << w << std::endl;
//       Data::Moves x;
//       try {
//         x = Strings::string_to_move(w);
//       } catch (const std::exception &e) {
//         std::cout << e.what() << std::endl;
//         continue;
//       }

//       std::cout << (int)x << std::endl;
//     }
//   }
// }

// int main(int argc, char **argv) { return main_loop(argc, argv); }