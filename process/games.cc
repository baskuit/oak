#include <games.h>

namespace Process {
namespace Games {

std::string Program::prompt() const noexcept {
  std::string r{};
  switch (depth()) {
  case 4:
    r = "/4" + r;
  case 3:
    r = "/3" + r;
  case 2:
    r = "/2" + r;
  case 1:
    r = "/1" + r;
  case 0:
    r = "/0" + r;
  default:
    r = "games" + r;
    r += "$ ";
  }
  return r;
}

bool Program::handle_command(
    const std::span<const std::string> words) noexcept {
  if (words.empty()) {
    return false;
  }
  const auto &command = words.front();
  if (command == "print" || command == "p") {
    print();
    return true;
  }
  err("games: command '", command, "' not recognized");
  return false;
}

bool Program::save(std::filesystem::path path) noexcept { return false; }
bool Program::load(std::filesystem::path path) noexcept { return false; }

void Program::print() const noexcept {
  switch (depth()) {
  case 0:
    log(data.histories.size(), " games:");
    for (const auto &[key, value] : data.histories) {
      log(key);
    }
    return;
  case 1:
    log("TODO print game");
    return;
  case 2:
    log("TODO print state");
    return;
  case 3:
    log("TODO print node");
    return;
  case 4:
    log("TODO print output");
    return;
  }
}

} // namespace Games
} // namespace Process