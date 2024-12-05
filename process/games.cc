#include <games.h>

namespace Process {
namespace Games {

std::string Program::prompt() const noexcept {
  std::string p{"games"};
  p += "$ ";
  return p;
}

bool Program::handle_command(
    const std::span<const std::string> words) noexcept {
  if (words.empty()) {
    return false;
  }
  const auto &command = words.front();
  err("Sides: command '", command, "' not recognized");
  return false;
}

bool Program::save(std::filesystem::path path) noexcept {
  return false;
}
bool Program::load(std::filesystem::path path) noexcept {
  return false;
}

}
}