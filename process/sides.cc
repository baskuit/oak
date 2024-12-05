#include <sides.h>

namespace Process {
namespace Sides {

std::string Program::prompt() const noexcept {
  std::string p{" sides"};
  if (mgmt.cli_key.has_value()) {
    p += "/" + mgmt.cli_key.value();
  }
  if (mgmt.cli_slot.has_value()) {
    const auto slot = mgmt.cli_slot.value();
    p += "/" + (slot == 0 ? "Active" : std::to_string(slot));
  }
  p += "$ ";
  return p;
}

bool Program::handle_command(
    const std::span<const std::string> words) noexcept {
  if (words.size() == 0) {
    return false;
  }
  const auto &command = words[0];
  if (command == "show" || command == "print" || command == "p") {
  }
  err("Sides: command not recognized");
  return false;
}
} // namespace Sides
} // namespace Process