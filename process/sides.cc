#include <sides.h>

#include <util/fs.h>

namespace Process {
namespace Sides {

std::string Program::prompt() const noexcept {
  std::string p{"sides"};
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
  if (words.empty()) {
    return false;
  }
  const auto &command = words.front();
  if (command == "print" || command == "p") {
    print();
  } else if (command == "save") {
    if (words.size() < 2) {
      err("save: No path.");
      return false;
    }
    std::filesystem::path path = words[1];
    const auto success = save(path);
    if (!success) {
      err("save: Failed to save.");
    } else {
      log("save: Successfully saved to ", path.string(), ".");
    }
    return success;
  } else if (command == "load") {
    if (words.size() < 2) {
      err("load: No path.");
      return false;
    }
    std::filesystem::path path = words[1];
    const auto success = load(path);
    if (!success) {
      err("load: Failed to load.");
    } else {
      log("load: Successfully loaded from ", path.string(), ".");
    }
    return success;
  }
  err("Sides: command not recognized");
  return false;
}

bool Program::save(std::filesystem::path path) noexcept {
  return FS::save(path, data.sides);
}
bool Program::load(std::filesystem::path path) noexcept {
  return FS::load(path, data.sides);
}

bool Program::add(std::string key) noexcept {
  if (data.sides.contains(key)) {
    err("add: ", key, " already present.");
    return false;
  }
  return false;
}

} // namespace Sides
} // namespace Process