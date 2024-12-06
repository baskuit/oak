#pragma once

#include <filesystem>
#include <ostream>
#include <span>
#include <string>

namespace Process {

template <bool multi_threaded = false, bool filesystem = false>
class ProgramBase {
  std::ostream *mOut;
  std::ostream *mErr;

public:
  ProgramBase(std::ostream *out, std::ostream *err) : mOut{out}, mErr{err} {}

  virtual std::string prompt() const noexcept = 0;
  virtual bool handle_command(const std::span<const std::string>) noexcept = 0;

  virtual bool save(std::filesystem::path) noexcept = 0;
  virtual bool load(std::filesystem::path) noexcept = 0;

  template <typename... Args> void log(const Args &...messages) const {
    ((*mOut << messages << " "), ...) << std::endl;
  }

  template <typename... Args> void err(const Args &...messages) const {
    ((*mErr << messages << " "), ...) << std::endl;
  }

  template <typename... Args> void log_(const Args &...messages) const {
    ((*mOut << messages << " "), ...);
  }

  template <typename... Args> void err_(const Args &...messages) const {
    ((*mErr << messages << " "), ...);
  }
};

} // namespace Process