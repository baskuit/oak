#pragma once

#include <bit>
#include <filesystem>
#include <fstream>
#include <map>

namespace FS {

template <typename T, template <typename...> typename Container>
void write_container(std::fstream &file, const Container<T> &container) {
  uint64_t s = container.size();
  file.write(std::bit_cast<const char *>(&s), 8);
  for (const auto &x : container) {
    file.write(std::bit_cast<const char *>(&x), sizeof(T));
  }
}

template <typename T, template <typename...> typename Container>
bool read_container(std::fstream &file, Container<T> &container) {
  uint64_t s;
  file.read(std::bit_cast<char *>(&s), 8);
  if (const auto g = file.gcount(); g != 8) {
    return false;
  }
  container.resize(s);
  file.read(container.data(), s * sizeof(T));
  if (const auto g = file.gcount(); g != s * sizeof(T)) {
    return false;
  }
}

template <typename T> bool try_read(std::fstream &file, T &t) {
  file.read(std::bit_cast<char *>(&t), sizeof(T));
  if (const auto g = file.gcount(); g != sizeof(T)) {
    return false;
  }
  return true;
}

template <typename Key, typename Value, template <typename...> typename Map>
bool load(const std::filesystem::path path, Map<Key, Value> &map) {
  std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  file.seekg(0, std::ios::beg);
  while (file.peek() != EOF) {
    Key key;
    size_t s;
    Value value;

    file.read(std::bit_cast<char *>(&s), sizeof(size_t));
    if (const auto g = file.gcount(); g != sizeof(size_t)) {
      return false;
    }
    std::vector<char> buffer{};
    buffer.resize(s);
    file.read(buffer.data(), s * sizeof(typename Key::value_type));
    if (const auto g = file.gcount(); g != s) {
      return false;
    }
    key = Key{buffer.data(), s};
    // file.seekg(0, std::ios::cur);

    file.read(std::bit_cast<char *>(&value), sizeof(Value));
    if (const auto g = file.gcount(); g != sizeof(Value)) {
      return false;
    }
    map[key] = value;
  }

  file.close();
  return true;
}

} // namespace FS