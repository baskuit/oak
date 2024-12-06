#pragma once

#include <bit>
#include <filesystem>
#include <fstream>
#include <map>

namespace FS {

template <typename Key, typename Value, template <typename...> typename Map>
bool save(const std::filesystem::path path, const Map<Key, Value> &map,
          const bool overwrite = true) {
  const auto mode =
      overwrite ? std::ios::binary : std::ios::binary | std::ios::trunc;
  std::ofstream file(path, mode);
  if (!file.is_open()) {
    return false;
  }

  size_t n = 0;
  size_t s;
  for (const auto &[key, value] : map) {
    s = key.size();
    file.write(std::bit_cast<const char *>(&s), sizeof(size_t));
    file.write(std::bit_cast<const char *>(key.data()), sizeof(typename Key::value_type) * s);
    file.write(std::bit_cast<const char *>(&value), sizeof(value));
    ++n;
  }

  size_t expected_size = n * (sizeof(Key) + sizeof(Value));
  // assert()

  file.close();
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